# backend/sql_agent.py
"""
SQL Agent with visualization capabilities using LangGraph.
Handles SQL queries, maintains conversation context, and generates tabular and visual outputs.
"""
from datetime import datetime
import os
import sys
import ast
import json
from datetime import datetime, date
import time
import re
from typing import Annotated, List, Dict
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import logging
import uuid
import traceback
from pydantic import BaseModel, Field
from backend.utils.viz_utils import choose_viz_type, format_data_for_viz
import pandas as pd

# Configure logging
logging.getLogger().handlers = []
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("sql_agent.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

logger.info(f"Python version: {sys.version}")

# Load environment variables
load_dotenv()
QWEN_API_KEY = os.getenv("QWEN_API_KEY")
if not QWEN_API_KEY:
    raise EnvironmentError("QWEN_API_KEY not set in environment")

QWEN_API_URL = os.getenv("QWEN_API_URL", "http://100.94.4.96:8000/v1")
QWEN_MODEL = os.getenv("QWEN_MODEL", "Qwen3-Coder-480B")
TEMPERATURE = float(os.getenv("QWEN_TEMPERATURE", 0.2))

class State(TypedDict):
    """State definition for the LangGraph workflow."""
    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    sql_result: List[Dict]
    viz_type: str
    viz_data: Dict
    tables: List[Dict]
    tool_history: List[Dict]  # Store tool call inputs/outputs
    status_messages: List[str]  # Store node processing status

class CheckResultInput(BaseModel):
    query_result: str = Field(description="The result of the executed SQL query")

def initialize_llm() -> ChatOpenAI:
    """Initialize the LLM with environment variables."""
    try:
        llm = ChatOpenAI(
            model=QWEN_MODEL,
            api_key=QWEN_API_KEY,
            base_url=QWEN_API_URL,
            temperature=TEMPERATURE
        )
        logger.info("LLM initialized successfully")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {traceback.format_exc()}")
        raise

llm = initialize_llm()

def initialize_tools(db: SQLDatabase) -> List:
    """Initialize SQL tools and custom check_result tool."""
    try:
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        tools = toolkit.get_tools()
        for tool in tools:
            if not tool.__doc__:
                logger.warning(f"Tool {tool.name} lacks docstring, may cause issues")
        
        list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
        get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")
        db_query_tool = next(tool for tool in tools if tool.name == "sql_db_query")
        query_check_tool = next(tool for tool in tools if tool.name == "sql_db_query_checker")
        
        def check_result(query_result: str) -> str:
            """Verify if the SQL query result is empty or irrelevant."""
            try:
                time.sleep(1)
                query_result_check_system = """You are a SQL expert. Check if the query result is empty or irrelevant.
                - If empty or irrelevant, suggest retrying with a different query.
                Result: {query_result}"""
                query_result_check_prompt = ChatPromptTemplate.from_messages([
                    ("system", query_result_check_system),
                    ("user", "{query_result}")
                ])
                query_result_check = query_result_check_prompt | llm
                result = query_result_check.invoke({"query_result": query_result}).content
                logger.debug(f"Check result output: {result}")
                return result
            except Exception as e:
                logger.error(f"Result check failed: {traceback.format_exc()}")
                return "Result check failed, proceed."

        check_result_tool = StructuredTool.from_function(
            func=check_result,
            name="check_result",
            description="Verify if the SQL query result is empty or irrelevant.",
            args_schema=CheckResultInput
        )

        return [list_tables_tool, get_schema_tool, db_query_tool, query_check_tool, check_result_tool]
    except Exception as e:
        logger.error(f"Failed to initialize tools: {traceback.format_exc()}")
        raise

system_prompt = """You are an expert SQL agent interacting with a SQLite database in a conversational manner.
The current date is {current_date}. For time-sensitive queries involving relative dates (e.g., 'near month', 'recent week', 'last year'), translate them into absolute dates using SQLite date functions like date('now', '-1 month','localtime') for 'last month', date('now', '-7 days','localtime') for 'last week', etc. Always use 'now' as the reference for current time in queries.
Maintain context from previous messages, including user questions, assistant responses, and tool outputs (e.g., table lists, schemas, query results).
For each new user question:
1. Check prior messages for known table or schema information to avoid redundant tool calls.
2. If necessary, call sql_db_list_tables to list tables or sql_db_schema for schemas.
3. Generate a syntactically correct SELECT query, selecting only necessary columns, limiting to 5 results unless specified.
4. Call sql_db_query_checker to validate the query.
5. Call sql_db_query to execute the query.
6. Call check_result to verify the result.
7. If the query fails or results are empty, rewrite and retry (max 2 retries).
8. Structure the final response as a natural language summary followed by a tabular representation.
   - Suggest logical sub-table groupings (e.g., by model or vendor) if the result set is large (>5 rows).
   - Format failure rates or percentages (columns with 'rate' or 'failure' in the name) as percentages (e.g., 0.123 -> 12.3%).
9. NEVER return a SQL query as text; ALWAYS use sql_db_query to execute it.
10. DO NOT use DML statements (INSERT, UPDATE, DELETE, DROP).
Example:
Question: Which country's customers spent the most?
Answer: The country whose customers spent the most is [Country], with a total of [TotalSpent].
Suggested Grouping: By Country
Table:
| Country | TotalSpent |
|---------|------------|
| [Country] | [TotalSpent] |
"""

def clean_column_name(col: str) -> str:
    """Clean SQL column names by removing aggregation functions, aliases, and quotes."""
    # Remove outer quotes if present
    col = col.strip().strip('"').strip("'").strip('"')
    
    # Remove SUM(), ROUND(), COUNT(), etc., and extract alias if present
    col = re.sub(r'^(SUM|ROUND|COUNT|AVG|MIN|MAX)\((.*?)\)(\s*AS\s*(.*?))?$', r'\4', col, flags=re.IGNORECASE).strip()
    col = re.sub(r'^(.*?)\s+AS\s+(.*?)$', r'\2', col, flags=re.IGNORECASE).strip()
    
    # Remove any remaining quotes
    col = col.strip().strip('"').strip("'").strip('"')
    
    return col if col else "Unknown"

def format_tables(sql_result: List[Dict], question: str, agent_suggestion: str = "") -> List[Dict]:
    """Format SQL results into sub-tables based on logical groupings or agent suggestion."""
    try:
        if not sql_result:
            return []
        
        # Clean column names
        cleaned_result = []
        for row in sql_result:
            cleaned_row = {clean_column_name(k): v for k, v in row.items()}
            cleaned_result.append(cleaned_row)
        
        df = pd.DataFrame(cleaned_result)
        tables = []
        
        # Identify columns for grouping
        categorical_cols = [col for col in df.columns if df[col].dtype == "object" and col.lower() not in ["rate", "failure"]]
        numeric_cols = [col for col in df.columns if df[col].dtype in ["int64", "float64"]]
        
        # Format numeric columns with 'rate' or 'failure' as percentages
        for col in numeric_cols:
            if "rate" in col.lower() or "failure" in col.lower():
                df[col] = df[col].apply(lambda x: f"{x*100:.1f}%" if isinstance(x, (int, float)) and pd.notnull(x) else x)
        
        # Use agent-suggested grouping if provided
        group_col = None
        if agent_suggestion and any(col.lower() in agent_suggestion.lower() for col in categorical_cols):
            group_col = next(col for col in categorical_cols if col.lower() in agent_suggestion.lower())
        elif categorical_cols and len(sql_result) > 5:
            group_col = categorical_cols[0]
        
        if group_col:
            for name, group in df.groupby(group_col):
                tables.append({
                    "title": f"Results for {group_col}: {name}",
                    "data": group.to_dict(orient="records")
                })
        else:
            tables.append({
                "title": "Query Results",
                "data": df.to_dict(orient="records")
            })
        
        return tables
    except Exception as e:
        logger.error(f"Table formatting failed: {traceback.format_exc()}")
        return [{"title": "Query Results", "data": cleaned_result}]

def agent_node(state: State, tools) -> Dict:
    """Agent node: Invokes the LLM with tools bound."""
    try:
        current_date = date.today()  # 获取当前日期（不含时间） datetime.date(2025, 10, 22)
        current_datetime = datetime.now()  # 获取当前日期时间 datetime.datetime(2025, 10, 22, 18, 55, 14, 139447)
        formatted_system_prompt = system_prompt.format(current_date=current_date)
        state["status_messages"].append("Analyzing question and generating SQL query...")
        prompt = ChatPromptTemplate.from_messages([
            ("system", formatted_system_prompt),
            MessagesPlaceholder("messages"),
        ])
        llm_with_tools = llm.bind_tools(tools)
        chain = prompt | llm_with_tools
        time.sleep(1)
        response = chain.invoke({"messages": state["messages"]})
        logger.debug(f"Agent response: {response}")
        # Extract grouping suggestion from response if present
        grouping_suggestion = ""
        if "Suggested Grouping" in response.content:
            grouping_suggestion = response.content.split("Suggested Grouping: ")[-1].split("\n")[0]
        return {"messages": [response], "grouping_suggestion": grouping_suggestion}
    except Exception as e:
        logger.error(f"Agent node failed: {traceback.format_exc()}")
        return {"messages": [AIMessage(content="Agent error occurred.")], "sql_result": [], "viz_type": "none", "viz_data": {}, "tables": [], "status_messages": state["status_messages"] + ["Error in agent processing."]}

def tool_node(state: State, tools) -> Dict:
    """Tool node: Executes the called tool and handles SQL query results."""
    try:
        messages = state["messages"]
        last_message = messages[-1]
        tool_call = last_message.tool_calls[0]
        tool = next(t for t in tools if t.name == tool_call["name"])
        state["status_messages"].append(f"Executing tool: {tool_call['name']}...")
        time.sleep(1)
        tool_result = tool.invoke(tool_call["args"])
        logger.debug(f"Tool {tool_call['name']} result: {tool_result} (type: {type(tool_result)})")
        sql_result = state.get("sql_result", [])
        # 在 tool_node 中，替换原有处理 sql_db_query 的 try/except 分支为以下实现片段
        if tool_call["name"] == "sql_db_query":
            try:
                # handle different tool_result types robustly
                parsed = None

                # If tool returned already a Python list/dict (some toolkits do), accept directly
                if isinstance(tool_result, (list, dict)):
                    parsed = tool_result
                elif isinstance(tool_result, str):
                    txt = tool_result.strip()
                    # 1. 尝试解析为 JSON
                    try:
                        parsed = json.loads(txt)
                    except Exception:
                        # 2. 尝试 ast.literal_eval（python literal）
                        try:
                            parsed = ast.literal_eval(txt)
                        except Exception:
                            # 3. 尝试从字符串中抽取首个 JSON/array 子串再解析
                            m = re.search(r'(\[.*\])', txt, flags=re.S)
                            if m:
                                try:
                                    parsed = ast.literal_eval(m.group(1))
                                except Exception:
                                    try:
                                        parsed = json.loads(m.group(1))
                                    except Exception:
                                        parsed = None
                            else:
                                m2 = re.search(r'(\{.*\})', txt, flags=re.S)
                                if m2:
                                    try:
                                        parsed = ast.literal_eval(m2.group(1))
                                    except Exception:
                                        try:
                                            parsed = json.loads(m2.group(1))
                                        except Exception:
                                            parsed = None
                else:
                    parsed = None

                # Normalize parsed into a list of dicts
                sql_result = []
                if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                    # already list of dicts
                    sql_result = parsed
                elif isinstance(parsed, list) and parsed and isinstance(parsed[0], (list, tuple)):
                    # list of tuples: need column names to zip
                    tuples = [list(t) for t in parsed]
                    query = tool_call["args"].get("query", "")
                    # extract select part preserving case and aliases
                    select_part = ""
                    if "SELECT" in query.upper() and "FROM" in query.upper():
                        # 保留原 query 的大小写，先找 SELECT 到 FROM
                        try:
                            # 使用分割保留原
                            select_part = re.split(r'\bSELECT\b', query, flags=re.IGNORECASE, maxsplit=1)[1]
                            select_part = re.split(r'\bFROM\b', select_part, flags=re.IGNORECASE, maxsplit=1)[0]
                        except Exception:
                            select_part = ""
                    # 切分列（避免括号内逗号被切分）
                    if select_part:
                        cols = re.split(r',\s*(?![^()]*\))', select_part)
                        cols = [clean_column_name(c.strip()) for c in cols if c.strip()]
                    else:
                        cols = []
                    # If columns count matches tuple length use them; otherwise fallback to generic names
                    for row in tuples:
                        if cols and len(cols) == len(row):
                            sql_result.append(dict(zip(cols, row)))
                        else:
                            # fallback: name columns as col0, col1...
                            sql_result.append({f"col{i}": row[i] for i in range(len(row))})
                elif isinstance(parsed, dict):
                    # single dict -> wrap
                    sql_result = [parsed]
                else:
                    # if parsed is None or empty, try to accept tool_result if it's already structured as Python object
                    if isinstance(tool_result, list):
                        sql_result = tool_result
                    else:
                        sql_result = []

                logger.debug(f"Parsed sql_result: {sql_result}")
            except Exception as e:
                logger.error(f"Failed to parse sql_db_query result: {traceback.format_exc()}")
                sql_result = []
        
        
        # if tool_call["name"] == "sql_db_query":
        #     try:
        #         if isinstance(tool_result, str):
        #             parsed_result = ast.literal_eval(tool_result)
        #             if isinstance(parsed_result, list) and all(isinstance(item, tuple) for item in parsed_result):
        #                 query = tool_call["args"]["query"]
        #                 # Extract column names from the query more robustly
        #                 query_upper = query.upper()
        #                 if "SELECT" in query_upper:
        #                     select_part = query_upper.split("SELECT")[1].split("FROM")[0].strip()
        #                     # Handle quoted column names
        #                     columns = re.findall(r'"[^"]*"|\'[^\']*\'|[^,\s]+', select_part)
        #                     columns = [col.strip().strip('"').strip("'") for col in columns]
        #                 else:
        #                     # Fallback: use the first row's keys if available
        #                     columns = list(sql_result[0].keys()) if sql_result else []
                        
        #                 # Clean column names
        #                 cleaned_columns = [clean_column_name(col) for col in columns]
        #                 sql_result = [dict(zip(cleaned_columns, row)) for row in parsed_result]
        #             else:
        #                 sql_result = parsed_result if isinstance(parsed_result, list) else []
        #         else:
        #             sql_result = tool_result if isinstance(tool_result, list) else []
        #         logger.debug(f"Parsed sql_result: {sql_result}")
        #     except Exception as e:
        #         logger.error(f"Failed to parse sql_db_query result: {traceback.format_exc()}")
        #         sql_result = []
        # Store tool call and result in history
        tool_history = state.get("tool_history", []) + [{"tool": tool_call["name"], "input": tool_call["args"], "output": str(tool_result)}]
        return {
            "messages": [ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"], name=tool_call["name"])],
            "sql_result": sql_result,
            "tool_history": tool_history,
            "status_messages": state["status_messages"]
        }
    except Exception as e:
        logger.error(f"Tool node failed: {traceback.format_exc()}")
        return {
            "messages": [ToolMessage(content="Tool error occurred.", tool_call_id=last_message.tool_calls[0]["id"], name=last_message.tool_calls[0]["name"])],
            "sql_result": [],
            "status_messages": state["status_messages"] + ["Error in tool execution."]
        }

def viz_node(state: State) -> Dict:
    """Visualization node: Determines viz_type and formats data based on question, history, and SQL result."""
    try:
        state["status_messages"].append("Generating visualization...")
        logger.debug(f"viz_node received sql_result: {state.get('sql_result', [])}")
        if not state.get("sql_result"):
            logger.warning("No SQL result for viz, skipping visualization")
            state["viz_type"] = "none"
            state["viz_data"] = {}
            state["tables"] = []
            return state
        
        # Extract conversation and tool history
        history = "\n".join([f"{msg.__class__.__name__}: {msg.content}" for msg in state["messages"] if isinstance(msg, (HumanMessage, AIMessage))])
        tool_history = "\n".join([f"Tool {h['tool']}: Input={h['input']}, Output={h['output']}" for h in state.get("tool_history", [])])
        state["viz_type"] = choose_viz_type(state["question"], state["sql_result"], history, tool_history)
        state["viz_data"] = format_data_for_viz(state["viz_type"], state["sql_result"])
        state["tables"] = format_tables(state["sql_result"], state["question"], state.get("grouping_suggestion", ""))
        logger.info(f"Viz generated: type={state['viz_type']}, data={state['viz_data']}, tables={len(state['tables'])}")
        return state
    except Exception as e:
        logger.error(f"Viz node failed: {traceback.format_exc()}")
        state["viz_type"] = "none"
        state["viz_data"] = {}
        state["tables"] = []
        state["status_messages"] = state["status_messages"] + ["Error in visualization."]
        return state

def should_continue(state: State) -> str:
    """Conditional edge: Determines next node based on agent response."""
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return "viz"

def build_graph(db: SQLDatabase) -> tuple:
    """Builds the LangGraph workflow with checkpointing."""
    try:
        tools = initialize_tools(db)
        checkpointer = MemorySaver()
        workflow = StateGraph(State)
        workflow.add_node("agent", lambda state: agent_node(state, tools))
        workflow.add_node("tools", lambda state: tool_node(state, tools))
        workflow.add_node("viz", viz_node)
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "viz": "viz"})
        workflow.add_edge("tools", "agent")
        workflow.add_edge("viz", END)
        return workflow.compile(checkpointer=checkpointer), tools
    except Exception as e:
        logger.error(f"Failed to build graph: {traceback.format_exc()}")
        raise

def process_query(graph, inputs: dict, config: RunnableConfig, status_placeholder=None) -> Dict:
    """Processes a single query through the graph and generates Chart.js visualization."""
    try:
        start_time = time.time()
        inputs["status_messages"] = []  # Initialize status messages
        response = graph.invoke(inputs, config)
        processing_time = time.time() - start_time
        logger.debug(f"Graph response: {response}")
        
        # Update status with final processing time
        if status_placeholder:
            status_placeholder.markdown(f"Processing complete in {processing_time:.2f} seconds.")
        
        # Generate Chart.js configuration
        chart_config = None
        if response["viz_type"] != "none" and response["viz_data"]:
            chart_config = response["viz_data"]
        
        # Filter messages to include only final AIMessage
        filtered_messages = [
            msg for msg in response["messages"]
            if isinstance(msg, (HumanMessage, AIMessage)) and not (isinstance(msg, AIMessage) and msg.tool_calls)
        ]
        final_message = next((msg for msg in reversed(response["messages"]) if isinstance(msg, AIMessage) and not msg.tool_calls), AIMessage(content="Error: No assistant response"))
        return {
            "answer": final_message.content,
            "viz_type": response.get("viz_type", "none"),
            "viz_data": response.get("viz_data", {}),
            "chart_config": chart_config,
            "tables": response.get("tables", []),
            "messages": filtered_messages,
            "processing_time": processing_time,
            "status_messages": response.get("status_messages", [])
        }
    except Exception as e:
        logger.error(f"Query processing failed: {traceback.format_exc()}")
        if status_placeholder:
            status_placeholder.error(f"Error: {str(e)}")
        return {
            "answer": f"Error occurred during processing: {str(e)}",
            "viz_type": "none",
            "viz_data": {},
            "chart_config": None,
            "tables": [],
            "messages": inputs["messages"] + [AIMessage(content=f"Error: {str(e)}")],
            "processing_time": time.time() - start_time,
            "status_messages": inputs.get("status_messages", []) + ["Error in query processing."]
        }

if __name__ == "__main__":
    try:
        db = SQLDatabase.from_uri("sqlite:///real_database.db")
        graph, _ = build_graph(db)
        print("欢迎使用SQL Agent with Viz！输入您的查询问题（输入 'exit' 或 'quit' 退出）。")
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        graph.invoke({"messages": [], "status_messages": []}, config)
        while True:
            question = input("您的查询: ")
            if question.lower() in ["exit", "quit"]:
                print("退出程序。")
                break
            try:
                inputs = {"messages": [HumanMessage(content=question)], "question": question, "status_messages": []}
                result = process_query(graph, inputs, config)
                print(f"答案: {result['answer']}")
                print(f"Processing time: {result['processing_time']:.2f} seconds")
                print(f"Viz Type: {result['viz_type']}, Data: {result['viz_data']}")
                for table in result['tables']:
                    print(f"\n{table['title']}:")
                    print(pd.DataFrame(table['data']).to_string(index=False))
            except Exception as e:
                print(f"错误: {e}")
                logger.error(f"Query error: {traceback.format_exc()}")
    except Exception as e:
        print(f"初始化错误: {e}")
        logger.error(f"Initialization error: {traceback.format_exc()}")