# sql_agent.py
"""
SQL Agent with visualization capabilities using LangGraph.
Handles SQL queries, maintains conversation context, and generates tabular and visual outputs with synchronous status updates.
"""
from datetime import datetime
import os
import sys
import json
import time
import ast
import re
from typing import Annotated, List, Dict, Callable
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
from backend.utils.viz_utils import choose_viz_type, format_data_for_viz, format_tables, build_chart_config
import pandas as pd

# Configure logging
logging.getLogger().handlers = []
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("sql_agent.log", encoding="utf-8"),
        logging.FileHandler("error.log", encoding="utf-8", mode="a"),  # Dedicated error log
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
    """State definition for the LangGraph workflow, aligned with State.py."""
    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    sql_result: List[Dict]
    viz_type: str
    viz_data: Dict
    tables: List[Dict]
    tool_history: List[Dict]
    status_messages: List[str]

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
8. Structure the final response as a natural language summary.
9. NEVER return a SQL query as text; ALWAYS use sql_db_query to execute it.
10. DO NOT use DML statements (INSERT, UPDATE, DELETE, DROP).
Example:
Question: Which country's customers spent the most?
Thought: I should look at the tables in the database to see what I can query. Then I should query the schema of the relevant tables.
Action: sql_db_list_tables
Observation: Albums, Artists, Customers, Employees, Genres, InvoiceLines, Invoices, MediaTypes, Playlists, PlaylistTrack, Tracks
Thought: To find out which country's customers spent the most, I'll need to query data from the Customers and Invoices tables.
Action: sql_db_schema
Args: Customers, Invoices
Observation: schema details...
Thought: Now I can construct my query.
Action: sql_db_query_checker
Args: SELECT c.Country, SUM(i.Total) AS TotalSpent FROM Customers c JOIN Invoices i ON c.CustomerId = i.CustomerId GROUP BY c.Country ORDER BY TotalSpent DESC LIMIT 5
Observation: The query looks good.
Action: sql_db_query
Args: SELECT c.Country, SUM(i.Total) AS TotalSpent FROM Customers c JOIN Invoices i ON c.CustomerId = i.CustomerId GROUP BY c.Country ORDER BY TotalSpent DESC LIMIT 5
Observation: [('USA', 523.06), ('Canada', 303.96), ('France', 195.10), ('Brazil', 190.10), ('Germany', 156.48)]
Final Answer: The country whose customers spent the most is the USA, with a total of $523.06."""

def agent_node(state: State, tools: List, status_callback: Callable[[str], None] = None) -> Dict:
    """Agent node: Decides on tool calls or final response, updates status via callback."""
    try:
        current_date = datetime.now().strftime("%Y-%m-%d")
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt.format(current_date=current_date)),
            MessagesPlaceholder(variable_name="messages")
        ])
        agent = prompt | llm.bind_tools(tools)
        result = agent.invoke(state["messages"])
        logger.debug(f"Agent node output: {result}")
        if status_callback:
            status_callback("Agent processed query.")
        return {"messages": [result], "status_messages": state["status_messages"] + ["Agent processed query."]}
    except Exception as e:
        logger.error(f"Agent node failed: {traceback.format_exc()}")
        if status_callback:
            status_callback("Error in agent decision.")
        return {
            "messages": [AIMessage(content="抱歉，代理决策错误，请稍后重试。")],
            "status_messages": state["status_messages"] + ["Error in agent decision."]
        }

def parse_select_columns(query: str) -> List[str]:
    """Parse column names or aliases from the SELECT clause of a SQL query."""
    try:
        match = re.search(r'SELECT\s+(.+?)\s+FROM', query, re.IGNORECASE | re.DOTALL)
        if not match:
            logger.warning("Could not parse SELECT clause")
            return []
        select_clause = match.group(1)
        columns = []
        current_col = ""
        paren_count = 0
        for char in select_clause:
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
            elif char == ',' and paren_count == 0:
                columns.append(current_col.strip())
                current_col = ""
                continue
            current_col += char
        if current_col.strip():
            columns.append(current_col.strip())
        
        parsed_columns = []
        for col in columns:
            alias_match = re.search(r'\bAS\s+([`"\']?)(.*?)\1\s*$', col, re.IGNORECASE)
            if alias_match:
                parsed_columns.append(alias_match.group(2))
            else:
                col_clean = re.sub(r'[`"\']', '', col.split()[-1])
                parsed_columns.append(col_clean)
        logger.debug(f"Parsed columns: {parsed_columns}")
        return parsed_columns
    except Exception as e:
        logger.error(f"Failed to parse SELECT columns: {traceback.format_exc()}")
        return []

def tool_node(state: State, tools: List, status_callback: Callable[[str], None] = None) -> Dict:
    """Tool node: Executes tools and updates state with sql_result, updates status via callback."""
    try:
        last_message = state["messages"][-1]
        tool_call = last_message.tool_calls[0]
        tool_name = tool_call["name"]
        tool_input = tool_call["args"]
        tool = next((t for t in tools if t.name == tool_name), None)
        if not tool:
            raise ValueError(f"Tool {tool_name} not found")
        
        if status_callback:
            status_callback(f"Executing tool: {tool_name}")
        logger.debug(f"Executing tool: {tool_name} with input: {tool_input}")
        result = tool.invoke(tool_input)
        
        tool_history_entry = {"tool": tool_name, "input": tool_input, "output": result}
        state["tool_history"] = state.get("tool_history", []) + [tool_history_entry]
        
        sql_result = state.get("sql_result", [])
        if tool_name == "sql_db_query":
            try:
                if isinstance(result, list):
                    if all(isinstance(row, dict) for row in result):
                        sql_result = result
                    else:
                        columns = parse_select_columns(tool_input.get("query", ""))
                        if not columns:
                            columns = ["column_" + str(i) for i in range(len(result[0]))] if result else []
                        sql_result = [
                            dict(zip(columns, (val if val is not None else "Unknown" for val in row)))
                            for row in result
                        ]
                elif isinstance(result, str):
                    try:
                        parsed_result = ast.literal_eval(result)
                        if isinstance(parsed_result, list) and all(isinstance(row, tuple) for row in parsed_result):
                            columns = parse_select_columns(tool_input.get("query", ""))
                            if not columns:
                                columns = ["model", "count", "percentage"]
                            sql_result = [
                                dict(zip(columns, (val if val is not None else "Unknown" for val in row)))
                                for row in parsed_result
                            ]
                        else:
                            logger.warning(f"Parsed result is not a list of tuples: {type(parsed_result)}")
                            sql_result = []
                    except (ValueError, SyntaxError) as e:
                        logger.error(f"Failed to parse string result: {result}, error: {e}")
                        sql_result = []
                else:
                    logger.warning(f"Unexpected SQL result format: {type(result)}")
                    sql_result = []
                logger.debug(f"Stored sql_result: {sql_result[:3]}...")
            except Exception as e:
                logger.error(f"Failed to parse SQL result: {traceback.format_exc()}")
                sql_result = []

        if status_callback:
            status_callback(f"Tool {tool_name} executed successfully.")
        return {
            "messages": [ToolMessage(content=str(result), tool_call_id=tool_call["id"], name=tool_name)],
            "tool_history": state["tool_history"],
            "sql_result": sql_result,
            "status_messages": state["status_messages"] + [f"Tool {tool_name} executed successfully."]
        }
    except Exception as e:
        logger.error(f"Tool node failed: {traceback.format_exc()}")
        if status_callback:
            status_callback("Error in tool execution.")
        return {
            "messages": [ToolMessage(content="Tool error.", tool_call_id=last_message.tool_calls[0]["id"], name=last_message.tool_calls[0]["name"])],
            "sql_result": state.get("sql_result", []),
            "status_messages": state["status_messages"] + ["Error in tool execution."]
        }

def viz_node(state: State, status_callback: Callable[[str], None] = None) -> Dict:
    """Visualization node: Chooses viz type, formats data and tables with LLM, and builds chart config."""
    try:
        if status_callback:
            status_callback("Generating visualization and tables...")
        logger.debug(f"viz_node received sql_result: {state.get('sql_result', [])[:3]}")
        if not state.get("sql_result"):
            logger.warning("No SQL result for viz, skipping visualization and table formatting")
            state["viz_type"] = "none"
            state["viz_data"] = {}
            state["tables"] = []
            return state
        
        history = "\n".join([f"{msg.__class__.__name__}: {msg.content}" for msg in state["messages"] if isinstance(msg, (HumanMessage, AIMessage))])
        tool_history = "\n".join([f"Tool {h['tool']}: Input={h['input']}, Output={h['output']}" for h in state.get("tool_history", [])])
        
        state["viz_type"] = choose_viz_type(state["question"], state["sql_result"], history, tool_history)
        formatted_data = format_data_for_viz(state["viz_type"], state["sql_result"], state["question"], history, tool_history)
        state["viz_data"] = build_chart_config(state["viz_type"], formatted_data)
        state["tables"] = format_tables(state["sql_result"], state["question"], history, tool_history)
        
        logger.info(f"Viz generated: type={state['viz_type']}, data={state['viz_data']}, tables={len(state['tables'])}")
        return state
    except Exception as e:
        logger.error(f"Viz node failed: {traceback.format_exc()}")
        state["viz_type"] = "none"
        state["viz_data"] = {}
        state["tables"] = []
        state["status_messages"] = state["status_messages"] + ["Error in visualization and table formatting."]
        return state

def should_continue(state: State) -> str:
    """Conditional edge: Determines next node based on agent response."""
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return "viz"

def filter_context(state: State) -> State:
    """Filter state to include only the latest relevant HumanMessage and AIMessage pair, plus all relevant tool history."""
    try:
        # Collect messages, keeping only the latest HumanMessage and AIMessage pair without tool_calls
        filtered_messages = []
        seen_human = False
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage) and not seen_human:
                filtered_messages.append(msg)
                seen_human = True
            elif isinstance(msg, AIMessage) and not (hasattr(msg, "tool_calls") and msg.tool_calls) and seen_human:
                filtered_messages.append(msg)
                break  # Stop after finding the first valid AIMessage following a HumanMessage
        filtered_messages.reverse()  # Restore original order

        # Keep all relevant tool history entries
        filtered_tool_history = [
            h for h in state.get("tool_history", [])
            if h["tool"] in ["sql_db_list_tables", "sql_db_schema", "sql_db_query", "sql_db_query_checker", "check_result"]
        ]
        logger.debug(f"Filtered messages: {len(filtered_messages)}, Filtered tool history: {len(filtered_tool_history)}")
        return {
            "messages": filtered_messages,
            "question": state["question"],
            "sql_result": [],
            "viz_type": "none",
            "viz_data": {},
            "tables": [],
            "tool_history": filtered_tool_history,
            "status_messages": []
        }
    except Exception as e:
        logger.error(f"Failed to filter context: {traceback.format_exc()}")
        return {
            "messages": state["messages"],
            "question": state["question"],
            "sql_result": [],
            "viz_type": "none",
            "viz_data": {},
            "tables": [],
            "tool_history": state.get("tool_history", []),
            "status_messages": state.get("status_messages", []) + ["Error in context filtering."]
        }

def build_graph(db: SQLDatabase, status_callback: Callable[[str], None] = None) -> tuple:
    """Builds the LangGraph workflow with checkpointing, accepting status_callback."""
    try:
        tools = initialize_tools(db)
        checkpointer = MemorySaver()
        workflow = StateGraph(State)
        workflow.add_node("agent", lambda state: agent_node(state, tools, status_callback))
        workflow.add_node("tools", lambda state: tool_node(state, tools, status_callback))
        workflow.add_node("viz", lambda state: viz_node(state, status_callback))
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "viz": "viz"})
        workflow.add_edge("tools", "agent")
        workflow.add_edge("viz", END)
        compiled_graph = workflow.compile(checkpointer=checkpointer)
        logger.info("LangGraph workflow built successfully")
        return compiled_graph, tools
    except Exception as e:
        logger.error(f"Failed to build graph: {traceback.format_exc()}")
        raise

def process_query(graph, inputs: dict, config: RunnableConfig, status_callback: Callable[[str], None] = None) -> Dict:
    """Processes a single query through the graph and generates Chart.js visualization."""
    try:
        start_time = time.time()
        inputs = filter_context(inputs)  # Filter context for follow-up questions
        response = graph.invoke(inputs, config)
        processing_time = time.time() - start_time
        logger.debug(f"Graph response: {response}")
        
        # Deduplicate AIMessage entries, keeping only the latest non-tool-call AIMessage
        unique_messages = []
        seen_contents = set()
        for msg in response["messages"]:
            if isinstance(msg, AIMessage) and not (hasattr(msg, "tool_calls") and msg.tool_calls):
                if msg.content not in seen_contents:
                    unique_messages.append(msg)
                    seen_contents.add(msg.content)
            else:
                unique_messages.append(msg)
        
        # Select final_message
        final_message = None
        for msg in reversed(unique_messages):
            if isinstance(msg, AIMessage) and not (hasattr(msg, "tool_calls") and msg.tool_calls):
                final_message = msg
                break
        if not final_message:
            logger.warning("No valid AIMessage found in response, using fallback")
            final_message = AIMessage(content="抱歉，未能生成有效回答，请稍后重试或联系支持。")
        
        chart_config = response.get("viz_data")
        filtered_messages = [
            msg for msg in unique_messages
            if isinstance(msg, (HumanMessage, AIMessage)) and not (isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls)
        ]
        return {
            "answer": final_message.content,
            "viz_type": response.get("viz_type", "none"),
            "viz_data": response.get("viz_data", {}),
            "chart_config": chart_config,
            "tables": response.get("tables", []),
            "messages": filtered_messages,
            "processing_time": processing_time,
            "status_messages": response.get("status_messages", []),
            "tool_history": response.get("tool_history", [])
        }
    except Exception as e:
        logger.error(f"Query processing failed: {traceback.format_exc()}")
        return {
            "answer": "抱歉，处理查询时发生错误，请稍后重试或联系支持。",
            "viz_type": "none",
            "viz_data": {},
            "chart_config": None,
            "tables": [],
            "messages": inputs["messages"] + [AIMessage(content="抱歉，处理查询时发生错误，请稍后重试或联系支持。")],
            "processing_time": time.time() - start_time,
            "status_messages": inputs.get("status_messages", []) + ["Error in query processing."],
            "tool_history": inputs.get("tool_history", []),
            "error": str(e)
        }

if __name__ == "__main__":
    try:
        db = SQLDatabase.from_uri("sqlite:///real_database.db")
        graph, _ = build_graph(db)  # Initialize without status_callback for CLI
        print("Welcome to SQL Agent with Viz! Enter your query (type 'exit' or 'quit' to exit).")
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        graph.invoke({"messages": [], "status_messages": [], "sql_result": [], "tool_history": []}, config)
        while True:
            question = input("Your query: ")
            if question.lower() in ["exit", "quit"]:
                print("Exiting program.")
                break
            try:
                inputs = {
                    "messages": [HumanMessage(content=question)],
                    "question": question,
                    "status_messages": [],
                    "sql_result": [],
                    "tool_history": []
                }
                result = process_query(graph, inputs, config, lambda msg: print(f"Status: {msg}"))
                print(f"Answer: {result['answer']}")
                print(f"Processing time: {result['processing_time']:.2f} seconds")
                print(f"Viz Type: {result['viz_type']}, Data: {result['viz_data']}")
                for table in result['tables']:
                    print(f"\n{table['title']}:")
                    print(pd.DataFrame(table['data']).to_string(index=False))
            except Exception as e:
                print("抱歉，处理查询时发生错误，请稍后重试或联系支持。")
                logger.error(f"Query error: {traceback.format_exc()}")
    except Exception as e:
        print("抱歉，初始化时发生错误，请稍后重试或联系支持。")
        logger.error(f"Initialization error: {traceback.format_exc()}")