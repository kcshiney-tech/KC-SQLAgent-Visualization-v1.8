# viz_utils.py
"""
Utilities for choosing and formatting visualization data and tables using LLM, based on SQL results and conversation context.
"""
import ast
from altair import value
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from typing import Dict, List
import logging
import traceback
from dotenv import load_dotenv
import os
import json
import re

from streamlit import context

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("viz_utils.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
QWEN_API_KEY = os.getenv("QWEN_API_KEY")
if not QWEN_API_KEY:
    logger.error("QWEN_API_KEY not set in environment")
    raise EnvironmentError("QWEN_API_KEY not set in environment")

QWEN_API_URL = os.getenv("QWEN_API_URL", "http://100.94.4.96:8000/v1")
QWEN_MODEL = os.getenv("QWEN_MODEL", "Qwen3-Coder-480B")
TEMPERATURE = float(os.getenv("QWEN_TEMPERATURE", 0.2))

def initialize_llm() -> ChatOpenAI:
    """Initialize the LLM with environment variables."""
    try:
        llm = ChatOpenAI(
            model=QWEN_MODEL,
            api_key=QWEN_API_KEY,
            base_url=QWEN_API_URL,
            temperature=TEMPERATURE
        )
        logger.info("LLM initialized successfully for visualization")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {traceback.format_exc()}")
        raise

llm = initialize_llm()

# Graph instructions adapted from reference
graph_instructions = {
    "bar": '''
Where data is: {
  title: string,
  xLabel: string,
  yLabel: string,
  labels: string[],
  values: {data: number[], label: string}[]
}
Examples:
1. data = {
  title: "Average Income by Month",
  xLabel: "Month",
  yLabel: "Income",
  labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
  values: [{data:[21.5, 25.0, 47.5, 64.8, 105.5, 133.2], label: 'Income'}]
}
2. data = {
  title: "Player Performance by Series",
  xLabel: "Series",
  yLabel: "Performance",
  labels: ['series A', 'series B', 'series C'],
  values: [{data:[10, 15, 20], label: 'American'}, {data:[20, 25, 30], label: 'European'}]
}
''',
    "horizontal_bar": '''
Where data is: {
  title: string,
  xLabel: string,
  yLabel: string,
  labels: string[],
  values: {data: number[], label: string}[]
}
Examples:
1. data = {
  title: "Average Income by Month",
  xLabel: "Month",
  yLabel: "Income",
  labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
  values: [{data:[21.5, 25.0, 47.5, 64.8, 105.5, 133.2], label: 'Income'}]
}
2. data = {
  title: "Player Performance by Series",
  xLabel: "Series",
  yLabel: "Performance",
  labels: ['series A', 'series B', 'series C'],
  values: [{data:[10, 15, 20], label: 'American'}, {data:[20, 25, 30], label: 'European'}]
}
''',
    "line": '''
Where data is: {
  title: string,
  xLabel: string,
  yLabel: string,
  xValues: number[] | string[],
  yValues: { data: number[]; label: string }[]
}
Examples:
1. data = {
  title: "Momentum by Mass",
  xLabel: "Mass",
  yLabel: "Momentum",
  xValues: ['2020', '2021', '2022', '2023', '2024'],
  yValues: [{ data: [2, 5.5, 2, 8.5, 1.5], label: 'Momentum'}]
}
2. data = {
  title: "Player Performance by Year",
  xLabel: "Year",
  yLabel: "Performance",
  xValues: ['2020', '2021', '2022', '2023', '2024'],
  yValues: [
    { data: [2, 5.5, 2, 8.5, 1.5], label: 'American' },
    { data: [2, 5.5, 2, 8.5, 1.5], label: 'European' }
  ]
}
''',
    "pie": '''
Where data is: {
  title: string,
  data: { label: string, value: number }[]
}
Example:
data = {
  title: "Market Share Distribution",
  data: [
    { label: 'series A', value: 10 },
    { label: 'series B', value: 15 },
    { label: 'series C', value: 20 }
  ]
}
''',
    "scatter": '''
Where data is: {
  title: string,
  xLabel: string,
  yLabel: string,
  series: { data: { x: number; y: number; id: number }[], label: string }[]
}
Examples:
1. data = {
  title: "Spending vs Quantity by Gender",
  xLabel: "Amount Spent",
  yLabel: "Quantity Bought",
  series: [
    {
      data: [
        { x: 100, y: 200, id: 1 },
        { x: 120, y: 100, id: 2 },
        { x: 170, y: 300, id: 3 }
      ],
      label: 'Men'
    },
    {
      data: [
        { x: 300, y: 300, id: 1 },
        { x: 400, y: 500, id: 2 },
        { x: 200, y: 700, id: 3 }
      ],
      label: 'Women'
    }
  ]
}
2. data = {
  title: "Height vs Weight of Players",
  xLabel: "Height",
  yLabel: "Weight",
  series: [
    {
      data: [
        { x: 180, y: 80, id: 1 },
        { x: 170, y: 70, id: 2 },
        { x: 160, y: 60, id: 3 }
      ],
      label: 'Players'
    }
  ]
}
''',
    "hierarchical_bar": '''
Where data is: {
  title: string,
  xLabel: string,
  yLabel: string,
  labels: string[],  // hierarchical labels like 'main.sub'
  values: {data: number[], label: string}[]
}
Examples:
1. data = {
  title: "Sales by Region and City",
  xLabel: "Location",
  yLabel: "Sales",
  labels: ['North.A', 'North.B', 'South.C', 'South.D'],
  values: [{data:[10, 20, 30, 40], label: 'Q1'}]
}
2. data = {
  title: "Faults by Model and Vendor",
  xLabel: "Model.Vendor",
  yLabel: "Count",
  labels: ['Model1.Vendor1', 'Model1.Vendor2', 'Model2.Vendor1'],
  values: [{data:[5, 3, 8], label: 'Date1'}, {data:[4, 6, 2], label: 'Date2'}]
}
'''
}

def _extract_json_substring(s: str) -> str:
    """Extract the complete JSON object or array from text, handling code blocks."""
    if not s or not isinstance(s, str):
        logger.debug("Empty or invalid input string for JSON extraction")
        return ""
    s_clean = re.sub(r"```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s_clean = re.sub(r"\s*```", "", s_clean, flags=re.IGNORECASE).strip()
    
    # Find the first complete JSON object or array
    stack = []
    start_idx = -1
    for i, char in enumerate(s_clean):
        if char in '{[':
            if stack and stack[-1] in '{[':
                stack.append(char)
            else:
                stack = [char]
                start_idx = i
        elif char in '}]':
            if stack:
                if (char == '}' and stack[-1] == '{') or (char == ']' and stack[-1] == '['):
                    stack.pop()
                    if not stack:
                        return s_clean[start_idx:i+1]
                else:
                    stack.append(char)
            else:
                continue
    logger.warning(f"No complete JSON found in: {s_clean[:100]}...")
    return ""

def parse_llm_response(content: str) -> Dict:
    """Parse LLM response to extract and validate JSON."""
    try:
        json_sub = _extract_json_substring(content)
        if not json_sub:
            logger.warning("No JSON content found in LLM response")
            return {}
        try:
            parsed = json.loads(json_sub)
            logger.debug(f"Successfully parsed JSON: {parsed}")
            return parsed
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(json_sub)
                if isinstance(parsed, (dict, list)):
                    logger.debug(f"Parsed via ast.literal_eval: {parsed}")
                    return parsed
                logger.warning("Parsed content is not a dictionary or list")
                return {}
            except Exception:
                logger.warning(f"Failed to parse JSON or literal: {json_sub}")
                return {}
    except Exception as e:
        logger.error(f"Failed to parse LLM response: {traceback.format_exc()}")
        return {}

def choose_viz_type(question: str, sql_result: List[Dict], history: str = "", tool_history: str = "") -> str:
    """Choose visualization type using LLM based on question, SQL result, and context."""
    try:
        viz_types = ["bar", "horizontal_bar", "line", "pie", "scatter", "hierarchical_bar", "none"]
        prompt = ChatPromptTemplate.from_template(
            """You are a data visualization expert. Based on:
            - User question: '{question}'
            - Conversation history: '{history}'
            - Tool history (SQL queries and results): '{tool_history}'
            - SQL query result (first 3 rows): {sample_result}
            Detect user intent for visualization:
            - Explicit intent: Keywords like 'chart', 'graph', 'visualize', 'plot', 'show me a', or '饼图' (pie chart).
            - Implicit intent: Queries involving trends ('trend', 'over time'), distributions ('distribution', 'proportion', '占比'), comparisons ('compare', 'vs', 'ranking'), or aggregations ('group by', 'top N').
            - Prioritize 'pie' if the user explicitly requests a pie chart (e.g., '画饼图').
            - Default to 'none' if no visual intent or unsuitable data.
            Select the most appropriate type from: {viz_types}.
            - 'bar': Comparisons across categories (>2 categories), e.g., "Sales by product".
            - 'horizontal_bar': Comparisons with few categories or large disparities, e.g., "Revenue of A vs B".
            - 'line': Trends over time, e.g., "Website visits over the year".
            - 'pie': Proportions, e.g., "Market share distribution" or explicit pie chart requests.
            - 'scatter': Correlations between two numeric variables, e.g., "Height vs weight".
            - 'hierarchical_bar': Hierarchical categories on x-axis, e.g., "faults by model and vendor" or multiple grouping levels.
            - 'none': Single values, empty results, or no visual intent.
            Ensure data structure compatibility (e.g., categorical + numeric for bar/pie, two numerics for scatter).
            Output JSON: {{"viz_type": "chosen_type"}}."""
        )
        
        chain = prompt | llm
        sample_result = json.dumps(sql_result[:3])
        logger.debug(f"Choosing viz type - Question: {question}, History: {history}, Tool History: {tool_history}, Sample Result: {sample_result}")
        response = chain.invoke({"question": question, "history": history, "tool_history": tool_history, "sample_result": sample_result, "viz_types": ", ".join(viz_types)})
        content = response.content if hasattr(response, "content") else str(response)
        parsed = parse_llm_response(content)
        viz_type = parsed.get("viz_type", "none").lower()
        if viz_type not in viz_types:
            logger.warning(f"Invalid viz_type '{viz_type}', defaulting to 'none'")
            viz_type = "none"
        logger.info(f"Chose viz_type: {viz_type}")
        return viz_type
    except Exception as e:
        logger.error(f"Viz type selection failed: {traceback.format_exc()}")
        return "none"

def format_data_for_viz(viz_type: str, sql_result: List[Dict], question: str, history: str = "", tool_history: str = "" , max_retries: int = 2) -> Dict:
    """Format SQL results for selected viz_type using LLM."""
    try:
        if not sql_result or viz_type == "none":
            logger.debug("No SQL result or viz_type is none, returning empty dict")
            return {}
        instructions = graph_instructions.get(viz_type, "")
        if not instructions:
            logger.warning(f"No instructions for viz_type {viz_type}, returning empty")
            return {}
        prompt = ChatPromptTemplate.from_template(
            """You are a data expert formatting SQL results for {viz_type} visualization. Based on:
            - User question: '{question}'
            - Conversation history: '{history}'
            - Tool history: '{tool_history}'
            - SQL results: {results}
            Format the data according to these instructions:
            {instructions}
            - Use meaningful title, xLabel, yLabel based on the question.
            - For hierarchical_bar, use '.' as separator in labels for levels (e.g., 'Model.Vendor').
            - Ensure data arrays match labels length.
            - Handle multiple series if present (e.g., by date).
            - Replace NULL with 0 or appropriate.
            Output ONLY the JSON object for 'data'."""
        )
        chain = prompt | llm
        for attempt in range(max_retries):
            try:
                response = chain.invoke({
                    "viz_type": viz_type,
                    "question": question,
                    "history": history,
                    "tool_history": tool_history,
                    "results": json.dumps(sql_result),
                    "instructions": instructions
                })
                content = response.content if hasattr(response, "content") else str(response)
                logger.debug(f"LLM formatting response (attempt {attempt + 1}): {content}")
                formatted_data = parse_llm_response(content)
                if not formatted_data:
                    logger.warning(f"Failed to parse formatted data on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        continue
                    raise ValueError("Invalid formatted data after retries")
                logger.info(f"Formatted viz data: {formatted_data}")
                return formatted_data
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise ValueError("Invalid formatted data after retries")
        return {}
    except Exception as e:
        logger.error(f"Data formatting failed: {traceback.format_exc()}")
        return {}

def format_tables(sql_result: List[Dict], question: str, history: str = "", tool_history: str = "") -> List[Dict]:
    """Format SQL results into tables using LLM for title and structure."""
    try:
        if not sql_result:
            logger.debug("No SQL result for table formatting, returning empty list")
            return []
        prompt = ChatPromptTemplate.from_template(
            """You are a data expert formatting SQL results into tables. Based on:
            - User question: '{question}'
            - Conversation history: '{history}'
            - Tool history: '{tool_history}'
            - SQL results: {results}
            Format the data into a list of tables, each with a title and data:
            {
                "title": "string",
                "data": { "column_name": "value" }[]
            }
            - Provide a meaningful table title based on the question.
            - Use human-readable column names, derived from the question and data (e.g., '型号', '故障次数', '占比').
            - Replace NULL/empty values with 'Unknown'.
            - If grouping is logical (e.g., by category), create multiple tables.
            Output ONLY the JSON object: [{"title": "string", "data": { "column_name": "value" }[]}]"""
        )
        chain = prompt | llm
        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = chain.invoke({
                    "question": question,
                    "history": history,
                    "tool_history": tool_history,
                    "results": json.dumps(sql_result)
                })
                content = response.content if hasattr(response, "content") else str(response)
                logger.debug(f"LLM table formatting response (attempt {attempt + 1}): {content}")
                tables = parse_llm_response(content)
                if not isinstance(tables, list):
                    logger.warning(f"Table formatting did not return a list on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        continue
                    return []
                logger.info(f"Formatted tables: {tables}")
                return tables
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    return []
        return []
    except Exception as e:
        logger.error(f"Table formatting failed: {traceback.format_exc()}")
        return []

def build_chart_config(viz_type: str, formatted_data: Dict) -> Dict:
    """Build Chart.js configuration from LLM-formatted data."""
    try:
        if not formatted_data or viz_type == "none":
            logger.debug("No formatted data or viz_type is none, returning empty config")
            return {}
        colors = ["#36A2EB", "#FF6384", "#FFCE56", "#4BC0C0", "#9966FF", "#FF9F40", "#66FF66", "#999999"] * 10
        title = formatted_data.get("title", "Chart")
        config = {
            "type": viz_type if viz_type != "horizontal_bar" and viz_type != "hierarchical_bar" else "bar",
            "data": {},
            "options": {
                "responsive": True,
                "maintainAspectRatio": False,
                "plugins": {
                    "title": {"display": True, "text": title},
                    "legend": {"display": True}
                }
            }
        }

        if viz_type in ["bar", "horizontal_bar", "hierarchical_bar"]:
            labels = formatted_data.get("labels", [])
            values = formatted_data.get("values", [])
            datasets = [
                {
                    "label": v["label"],
                    "data": [float(x) if x is not None else 0.0 for x in v["data"]],
                    "backgroundColor": colors[i % len(colors)],
                    "borderColor": colors[i % len(colors)],
                    "borderWidth": 1
                } for i, v in enumerate(values)
            ]
            config["data"] = {"labels": labels, "datasets": datasets}
            config["options"]["scales"] = {
                "x": {"title": {"display": True, "text": formatted_data.get("xLabel", "X Axis")}},
                "y": {"beginAtZero": True, "title": {"display": True, "text": formatted_data.get("yLabel", "Y Axis")}}
            }
            if viz_type == "horizontal_bar":
                config["options"]["indexAxis"] = "y"
            if viz_type == "hierarchical_bar":
                config["options"]["scales"]["x"]["type"] = "hierarchical"
                config["options"]["scales"]["x"]["separator"] = "."
                config["options"]["scales"]["x"]["levelPadding"] = 10  # Optional, adjust as needed

        elif viz_type == "line":
            config["data"] = {
                "labels": formatted_data.get("xValues", []),
                "datasets": [
                    {
                        "label": v["label"],
                        "data": [float(x) if x is not None else 0.0 for x in v["data"]],
                        "fill": False,
                        "borderColor": colors[i % len(colors)]
                    } for i, v in enumerate(formatted_data.get("yValues", []))
                ]
            }
            config["options"]["scales"] = {
                "x": {"title": {"display": True, "text": formatted_data.get("xLabel", "X Axis")}},
                "y": {"beginAtZero": True, "title": {"display": True, "text": formatted_data.get("yLabel", "Y Axis")}}
            }

        elif viz_type == "pie":
            data_list = formatted_data.get("data", [])
            labels = [d["label"] if d["label"] is not None else "Unknown" for d in data_list]
            values = [round(float(d["value"]), 2) if d["value"] is not None else 0.0 for d in data_list]
            config["data"] = {
                "labels": labels,
                "datasets": [{"data": values, "backgroundColor": colors[:len(values)]}]
            }
            config["options"]["plugins"]["legend"]["display"] = True

        elif viz_type == "scatter":
            series = formatted_data.get("series", [])
            datasets = [
                {
                    "label": s["label"],
                    "data": [{"x": float(d["x"]) if d["x"] is not None else 0.0, "y": float(d["y"]) if d["y"] is not None else 0.0, "id": d["id"]} for d in s["data"]],
                    "backgroundColor": colors[i % len(colors)]
                } for i, s in enumerate(series)
            ]
            config["data"] = {"datasets": datasets}
            config["options"]["scales"] = {
                "x": {"title": {"display": True, "text": formatted_data.get("xLabel", "X Axis")}},
                "y": {"title": {"display": True, "text": formatted_data.get("yLabel", "Y Axis")}}
            }

        logger.info(f"Built chart config for {viz_type}: {config}")
        return config
    except Exception as e:
        logger.error(f"Failed to build chart config: {traceback.format_exc()}")
        return {}