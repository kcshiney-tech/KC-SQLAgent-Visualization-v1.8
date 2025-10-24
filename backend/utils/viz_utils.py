# backend/utils/viz_utils.py
"""
Utilities for choosing and formatting visualization data based on SQL results and conversation context.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from typing import Dict, List
import logging
import traceback
from dotenv import load_dotenv
import os
import json

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

def choose_viz_type(question: str, sql_result: List[Dict], history: str = "", tool_history: str = "") -> str:
    """Choose visualization type based on question, SQL result, and conversation/tool history, considering user intent."""
    try:
        viz_types = ["bar", "line", "pie", "scatter", "none"]
        prompt = ChatPromptTemplate.from_template(
            """You are a data visualization expert. Based on:
            - User question: '{question}'
            - Conversation history: '{history}'
            - Tool history (SQL queries and results): '{tool_history}'
            - SQL query result (first 3 rows): {sample_result}
            First, detect user intent for visualization: 
            - Explicit intent: If the question or history contains keywords like 'chart', 'graph', 'visualize', 'plot', 'show me a', or explicitly requests a visual, proceed to select a viz type.
            - Implicit intent: If the query involves trends (e.g., 'trend', 'over time', 'change over'), distributions (e.g., 'distribution', 'proportion', 'percentage of'), comparisons (e.g., 'compare', 'vs', 'ranking'), or aggregations that benefit from visuals (e.g., 'group by', 'top N'), consider visualization even without explicit request, as charts can enhance understanding.
            - Otherwise, default to 'none' even if data suits visualization.
            If visualization is intended (explicit or implicit), select the most appropriate type from: {viz_types}.
            - Use 'bar' for comparisons or aggregations across categories (e.g., sums, counts, or percentages by category). For horizontal bar charts, 'bar' will be used with appropriate options.
            - Use 'line' for trends over time (e.g., values over dates or sequential data).
            - Use 'pie' for proportions or percentages of a total (e.g., distribution across categories).
            - Use 'scatter' for correlations between two numeric variables.
            - Use 'none' if no visualization is suitable or not intended (e.g., single value, empty result, unsuitable data, or no visual intent).
            Consider historical context and tool outputs (e.g., keywords like 'trend', 'compare', prior SQL results) to refine the choice.
            Ensure the selected type matches the data structure (e.g., at least one categorical and one numeric column for bar/pie, two numeric columns for scatter).
            Output a JSON object: {{"viz_type": "chosen_type"}}."""
        )
        chain = prompt | llm
        sample_result = json.dumps(sql_result[:3])
        logger.debug(f"Choosing viz type with question: {question}, history: {history}, tool_history: {tool_history}, sample_result: {sample_result}")
        response = chain.invoke({"question": question, "history": history, "tool_history": tool_history, "sample_result": sample_result, "viz_types": ", ".join(viz_types)})
        logger.debug(f"LLM response content: {response.content}")
        try:
            response_json = json.loads(response.content)
            viz_type = response_json.get("viz_type", "none").lower()
            if viz_type not in viz_types:
                logger.warning(f"Invalid viz type returned: {viz_type}, defaulting to 'bar' for comparison queries")
                viz_type = "bar" if any(keyword in (question.lower() + history.lower() + tool_history.lower()) for keyword in ["compare", "by", "group", "rank", "order", "分布"]) else "none"
        except json.JSONDecodeError:
            logger.warning("LLM returned invalid JSON, defaulting to 'bar' for comparison queries")
            viz_type = "bar" if any(keyword in (question.lower() + history.lower() + tool_history.lower()) for keyword in ["compare", "by", "group", "rank", "order", "分布"]) else "none"
        logger.info(f"Chosen viz type: {viz_type}")
        return viz_type
    except Exception as e:
        logger.error(f"Viz type choice failed: {traceback.format_exc()}")
        return "none"

def format_data_for_viz(viz_type: str, sql_result: List[Dict]) -> Dict:
    """Format SQL result for Chart.js based on visualization type."""
    try:
        if viz_type == "none" or not sql_result:
            logger.debug("No visualization or empty result, returning empty dict")
            return {}
        
        if not isinstance(sql_result, list) or not all(isinstance(row, dict) for row in sql_result):
            logger.error(f"Invalid SQL result format: {sql_result}")
            return {}
        
        keys = list(sql_result[0].keys())
        logger.debug(f"SQL result keys: {keys}")
        if len(keys) < 2:
            logger.error("Need at least 2 columns for visualization")
            return {}
        
        label_col = keys[0]
        secondary_label_col = keys[1] if len(keys) > 1 else None
        value_col = None
        for key in reversed(keys):
            if any(isinstance(row.get(key), (int, float)) and not isinstance(row.get(key), bool) for row in sql_result):
                value_col = key
                break
        if not value_col:
            logger.warning("No numeric column found for values, returning empty viz data")
            return {}
        
        is_numeric = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool)
        
        if viz_type == "bar":
            if secondary_label_col:
                labels = [f"{row[label_col]} ({row[secondary_label_col]})" for row in sql_result]
            else:
                labels = [str(row[label_col]) for row in sql_result]
            values = [float(row[value_col]) if is_numeric(row.get(value_col)) else 0 for row in sql_result]
            if not any(values):
                logger.warning("All values are zero or non-numeric, returning empty viz data")
                return {}
            
            # Improved horizontal bar detection
            # is_horizontal = (
            #     len(labels) > 10 or  # Many categories suggest horizontal
            #     any(keyword in (question.lower() + history.lower() + tool_history.lower()) for keyword in ["分布", "按", "by", "group", "compare", "vs"]) or
            #     "厂商" in str(label_col) or "型号" in str(label_col)  # Vendor/model distribution
            # )
            is_horizontal = False
            
            chart_config = {
                "type": "bar",
                "data": {
                    "labels": labels,
                    "datasets": [{
                        "label": value_col,
                        "data": values,
                        "backgroundColor": ["#36A2EB", "#FF6384", "#FFCE56", "#4BC0C0", "#9966FF", "#FF9F40", "#FF6384", "#C9CBCF", "#4BC0C0", "#FFCE56"] * 5,  # Repeat colors for more data points
                        "borderColor": ["#36A2EB", "#FF6384", "#FFCE56", "#4BC0C0", "#9966FF", "#FF9F40", "#FF6384", "#C9CBCF", "#4BC0C0", "#FFCE56"] * 5,
                        "borderWidth": 1
                    }]
                },
                "options": {
                    "indexAxis": "y" if is_horizontal else "x",
                    "responsive": True,
                    "maintainAspectRatio": False,
                    "scales": {
                        ("y" if is_horizontal else "x"): {
                            "beginAtZero": True,
                            "title": {
                                "display": True,
                                "text": value_col
                            }
                        },
                        ("x" if is_horizontal else "y"): {
                            "title": {
                                "display": True,
                                "text": label_col + (f" ({secondary_label_col})" if secondary_label_col else "")
                            }
                        }
                    },
                    "plugins": {
                        "title": {
                            "display": True,
                            "text": f"{value_col} by {label_col}" + (f" ({secondary_label_col})" if secondary_label_col else "")
                        },
                        "legend": {
                            "display": True
                        }
                    }
                }
            }
            return chart_config
        elif viz_type == "line":
            if secondary_label_col:
                x = [f"{row[label_col]} ({row[secondary_label_col]})" for row in sql_result]
            else:
                x = [str(row[label_col]) for row in sql_result]
            y = [float(row[value_col]) if is_numeric(row.get(value_col)) else 0 for row in sql_result]
            if not any(y):
                logger.warning("All y-values are zero or non-numeric, returning empty viz data")
                return {}
            return {
                "type": "line",
                "data": {
                    "labels": x,
                    "datasets": [{
                        "label": value_col,
                        "data": y,
                        "borderColor": "#36A2EB",
                        "fill": False
                    }]
                },
                "options": {
                    "responsive": True,
                    "maintainAspectRatio": False,
                    "scales": {
                        "x": {
                            "title": {
                                "display": True,
                                "text": label_col + (f" ({secondary_label_col})" if secondary_label_col else "")
                            }
                        },
                        "y": {
                            "beginAtZero": True,
                            "title": {
                                "display": True,
                                "text": value_col
                            }
                        }
                    },
                    "plugins": {
                        "title": {
                            "display": True,
                            "text": f"{value_col} over {label_col}" + (f" ({secondary_label_col})" if secondary_label_col else "")
                        }
                    }
                }
            }
        elif viz_type == "pie":
            if secondary_label_col:
                labels = [f"{row[label_col]} ({row[secondary_label_col]})" for row in sql_result]
            else:
                labels = [str(row[label_col]) for row in sql_result]
            values = [float(row[value_col]) if is_numeric(row.get(value_col)) else 0 for row in sql_result]
            if not any(values):
                logger.warning("All values are zero or non-numeric, returning empty viz data")
                return {}
            return {
                "type": "pie",
                "data": {
                    "labels": labels,
                    "datasets": [{
                        "label": value_col,
                        "data": values,
                        "backgroundColor": ["#36A2EB", "#FF6384", "#FFCE56", "#4BC0C0", "#9966FF"],
                        "borderColor": ["#36A2EB", "#FF6384", "#FFCE56", "#4BC0C0", "#9966FF"],
                        "borderWidth": 1
                    }]
                },
                "options": {
                    "responsive": True,
                    "maintainAspectRatio": False,
                    "plugins": {
                        "title": {
                            "display": True,
                            "text": f"{value_col} by {label_col}" + (f" ({secondary_label_col})" if secondary_label_col else "")
                        },
                        "legend": {
                            "display": True
                        }
                    }
                }
            }
        elif viz_type == "scatter":
            numeric_cols = [key for key in keys if any(is_numeric(row.get(key)) for row in sql_result)]
            if len(numeric_cols) < 2:
                logger.warning("Need at least two numeric columns for scatter plot, returning empty viz data")
                return {}
            x_key, y_key = numeric_cols[-2], numeric_cols[-1]
            series = []
            for i, row in enumerate(sql_result):
                x_val = row.get(x_key)
                y_val = row.get(y_key)
                if is_numeric(x_val) and is_numeric(y_val):
                    series.append({"x": float(x_val), "y": float(y_val)})
            if not series:
                logger.warning("No valid numeric pairs for scatter plot, returning empty viz data")
                return {}
            return {
                "type": "scatter",
                "data": {
                    "datasets": [{
                        "label": f"{x_key} vs {y_key}",
                        "data": series,
                        "backgroundColor": "#36A2EB",
                        "borderColor": "#36A2EB",
                        "pointRadius": 5
                    }]
                },
                "options": {
                    "responsive": True,
                    "maintainAspectRatio": False,
                    "scales": {
                        "x": {
                            "title": {
                                "display": True,
                                "text": x_key
                            }
                        },
                        "y": {
                            "title": {
                                "display": True,
                                "text": y_key
                            }
                        }
                    },
                    "plugins": {
                        "title": {
                            "display": True,
                            "text": f"{y_key} vs {x_key}"
                        }
                    }
                }
            }
        else:
            logger.error(f"Unsupported viz type: {viz_type}")
            return {}
    except Exception as e:
        logger.error(f"Data formatting failed: {traceback.format_exc()}")
        return {}