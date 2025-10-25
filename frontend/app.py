# app.py
"""
Streamlit frontend for the SQL Agent with data visualization.
Provides a ChatGPT-like interface with session management and tabular output.
"""
import streamlit as st
import streamlit.components.v1 as components
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import uuid
import pandas as pd
import json
import time
from backend.sql_agent import build_graph, process_query
from langchain_community.utilities import SQLDatabase
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage
from backend.database_builder import DatabaseBuilder
from backend.data_loader import ExcelDataSourceLoader, CSVDataSourceLoader
import logging
import traceback

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("frontend.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

DB_PATH = "real_database.db"

# Initialize database and graph
try:
    db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
    graph, _ = build_graph(db)
    logger.info("Database and graph initialized successfully")
except Exception as e:
    st.error(f"Database/Graph initialization failed: {e}")
    logger.error(f"Initialization failed: {traceback.format_exc()}")
    st.stop()

@st.cache_data
def load_excel_sheets(file_content, file_name):
    """Cache Excel file sheet parsing for performance."""
    try:
        xl = pd.ExcelFile(file_content)
        logger.info(f"Cached sheets for {file_name}: {xl.sheet_names}")
        return xl.sheet_names
    except Exception as e:
        logger.error(f"Failed to load sheets for {file_name}: {traceback.format_exc()}")
        return []

def rebuild_db(uploaded_files, sheet_configs=None):
    """Rebuild the database from uploaded files."""
    try:
        if not uploaded_files:
            raise ValueError("No files uploaded")
        loaders = []
        for uploaded_file in uploaded_files:
            file_size = len(uploaded_file.getvalue()) / 1024 / 1024
            if file_size > 10:
                st.warning(f"File {uploaded_file.name} ({file_size:.2f}MB) exceeds 10MB limit.")
                raise ValueError(f"File {uploaded_file.name} exceeds 10MB limit")
            if not uploaded_file.name.endswith((".xlsx", ".csv")):
                raise ValueError(f"File {uploaded_file.name} must be .xlsx or .csv")
            unique_id = str(uuid.uuid4())
            file_ext = uploaded_file.name.split(".")[-1]
            file_path = f"temp_{unique_id}.{file_ext}"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            logger.info(f"Processing file: {uploaded_file.name}, size: {file_size:.2f}MB")
            if file_path.endswith(".xlsx"):
                available_sheets = load_excel_sheets(uploaded_file.getvalue(), uploaded_file.name)
                sheets = sheet_configs.get(uploaded_file.name, [(name, None) for name in available_sheets])
                logger.info(f"Selected sheets for {uploaded_file.name}: {sheets}")
                loader = ExcelDataSourceLoader(file_path, sheets=sheets)
            elif file_path.endswith(".csv"):
                loader = CSVDataSourceLoader(file_path, table_name=uploaded_file.name.split(".")[0])
            else:
                raise ValueError(f"Unsupported file type: {uploaded_file.name}")
            loaders.append(loader)
            os.remove(file_path)
        builder = DatabaseBuilder(DB_PATH)
        result = builder.build_database(loaders, rebuild=True)
        if result["status"] != 0:
            raise ValueError(f"Database rebuild failed: {result['errors']}")
        st.success("Database rebuilt successfully!")
        global db, graph
        db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
        graph, _ = build_graph(db)
        logger.info("Database and graph reinitialized after rebuild")
    except Exception as e:
        st.error(f"Database rebuild error: {e}")
        logger.error(f"Rebuild failed: {traceback.format_exc()}")

# Define routine dashboard queries
dashboard_modules = {
    "质量模块": [
        # {"query": "近一个月的光模块故障数（从ROCE事件和事件监控的光模块故障表获取数据），按厂商、型号分布", "title": "近一个月光模块故障"},
        # {"query": "近一个月的网络设备故障率，按厂商、型号分布", "title": "近一个月网络设备故障"}
    ],
    "容量模块": [
        # {"query": "近两个月光模块故障数（从ROCE事件和事件监控的光模块故障表获取数据），按厂商、型号分布", "title": "近两个月光模块故障"}
    ]
}

def render_dashboard():
    """Render fixed dashboard with routine queries."""
    st.header("Data Dashboard")
    for module, queries in dashboard_modules.items():
        with st.expander(f"Module: {module}"):
            for q in queries:
                st.subheader(q["title"])
                with st.spinner(f"Running: {q['query']}"):
                    inputs = {
                        "messages": [HumanMessage(content=q["query"])],
                        "question": q["query"],
                        "tool_history": [],
                        "status_messages": [],
                        "sql_result": []
                    }
                    config = {"configurable": {"thread_id": "dashboard"}}
                    result = process_query(graph, inputs, config)
                    st.markdown(result['answer'])
                    if result["tables"]:
                        for table in result["tables"]:
                            st.write(f"**{table['title']}**")
                            st.dataframe(pd.DataFrame(table["data"]), width='stretch')
                    if result["chart_config"]:
                        chart_id = f"chart_{uuid.uuid4().hex}"
                        chart_json = json.dumps(result["chart_config"])
                        html = f"""
                        <html>
                        <head>
                            <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js"></script>
                        </head>
                        <body>
                            <canvas id="{chart_id}" style="width:100%; max-width:800px; height:400px;"></canvas>
                            <script>
                                document.addEventListener('DOMContentLoaded', function() {{
                                    try {{
                                        var ctx = document.getElementById('{chart_id}').getContext('2d');
                                        var myChart = new Chart(ctx, {chart_json});
                                    }} catch (e) {{
                                        console.error('Chart.js error: ' + e.message);
                                    }}
                                }});
                            </script>
                        </body>
                        </html>
                        """
                        try:
                            components.html(html, height=450, scrolling=False)
                        except Exception as e:
                            st.error(f"Failed to render chart: {str(e)}. Please ensure Streamlit is version 0.87.0 or higher.")
                            logger.error(f"Chart rendering failed: {traceback.format_exc()}")

# Streamlit UI
st.title("灵图SQL视图")

# Tabs for Chat and Dashboard
tab1, tab2 = st.tabs(["Chat", "Dashboard"])

with tab1:
    with st.sidebar:
        st.header("Chat History")
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = {}
            st.session_state.first_questions = {}
            st.session_state.current_thread_id = str(uuid.uuid4())
            st.session_state.chat_history[st.session_state.current_thread_id] = []
            st.session_state.first_questions[st.session_state.current_thread_id] = "New Chat"
            logger.info(f"Initialized new chat session: {st.session_state.current_thread_id}")

    # Handle rerun
    if st.session_state.get("needs_rerun", False):
        st.session_state.needs_rerun = False
        st.rerun()

    # Chat interface
    chat_container = st.container()
    with chat_container:
        st.header("灵图SQL对话")
        for message in st.session_state.chat_history[st.session_state.current_thread_id]:
            if isinstance(message, (HumanMessage, AIMessage)):
                role = "user" if isinstance(message, HumanMessage) else "assistant"
                with st.chat_message(role):
                    st.markdown(message.content)
                    if isinstance(message, AIMessage) and hasattr(message, "tables"):
                        for table in message.tables:
                            st.write(f"**{table['title']}**")
                            st.dataframe(pd.DataFrame(table["data"]), width='stretch')
                    if isinstance(message, AIMessage) and hasattr(message, "chart_config") and message.chart_config:
                        chart_id = f"chart_{uuid.uuid4().hex}"
                        chart_json = json.dumps(message.chart_config)
                        html = f"""
                        <html>
                        <head>
                            <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js"></script>
                        </head>
                        <body>
                            <canvas id="{chart_id}" style="width:100%; max-width:800px; height:400px;"></canvas>
                            <script>
                                document.addEventListener('DOMContentLoaded', function() {{
                                    try {{
                                        var ctx = document.getElementById('{chart_id}').getContext('2d');
                                        var myChart = new Chart(ctx, {chart_json});
                                    }} catch (e) {{
                                        console.error('Chart.js error: ' + e.message);
                                    }}
                                }});
                            </script>
                        </body>
                        </html>
                        """
                        try:
                            components.html(html, height=450, scrolling=False)
                        except Exception as e:
                            st.error(f"Failed to render chart: {str(e)}. Please ensure Streamlit is version 0.87.0 or higher.")
                            logger.error(f"Chart rendering failed: {traceback.format_exc()}")

    # Chat input and processing
    prompt = st.chat_input("Enter your query (e.g., 'Which country's customers spent the most?')")
    if prompt:
        if not st.session_state.chat_history[st.session_state.current_thread_id]:
            st.session_state.first_questions[st.session_state.current_thread_id] = prompt[:50] + "..." if len(prompt) > 50 else prompt
        user_message = HumanMessage(content=prompt)
        st.session_state.chat_history[st.session_state.current_thread_id].append(user_message)
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                status_placeholder = st.empty()
                status_placeholder.markdown("Starting query processing...")
                with st.spinner("Processing..."):
                    try:
                        inputs = {
                            "messages": st.session_state.chat_history[st.session_state.current_thread_id],
                            "question": prompt,
                            "tool_history": [],
                            "status_messages": [],
                            "sql_result": []
                        }
                        config = {"configurable": {"thread_id": st.session_state.current_thread_id}}
                        result = process_query(graph, inputs, config, status_placeholder)
                        for status in result["status_messages"]:
                            status_placeholder.markdown(status)
                            time.sleep(0.5)
                        st.markdown(result['answer'])
                        if result["tables"]:
                            for table in result["tables"]:
                                st.write(f"**{table['title']}**")
                                st.dataframe(pd.DataFrame(table["data"]), width='stretch')
                        if result["chart_config"]:
                            chart_id = f"chart_{uuid.uuid4().hex}"
                            chart_json = json.dumps(result["chart_config"])
                            html = f"""
                            <html>
                            <head>
                                <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js"></script>
                            </head>
                            <body>
                                <canvas id="{chart_id}" style="width:100%; max-width:800px; height:400px;"></canvas>
                                <script>
                                    document.addEventListener('DOMContentLoaded', function() {{
                                        try {{
                                            var ctx = document.getElementById('{chart_id}').getContext('2d');
                                            var myChart = new Chart(ctx, {chart_json});
                                        }} catch (e) {{
                                            console.error('Chart.js error: ' + e.message);
                                        }}
                                    }});
                                </script>
                            </body>
                            </html>
                            """
                            try:
                                components.html(html, height=450, scrolling=False)
                            except Exception as e:
                                st.error(f"Failed to render chart: {str(e)}. Please ensure Streamlit is version 0.87.0 or higher.")
                                logger.error(f"Chart rendering failed: {traceback.format_exc()}")
                        st.markdown(f"**Processing Time**: {result['processing_time']:.2f} seconds")
                        assistant_message = AIMessage(content=result['answer'])
                        if result["tables"]:
                            assistant_message.tables = result["tables"]
                        if result["chart_config"]:
                            assistant_message.chart_config = result["chart_config"]
                        st.session_state.chat_history[st.session_state.current_thread_id] = result["messages"] + [assistant_message]
                    except Exception as e:
                        status_placeholder.error(f"Query error: {e}")
                        logger.error(f"Query failed: {traceback.format_exc()}")
                        assistant_message = AIMessage(content=f"Error: {str(e)}")
                        st.session_state.chat_history[st.session_state.current_thread_id].append(assistant_message)

with tab2:
    render_dashboard()