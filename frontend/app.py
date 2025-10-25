# app.py
"""
Streamlit frontend for the SQL Agent with visualization capabilities.
Provides a ChatGPT-like interface with session management, synchronous Chinese tool messages, and historical conversation sidebar.
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
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("frontend.log", encoding="utf-8"),
        logging.FileHandler("error.log", encoding="utf-8", mode="a"),  # Dedicated error log
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
    st.error("抱歉，初始化时发生错误，请稍后重试或联系支持。")
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
                st.warning(f"文件 {uploaded_file.name} ({file_size:.2f}MB) 超过10MB限制。")
                raise ValueError(f"File {uploaded_file.name} exceeds 10MB limit")
            if not uploaded_file.name.endswith((".xlsx", ".csv")):
                raise ValueError(f"文件 {uploaded_file.name} 必须是 .xlsx 或 .csv")
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
        st.success("数据库重建成功！")
        global db, graph
        db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
        graph, _ = build_graph(db)
        logger.info("Database and graph reinitialized after rebuild")
    except Exception as e:
        st.error("抱歉，数据库重建时发生错误，请稍后重试或联系支持。")
        logger.error(f"Rebuild failed: {traceback.format_exc()}")

# Define routine dashboard queries (customize based on your DB schema)
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
    st.header("数据仪表板")
    for module, queries in dashboard_modules.items():
        with st.expander(f"模块: {module}"):
            for q in queries:
                st.subheader(q["title"])
                with st.spinner(f"正在运行: {q['query']}"):
                    inputs = {
                        "messages": [HumanMessage(content=q["query"])],
                        "question": q["query"],
                        "tool_history": [],
                        "status_messages": []
                    }
                    config = {"configurable": {"thread_id": "dashboard"}}
                    result = process_query(graph, inputs, config, lambda msg: st.markdown(translate_status_message(msg)))
                    if "error" in result:
                        st.error("抱歉，仪表板查询时发生错误，请稍后重试或联系支持。")
                        logger.error(f"Dashboard query error: {result['error']}")
                    else:
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
                                st.error("抱歉，图表渲染失败，请稍后重试或联系支持。")
                                logger.error(f"Chart rendering failed: {traceback.format_exc()}")

# Translate English status messages to Chinese for frontend display
def translate_status_message(message: str) -> str:
    translations = {
        "Agent processed query.": "灵图正在处理查询~",
        "Executing tool: sql_db_list_tables": "正在查看数据库表",
        "Executing tool: sql_db_schema": "正在获取表结构",
        "Executing tool: sql_db_query": "正在执行SQL查询",
        "Executing tool: sql_db_query_checker": "正在检查SQL查询",
        "Executing tool: check_result": "正在检查查询结果",
        "Tool sql_db_list_tables executed successfully.": "工具列出数据表执行成功。",
        "Tool sql_db_schema executed successfully.": "工具获取表结构执行成功。",
        "Tool sql_db_query executed successfully.": "工具执行SQL查询成功。",
        "Tool sql_db_query_checker executed successfully.": "工具检查SQL查询成功。",
        "Tool check_result executed successfully.": "工具检查查询结果成功。",
        "Generating visualization and tables...": "正在生成可视化和表格...",
        "Error in agent decision.": "抱歉，代理决策错误，请稍后重试。",
        "Error in tool execution.": "抱歉，工具执行错误，请稍后重试。",
        "Error in visualization and table formatting.": "抱歉，可视化或表格生成错误，请稍后重试。",
        "Error in query processing.": "抱歉，查询处理错误，请稍后重试。",
    }
    return translations.get(message, "灵图正在处理查询~")

# Stream response to frontend
def stream_response(result: dict, status_placeholder, answer_placeholder, chart_placeholder, table_placeholder):
    if "error" in result:
        status_placeholder.error("抱歉，处理查询时发生错误，请稍后重试或联系支持。")
        logger.error(f"Streaming error: {result['error']}")
        return

    # Display answer if available
    answer = result.get("answer", "")
    if answer:
        with answer_placeholder.container():
            st.markdown("**回答:**")
            st.markdown(answer)
        status_placeholder.empty()  # Clear tool messages
        logger.debug("Answer streamed, tool messages cleared")

    # Display chart if available
    if result.get("viz_data") and result.get("viz_type") != "none":
        with chart_placeholder.container():
            st.markdown("**图表:**")
            chart_id = f"chart_{uuid.uuid4().hex}"
            chart_json = json.dumps(result["viz_data"])
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
                st.error("抱歉，图表渲染失败，请稍后重试或联系支持。")
                logger.error(f"Chart rendering failed: {traceback.format_exc()}")

    # Display tables if available
    if result.get("tables"):
        with table_placeholder.container():
            st.markdown("**表格:**")
            for table in result["tables"]:
                st.markdown(f"**{table['title']}**")
                df = pd.DataFrame(table["data"])
                st.dataframe(df, width='stretch')
                logger.debug(f"Table displayed: {table['title']}")

    # Display processing time
    status_placeholder.markdown(f"处理时间: {result['processing_time']:.2f}秒")

# Streamlit UI
st.title("灵图SQL视图")

# Tabs for Chat and Dashboard
tab1, tab2 = st.tabs(["对话", "仪表板"])

with tab1:
    # Sidebar for session management
    with st.sidebar:
        st.header("聊天历史")
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = {}
            st.session_state.first_questions = {}
            st.session_state.chat_creation_times = {}
            st.session_state.current_thread_id = str(uuid.uuid4())
            st.session_state.tool_history = {}
            st.session_state.chat_history[st.session_state.current_thread_id] = []
            st.session_state.first_questions[st.session_state.current_thread_id] = "新建对话"
            st.session_state.chat_creation_times[st.session_state.current_thread_id] = datetime.now()
            st.session_state.tool_history[st.session_state.current_thread_id] = []
            logger.info(f"Initialized first conversation: {st.session_state.current_thread_id}")

        # New chat button
        if st.button("新建聊天 +"):
            new_thread_id = str(uuid.uuid4())
            st.session_state.chat_history[new_thread_id] = []
            st.session_state.first_questions[new_thread_id] = "新建对话"
            st.session_state.chat_creation_times[new_thread_id] = datetime.now()
            st.session_state.tool_history[new_thread_id] = []
            st.session_state.current_thread_id = new_thread_id
            logger.info(f"Created new conversation: {new_thread_id}")
            st.rerun()

        # Group chats by time range
        now = datetime.now()
        groups = {
            "今天": [],
            "昨天": [],
            "过去7天": [],
            "过去30天": [],
            "更早": []
        }

        for thread_id, creation_time in sorted(st.session_state.chat_creation_times.items(), key=lambda x: x[1], reverse=True):
            delta = now - creation_time
            if delta < timedelta(days=1):
                groups["今天"].append((thread_id, creation_time))
            elif delta < timedelta(days=2):
                groups["昨天"].append((thread_id, creation_time))
            elif delta < timedelta(days=7):
                groups["过去7天"].append((thread_id, creation_time))
            elif delta < timedelta(days=30):
                groups["过去30天"].append((thread_id, creation_time))
            else:
                groups["更早"].append((thread_id, creation_time))

        # Display grouped history
        for group_name, threads in groups.items():
            if threads:
                st.subheader(group_name)
                for thread_id, creation_time in threads:
                    label = st.session_state.first_questions.get(thread_id, "未知")
                    if st.button(f"{label} - {creation_time.strftime('%Y-%m-%d %H:%M')}", key=thread_id):
                        st.session_state.current_thread_id = thread_id
                        logger.info(f"Switched to conversation: {thread_id}")
                        st.rerun()

    # Chat interface
    chat_container = st.container()
    with chat_container:
        st.header("灵图SQL对话")
        
        # Display chat history for current thread
        for message in st.session_state.chat_history[st.session_state.current_thread_id]:
            if isinstance(message, (HumanMessage, AIMessage)):
                role = "user" if isinstance(message, HumanMessage) else "assistant"
                with st.chat_message(role):
                    st.markdown(message.content)
                    if isinstance(message, AIMessage) and hasattr(message, "tables"):
                        for table in message.tables:
                            st.markdown(f"**{table['title']}**")
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
                            st.error("抱歉，图表渲染失败，请稍后重试或联系支持。")
                            logger.error(f"Chart rendering failed: {traceback.format_exc()}")

    # Chat input and processing
    prompt = st.chat_input("输入您的查询 (例如: '哪个国家的客户消费最多？')")
    if prompt:
        # Update first question for new sessions
        if not st.session_state.chat_history[st.session_state.current_thread_id]:
            st.session_state.first_questions[st.session_state.current_thread_id] = prompt[:50] + "..." if len(prompt) > 50 else prompt
        
        user_message = HumanMessage(content=prompt)
        st.session_state.chat_history[st.session_state.current_thread_id].append(user_message)
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                status_placeholder = st.empty()
                answer_placeholder = st.empty()
                chart_placeholder = st.empty()
                table_placeholder = st.empty()
                status_placeholder.markdown("开始查询处理...")
                with st.spinner("处理中..."):
                    try:
                        # Filter context for follow-up questions
                        filtered_messages = [
                            msg for msg in st.session_state.chat_history[st.session_state.current_thread_id]
                            if isinstance(msg, (HumanMessage, AIMessage)) and not (isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls)
                        ]
                        inputs = {
                            "messages": filtered_messages + [user_message],
                            "question": prompt,
                            "tool_history": st.session_state.tool_history.get(st.session_state.current_thread_id, []),
                            "status_messages": []
                        }
                        config = {"configurable": {"thread_id": st.session_state.current_thread_id}}
                        # Pass callback to update status messages in real-time
                        result = process_query(graph, inputs, config, lambda msg: status_placeholder.markdown(translate_status_message(msg)))
                        stream_response(result, status_placeholder, answer_placeholder, chart_placeholder, table_placeholder)
                        # Update chat history with final message
                        assistant_message = AIMessage(content=result['answer'])
                        if result["tables"]:
                            assistant_message.tables = result["tables"]
                        if result["viz_data"]:
                            assistant_message.chart_config = result["viz_data"]
                        st.session_state.chat_history[st.session_state.current_thread_id] = result["messages"] + [assistant_message]
                        # Update tool history
                        st.session_state.tool_history[st.session_state.current_thread_id] = [
                            h for h in result.get("tool_history", [])
                            if h["tool"] in ["sql_db_list_tables", "sql_db_schema", "sql_db_query", "sql_db_query_checker", "check_result"]
                        ]
                    except Exception as e:
                        status_placeholder.error("抱歉，处理查询时发生错误，请稍后重试或联系支持。")
                        logger.error(f"Query failed: {traceback.format_exc()}")
                        assistant_message = AIMessage(content="抱歉，处理查询时发生错误，请稍后重试或联系支持。")
                        st.session_state.chat_history[st.session_state.current_thread_id].append(assistant_message)

with tab2:
    render_dashboard()

if prompt and prompt.lower() in ["exit", "quit"]:
    st.write("退出程序。")
    logger.info("User exited the program")
    st.stop()