# app.py
"""
Streamlit frontend for the SQL Agent with chat interface.
支持多层级柱状图（ECharts）+ 其他图表（Chart.js）
"""
from httpx import delete
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

# ---------- 独立的多层级图表模块 ----------
from hierarchical_chart import render_hierarchical_bar

st.set_page_config(layout="wide")  # 添加此行
# 禁用Streamlit默认的 metrics 跟踪（解决fivetran连接错误）
# st.set_option('client.showErrorDetails', False)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("frontend.log", encoding="utf-8"),
        logging.FileHandler("error.log", encoding="utf-8", mode="a"),
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

    if result.get("answer", ""):
        with answer_placeholder.container():
            st.markdown("**回答:**")
            st.markdown(result['answer'])
        status_placeholder.empty()
        logger.debug("Answer streamed, tool messages cleared")

    if result.get("viz_data") and result.get("viz_type") != "none":
        with chart_placeholder.container():
            st.markdown("**图表:**")
            try:
                viz_type = result.get("viz_type")
                viz_data = result.get("viz_data")

                # if viz_type == "hierarchical_bar":
                #     # 使用独立模块渲染 ECharts 多层级图
                #     html = render_hierarchical_bar(viz_data, height=520)
                #     components.html(html, height=520, scrolling=True)
                # 在 stream_response 中
                if viz_type == "hierarchical_bar":
                    raw_viz = result["viz_data"].get("raw_data") or result["viz_data"]
                    if "raw_data" not in result["viz_data"] and "data" in raw_viz:
                        cfg = raw_viz
                        raw_viz = {
                            "title": cfg["options"]["plugins"]["title"]["text"],
                            "xLabel": cfg["options"]["scales"]["x"]["title"]["text"],
                            "yLabel": cfg["options"]["scales"]["y"]["title"]["text"],
                            "labels": cfg["data"]["labels"],
                            "values": [{"label": ds["label"], "data": ds["data"]} for ds in cfg["data"]["datasets"]]
                        }
                    # html = render_hierarchical_bar(raw_viz, height=900)
                    # components.html(html, height=900, scrolling=True)
                    html = render_hierarchical_bar(raw_viz, height=700)  # 高度 600px，清晰不占屏
                    components.html(
                        html,
                        height=700,
                        width=1400,        # 强制宽度 1400px（可根据屏幕调）
                        scrolling=False    # 关闭滚动条，图表自适应
                        # component_iframe_attrs={"style": "width: 100% !important; min-width: 2000px;"}
                    )
                    
                else:
                    # 其他图表使用 Chart.js（保持原有逻辑）
                    chart_id = f"chart_{uuid.uuid4().hex}"
                    chart_json = json.dumps(viz_data)

                    html = f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.js"></script>
                        <script src="https://cdn.jsdelivr.net/npm/@kurkle/color@0.3.2/dist/color.umd.min.js"></script>
                        <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-hierarchical@4.4.2/build/index.umd.min.js"></script>
                    </head>
                    <body>
                        <div style="width: 100%; height: 480px; overflow: auto;">
                            <canvas id="{chart_id}"></canvas>
                        </div>
                        <script>
                        document.addEventListener('DOMContentLoaded', function () {{
                            try {{
                                function registerHierarchicalPlugin() {{
                                    if (!window.Chart) throw new Error('Chart.js not loaded');

                                    const candidateNames = [
                                        'chartjs-plugin-hierarchical',
                                        'chartjsPluginHierarchical',
                                        'ChartjsPluginHierarchical',
                                        'ChartHierarchicalPlugin',
                                        'HierarchicalPlugin',
                                        'HierarchicalScale',
                                    ];

                                    const candidates = candidateNames.map(n => window[n]).filter(x => !!x);
                                    const maybeDefaultCandidates = candidateNames
                                        .map(n => (window[n] && window[n].default) ? window[n].default : null)
                                        .filter(x => !!x);
                                    const allCandidates = Array.from(new Set([...candidates, ...maybeDefaultCandidates]));

                                    if (window.HierarchicalScale) {{
                                        try {{
                                            Chart.register(window.HierarchicalScale);
                                            console.log('Registered HierarchicalScale from window.HierarchicalScale');
                                            return true;
                                        }} catch (e) {{
                                            console.warn('Failed to register window.HierarchicalScale:', e);
                                        }}
                                    }}

                                    for (const p of allCandidates) {{
                                        try {{
                                            if (typeof p === 'object') {{
                                                if (p.HierarchicalScale) {{
                                                    Chart.register(p.HierarchicalScale);
                                                    console.log('Registered candidate.HierarchicalScale');
                                                    return true;
                                                }}
                                                if (p.scale && p.scale.HierarchicalScale) {{
                                                    Chart.register(p.scale.HierarchicalScale);
                                                    console.log('Registered candidate.scale.HierarchicalScale');
                                                    return true;
                                                }}
                                                try {{
                                                    Chart.register(p);
                                                    console.log('Registered candidate plugin object');
                                                    return true;
                                                }} catch (err) {{
                                                    console.warn('Attempt to Chart.register(candidate) failed:', err);
                                                }}
                                            }}
                                            if (typeof p === 'function') {{
                                                try {{
                                                    p(Chart);
                                                    console.log('Called candidate install function with Chart');
                                                    return true;
                                                }} catch (err) {{
                                                    try {{
                                                        Chart.register(p);
                                                        console.log('Registered candidate function as plugin');
                                                        return true;
                                                    }} catch (err2) {{
                                                        console.warn('Failed to register function candidate:', err2);
                                                    }}
                                                }}
                                            }}
                                        }} catch (e) {{
                                            console.warn('Candidate plugin registration attempt failed, trying next. Error:', e);
                                        }}
                                    }}

                                    console.warn('No hierarchical plugin found or registration failed. Candidates checked:', allCandidates.length);
                                    return false;
                                }}

                                const registered = registerHierarchicalPlugin();

                                var ctx = document.getElementById('{chart_id}').getContext('2d');
                                var config = {chart_json};

                                try {{
                                    var xScale = config.options && config.options.scales && config.options.scales.x;
                                    if (xScale && xScale._hierarchical) {{
                                        if (registered && Chart.registry.getScale && Chart.registry.getScale('hierarchical')) {{
                                            xScale.type = 'hierarchical';
                                            xScale.hierarchical = xScale.hierarchical || {{}};
                                            Object.assign(xScale.hierarchical, xScale._hierarchical);
                                            delete xScale._hierarchical;
                                        }} else if (registered) {{
                                            xScale.type = 'hierarchical';
                                            xScale.hierarchical = xScale.hierarchical || {{}};
                                            Object.assign(xScale.hierarchical, xScale._hierarchical);
                                            delete xScale._hierarchical;
                                        }} else {{
                                            console.warn('Hierarchical plugin not available; using category axis.');
                                            xScale.type = 'category';
                                        }}
                                    }}
                                }} catch (err) {{
                                    console.warn('Error while applying hierarchical config migration:', err);
                                }}

                                if (config.options && config.options.plugins && config.options.plugins.zoom && !window.ChartZoom) {{
                                    delete config.options.plugins.zoom;
                                }}

                                var myChart = new Chart(ctx, config);
                                console.log('Chart created successfully');

                            }} catch (e) {{
                                console.error('Chart creation error:', e);
                                try {{
                                    var ctx = document.getElementById('{chart_id}').getContext('2d');
                                    var config = {chart_json};
                                    config.type = 'bar';
                                    if (config.options && config.options.scales && config.options.scales.x) {{
                                        config.options.scales.x.type = 'category';
                                        if (config.options.scales.x._hierarchical) delete config.options.scales.x._hierarchical;
                                        if (config.options.scales.x.hierarchical) delete config.options.scales.x.hierarchical;
                                    }}
                                    var myChart = new Chart(ctx, config);
                                    console.log('Fallback chart created');
                                }} catch (fallbackError) {{
                                    console.error('Fallback also failed:', fallbackError);
                                }}
                            }}
                        }});
                        </script>
                    </body>
                    </html>
                    """
                    components.html(html, height=500, scrolling=False)
                    st.code(f"DEBUG viz_data: {json.dumps(result.get('viz_data'), ensure_ascii=False)[:1000]}")

            except Exception as e:
                st.error("抱歉，图表渲染失败，请稍后重试或联系支持。")
                logger.error(f"Chart rendering failed: {traceback.format_exc()}")

    if result.get("tables"):
        with table_placeholder.container():
            st.markdown("**表格:**")
            for table in result["tables"]:
                if table.get("data"):
                    st.markdown(f"**{table['title']}**")
                    st.dataframe(pd.DataFrame(table["data"]))

# ------------------- SESSION STATE 初始化（关键修复） -------------------
def _ensure_thread_state(thread_id: str):
    """确保当前 thread_id 对应的所有字典都已初始化"""
    if thread_id not in st.session_state.chat_history:
        st.session_state.chat_history[thread_id] = []
    if thread_id not in st.session_state.tool_history:
        st.session_state.tool_history[thread_id] = []
    if thread_id not in st.session_state.first_questions:
        st.session_state.first_questions[thread_id] = ""

# 基础字典
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}
if "tool_history" not in st.session_state:
    st.session_state.tool_history = {}
if "first_questions" not in st.session_state:
    st.session_state.first_questions = {}

# 当前线程
if "current_thread_id" not in st.session_state:
    st.session_state.current_thread_id = str(uuid.uuid4())

# 关键：每次页面加载都确保当前 thread 已初始化
_ensure_thread_state(st.session_state.current_thread_id)

# ------------------- 侧边栏：文件上传 & DB 重建 -------------------
with st.sidebar:
    st.title("文件上传与数据库重建")
    uploaded_files = st.file_uploader("上传 Excel 或 CSV 文件", accept_multiple_files=True, type=["xlsx", "csv"])
    sheet_configs = {}
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name.endswith(".xlsx"):
                sheets = load_excel_sheets(uploaded_file.getvalue(), uploaded_file.name)
                if sheets:
                    st.markdown(f"**{uploaded_file.name} 工作表选择:**")
                    selected_sheets = []
                    for sheet in sheets:
                        selected = st.checkbox(sheet, value=True, key=f"{uploaded_file.name}_{sheet}")
                        if selected:
                            selected_sheets.append((sheet, None))
                    sheet_configs[uploaded_file.name] = selected_sheets
                else:
                    st.warning(f"无法加载 {uploaded_file.name} 的工作表。")
    if st.button("重建数据库"):
        rebuild_db(uploaded_files, sheet_configs)

# ------------------- 侧边栏：对话线程管理 -------------------
with st.sidebar:
    st.title("对话线程")
    if st.button("新对话"):
        new_thread = str(uuid.uuid4())
        st.session_state.current_thread_id = new_thread
        _ensure_thread_state(new_thread)   # 立即初始化
        st.rerun()

    # 按创建顺序倒序显示（最近的在上面）
    for thread_id in reversed(list(st.session_state.chat_history.keys())):
        first_q = st.session_state.first_questions.get(thread_id, "未知问题")
        label = f"{first_q[:30]}... ({thread_id[:8]})" if len(first_q) > 30 else f"{first_q} ({thread_id[:8]})"
        if st.button(label, key=f"switch_{thread_id}"):
            st.session_state.current_thread_id = thread_id
            _ensure_thread_state(thread_id)
            st.rerun()

# ------------------- 主聊天界面 -------------------
st.title("SQL Agent with Viz - Streamlit Frontend")
chat_container = st.container()

# 显示历史消息
with chat_container:
    for msg in st.session_state.chat_history[st.session_state.current_thread_id]:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(msg.content)
                if hasattr(msg, "chart_config") and msg.chart_config:
                    st.markdown("**图表:**")
                    components.html(msg.chart_config, height=500)
                if hasattr(msg, "tables") and msg.tables:
                    st.markdown("**表格:**")
                    for table in msg.tables:
                        st.markdown(f"**{table['title']}**")
                        st.dataframe(pd.DataFrame(table["data"]))

# ------------------- 输入框 -------------------
prompt = st.chat_input("输入您的查询 (例如: '2025年每个月，QYZNJ机房，光模块的故障数，按光模块型号和厂商分布，画折线图？')")
if prompt:
    user_ip = st.query_params.get("user_ip", "unknown")
    logger.info(f"Query from IP {user_ip}, thread_id {st.session_state.current_thread_id}: {prompt}")

    # 记录首次问题（用于侧边栏展示）
    if not st.session_state.first_questions[st.session_state.current_thread_id]:
        st.session_state.first_questions[st.session_state.current_thread_id] = (
            prompt[:50] + "..." if len(prompt) > 50 else prompt
        )

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
                    # 过滤仅保留 Human/AI 消息（不带 tool_calls）
                    filtered_messages = [
                        msg for msg in st.session_state.chat_history[st.session_state.current_thread_id]
                        if isinstance(msg, (HumanMessage, AIMessage))
                        and not (isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None))
                    ]

                    inputs = {
                        "messages": filtered_messages + [user_message],
                        "question": prompt,
                        "tool_history": st.session_state.tool_history.get(st.session_state.current_thread_id, []),
                        "status_messages": []
                    }
                    config = {"configurable": {"thread_id": st.session_state.current_thread_id}}
                    result = process_query(
                        graph, inputs, config,
                        lambda msg: status_placeholder.markdown(translate_status_message(msg))
                    )
                    stream_response(result, status_placeholder, answer_placeholder, chart_placeholder, table_placeholder)

                    # 保存 assistant 消息（含图表/表格）
                    assistant_message = AIMessage(content=result.get("answer", ""))
                    if result.get("tables"):
                        assistant_message.tables = result["tables"]
                    if result.get("viz_data"):
                        assistant_message.chart_config = result["viz_data"]

                    # 更新历史（保留所有消息，供后续上下文）
                    st.session_state.chat_history[st.session_state.current_thread_id] = (
                        result.get("messages", []) + [assistant_message]
                    )
                    # 只保留关键工具历史
                    st.session_state.tool_history[st.session_state.current_thread_id] = [
                        h for h in result.get("tool_history", [])
                        if h["tool"] in [
                            "sql_db_list_tables", "sql_db_schema",
                            "sql_db_query", "sql_db_query_checker", "check_result"
                        ]
                    ]

                except Exception as e:
                    status_placeholder.error("抱歉，处理查询时发生错误，请稍后重试或联系支持。")
                    logger.error(f"Query failed for IP {user_ip}: {traceback.format_exc()}")
                    err_msg = AIMessage(content="抱歉，处理查询时发生错误，请稍后重试或联系支持。")
                    st.session_state.chat_history[st.session_state.current_thread_id].append(err_msg)

# ------------------- 退出指令 -------------------
if prompt and prompt.lower() in ["exit", "quit"]:
    st.write("退出程序。")
    logger.info("User exited the program")
    st.stop()