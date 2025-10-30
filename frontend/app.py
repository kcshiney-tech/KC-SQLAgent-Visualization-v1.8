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
                    # 调用新版 render_hierarchical_bar，取得 html 与计算高度
                    html, computed_height = render_hierarchical_bar(raw_viz, height=700)
                    # 不传固定宽度，交由 Streamlit 容器自适应（配合 st.set_page_config(layout="wide")）
                    components.html(
                        html,
                        height=computed_height,
                        width=1600,
                        scrolling=True   # 打开滚动以保证在高度/宽度超出时仍然可访问
                    )
                    
                else:
                    # -----------------------------
                    # 兼容各种非-hierarchical 的 viz_data 输入格式，构建 Chart.js config
                    # -----------------------------
                    chart_id = f"chart_{uuid.uuid4().hex}"

                    # 原始 viz_data（可能是 Chart.js config，也可能是内部简化格式）
                    data_in = viz_data

                    # 如果已经是 Chart.js 完整 config（含 data.datasets 或 data.labels），直接使用
                    is_chartjs_cfg = isinstance(data_in, dict) and ("data" in data_in and ("datasets" in data_in["data"] or "labels" in data_in["data"]))
                    # 另外支持 legacy keys: 'type' + 'data' as Chart.js config
                    if is_chartjs_cfg:
                        chart_cfg = data_in.copy()
                    else:
                        # 兼容常见内部格式 -> 转换成 Chart.js config
                        # 支持的内部格式包括（但不限于）：
                        # 1) { "xValues": [...], "yValues":[ {"label":..,"data":[...]}, ... ], "title":..., "yLabel":... }
                        # 2) { "labels": [...], "values": [ { "label": ..., "data": [...] }, ... ], "type": "bar" }
                        # 3) 简单 single-series: { "labels": [...], "values": [...] }  (values 为一维数组)
                        chart_type = data_in.get("type", "line")  # 默认 line
                        labels = None
                        datasets = []
                        if not isinstance(data_in, dict):
                            data_in = {}

                        # case A: xValues + yValues (来自时间序列)
                        if isinstance(data_in, dict) and "xValues" in data_in and "yValues" in data_in:
                            labels = data_in.get("xValues")
                            raw_series = data_in.get("yValues", [])
                            for i, s in enumerate(raw_series):
                                lab = s.get("label", f"series_{i}")
                                series_data = s.get("data", [])
                                datasets.append({
                                    "label": lab,
                                    "data": series_data,
                                    "fill": False
                                })
                        # case B: labels + values (values may be list-of-dicts or list-of-numbers)
                        elif isinstance(data_in, dict) and "labels" in data_in and "values" in data_in:
                            labels = data_in.get("labels")
                            vals = data_in.get("values")
                            # values may be list-of-dicts with label/data
                            if len(vals) > 0 and isinstance(vals[0], dict) and "data" in vals[0]:
                                for v in vals:
                                    datasets.append({
                                        "label": v.get("label", ""),
                                        "data": v.get("data", []),
                                        "fill": False
                                    })
                            else:
                                # single-series numeric array
                                datasets.append({
                                    "label": data_in.get("yLabel", "value"),
                                    "data": vals,
                                    "fill": False
                                })
                        # case C: older chart config already present under data.key (safety)
                        elif isinstance(data_in, dict) and isinstance(data_in.get("data"), dict) and ("labels" in data_in.get("data") or "datasets" in data_in.get("data")):
                        # elif isinstance(data_in.get("data"), dict) and ("labels" in data_in.get("data") or "datasets" in data_in.get("data")):
                            chart_cfg = data_in.copy()
                        else:
                            # 兜底：尝试提取 anything that looks like labels + series
                            maybe_labels = data_in.get("x") or data_in.get("labels") or data_in.get("xValues")
                            maybe_vals = data_in.get("y") or data_in.get("values") or data_in.get("yValues")
                            if maybe_labels and maybe_vals:
                                labels = maybe_labels
                                # try to coerce maybe_vals into datasets
                                if len(maybe_vals) > 0 and isinstance(maybe_vals[0], dict) and "data" in maybe_vals[0]:
                                    for v in maybe_vals:
                                        datasets.append({"label": v.get("label", ""), "data": v.get("data", []), "fill": False})
                                elif isinstance(maybe_vals, list) and all(isinstance(x, (int, float)) for x in maybe_vals):
                                    datasets.append({"label": data_in.get("yLabel", "value"), "data": maybe_vals, "fill": False})
                            else:
                                # 实在无法识别，退回显示错误信息的简单空图（避免前端 JS 崩溃）
                                chart_cfg = {
                                    "type": chart_type,
                                    "data": {"labels": [], "datasets": []},
                                    "options": {"responsive": True, "maintainAspectRatio": False}
                                }

                        # 如果未在早期分支中直接构建 chart_cfg，则在此合成
                        if 'chart_cfg' not in locals():
                            chart_cfg = {
                                "type": chart_type,
                                "data": {
                                    "labels": labels or [],
                                    "datasets": datasets
                                },
                                "options": {
                                    "responsive": True,
                                    "maintainAspectRatio": False,
                                    "plugins": {
                                        "title": {"display": True, "text": data_in.get("title", "")},
                                        "legend": {"display": True, "position": "top"}
                                    },
                                    "scales": {
                                        "x": {"title": {"display": bool(data_in.get("xLabel")), "text": data_in.get("xLabel", "")}},
                                        "y": {"title": {"display": bool(data_in.get("yLabel")), "text": data_in.get("yLabel", "")}}
                                    }
                                }
                            }

                    # 最终 chart_cfg 保证为 Chart.js 格式
                    chart_json = json.dumps(chart_cfg, ensure_ascii=False)

                    # 生成 HTML（保留之前的注册 plugin + 回退逻辑）
                    html = f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <meta charset="utf-8" />
                        <meta name="viewport" content="width=device-width, initial-scale=1"/>
                        <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.js"></script>
                        <script src="https://cdn.jsdelivr.net/npm/@kurkle/color@0.3.2/dist/color.umd.min.js"></script>
                    </head>
                    <body>
                        <div style="width:100%; height:480px; overflow:auto;">
                            <canvas id="{chart_id}"></canvas>
                        </div>
                        <script>
                        document.addEventListener('DOMContentLoaded', function () {{
                            try {{
                                var ctx = document.getElementById('{chart_id}').getContext('2d');
                                var config = {chart_json};
                                // ensure responsive + no aspect ratio if embed in resizable iframe
                                config.options = config.options || {{}};
                                config.options.responsive = true;
                                config.options.maintainAspectRatio = false;

                                var myChart = new Chart(ctx, config);
                            }} catch (e) {{
                                console.error('Chart creation error:', e);
                                // fallback: render minimal empty chart to avoid blank iframe
                                try {{
                                    var ctx = document.getElementById('{chart_id}').getContext('2d');
                                    var config = {{ type: 'bar', data: {{ labels: [], datasets: [] }}, options: {{ responsive: true, maintainAspectRatio: false }} }};
                                    new Chart(ctx, config);
                                }} catch (err) {{
                                    console.error('Fallback also failed', err);
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
st.title("SQL Agent")
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