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
                viz_type = (result.get("viz_type") or "").lower()
                viz_data = result.get("viz_data") or {}

                # --- hierarchical_bar (ECharts) passthrough (不变) ---
                if viz_type == "hierarchical_bar":
                    raw_viz = viz_data.get("raw_data") if isinstance(viz_data, dict) and "raw_data" in viz_data else viz_data
                    if isinstance(raw_viz, dict) and "data" in raw_viz and isinstance(raw_viz["data"], dict) and "labels" in raw_viz["data"]:
                        cfg = raw_viz
                        raw_viz = {
                            "title": cfg.get("options", {}).get("plugins", {}).get("title", {}).get("text", cfg.get("title", "")),
                            "xLabel": cfg.get("options", {}).get("scales", {}).get("x", {}).get("title", {}).get("text", cfg.get("xLabel", "")),
                            "yLabel": cfg.get("options", {}).get("scales", {}).get("y", {}).get("title", {}).get("text", cfg.get("yLabel", "")),
                            "labels": cfg["data"].get("labels", []),
                            "values": [{"label": ds.get("label", f"series_{i}"), "data": ds.get("data", [])} for i, ds in enumerate(cfg["data"].get("datasets", []))]
                        }
                    html, computed_height = render_hierarchical_bar(raw_viz, height=700)
                    components.html(html, height=computed_height, width=1600, scrolling=True)
                    result["viz"] = None


                # If result says "pie", force build proper pie config from common input shapes
                # if result.get("viz") and viz_type != "hierarchical_bar":
                if viz_type != "hierarchical_bar" and viz_data:  # 移除对result.get("viz")的依赖，直接检查图表类型和数据
                    # --- for other charts, prefer the viz_type from result (fix pie case) ---
                    # attempt to get a normal chart config from viz_data (raw_data preferred)
                    data_in = viz_data.get("raw_data") if isinstance(viz_data, dict) and "raw_data" in viz_data else viz_data
                    if viz_type == "pie":
                        # three common shapes:
                        # A) { "data": [ {"label": "...", "value": N}, ... ] }
                        # B) chartjs-like: { "type":"pie", "data": {"labels": [...], "datasets":[{"data":[...]}] } }
                        # C) simplified: { "labels":[...], "values":[...] } or { "labels": [...], "values":[{"label","value"}] }
                        labels = []
                        values = []
                        if isinstance(data_in, dict) and isinstance(data_in.get("data"), list) and all(isinstance(x, dict) and "label" in x and "value" in x for x in data_in["data"]):
                            labels = [str(x["label"]) for x in data_in["data"]]
                            values = [float(x["value"]) for x in data_in["data"]]
                        elif isinstance(data_in, dict) and isinstance(data_in.get("data"), dict):
                            d = data_in["data"]
                            if "labels" in d and "datasets" in d and isinstance(d["datasets"], list) and len(d["datasets"])>0:
                                labels = d.get("labels", [])
                                values = d["datasets"][0].get("data", [])
                        elif isinstance(data_in, dict) and "data" in data_in and isinstance(data_in["data"], list):
                            # fallback: possibly list of {"label","value"}
                            arr = data_in["data"]
                            if all(isinstance(x, dict) and "label" in x and ("value" in x or "data" in x) for x in arr):
                                labels = [str(x.get("label")) for x in arr]
                                values = [float(x.get("value", x.get("data", 0))) for x in arr]
                        elif isinstance(data_in, dict) and "labels" in data_in and "values" in data_in:
                            if all(isinstance(x, (int,float)) for x in data_in["values"]):
                                labels = data_in["labels"]
                                values = data_in["values"]
                            elif all(isinstance(x, dict) and ("value" in x or "data" in x) for x in data_in["values"]):
                                labels = [x.get("label") for x in data_in["values"]]
                                values = [float(x.get("value", x.get("data", 0))) for x in data_in["values"]]

                        labels = labels or []
                        values = values or []

                        # default color palette
                        colors = ["#36A2EB", "#FF6384", "#FFCE56", "#4BC0C0", "#9966FF", "#FF9F40", "#66FF66", "#999999"]
                        bg = [colors[i % len(colors)] for i in range(len(values))]

                        chart_cfg = {
                            "type": "pie",
                            "data": {"labels": labels, "datasets": [{"data": values, "backgroundColor": bg}]},
                            "options": {
                                "responsive": True,
                                "maintainAspectRatio": False,
                                "plugins": {"title": {"display": True, "text": viz_data.get("title", "")}, "legend": {"display": True, "position": "top"}}
                            },
                            "raw_data": viz_data
                        }

                    else:
                        # existing synthesis logic for non-pie charts (reuse your previous branch)
                        is_chartjs_cfg = isinstance(data_in, dict) and isinstance(data_in.get("data"), dict) and ("datasets" in data_in["data"] or "labels" in data_in["data"])
                        if is_chartjs_cfg:
                            chart_cfg = data_in.copy()
                        else:
                            chart_type = viz_type if isinstance(viz_type, str) and viz_type else data_in.get("type", "line")
                            labels = None
                            datasets = []

                            if isinstance(data_in, dict) and "xValues" in data_in and "yValues" in data_in:
                                labels = data_in.get("xValues", [])
                                raw_series = data_in.get("yValues", [])
                                for i, s in enumerate(raw_series):
                                    lab = s.get("label", f"series_{i}")
                                    series_data = s.get("data", [])
                                    ds = {"label": lab, "data": series_data, "fill": False}
                                    if isinstance(s.get("color"), str):
                                        ds["borderColor"] = s["color"]
                                    datasets.append(ds)
                            elif isinstance(data_in, dict) and "labels" in data_in and "values" in data_in:
                                labels = data_in.get("labels", [])
                                vals = data_in.get("values", [])
                                if len(vals) > 0 and isinstance(vals[0], dict) and "data" in vals[0]:
                                    for v in vals:
                                        ds = {"label": v.get("label", ""), "data": v.get("data", []), "fill": False}
                                        if v.get("borderColor"):
                                            ds["borderColor"] = v.get("borderColor")
                                        datasets.append(ds)
                                else:
                                    datasets.append({"label": data_in.get("yLabel", "value"), "data": vals, "fill": False})
                            elif isinstance(data_in, dict) and isinstance(data_in.get("data"), dict) and ("labels" in data_in["data"] or "datasets" in data_in["data"]):
                                chart_cfg = data_in.copy()
                            else:
                                maybe_labels = data_in.get("x") or data_in.get("labels") or data_in.get("xValues")
                                maybe_vals = data_in.get("y") or data_in.get("values") or data_in.get("yValues")
                                if maybe_labels and maybe_vals:
                                    labels = maybe_labels
                                    if len(maybe_vals) > 0 and isinstance(maybe_vals[0], dict) and "data" in maybe_vals[0]:
                                        for v in maybe_vals:
                                            datasets.append({"label": v.get("label", ""), "data": v.get("data", []), "fill": False})
                                    elif isinstance(maybe_vals, list) and all(isinstance(x, (int, float)) for x in maybe_vals):
                                        datasets.append({"label": data_in.get("yLabel", "value"), "data": maybe_vals, "fill": False})

                            if 'chart_cfg' not in locals():
                                chart_cfg = {
                                    "type": chart_type or "line",
                                    "data": {"labels": labels or [], "datasets": datasets},
                                    "options": {
                                        "responsive": True,
                                        "maintainAspectRatio": False,
                                        "plugins": {"title": {"display": True, "text": data_in.get("title", "") if isinstance(data_in, dict) else ""}}
                                    }
                                }
                                # add default scales for common charts
                                chart_cfg["options"]["scales"] = {
                                    "x": {"title": {"display": bool(data_in.get("xLabel") if isinstance(data_in, dict) else False), "text": data_in.get("xLabel", "") if isinstance(data_in, dict) else ""}},
                                    "y": {"beginAtZero": True, "title": {"display": bool(data_in.get("yLabel") if isinstance(data_in, dict) else False), "text": data_in.get("yLabel", "") if isinstance(data_in, dict) else ""}}
                                }

                    # Finally: render chart_cfg via Chart.js in an iframe html (like before)
                    chart_json = json.dumps(chart_cfg, ensure_ascii=False)
                    chart_id = f"chart_{uuid.uuid4().hex}"
                    # ----------------- 替换前的 chart HTML 生成处，改为内联 Chart.js & Color -----------------
                    import pathlib

                    # 计算 frontend/static/js 目录（以 app.py 所在目录为基准）
                    _here = pathlib.Path(__file__).resolve()
                    # 假设 app.py 在 frontend/ 目录下，则 parents[0] 即项目 frontend；调整 parents[n] 如有不同
                    js_dir = _here.parent / "static" / "js"

                    def _read_js_text(name):
                        p = js_dir / name
                        try:
                            return p.read_text(encoding="utf-8")
                        except Exception as e:
                            # 若读不到文件，插入一段警告脚本，避免页面崩溃
                            return f'console.warn("缺少本地JS文件: {name} - {e}");'

                    chart_js_text = _read_js_text("chart.umd.js")
                    color_js_text = _read_js_text("color.min.js")

                    # 把 chart_js_text、color_js_text 内联到 iframe HTML 中
                    chart_json = json.dumps(chart_cfg, ensure_ascii=False)
                    chart_id = f"chart_{uuid.uuid4().hex}"

                    html = f"""
                    <!DOCTYPE html><html><head><meta charset='utf-8'/><meta name='viewport' content='width=device-width, initial-scale=1'/>
                    <!-- inline color.min.js -->
                    <script>
                    {color_js_text}
                    </script>
                    <!-- inline chart.umd.js -->
                    <script>
                    {chart_js_text}
                    </script>
                    </head><body>
                    <div style="width:100%; height:480px; overflow:auto;">
                    <canvas id="{chart_id}"></canvas>
                    </div>
                    <script>
                    document.addEventListener('DOMContentLoaded', function () {{
                    try {{
                        var ctx = document.getElementById('{chart_id}').getContext('2d');
                        var config = {chart_json};
                        config.options = config.options || {{}};
                        config.options.responsive = true;
                        config.options.maintainAspectRatio = false;
                        new Chart(ctx, config);
                    }} catch (e) {{
                        console.error('Chart creation error:', e);
                    }}
                    }});</script></body></html>
                    """
                    components.html(html, height=500, scrolling=False)
                    logger.debug("Viz data (backend only): %s", json.dumps(result.get("viz_data"), ensure_ascii=False)[:4000])

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
# 替换 app.py 中原有 prompt 处理与 process_query 同步调用的部分
prompt = st.chat_input("输入您的查询 (例如: '2025年每个月，QYZNJ机房，光模块的故障数，按光模块型号和厂商分布，画折线图？')")

if prompt:
    user_ip = st.query_params.get("user_ip", "unknown")
    logger.info(f"Query from IP {user_ip}, thread_id {st.session_state.current_thread_id}: {prompt}")

    # 记录首次问题（用于侧边栏展示）
    if not st.session_state.first_questions[st.session_state.current_thread_id]:
        st.session_state.first_questions[st.session_state.current_thread_id] = (
            prompt[:50] + "." if len(prompt) > 50 else prompt
        )

    user_message = HumanMessage(content=prompt)
    st.session_state.chat_history[st.session_state.current_thread_id].append(user_message)

    # UI placeholders
    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)

        # Show assistant message container (will be updated by embedded component)
        with st.chat_message("assistant"):
            status_placeholder = st.empty()
            answer_placeholder = st.empty()
            chart_placeholder = st.empty()
            table_placeholder = st.empty()
            status_placeholder.markdown("准备发送任务到后端...")

            # Start SSE backend task
            try:
                SSE_API_BASE = "http://localhost:8000"  # 如果部署到域名或不同端口，请修改为你的后端地址
                payload = {"user_id": "streamlit_user", "query": prompt, "params": {}}
                import requests
                resp = requests.post(f"{SSE_API_BASE}/start_task", json=payload, timeout=15)
                if resp.status_code != 200 and resp.status_code != 201:
                    status_placeholder.error(f"启动任务失败: {resp.status_code} {resp.text}")
                else:
                    task_id = resp.json().get("task_id")
                    status_placeholder.markdown(f"任务已启动：{task_id}，正在连接事件流...")

                    # 直接把 SSE 前端逻辑放到一个 HTML component 中，避免 Streamlit 主线程轮询复杂化
                    component_html_template = f"""
                    <!-- component_html (replace prior component_html string in app.py) -->
                    <!doctype html>
                    <html>
                    <head>
                    <meta charset="utf-8">
                    <title>Agent Stream (compact)</title>
                    <style>
                        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial; margin:8px; }}
                        #container {{ max-width: 100%; }}
                        #mini {{ display:flex; gap:8px; align-items:center; }}
                        #openBtn {{ padding:6px 10px; border-radius:6px; border:1px solid #ddd; background:#f7f7f7; cursor:pointer; }}
                        #statusInline {{ color:#666; font-size:13px; }}
                        #panel {{ display:none; border:1px solid #eee; padding:10px; margin-top:10px; border-radius:6px; box-shadow: 0 1px 2px rgba(0,0,0,0.03); background:#fff; }}
                        #answer {{ white-space: pre-wrap; min-height:40px; margin-bottom:8px; }}
                        .sectionTitle {{ font-weight:600; margin-top:8px; margin-bottom:6px; font-size:14px; color:#222; }}
                        #table {{ max-height:300px; overflow:auto; border-top:1px solid #f0f0f0; padding-top:8px; }}
                        #chart {{ margin-top:8px; }}
                        .muted {{ color:#888; font-size:13px; }}
                        .hidden {{ display:none; }}
                        .error {{ color:#b00020; font-weight:600; }}
                    </style>
                    </head>
                    <body>
                    <div id="container">
                        <div id="mini">
                        <div id="openBtn">打开实时结果</div>
                        <div id="statusInline">状态：等待任务启动...</div>
                        </div>

                        <div id="panel" role="region" aria-live="polite">
                        <div class="sectionTitle">即时回答</div>
                        <div id="answer" class="muted">尚无回答</div>

                        <div id="tableWrapper" class="hidden">
                            <div class="sectionTitle">表格</div>
                            <div id="table">尚无数据</div>
                        </div>

                        <div id="chartWrapper" class="hidden">
                            <div class="sectionTitle">图表</div>
                            <div id="chart">尚无图表</div>
                        </div>

                        <div id="finalArea" class="muted" style="margin-top:8px;">总耗时：-</div>
                        <div id="errorArea" class="error hidden"></div>
                        </div>
                    </div>

                    <script>
                        (function(){{
                        const SSE_API_BASE = "{{SSE_API_BASE}}";
                        const TASK_ID = "{{task_id}}";
                        const eventsUrl = SSE_API_BASE + "/events?task_id=" + TASK_ID;

                        const openBtn = document.getElementById('openBtn');
                        const statusInline = document.getElementById('statusInline');
                        const panel = document.getElementById('panel');
                        const answerEl = document.getElementById('answer');
                        const tableWrapper = document.getElementById('tableWrapper');
                        const chartWrapper = document.getElementById('chartWrapper');
                        const tableEl = document.getElementById('table');
                        const chartEl = document.getElementById('chart');
                        const finalArea = document.getElementById('finalArea');
                        const errorArea = document.getElementById('errorArea');

                        let evtSource = null;
                        let connected = false;
                        let reconnectAttempts = 0;

                        function showPanel() {{
                            panel.style.display = 'block';
                        }}

                        function hidePanel() {{
                            panel.style.display = 'none';
                        }}

                        function safeParse(data) {{
                            try {{ return JSON.parse(data); }} catch(e) {{ return null; }}
                        }}

                        function renderTable(payload) {{
                            try {{
                            const cols = payload.columns || (payload.rows && payload.rows.length ? Object.keys(payload.rows[0]) : []);
                            const rows = payload.rows || [];
                            if(!cols.length) {{
                                tableEl.innerText = '无列数据';
                                return;
                            }}
                            let html = '<table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse;width:100%"><thead><tr>';
                            for (let c of cols) html += `<th style="background:#fafafa;text-align:left">${{c}}</th>`;
                            html += '</tr></thead><tbody>';
                            for (let r of rows) {{
                                html += '<tr>';
                                for (let c of cols) html += `<td>${{r[c] !== undefined ? r[c] : ''}}</td>`;
                                html += '</tr>';
                            }}
                            html += '</tbody></table>';
                            tableEl.innerHTML = html;
                            }} catch (e) {{
                            tableEl.innerText = '渲染表格失败: ' + e;
                            }}
                        }}

                        function renderViz(payload) {{
                            try {{
                            // Minimal rendering: label/value list for bar chart; extend if you embed ECharts later
                            if(payload && payload.chart && payload.chart.type === 'bar') {{
                                const labels = payload.chart.data.labels || [];
                                const values = payload.chart.data.values || [];
                                let html = '<div>';
                                for(let i=0;i<labels.length;i++) {{
                                html += `<div style="display:flex; justify-content:space-between; padding:4px 0;"><div>${{labels[i]}}</div><div>${{values[i]}}</div></div>`;
                                }}
                                html += '</div>';
                                chartEl.innerHTML = html;
                            }} else {{
                                chartEl.innerText = JSON.stringify(payload, null, 2);
                            }}
                            }} catch(e) {{
                            chartEl.innerText = '渲染图表失败: ' + e;
                            }}
                        }}

                        function startSSE() {{
                            if(evtSource) {{
                            try {{ evtSource.close(); }} catch(e) {{}}
                            evtSource = null;
                            }}
                            evtSource = new EventSource(eventsUrl);
                            connected = false;
                            evtSource.onopen = function() {{
                            connected = true;
                            reconnectAttempts = 0;
                            statusInline.textContent = '状态：已连接事件流';
                            showPanel();
                            }};
                            evtSource.onmessage = function(e) {{
                            const ev = safeParse(e.data);
                            if(!ev) return;
                            // Ignore pings
                            if(ev.type === 'ping') {{
                                // console.debug('ping', ev);
                                return;
                            }}
                            // Show panel on first meaningful event
                            if(!connected) {{ connected = true; showPanel(); }}
                            if(ev.type === 'status') {{
                                statusInline.textContent = '状态：' + (ev.message || '');
                                return;
                            }}
                            if(ev.type === 'text_chunk') {{
                                if(answerEl.classList.contains('muted')) answerEl.classList.remove('muted');
                                if(answerEl.textContent === '尚无回答') answerEl.textContent = '';
                                answerEl.textContent += ev.chunk;
                                return;
                            }}
                            if(ev.type === 'text_end') {{
                                statusInline.textContent = '状态：文本生成完成';
                                return;
                            }}
                            if(ev.type === 'tool_event') {{
                                // optionally log tool events to console for debugging
                                console.log('tool_event', ev);
                                return;
                            }}
                            if(ev.type === 'table_ready') {{
                                tableWrapper.classList.remove('hidden');
                                renderTable(ev.payload || {{}});
                                statusInline.textContent = '状态：表格就绪';
                                return;
                            }}
                            if(ev.type === 'viz_ready') {{
                                chartWrapper.classList.remove('hidden');
                                renderViz(ev.payload || {{}});
                                statusInline.textContent = '状态：图表就绪';
                                return;
                            }}
                            if(ev.type === 'final') {{
                                finalArea.textContent = '总耗时：' + (ev.processing_time ? ev.processing_time.toFixed(2) + ' 秒' : '-');
                                if(ev.status && ev.status !== 'success') {{
                                errorArea.classList.remove('hidden');
                                errorArea.textContent = '处理出错: ' + (ev.error || '未知错误');
                                statusInline.textContent = '状态：已终止（错误）';
                                }} else {{
                                statusInline.textContent = '状态：已完成';
                                }}
                                try {{ evtSource.close(); }} catch(e){{}}
                                return;
                            }}
                            }};
                            evtSource.onerror = function(err) {{
                            statusInline.textContent = '状态：事件流错误或断开（将尝试重连）';
                            console.warn('SSE error', err);
                            // Auto reconnect with backoff
                            try {{ evtSource.close(); }} catch(e){{}}
                            evtSource = null;
                            reconnectAttempts++;
                            const backoff = Math.min(30, 1 + reconnectAttempts * 2);
                            setTimeout(() => startSSE(), backoff * 1000);
                            }};
                        }}

                        // initial actions: clicking open button triggers SSE connect and shows panel
                        openBtn.addEventListener('click', function(){{
                            statusInline.textContent = '状态：正在连接事件流...';
                            startSSE();
                            // hide button to avoid re-clicks
                            openBtn.style.display = 'none';
                        }});

                        // Optional: auto-open immediately
                        // openBtn.click();
                        }})();
                    </script>
                    </body>
                    </html>
                    """
                    component_html = component_html_template.replace("{SSE_API_BASE}", SSE_API_BASE).replace("{task_id}", task_id)
            except Exception as ex:
                status_placeholder.error(f"任务启动失败: {ex}")

# ------------------- 退出指令 -------------------
if prompt and prompt.lower() in ["exit", "quit"]:
    st.write("退出程序。")
    logger.info("User exited the program")
    st.stop()