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
prompt = st.chat_input("输入您的查询 (例如: '2025年每个月，QYZNJ机房，光模块的故障数，按光模块型号和厂商分布，画折线图？')")

# === 替换 app.py 中整个 if prompt: 块 ===
if prompt:
    user_ip = st.query_params.get("user_ip", "unknown")
    logger.info(f"Query from IP {user_ip}, thread_id {st.session_state.current_thread_id}: {prompt}")

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
            # === Grok 风格“思考中”折叠框（动态标题）===
            expander_title = st.empty()
            thinking_expander = st.expander("思考中...", expanded=True)
            with thinking_expander:
                status_lines = st.empty()

            # === 内容容器 ===
            answer_container = st.empty()
            chart_container = st.empty()
            table_container = st.empty()
            timing_container = st.empty()

            # === 耗时记录 ===
            timings = {
                "start": time.time(),
                "answer_start": None,
                "answer_end": None,
                "chart_start": None,
                "chart_end": None,
                "table_start": None,
                "table_end": None,
                "tool_times": []
            }

            intermediate = {
                "answer": "",
                "viz_data": None,
                "viz_type": "none",
                "tables": [],
                "tool_history": [],
                "status_lines": []
            }

            # === 追加状态行并更新标题 ===
            def add_status(text: str, phase: str = None):
                intermediate["status_lines"].append(text)
                status_lines.markdown("\n".join(intermediate["status_lines"]))
                if phase:
                    expander_title.markdown(f"**{phase}**")

            # === 手动流式处理 ===
            try:
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

                last_status = []

                for chunk in graph.stream(inputs, config, stream_mode="values"):
                    state = chunk

                    # === 1. 工具执行状态 ===
                    new_status = state.get("status_messages", [])
                    if new_status and new_status != last_status:
                        latest_msg = new_status[-1]
                        translated = translate_status_message(latest_msg)
                        add_status(f"- {translated}")
                        last_status = new_status

                    # === 2. 回答流式输出 ===
                    final_ai_msg = None
                    for msg in reversed(state.get("messages", [])):
                        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                            final_ai_msg = msg
                            break
                    if final_ai_msg and final_ai_msg.content:
                        new_text = final_ai_msg.content
                        if len(new_text) > len(intermediate["answer"]):
                            delta = new_text[len(intermediate["answer"]):]
                            intermediate["answer"] = new_text
                            if timings["answer_start"] is None:
                                timings["answer_start"] = time.time()
                                # 回答开始：追加图表提示 + 更新标题
                                add_status("- 正在生成图表~", "正在生成图表...")
                            # 流式显示
                            for char in delta:
                                answer_container.markdown(f"**回答：**\n{intermediate['answer']}")

                    # === 3. 图表生成 ===
                    if state.get("viz_data") and state.get("viz_type") != "none":
                        if intermediate["viz_data"] is None:
                            intermediate["viz_data"] = state["viz_data"]
                            intermediate["viz_type"] = state["viz_type"]
                            timings["chart_start"] = time.time()
                            # 渲染图表
                            html, height = render_hierarchical_bar(state["viz_data"])
                            chart_container.empty()
                            chart_container.markdown("**图表：**")
                            components.html(html, height=height, scrolling=True)
                            timings["chart_end"] = time.time()
                            # 图表完成：追加表格提示 + 更新标题
                            add_status("- 正在生成表格~", "正在生成表格...")

                    # === 4. 表格生成 ===
                    if state.get("tables"):
                        new_tables = state["tables"]
                        if new_tables != intermediate["tables"]:
                            intermediate["tables"] = new_tables
                            timings["table_start"] = time.time()
                            table_container.empty()
                            table_container.markdown("**表格：**")
                            for table in new_tables:
                                if table.get("data"):
                                    table_container.markdown(f"**{table['title']}**")
                                    table_container.dataframe(pd.DataFrame(table["data"]))
                            timings["table_end"] = time.time()
                            # 表格完成：更新标题为“已完成”
                            expander_title.markdown("**已完成**")

                    # === 5. 工具历史 ===
                    if state.get("tool_history"):
                        intermediate["tool_history"] = state["tool_history"]

                # 结束
                timings["answer_end"] = timings["answer_end"] or time.time()

            except Exception as e:
                logger.error(f"Stream error: {traceback.format_exc()}")
                answer_container.error("抱歉，处理查询时发生错误。")
                expander_title.markdown("**处理失败**")

            # === 最终耗时（精准分段）===
            total_time = time.time() - timings["start"]
            thinking_time = sum(t["time"] for t in timings["tool_times"]) if timings["tool_times"] else 0
            answer_time = (timings["answer_end"] - timings["answer_start"]) if timings["answer_start"] else 0
            chart_time = (timings["chart_end"] - timings["chart_start"]) if timings["chart_start"] and timings["chart_end"] else 0
            table_time = (timings["table_end"] - timings["table_start"]) if timings["table_start"] and timings["table_end"] else 0

            timing_parts = [f"**总耗时:** {total_time:.2f}s"]
            if thinking_time > 0:
                timing_parts.append(f"思考: {thinking_time:.2f}s")
            if answer_time > 0:
                timing_parts.append(f"回答: {answer_time:.2f}s")
            if chart_time > 0:
                timing_parts.append(f"图表: {chart_time:.2f}s")
            if table_time > 0:
                timing_parts.append(f"表格: {table_time:.2f}s")

            timing_container.caption(" | ".join(timing_parts))

            # === 保存历史 ===
            assistant_message = AIMessage(content=intermediate["answer"])
            if intermediate.get("tables"):
                assistant_message.tables = intermediate["tables"]
            if intermediate.get("viz_data"):
                assistant_message.chart_config = intermediate["viz_data"]

            st.session_state.chat_history[st.session_state.current_thread_id].append(assistant_message)
            st.session_state.tool_history[st.session_state.current_thread_id] = intermediate["tool_history"]

    # === 保留退出逻辑 ===
    if prompt and prompt.lower() in ["exit", "quit"]:
        st.write("退出程序。")
        logger.info("User exited the program")
        st.stop()