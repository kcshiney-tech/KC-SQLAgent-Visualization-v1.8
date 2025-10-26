# dashboard.py
"""
Streamlit Dashboard for SQL Agent with modern design: Card-based layout, interactivity, and data storytelling.
Concurrent query execution using ThreadPoolExecutor.
"""
import streamlit as st
import streamlit.components.v1 as components
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import uuid
import pandas as pd
import json
import logging
import traceback
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from backend.sql_agent import build_graph, process_query
from langchain_community.utilities import SQLDatabase
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage
import re  # 用于提取KPI数字
import time  # 用于时间检查

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("dashboard.log", encoding="utf-8"),
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

# Define routine dashboard queries with timer intervals (seconds)
def get_dashboard_queries(date_filter="近一个月"):
    """Generate queries based on date filter, with refresh intervals."""
    return {
        "质量模块": [
            {
                "query": f"{date_filter}光模块故障数，按厂商、型号分布，画柱状图",
                "title": f"{date_filter}光模块故障分布",
                "kpi": "总故障数",
                "refresh_interval": 60  # 每1分钟
            },
            {
                "query": f"{date_filter}不同集群的光模块故障数，首先按照集群分类，然后按光模块型号分类，最后按厂商细分。画柱状图。未知集群不统计。",
                "title": f"{date_filter}光模块故障按集群分布",
                "kpi": "平均故障率",
                "refresh_interval": 90  # 每1.5分钟
            }
        ],
        "容量模块": [
            {
                "query": f"{date_filter.replace('一个月', '两个月')}光模块故障数，按厂商、型号分布，画柱状图",
                "title": f"{date_filter.replace('一个月', '两个月')}光模块故障分布",
                "kpi": "总故障数",
                "refresh_interval": 50  # 每1分钟
            }
        ]
    }

def process_dashboard_query(query_info, graph, thread_id):
    """Process a single dashboard query using SQL Agent."""
    try:
        inputs = {
            "messages": [HumanMessage(content=query_info["query"])],
            "question": query_info["query"],
            "tool_history": [],
            "status_messages": []
        }
        config = {"configurable": {"thread_id": thread_id}}
        result = process_query(graph, inputs, config, lambda msg: None)
        return {"title": query_info["title"], "kpi": query_info["kpi"], "result": result, "last_refresh": time.time()}
    except Exception as e:
        logger.error(f"Query failed for {query_info['title']}: {traceback.format_exc()}")
        return {"title": query_info["title"], "kpi": query_info["kpi"], "result": {"error": str(e)}, "last_refresh": time.time()}

def render_chart(result):
    """Render Chart.js chart if available."""
    if result.get("viz_data") and result.get("viz_type") != "none":
        chart_id = f"chart_{uuid.uuid4().hex}"
        chart_json = json.dumps(result["viz_data"])
        html = f"""
        <html>
        <head>
            <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js"></script>
        </head>
        <body>
            <canvas id="{chart_id}" style="width:100%; height:300px;"></canvas>
            <script>
                document.addEventListener('DOMContentLoaded', function() {{
                    try {{
                        var ctx = document.getElementById('{chart_id}').getContext('2d');
                        new Chart(ctx, {chart_json});
                    }} catch (e) {{
                        console.error('Chart.js error: ' + e.message);
                    }}
                }});
            </script>
        </body>
        </html>
        """
        components.html(html, height=350, scrolling=False)
    else:
        st.info("无可用图表。")

def render_summary(answer):
    """Render summary from LLM answer (first 3 lines or 200 chars)."""
    if answer:
        # 提取前3行
        lines = answer.split('\n')[:3]
        summary = '\n'.join(lines) if len(lines) > 1 else answer[:200] + "..."
        st.markdown(summary)

def render_table(result):
    """Render tables if available (collapsible)."""
    if result.get("tables"):
        with st.expander("查看详细表格"):
            for table in result["tables"]:
                st.markdown(f"**{table['title']}**")
                df = pd.DataFrame(table["data"])
                st.dataframe(df, use_container_width=True, hide_index=True)

def extract_kpi(answer, kpi_label):
    """Extract KPI value from answer with better matching."""
    # 改进匹配：搜索如 "总故障数: 123" 或第一个数字
    pattern = rf"{kpi_label}\s*[:：]?\s*(\d+)"
    match = re.search(pattern, answer, re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r'\d+', answer)
    return match.group(0) if match else "N/A"

def render_dashboard_card(query_info, result):
    """Render a single dashboard card."""
    with st.container(border=True):
        st.markdown(f"### {query_info['title']}")
        
        # KPI + Summary columns
        col1, col2 = st.columns([1, 3])
        with col1:
            kpi_value = extract_kpi(result.get("answer", ""), query_info["kpi"])
            st.metric(label=query_info["kpi"], value=kpi_value, help=f"从分析中提取的{query_info['kpi']}（如果不准确，请检查数据源）")
        with col2:
            render_summary(result.get("answer", ""))
        
        # Chart
        render_chart(result)
        
        # Table (collapsible)
        render_table(result)
        
        # Refresh button for single card (no rerun, update state)
        if st.button(f"🔄 刷新 {query_info['title']}", key=f"refresh_{query_info['title']}"):
            with st.spinner(f"刷新 {query_info['title']}..."):
                new_result = process_dashboard_query(query_info, graph, f"dashboard_{uuid.uuid4().hex}")
                st.session_state.dashboard_data[query_info['title']] = new_result
            st.success(f"{query_info['title']} 已刷新！")

# # 添加简单认证
# def authenticate():
#     """简单用户名/密码认证。"""
#     if "authenticated" not in st.session_state:
#         st.session_state.authenticated = False
#     if not st.session_state.authenticated:
#         username = st.text_input("用户名")
#         password = st.text_input("密码", type="password")
#         if st.button("登录"):
#             if username == "admin" and password == "password":
#                 st.session_state.authenticated = True
#                 st.rerun()
#             else:
#                 st.error("无效凭证")
#                 return False
#     return st.session_state.authenticated

# Streamlit UI
st.set_page_config(page_title="灵图SQL仪表板", layout="wide")

st.title("🚀 灵图SQL数据仪表板")
st.markdown("**实时监控关键指标，支持交互过滤和刷新。**")

# if not authenticate():
#     st.stop()

# 侧边栏：模块分页选择
with st.sidebar:
    st.header("导航")
    selected_module = st.radio("选择模块", list(get_dashboard_queries().keys()) + ["所有模块"], index=0)
    
    st.header("全局设置")
    date_range = st.selectbox("时间范围", ["近一个月", "近两个月", "近三个月", "自定义"])
    if st.button("🔄 刷新全部", type="primary"):
        st.session_state.dashboard_data = {}  # 清空缓存
        st.rerun()

# 加载数据（并发）
dashboard_queries = get_dashboard_queries(date_range)
if "dashboard_data" not in st.session_state:
    st.session_state.dashboard_data = {}

with st.spinner("加载仪表板数据..."):
    all_queries = []
    modules_to_load = [selected_module] if selected_module != "所有模块" else list(dashboard_queries.keys())
    for module in modules_to_load:
        for query_info in dashboard_queries[module]:
            title = query_info["title"]
            if title not in st.session_state.dashboard_data or time.time() - st.session_state.dashboard_data[title].get("last_refresh", 0) > query_info["refresh_interval"]:
                all_queries.append((query_info, f"dashboard_{uuid.uuid4().hex}"))

    if all_queries:
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_query = {
                executor.submit(process_dashboard_query, q_info, graph, thread_id): q_info
                for q_info, thread_id in all_queries
            }
            progress_bar = st.progress(0)
            completed = 0
            for future in as_completed(future_to_query):
                result = future.result()
                st.session_state.dashboard_data[result["title"]] = result
                completed += 1
                progress_bar.progress(completed / len(all_queries))

# 渲染卡片网格（根据侧边栏分页）
for module_name in modules_to_load:
    if module_name in dashboard_queries:
        st.markdown(f"## {module_name}")
        cols = st.columns(min(len(dashboard_queries[module_name]), 3))  # 最多3列
        for i, query_info in enumerate(dashboard_queries[module_name]):
            with cols[i]:
                result = st.session_state.dashboard_data.get(query_info["title"], {"result": {"error": "加载中..."}})
                render_dashboard_card(query_info, result["result"])

# 页脚
st.markdown("---")
st.markdown("*数据来源于 SQL Agent 实时查询 | 更新时间: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "*")