# dashboard.py
"""
Streamlit frontend for the SQL Agent Dashboard with concurrent query execution.
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from backend.sql_agent import build_graph, process_query
from langchain_community.utilities import SQLDatabase
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage

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

# Define routine dashboard queries
dashboard_modules = {
    "质量模块": [
        {"query": "近一个月的光模块故障数，按厂商、型号分布，画柱状图", "title": "近一个月光模块故障"},
        {"query": "查询一下，2025年，不同集群的光模块故障数，首先按照集群分类，然后按光模块型号分类，最后按厂商细分。画柱状图。", "title": "2025年不同集群光模块故障数"},
    ],
    "容量模块": [
        {"query": "近两个月光模块故障数，按厂商、型号分布，画柱状图", "title": "近两个月光模块故障"}
    ]
}

def process_dashboard_query(query_info, graph, thread_id):
    """Process a single dashboard query."""
    try:
        inputs = {
            "messages": [HumanMessage(content=query_info["query"])],
            "question": query_info["query"],
            "tool_history": [],
            "status_messages": []
        }
        config = {"configurable": {"thread_id": thread_id}}
        result = process_query(graph, inputs, config, lambda msg: None)  # No status updates for dashboard
        return {"title": query_info["title"], "result": result}
    except Exception as e:
        logger.error(f"Query failed for {query_info['title']}: {traceback.format_exc()}")
        return {"title": query_info["title"], "result": {"error": str(e)}}

def render_dashboard():
    """Render dashboard with concurrent query execution."""
    st.header("数据仪表板")
    
    # Collect all queries to execute
    all_queries = []
    for module, queries in dashboard_modules.items():
        for query_info in queries:
            all_queries.append((query_info, f"dashboard_{uuid.uuid4().hex}"))

    # Execute queries concurrently
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_query = {
            executor.submit(process_dashboard_query, query_info, graph, thread_id): query_info
            for query_info, thread_id in all_queries
        }
        for future in as_completed(future_to_query):
            query_info = future_to_query[future]
            try:
                result_data = future.result()
                module = next(m for m, q in dashboard_modules.items() if query_info in q)
                with st.expander(f"模块: {module}"):
                    st.subheader(result_data["title"])
                    result = result_data["result"]
                    if "error" in result:
                        st.error(f"查询错误: {result['error']}")
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
            except Exception as e:
                st.error(f"处理 {query_info['title']} 时发生错误: {str(e)}")
                logger.error(f"Future failed for {query_info['title']}: {traceback.format_exc()}")

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
st.title("灵图SQL仪表板")

# if not authenticate():
#     st.stop()

render_dashboard()