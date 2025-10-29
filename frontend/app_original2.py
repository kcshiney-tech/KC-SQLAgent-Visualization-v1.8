# app.py
"""
Streamlit frontend for the SQL Agent with chat interface.
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
                chart_id = f"chart_{uuid.uuid4().hex}"
                chart_json = json.dumps(result["viz_data"])
                
                # 使用兼容的 Chart.js 版本和正确的依赖
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

                                    // 候选对象：穷举常见 UMD 全局名及 default 导出
                                    const candidateNames = [
                                        'chartjs-plugin-hierarchical',
                                        'chartjsPluginHierarchical',
                                        'ChartjsPluginHierarchical',
                                        'ChartHierarchicalPlugin',
                                        'HierarchicalPlugin',
                                        'HierarchicalScale',
                                    ];

                                    const candidates = candidateNames.map(n => window[n]).filter(x => !!x);

                                    // 若 script 将模块作为 single default export：window['chartjs-plugin-hierarchical'].default
                                    const maybeDefaultCandidates = candidateNames
                                        .map(n => (window[n] && window[n].default) ? window[n].default : null)
                                        .filter(x => !!x);

                                    const allCandidates = Array.from(new Set([...candidates, ...maybeDefaultCandidates]));

                                    // 额外尝试：一些打包可能把 scale 放在 window.HierarchicalScale 或在插件对象里
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
                                            // 如果是一个对象，先尝试注册其中的 HierarchicalScale 成员
                                            if (typeof p === 'object') {{
                                                if (p.HierarchicalScale) {{
                                                    Chart.register(p.HierarchicalScale);
                                                    console.log('Registered candidate.HierarchicalScale');
                                                    return true;
                                                }}
                                                // 有些包把 scale 直接命名为 HierarchicalScale 或 scale
                                                if (p.scale && p.scale.HierarchicalScale) {{
                                                    Chart.register(p.scale.HierarchicalScale);
                                                    console.log('Registered candidate.scale.HierarchicalScale');
                                                    return true;
                                                }}
                                                // 有些包直接就是 plugin 对象
                                                try {{
                                                    Chart.register(p);
                                                    console.log('Registered candidate plugin object');
                                                    return true;
                                                }} catch (err) {{
                                                    // 继续尝试下一个
                                                    console.warn('Attempt to Chart.register(candidate) failed:', err);
                                                }}
                                            }}
                                            // 如果是函数（UMD 直接导出为注册函数）
                                            if (typeof p === 'function') {{
                                                try {{
                                                    // 某些 UMD 导出是一个 install 函数（接收 Chart）
                                                    p(Chart);
                                                    console.log('Called candidate install function with Chart');
                                                    return true;
                                                }} catch (err) {{
                                                    // 尝试直接 register 作为 plugin
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

                                // 获取并处理配置
                                var ctx = document.getElementById('{chart_id}').getContext('2d');
                                var config = {chart_json};

                                // 安全处理：如果存在临时 _hierarchical 配置并且插件已注册，则把 type 改为 'hierarchical'
                                try {{
                                    var xScale = config.options && config.options.scales && config.options.scales.x;
                                    if (xScale && xScale._hierarchical) {{
                                        if (registered && Chart.registry.getScale && Chart.registry.getScale('hierarchical')) {{
                                            // 插件已注册并提供 hierarchical scale：迁移配置
                                            xScale.type = 'hierarchical';
                                            xScale.hierarchical = xScale.hierarchical || {{}};
                                            // 将所有 _hierarchical 字段迁移过去
                                            Object.assign(xScale.hierarchical, xScale._hierarchical);
                                            // 清理临时字段
                                            delete xScale._hierarchical;
                                        }} else if (registered) {{
                                            // 注册成功但 Chart.registry 还找不到 scale，仍旧尝试直接设置 type 并 hope for the best
                                            xScale.type = 'hierarchical';
                                            xScale.hierarchical = xScale.hierarchical || {{}};
                                            Object.assign(xScale.hierarchical, xScale._hierarchical);
                                            delete xScale._hierarchical;
                                        }} else {{
                                            // 插件未注册，保留 category，不做 hierarchical
                                            console.warn('Hierarchical plugin not available; using category axis.');
                                            xScale.type = 'category';
                                        }}
                                    }}
                                }} catch (err) {{
                                    console.warn('Error while applying hierarchical config migration:', err);
                                    // fallback 保持原有 config
                                }}

                                // 移除可能导致问题的 zoom 插件配置（如果未引入）
                                if (config.options && config.options.plugins && config.options.plugins.zoom && !window.ChartZoom) {{
                                    delete config.options.plugins.zoom;
                                }}

                                // 最终创建 chart
                                var myChart = new Chart(ctx, config);
                                console.log('Chart created successfully');

                            }} catch (e) {{
                                console.error('Chart creation error:', e);
                                // 备用方案：使用普通柱状图
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
            except Exception as e:
                st.error("抱歉，图表渲染失败，请稍后重试或联系支持。")
                logger.error(f"Chart rendering failed: {traceback.format_exc()}")

    if result.get("tables"):
        with table_placeholder.container():
            st.markdown("**表格:**")
            for table in result["tables"]:
                st.markdown(f"**{table['title']}**")
                df = pd.DataFrame(table["data"])
                st.dataframe(df, width='stretch')
                logger.debug(f"Table displayed: {table['title']}")

    status_placeholder.markdown(f"处理时间: {result['processing_time']:.2f}秒")

st.title("🚀 灵图SQL对话")


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

    if st.button("新建聊天 +"):
        new_thread_id = str(uuid.uuid4())
        st.session_state.chat_history[new_thread_id] = []
        st.session_state.first_questions[new_thread_id] = "新建对话"
        st.session_state.chat_creation_times[new_thread_id] = datetime.now()
        st.session_state.tool_history[new_thread_id] = []
        st.session_state.current_thread_id = new_thread_id
        logger.info(f"Created new conversation: {new_thread_id}")
        st.rerun()

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
    # st.header("灵图SQL对话")
    
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

                                    // 候选对象：穷举常见 UMD 全局名及 default 导出
                                    const candidateNames = [
                                        'chartjs-plugin-hierarchical',
                                        'chartjsPluginHierarchical',
                                        'ChartjsPluginHierarchical',
                                        'ChartHierarchicalPlugin',
                                        'HierarchicalPlugin',
                                        'HierarchicalScale',
                                    ];

                                    const candidates = candidateNames.map(n => window[n]).filter(x => !!x);

                                    // 若 script 将模块作为 single default export：window['chartjs-plugin-hierarchical'].default
                                    const maybeDefaultCandidates = candidateNames
                                        .map(n => (window[n] && window[n].default) ? window[n].default : null)
                                        .filter(x => !!x);

                                    const allCandidates = Array.from(new Set([...candidates, ...maybeDefaultCandidates]));

                                    // 额外尝试：一些打包可能把 scale 放在 window.HierarchicalScale 或在插件对象里
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
                                            // 如果是一个对象，先尝试注册其中的 HierarchicalScale 成员
                                            if (typeof p === 'object') {{
                                                if (p.HierarchicalScale) {{
                                                    Chart.register(p.HierarchicalScale);
                                                    console.log('Registered candidate.HierarchicalScale');
                                                    return true;
                                                }}
                                                // 有些包把 scale 直接命名为 HierarchicalScale 或 scale
                                                if (p.scale && p.scale.HierarchicalScale) {{
                                                    Chart.register(p.scale.HierarchicalScale);
                                                    console.log('Registered candidate.scale.HierarchicalScale');
                                                    return true;
                                                }}
                                                // 有些包直接就是 plugin 对象
                                                try {{
                                                    Chart.register(p);
                                                    console.log('Registered candidate plugin object');
                                                    return true;
                                                }} catch (err) {{
                                                    // 继续尝试下一个
                                                    console.warn('Attempt to Chart.register(candidate) failed:', err);
                                                }}
                                            }}
                                            // 如果是函数（UMD 直接导出为注册函数）
                                            if (typeof p === 'function') {{
                                                try {{
                                                    // 某些 UMD 导出是一个 install 函数（接收 Chart）
                                                    p(Chart);
                                                    console.log('Called candidate install function with Chart');
                                                    return true;
                                                }} catch (err) {{
                                                    // 尝试直接 register 作为 plugin
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

                                // 获取并处理配置
                                var ctx = document.getElementById('{chart_id}').getContext('2d');
                                var config = {chart_json};

                                // 安全处理：如果存在临时 _hierarchical 配置并且插件已注册，则把 type 改为 'hierarchical'
                                try {{
                                    var xScale = config.options && config.options.scales && config.options.scales.x;
                                    if (xScale && xScale._hierarchical) {{
                                        if (registered && Chart.registry.getScale && Chart.registry.getScale('hierarchical')) {{
                                            // 插件已注册并提供 hierarchical scale：迁移配置
                                            xScale.type = 'hierarchical';
                                            xScale.hierarchical = xScale.hierarchical || {{}};
                                            // 将所有 _hierarchical 字段迁移过去
                                            Object.assign(xScale.hierarchical, xScale._hierarchical);
                                            // 清理临时字段
                                            delete xScale._hierarchical;
                                        }} else if (registered) {{
                                            // 注册成功但 Chart.registry 还找不到 scale，仍旧尝试直接设置 type 并 hope for the best
                                            xScale.type = 'hierarchical';
                                            xScale.hierarchical = xScale.hierarchical || {{}};
                                            Object.assign(xScale.hierarchical, xScale._hierarchical);
                                            delete xScale._hierarchical;
                                        }} else {{
                                            // 插件未注册，保留 category，不做 hierarchical
                                            console.warn('Hierarchical plugin not available; using category axis.');
                                            xScale.type = 'category';
                                        }}
                                    }}
                                }} catch (err) {{
                                    console.warn('Error while applying hierarchical config migration:', err);
                                    // fallback 保持原有 config
                                }}

                                // 移除可能导致问题的 zoom 插件配置（如果未引入）
                                if (config.options && config.options.plugins && config.options.plugins.zoom && !window.ChartZoom) {{
                                    delete config.options.plugins.zoom;
                                }}

                                // 最终创建 chart
                                var myChart = new Chart(ctx, config);
                                console.log('Chart created successfully');

                            }} catch (e) {{
                                console.error('Chart creation error:', e);
                                // 备用方案：使用普通柱状图
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
                    try:
                        components.html(html, height=480, scrolling=False)
                    except Exception as e:
                        st.error("抱歉，图表渲染失败，请稍后重试或联系支持。")
                        logger.error(f"Chart rendering failed: {traceback.format_exc()}")

            
prompt = st.chat_input("输入您的查询 (例如: '2025年每个月，QYZNJ机房，光模块的故障数，按光模块型号和厂商分布，画折线图？')")
if prompt:
    # if not check_rate_limit():
    #     st.stop()
    user_ip = st.query_params.get("user_ip", "unknown")
    logger.info(f"Query from IP {user_ip}, thread_id {st.session_state.current_thread_id}: {prompt}")
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
                    result = process_query(graph, inputs, config, lambda msg: status_placeholder.markdown(translate_status_message(msg)))
                    stream_response(result, status_placeholder, answer_placeholder, chart_placeholder, table_placeholder)
                    assistant_message = AIMessage(content=result['answer'])
                    if result["tables"]:
                        assistant_message.tables = result["tables"]
                    if result["viz_data"]:
                        assistant_message.chart_config = result["viz_data"]
                    st.session_state.chat_history[st.session_state.current_thread_id] = result["messages"] + [assistant_message]
                    st.session_state.tool_history[st.session_state.current_thread_id] = [
                        h for h in result.get("tool_history", [])
                        if h["tool"] in ["sql_db_list_tables", "sql_db_schema", "sql_db_query", "sql_db_query_checker", "check_result"]
                    ]
                except Exception as e:
                    status_placeholder.error("抱歉，处理查询时发生错误，请稍后重试或联系支持。")
                    logger.error(f"Query failed for IP {user_ip}: {traceback.format_exc()}")
                    assistant_message = AIMessage(content="抱歉，处理查询时发生错误，请稍后重试或联系支持。")
                    st.session_state.chat_history[st.session_state.current_thread_id].append(assistant_message)

if prompt and prompt.lower() in ["exit", "quit"]:
    st.write("退出程序。")
    logger.info("User exited the program")
    st.stop()