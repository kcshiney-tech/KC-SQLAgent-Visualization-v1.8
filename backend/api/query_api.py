# backend/api/query_api.py
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from backend.sql_agent import build_graph, SQLDatabase
from langchain_core.messages import HumanMessage
import json
import time
import logging

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

app = FastAPI()
db = SQLDatabase.from_uri("sqlite:///real_database.db")
graph, _ = build_graph(db)

async def event_generator(request: Request, body: dict):
    thread_id = body.get("thread_id", "default")
    question = body["question"]
    tool_history = body.get("tool_history", [])
    config = {"configurable": {"thread_id": thread_id}}

    inputs = {
        "messages": [HumanMessage(content=question)],
        "question": question,
        "tool_history": tool_history,
        "status_messages": []
    }

    start_time = time.time()
    try:
        async for event in graph.astream_events(inputs, config=config, version="v2"):
        # async for event in graph.astream_events(inputs, config, version="v2"):
            if await request.is_disconnected():
                break

            # 1. 回答流式输出
            if event["event"] == "on_chat_model_stream" and "chunk" in event["data"]:
                chunk = event["data"]["chunk"]
                if hasattr(chunk, "content") and chunk.content:
                    yield json.dumps({
                        "type": "answer_chunk",
                        "content": chunk.content
                    }, ensure_ascii=False) + "\n"

            # 2. 工具执行状态
            elif event["event"] == "on_tool_start":
                tool_name = event["name"]
                input_data = event["data"].get("input", {})
                query = str(input_data.get("query", ""))[:100]
                yield json.dumps({
                    "type": "status",
                    "message": f"正在执行: {tool_name} {query}..."
                }, ensure_ascii=False) + "\n"

            # 3. 可视化完成（最终结果）
            elif event["name"] == "viz":
                data = event["data"]
                total_time = time.time() - start_time
                yield json.dumps({
                    "type": "viz_complete",
                    "viz_type": data.get("viz_type", "none"),
                    "viz_data": data.get("viz_data", {}),
                    "tables": data.get("tables", []),
                    "total_time": round(total_time, 2),
                    "tool_history": data.get("tool_history", tool_history)
                }, ensure_ascii=False) + "\n"
                break

    except Exception as e:
        logger.error(f"SSE Error: {e}")
        yield json.dumps({
            "type": "error",
            "message": "后端处理异常"
        }, ensure_ascii=False) + "\n"

@app.post("/stream")
async def stream_endpoint(request: Request):
    body = await request.json()
    return StreamingResponse(
        event_generator(request, body),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )