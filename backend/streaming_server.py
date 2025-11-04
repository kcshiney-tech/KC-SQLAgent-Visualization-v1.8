# streaming_server.py
import time
import json
import uuid
import asyncio
from typing import Dict, Any
from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse  # or use fastapi-utils EventSourceResponse alternative
import logging
import os

# 假定你的 sql_agent.py 在同目录并导出了 agent_node, tool_node, viz_node, build_graph, initialize_tools
from sql_agent import agent_node, tool_node, viz_node, build_graph, initialize_tools, filter_context, State

logger = logging.getLogger("streaming_server")
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# 简易 token 验证（生产请换成 OAuth2/JWT/企业 SSO）
API_TOKEN = os.getenv("STREAM_API_TOKEN", "change_me")

def verify_token(token: str):
    if not token or token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

async def run_agent_and_stream(question: str, client_id: str, send):
    """
    执行 agent_node、tools（如果需要），viz_node 并通过 send(event) 发送事件
    send: async callable send(dict_event) -> sends to client
    """
    start_ts = time.time()
    # 初始 state
    state = {
        "messages": [ ],  # for agent_node we will place HumanMessage inside if needed
        "question": question,
        "sql_result": [],
        "viz_type": "none",
        "viz_data": {},
        "tables": [],
        "tool_history": [],
        "status_messages": []
    }
    # wrap HumanMessage (string) into required type or pass as message content only
    from langgraph.graph.message import HumanMessage, AIMessage, ToolMessage
    state["messages"] = [HumanMessage(content=question)]

    # 1) call agent_node: get AIMessage (answer) and possibly tool_calls
    send({"type":"status","msg":"running_agent"})
    agent_out = agent_node(state, initialize_tools(None))
    # agent_node returns {"messages":[AIMessage...], "status_messages": [...]}
    msgs = agent_out.get("messages", [])
    # find AIMessage content
    answer_text = None
    for m in msgs:
        if hasattr(m, "content"):
            answer_text = m.content
            break
    if answer_text is None:
        answer_text = "抱歉，未获得代理回答。"
    await send({"type":"answer", "content": answer_text, "client_id": client_id, "elapsed": time.time()-start_ts})

    # 2) if agent suggests tools (tool_calls), run tool_node(s)
    # NOTE: your agent_node populates tool_calls inside AIMessage.tool_calls; we will try to detect that.
    last_ai = msgs[-1] if msgs else None
    if last_ai and getattr(last_ai, "tool_calls", None):
        for tool_call in last_ai.tool_calls:
            # send status
            await send({"type":"status","msg":f"executing_tool:{tool_call['name']}"})
            # prepare a state that includes this tool call as last message (tool_node expects it)
            state_for_tool = {
                "messages": [last_ai],
                "question": question,
                "sql_result": state.get("sql_result", []),
                "tool_history": state.get("tool_history", []),
                "status_messages": []
            }
            tool_out = tool_node(state_for_tool, initialize_tools(None))
            # tool_out includes sql_result and tool_history
            await send({"type":"tool", "tool": tool_call["name"], "output": tool_out.get("sql_result", tool_out.get("tool_history")), "elapsed": time.time()-start_ts})
            # update global state
            state["sql_result"] = tool_out.get("sql_result", state.get("sql_result"))
            state["tool_history"] = (state.get("tool_history", [])) + tool_out.get("tool_history", [])

    # 3) finally run viz_node (build viz_data/tables)
    await send({"type":"status","msg":"building_visualization"})
    viz_state = {"messages": state["messages"], "question": question, "sql_result": state.get("sql_result", []), "tool_history": state.get("tool_history", []), "status_messages": []}
    viz_out = viz_node(viz_state)
    # viz_out should contain viz_type, viz_data, tables
    await send({"type":"viz", "viz_type": viz_out.get("viz_type"), "viz_data": viz_out.get("viz_data"), "tables": viz_out.get("tables"), "elapsed": time.time()-start_ts})

    await send({"type":"done","elapsed": time.time()-start_ts})

@app.post("/query_stream")
async def query_stream(request: Request, authorization: str = Header(None)):
    """
    接收前端 query，返回 SSE stream（先 answer，再 tools 输出，再 viz）。
    前端以 EventSource 监听：
      const es = new EventSource('/query_stream?question=xxx', { headers:{Authorization: 'Bearer ...'} })
      // but EventSource doesn't support custom headers; better pass token as query param or use fetch -> EventSource polyfill
    NOTE: For production prefer WebSocket if you need bi-directional control/abort.
    """
    token = authorization.split()[-1] if authorization else None
    verify_token(token)

    body = await request.json()
    question = body.get("question") or body.get("q") or ""
    client_id = str(uuid.uuid4())

    async def event_generator():
        queue = asyncio.Queue()

        async def send(ev: Dict[str, Any]):
            # ensure JSON-serializable
            await queue.put(json.dumps(ev, ensure_ascii=False))

        # start background task that runs the agent/process
        loop = asyncio.get_event_loop()
        loop.create_task(run_agent_and_stream(question, client_id, send))

        while True:
            msg = await queue.get()
            yield msg + "\n\n"

            # end when done message received
            try:
                payload = json.loads(msg)
                if payload.get("type") == "done":
                    break
            except Exception:
                pass

    return EventSourceResponse(event_generator())

# Simple health check
@app.get("/health")
async def health():
    return {"ok": True}
