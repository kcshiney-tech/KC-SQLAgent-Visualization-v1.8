# sse_api.py (完整文件 — replace your existing sse_api.py)
import asyncio
import json
import time
import uuid
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# adjust imports to your layout
from backend.event_bus import get_queue, emit_event  # ensure backend/event_bus.py exists
import backend.sql_agent as sql_agent

app = FastAPI(title="SQL Agent SSE API (improved)")

# Allow CORS from frontend (adjust origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # set to your domain(s) in production
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

class StartTaskRequest(BaseModel):
    user_id: str
    query: str
    params: Dict[str, Any] = {}

@app.post("/start_task")
async def start_task(payload: StartTaskRequest):
    """
    Start a background task; returns task_id and events_url.
    """
    task_id = str(uuid.uuid4())
    # ensure queue exists
    await get_queue(task_id)
    # schedule background runner
    asyncio.get_event_loop().create_task(_run_task_wrapper(task_id, payload.dict()))
    events_url = f"/events?task_id={task_id}"
    return JSONResponse({"task_id": task_id, "events_url": events_url})

async def _run_task_wrapper(task_id: str, payload: Dict[str, Any]):
    start = time.time()
    try:
        await emit_event(task_id, {"type": "status", "message": "task_started"})
        await sql_agent.run_task(task_id, payload, emit_event)
        processing_time = time.time() - start
        await emit_event(task_id, {"type": "final", "status": "success", "processing_time": processing_time})
    except Exception as e:
        processing_time = time.time() - start
        # send error event (avoid sending raw trace to client in production)
        try:
            await emit_event(task_id, {"type": "final", "status": "error", "error": str(e), "processing_time": processing_time})
        except Exception:
            pass

@app.get("/events")
async def events(request: Request, task_id: str):
    """
    SSE endpoint. Streams events for the given task_id.
    Keeps connection alive with heartbeats.
    """
    q = await get_queue(task_id)

    async def event_generator():
        heartbeat_interval = 15.0  # seconds
        last_sent = time.time()
        try:
            while True:
                # If client disconnected, break
                if await request.is_disconnected():
                    break
                try:
                    # wait for next event with timeout to send heartbeat
                    event = await asyncio.wait_for(q.get(), timeout=heartbeat_interval)
                    payload = json.dumps(event, default=str, ensure_ascii=False)
                    yield f"data: {payload}\n\n"
                    last_sent = time.time()
                    if isinstance(event, dict) and event.get("type") == "final":
                        break
                except asyncio.TimeoutError:
                    # send heartbeat ping to keep connection alive
                    ping = {"type": "ping", "ts": time.time()}
                    yield f"data: {json.dumps(ping)}\n\n"
                    last_sent = time.time()
                    # continue waiting for real events
        except asyncio.CancelledError:
            return

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no"  # for nginx: disable response buffering for SSE
    }
    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=headers)
