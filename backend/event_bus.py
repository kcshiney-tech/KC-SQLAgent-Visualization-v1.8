# event_bus.py
# Simple asyncio-based in-memory event bus for single-host deployment.
# For multi-process/multi-host production, replace with Redis pub/sub or similar.

import asyncio
from typing import Dict

_QUEUES: Dict[str, asyncio.Queue] = {}
_LOCK = asyncio.Lock()

async def get_queue(task_id: str) -> asyncio.Queue:
    async with _LOCK:
        q = _QUEUES.get(task_id)
        if q is None:
            q = asyncio.Queue()
            _QUEUES[task_id] = q
    return q

async def emit_event(task_id: str, event: dict):
    """
    Put event into queue for the given task_id.
    Event must be JSON-serializable.
    """
    q = await get_queue(task_id)
    await q.put(event)

async def drain_queue(task_id: str):
    q = await get_queue(task_id)
    items = []
    while not q.empty():
        items.append(q.get_nowait())
    return items
