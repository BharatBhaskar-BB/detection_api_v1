"""WebSocket endpoints for pipeline progress streaming."""

import asyncio
from collections import defaultdict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

from app.pipeline_service.auth import authenticate_pipeline_request, verify_scan_access

router = APIRouter(tags=["pipeline-websocket"])


class ConnectionManager:
    """Manages WebSocket connections grouped by scan ID."""

    def __init__(self) -> None:
        self._connections: dict[str, list[WebSocket]] = defaultdict(list)
        self._last_progress: dict[str, dict] = {}

    async def connect(self, scan_id: str, ws: WebSocket) -> None:
        await ws.accept()
        self._connections[scan_id].append(ws)
        logger.info(f"Pipeline WS connected: scan={scan_id} (total={len(self._connections[scan_id])})")
        cached = self._last_progress.get(scan_id)
        if cached:
            try:
                await ws.send_json(cached)
            except Exception:
                pass

    def disconnect(self, scan_id: str, ws: WebSocket) -> None:
        try:
            self._connections[scan_id].remove(ws)
        except (KeyError, ValueError):
            return
        if not self._connections[scan_id]:
            del self._connections[scan_id]

    async def broadcast(self, scan_id: str, data: dict) -> None:
        msg_type = data.get("type")
        if msg_type in ("progress", "complete", "error"):
            self._last_progress[scan_id] = data
        if msg_type in ("complete", "error"):
            asyncio.get_event_loop().call_later(60, self._last_progress.pop, scan_id, None)

        dead: list[WebSocket] = []
        for ws in self._connections.get(scan_id, []):
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(scan_id, ws)


manager = ConnectionManager()


@router.websocket("/ws/{scan_id}")
async def websocket_endpoint(ws: WebSocket, scan_id: str):
    token = ws.query_params.get("token")
    pipeline_token = ws.query_params.get("pipeline_token")
    authorization = f"Bearer {token}" if token else None

    try:
        user_id = await authenticate_pipeline_request(pipeline_token, authorization)
        await verify_scan_access(scan_id, user_id)
    except Exception:
        await ws.close(code=1008)
        return

    await manager.connect(scan_id, ws)
    try:
        while True:
            data = await ws.receive_text()
            if data == "ping":
                await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        manager.disconnect(scan_id, ws)
    except Exception:
        manager.disconnect(scan_id, ws)
