"""WebSocket handler — real-time scan progress updates."""

import asyncio
from collections import defaultdict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

router = APIRouter(tags=["websocket"])


class ConnectionManager:
    """Manages WebSocket connections grouped by scan_id."""

    def __init__(self) -> None:
        self._connections: dict[str, list[WebSocket]] = defaultdict(list)
        self._last_progress: dict[str, dict] = {}  # cached latest progress per scan

    async def connect(self, scan_id: str, ws: WebSocket) -> None:
        await ws.accept()
        self._connections[scan_id].append(ws)
        logger.info(f"WS connected: scan={scan_id} (total={len(self._connections[scan_id])})")
        # Replay cached progress so reconnecting clients catch up
        cached = self._last_progress.get(scan_id)
        if cached:
            try:
                await ws.send_json(cached)
            except Exception:
                pass

    def disconnect(self, scan_id: str, ws: WebSocket) -> None:
        self._connections[scan_id].remove(ws)
        if not self._connections[scan_id]:
            del self._connections[scan_id]
        logger.info(f"WS disconnected: scan={scan_id}")

    async def broadcast(self, scan_id: str, data: dict) -> None:
        """Send a JSON message to all connections watching this scan."""
        # Cache progress/complete/error for reconnecting clients
        msg_type = data.get("type")
        if msg_type in ("progress", "complete", "error"):
            self._last_progress[scan_id] = data
        # Clean up cache on terminal states
        if msg_type in ("complete", "error"):
            # Keep it briefly for late reconnects, then discard
            asyncio.get_event_loop().call_later(60, self._last_progress.pop, scan_id, None)
        dead: list[WebSocket] = []
        for ws in self._connections.get(scan_id, []):
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(scan_id, ws)

    async def broadcast_to_user_scans(self, scan_ids: list[str], data: dict) -> None:
        """Send message to all scan connections (used for dashboard updates)."""
        for scan_id in scan_ids:
            await self.broadcast(scan_id, data)


# Singleton — imported by pipeline orchestrator to push updates
manager = ConnectionManager()


@router.websocket("/ws/{scan_id}")
async def websocket_endpoint(ws: WebSocket, scan_id: str):
    await manager.connect(scan_id, ws)
    try:
        while True:
            # Keep connection alive; client sends pings
            data = await ws.receive_text()
            if data == "ping":
                await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        manager.disconnect(scan_id, ws)
    except Exception:
        manager.disconnect(scan_id, ws)
