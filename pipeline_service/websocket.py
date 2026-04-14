"""WebSocket endpoints for pipeline progress streaming."""

import asyncio
import json
import threading
import time
import uuid
from collections import defaultdict

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from loguru import logger
import pika
from pika.exceptions import AMQPConnectionError

from app.config import get_settings
from pipeline_service.auth import authenticate_pipeline_request, verify_scan_access

router = APIRouter(tags=["pipeline-websocket"])
settings = get_settings()
_INSTANCE_ID = str(uuid.uuid4())
_WS_EXCHANGE = "pipeline_ws_events"
_ws_stop_event = threading.Event()
_ws_consumer_thread: threading.Thread | None = None


def _ws_bus_enabled() -> bool:
    return bool(settings.PIPELINE_QUEUE_ENABLED)


def _ws_connect() -> pika.BlockingConnection:
    params = pika.URLParameters(settings.RABBITMQ_URL)
    return pika.BlockingConnection(params)


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

    async def _broadcast_local(self, scan_id: str, data: dict) -> None:
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

    async def broadcast(self, scan_id: str, data: dict) -> None:
        await self._broadcast_local(scan_id, data)
        if _ws_bus_enabled():
            try:
                await asyncio.to_thread(_publish_ws_event, scan_id, data)
            except Exception:
                logger.warning("WS bus publish skipped: RabbitMQ unavailable")


manager = ConnectionManager()


def _publish_ws_event(scan_id: str, data: dict) -> None:
    conn = _ws_connect()
    try:
        ch = conn.channel()
        ch.exchange_declare(exchange=_WS_EXCHANGE, exchange_type="fanout", durable=False)
        payload = {
            "scan_id": scan_id,
            "data": data,
            "source": _INSTANCE_ID,
        }
        ch.basic_publish(exchange=_WS_EXCHANGE, routing_key="", body=json.dumps(payload).encode("utf-8"))
    finally:
        conn.close()


def _consume_ws_events(loop: asyncio.AbstractEventLoop) -> None:
    while not _ws_stop_event.is_set():
        conn: pika.BlockingConnection | None = None
        try:
            conn = _ws_connect()
            ch = conn.channel()
            ch.exchange_declare(exchange=_WS_EXCHANGE, exchange_type="fanout", durable=False)
            q = ch.queue_declare(queue="", exclusive=True)
            queue_name = q.method.queue
            ch.queue_bind(exchange=_WS_EXCHANGE, queue=queue_name)

            def _on_message(channel, method, properties, body):
                del properties
                try:
                    payload = json.loads(body.decode("utf-8"))
                    source = payload.get("source")
                    scan_id = payload.get("scan_id")
                    data = payload.get("data")
                    if source == _INSTANCE_ID or not scan_id or not isinstance(data, dict):
                        channel.basic_ack(delivery_tag=method.delivery_tag)
                        return

                    fut = asyncio.run_coroutine_threadsafe(manager._broadcast_local(scan_id, data), loop)
                    fut.result()
                    channel.basic_ack(delivery_tag=method.delivery_tag)
                except Exception:
                    logger.exception("WS bus consume error")
                    channel.basic_ack(delivery_tag=method.delivery_tag)

            ch.basic_consume(queue=queue_name, on_message_callback=_on_message, auto_ack=False)

            while not _ws_stop_event.is_set() and conn.is_open:
                conn.process_data_events(time_limit=1)
        except AMQPConnectionError:
            logger.warning("WS bus RabbitMQ not ready; retrying in 2s")
            time.sleep(2)
        except Exception:
            logger.exception("WS bus consumer error; retrying")
            time.sleep(2)
        finally:
            try:
                if conn and conn.is_open:
                    conn.close()
            except Exception:
                pass


def start_ws_bus(loop: asyncio.AbstractEventLoop) -> None:
    global _ws_consumer_thread
    if not _ws_bus_enabled():
        return
    if _ws_consumer_thread and _ws_consumer_thread.is_alive():
        return
    _ws_stop_event.clear()
    _ws_consumer_thread = threading.Thread(target=_consume_ws_events, args=(loop,), daemon=True, name="pipeline-ws-bus")
    _ws_consumer_thread.start()
    logger.info("WS bus consumer started")


def stop_ws_bus() -> None:
    _ws_stop_event.set()
    if _ws_consumer_thread and _ws_consumer_thread.is_alive():
        _ws_consumer_thread.join(timeout=3)


@router.websocket("/ws/{scan_id}")
async def websocket_endpoint(ws: WebSocket, scan_id: str):
    token = ws.query_params.get("token")
    pipeline_token = ws.query_params.get("pipeline_token")
    authorization = f"Bearer {token}" if token else None

    try:
        user_id = await authenticate_pipeline_request(pipeline_token, authorization)
        if user_id is not None:
            try:
                await verify_scan_access(scan_id, user_id)
            except HTTPException as exc:
                # `/ws/{scan_id}` is also reused by `/upload` jobs where the id
                # is a job_id (not a DB scan_id). In that case, fall back to
                # upload ownership persisted by pipeline routes.
                if exc.status_code != 403:
                    raise
                from pipeline_service.routes import _load_job

                state = _load_job(scan_id)
                owner_id = state.get("owner_user_id") if state else None
                if owner_id != user_id:
                    raise
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
