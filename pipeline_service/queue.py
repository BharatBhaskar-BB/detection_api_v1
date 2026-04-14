"""RabbitMQ queue helpers for pipeline job dispatch/consumption."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Awaitable, Callable
from typing import Any

import pika
from pika.exceptions import AMQPConnectionError
from loguru import logger

from app.config import get_settings

settings = get_settings()

JobHandler = Callable[[dict[str, Any]], Awaitable[None]]

_handler: JobHandler | None = None
_consumer_thread: threading.Thread | None = None
_stop_event = threading.Event()


def _connect() -> pika.BlockingConnection:
    params = pika.URLParameters(settings.RABBITMQ_URL)
    return pika.BlockingConnection(params)


def _declare_queue(channel: pika.adapters.blocking_connection.BlockingChannel) -> None:
    channel.queue_declare(queue=settings.PIPELINE_QUEUE_NAME, durable=True)


def configure_job_handler(handler: JobHandler) -> None:
    global _handler
    _handler = handler


def queue_enabled() -> bool:
    return bool(settings.PIPELINE_QUEUE_ENABLED)


def publish_pipeline_job(payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    conn = _connect()
    try:
        ch = conn.channel()
        _declare_queue(ch)
        ch.basic_publish(
            exchange="",
            routing_key=settings.PIPELINE_QUEUE_NAME,
            body=body,
            properties=pika.BasicProperties(delivery_mode=2),
        )
    finally:
        conn.close()


async def publish_pipeline_job_async(payload: dict[str, Any]) -> None:
    await asyncio.to_thread(publish_pipeline_job, payload)


def _consume_forever(loop: asyncio.AbstractEventLoop) -> None:
    while not _stop_event.is_set():
        conn: pika.BlockingConnection | None = None
        try:
            conn = _connect()
            ch = conn.channel()
            _declare_queue(ch)
            ch.basic_qos(prefetch_count=1)

            def _on_message(
                channel: pika.adapters.blocking_connection.BlockingChannel,
                method: pika.spec.Basic.Deliver,
                properties: pika.spec.BasicProperties,
                body: bytes,
            ) -> None:
                del properties
                if _handler is None:
                    logger.error("Pipeline queue handler is not configured")
                    channel.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
                    return

                try:
                    payload = json.loads(body.decode("utf-8"))
                except Exception:
                    logger.exception("Invalid queue payload")
                    channel.basic_ack(delivery_tag=method.delivery_tag)
                    return

                fut = asyncio.run_coroutine_threadsafe(_handler(payload), loop)
                try:
                    fut.result()
                except Exception:
                    logger.exception("Queued pipeline job failed")
                    channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
                else:
                    channel.basic_ack(delivery_tag=method.delivery_tag)

            ch.basic_consume(queue=settings.PIPELINE_QUEUE_NAME, on_message_callback=_on_message, auto_ack=False)

            while not _stop_event.is_set() and conn.is_open:
                conn.process_data_events(time_limit=1)
        except AMQPConnectionError:
            logger.warning("RabbitMQ not ready yet; retrying consumer connection in 2s")
            time.sleep(2)
        except Exception:
            logger.exception("RabbitMQ consumer error; retrying")
            time.sleep(2)
        finally:
            try:
                if conn and conn.is_open:
                    conn.close()
            except Exception:
                pass


def start_consumer(loop: asyncio.AbstractEventLoop) -> None:
    global _consumer_thread
    if not queue_enabled():
        logger.info("Pipeline queue is disabled; consumer not started")
        return
    if _consumer_thread and _consumer_thread.is_alive():
        return

    _stop_event.clear()
    _consumer_thread = threading.Thread(target=_consume_forever, args=(loop,), daemon=True, name="pipeline-rabbit-consumer")
    _consumer_thread.start()
    logger.info(f"RabbitMQ consumer started on queue={settings.PIPELINE_QUEUE_NAME}")


def stop_consumer() -> None:
    _stop_event.set()
    if _consumer_thread and _consumer_thread.is_alive():
        _consumer_thread.join(timeout=3)
