"""Pipeline backend entrypoint (Gemini + sockets + processing)."""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.config import get_settings
from pipeline_service.queue import configure_job_handler, start_consumer, stop_consumer
from pipeline_service.routes import consume_pipeline_job
from pipeline_service.routes import router as pipeline_router
from pipeline_service.websocket import start_ws_bus, stop_ws_bus
from pipeline_service.websocket import router as pipeline_ws_router

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Pipeline Backend service…")
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_job_handler(consume_pipeline_job)
    start_consumer(asyncio.get_running_loop())
    start_ws_bus(asyncio.get_running_loop())
    yield
    stop_ws_bus()
    stop_consumer()
    logger.info("Shutting down Pipeline Backend service")


app = FastAPI(
    title=f"{settings.APP_NAME} Pipeline Backend",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/pipeline/docs" if settings.PIPELINE_ENABLE_DOCS else None,
    redoc_url="/pipeline/redoc" if settings.PIPELINE_ENABLE_DOCS else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pipeline_router, prefix=settings.PIPELINE_API_PREFIX)
app.include_router(pipeline_ws_router, prefix=settings.PIPELINE_API_PREFIX)


@app.get("/pipeline/v1/health")
async def health():
    return {"status": "ok", "service": "pipeline_backend"}
