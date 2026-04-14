"""Pipeline backend entrypoint (Gemini + sockets + processing)."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.config import get_settings
from app.pipeline_service.routes import router as pipeline_router
from app.pipeline_service.websocket import router as pipeline_ws_router

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Pipeline Backend service…")
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    yield
    logger.info("Shutting down Pipeline Backend service")


app = FastAPI(
    title=f"{settings.APP_NAME} Pipeline Backend",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/pipeline/docs" if settings.DEBUG else None,
    redoc_url="/pipeline/redoc" if settings.DEBUG else None,
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
