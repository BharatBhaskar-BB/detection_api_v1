"""BundleBox — FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.config import get_settings

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    logger.info(f"Starting {settings.APP_NAME}…")

    # Create upload/output directories
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Init database tables (dev only)
    from app.database import init_db
    await init_db()
    logger.info("Database initialized")

    yield

    logger.info(f"Shutting down {settings.APP_NAME}")


app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs" if settings.DEBUG else None,
    redoc_url="/api/redoc" if settings.DEBUG else None,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount Routers ────────────────────────────────────────────────────────────

from app.api.auth import router as auth_router
from app.api.pipeline_callback import router as pipeline_callback_router
from app.api.scans import router as scans_router
from app.api.push import router as push_router
from app.api.phone_auth import router as phone_auth_router
from app.api.websocket import router as websocket_router

app.include_router(auth_router, prefix=settings.API_V1_PREFIX)
app.include_router(scans_router, prefix=settings.API_V1_PREFIX)
app.include_router(push_router, prefix=settings.API_V1_PREFIX)
app.include_router(phone_auth_router, prefix=settings.API_V1_PREFIX)
app.include_router(pipeline_callback_router, prefix=settings.API_V1_PREFIX)
app.include_router(websocket_router, prefix=settings.API_V1_PREFIX)


@app.get("/api/v1/health")
async def health():
    return {"status": "ok", "app": settings.APP_NAME}
