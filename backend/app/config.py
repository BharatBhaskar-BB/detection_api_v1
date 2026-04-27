"""BundleBox Backend — Configuration via environment variables."""

from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────────────
    APP_NAME: str = "BundleBox"
    DEBUG: bool = False
    API_V1_PREFIX: str = "/api/v1"
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    # ── Auth ─────────────────────────────────────────────────
    SECRET_KEY: str = "CHANGE-ME-in-production-use-openssl-rand-hex-32"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days

    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""
    GOOGLE_REDIRECT_URI: str = "http://localhost:5173/auth/callback"

    # ── Database ─────────────────────────────────────────────
    DATABASE_URL: str = "sqlite+aiosqlite:///./bundlebox.db"
    # Production: "postgresql+asyncpg://user:pass@host:5432/bundlebox"

    # ── Redis / Celery ───────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"
    RABBITMQ_URL: str = "amqp://guest:guest@rabbitmq:5672/%2F"
    PIPELINE_QUEUE_NAME: str = "pipeline_jobs"
    PIPELINE_QUEUE_ENABLED: bool = True

    # ── Storage ──────────────────────────────────────────────
    STORAGE_BACKEND: str = "local"  # "local" | "gcs"
    UPLOAD_DIR: Path = Path("uploads")
    OUTPUT_DIR: Path = Path("outputs")
    GCS_BUCKET: str = ""

    # ── MongoDB (optional result persistence) ────────────────
    STORE_DB: bool = True
    MONGODB_URI: str = ""
    MONGO_DB_NAME: str = "bundlebox"
    MONGO_RESULTS_COLLECTION: str = "pipeline_results"

    # ── Pipeline / ML ────────────────────────────────────────
    DETECTOR_TYPE: str = "gdino"  # "gdino" | "sam3"
    GDINO_WEIGHTS_PATH: str = "/models/groundingdino_swint_ogc.pth"
    SAM3_WEIGHTS_PATH: str = "/models/sam3.pt"
    SAM3_IMGSZ: int = 512
    CLIP_MODEL: str = "ViT-B/32"
    VIDEO_STRIDE: int = 30
    MAX_VIDEO_SIZE_MB: int = 500
    PROMPT_BATCH_SIZE: int = 20

    # Scene detection (legacy — kept for backward compat)
    SCENE_HIST_THRESHOLD: float = 0.3
    SCENE_SSIM_THRESHOLD: float = 0.4
    SCENE_CLIP_THRESHOLD: float = 0.15
    SCENE_MIN_FRAMES: int = 4
    SCENE_MAX_SCENES: int = 8

    # ── V2: Smart Frame Selection ────────────────────────────
    FRAME_SAMPLE_INTERVAL_S: int = 3        # dense sample every N seconds
    FRAME_BLUR_THRESHOLD: float = 50.0      # Laplacian variance min (reject motion blur)
    FRAME_FACE_AREA_THRESHOLD: float = 0.25 # reject if face > 25% of frame
    FRAME_EDGE_THRESHOLD: float = 10.0      # reject blank walls/ceilings

    # ── V2: Audio Transcription ──────────────────────────────
    WHISPER_MODEL: str = "medium"            # tiny/base/small/medium/large-v3
    WHISPER_DEVICE: str = "auto"             # auto/cpu/cuda

    # ── V2: LLM Inventory Draft ──────────────────────────────
    LLM_BATCH_FRAME_COUNT: int = 15         # frames per LLM batch call

    # ── OpenAI ───────────────────────────────────────────────
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o"

    # ── Gemini ───────────────────────────────────────────────
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash"
    GEMINI_COMPARISON_MODELS: str = "gemini-2.5-flash-lite,gemini-2.5-pro"  # CSV of models to run in background
    RESULTS_DIR: str = "/app/results"  # CSV comparison results directory

    # ── Pipeline Backend (separate service) ──────────────────
    PIPELINE_BACKEND_URL: str = "http://pipeline_backend:8100"
    PIPELINE_API_PREFIX: str = "/api/detection/v1"
    PIPELINE_SHARED_TOKEN: str = ""
    PIPELINE_ENABLE_DOCS: bool = True

    # ── External JWT (mobile app auth) ───────────────────────
    JWT_SECRET: str = ""

    # ── Web Push (VAPID) ─────────────────────────────────────
    VAPID_PRIVATE_KEY: str = ""
    VAPID_PUBLIC_KEY: str = ""
    VAPID_CLAIMS_EMAIL: str = "mailto:admin@bundlebox.app"

    # ── Twilio (WhatsApp OTP) ─────────────────────────────────
    TWILIO_ACCOUNT_SID: str = ""
    TWILIO_API_KEY_SID: str = ""
    TWILIO_API_KEY_SECRET: str = ""
    TWILIO_WHATSAPP_FROM: str = ""  # e.g. "whatsapp:+14155238886"
    OTP_EXPIRY_SECONDS: int = 300  # 5 minutes

    @property
    def async_database_url(self) -> str:
        return self.DATABASE_URL


@lru_cache
def get_settings() -> Settings:
    return Settings()
