"""SQLAlchemy ORM models — User, Scan, Room, Video, ScanResult, InventoryItem, PipelineCost."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum as SAEnum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    JSON,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid.uuid4().hex


# ── User ─────────────────────────────────────────────────────────────────────


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    email: Mapped[str | None] = mapped_column(String(320), unique=True, index=True, nullable=True)
    phone: Mapped[str | None] = mapped_column(String(20), unique=True, index=True, nullable=True)
    name: Mapped[str] = mapped_column(String(200), default="")
    picture: Mapped[str | None] = mapped_column(String(500), nullable=True)
    hashed_password: Mapped[str | None] = mapped_column(String(200), nullable=True)
    google_id: Mapped[str | None] = mapped_column(String(100), unique=True, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    scans: Mapped[list["Scan"]] = relationship(back_populates="user", cascade="all, delete-orphan")


# ── Scan ─────────────────────────────────────────────────────────────────────


class Scan(Base):
    __tablename__ = "scans"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    display_id: Mapped[str] = mapped_column(String(50))  # "2026-03-22 Scan-1"
    scan_mode: Mapped[str] = mapped_column(String(20), default="room_by_room")  # room_by_room | all_at_once
    llm_provider: Mapped[str] = mapped_column(String(20), default="openai")  # openai | gemini
    detector_type: Mapped[str] = mapped_column(String(20), default="gdino")  # gdino | sam3
    status: Mapped[str] = mapped_column(
        String(20), default="recording"
    )  # recording | submitted | processing | completed | failed
    progress: Mapped[float] = mapped_column(Float, default=0.0)  # 0.0 → 1.0
    progress_message: Mapped[str] = mapped_column(String(200), default="")
    pipeline_step: Mapped[str] = mapped_column(String(50), default="")
    total_items: Mapped[int] = mapped_column(Integer, default=0)
    total_rooms: Mapped[int] = mapped_column(Integer, default=0)
    processing_time_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    move_summary: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # structured move summary

    user: Mapped["User"] = relationship(back_populates="scans")
    rooms: Mapped[list["Room"]] = relationship(back_populates="scan", cascade="all, delete-orphan")
    videos: Mapped[list["Video"]] = relationship(back_populates="scan", cascade="all, delete-orphan")
    inventory_items: Mapped[list["InventoryItem"]] = relationship(
        back_populates="scan", cascade="all, delete-orphan"
    )
    scan_results: Mapped[list["ScanResult"]] = relationship(
        back_populates="scan", cascade="all, delete-orphan"
    )
    pipeline_costs: Mapped[list["PipelineCost"]] = relationship(
        back_populates="scan", cascade="all, delete-orphan"
    )


# ── Room ─────────────────────────────────────────────────────────────────────


class Room(Base):
    __tablename__ = "rooms"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    scan_id: Mapped[str] = mapped_column(ForeignKey("scans.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(100))  # "Living Room", "Kitchen", custom
    emoji: Mapped[str] = mapped_column(String(10), default="🏠")
    is_custom: Mapped[bool] = mapped_column(Boolean, default=False)
    has_video: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending | processing | completed | failed
    order: Mapped[int] = mapped_column(Integer, default=0)

    scan: Mapped["Scan"] = relationship(back_populates="rooms")


# ── Video ────────────────────────────────────────────────────────────────────


class Video(Base):
    __tablename__ = "videos"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    scan_id: Mapped[str] = mapped_column(ForeignKey("scans.id", ondelete="CASCADE"), index=True)
    room_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    filename: Mapped[str] = mapped_column(String(300))
    storage_path: Mapped[str] = mapped_column(String(500))
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    duration_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    scan: Mapped["Scan"] = relationship(back_populates="videos")


# ── Inventory Item ───────────────────────────────────────────────────────────


class InventoryItem(Base):
    __tablename__ = "inventory_items"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    scan_id: Mapped[str] = mapped_column(ForeignKey("scans.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    count: Mapped[int] = mapped_column(Integer, default=1)
    room_name: Mapped[str] = mapped_column(String(100), default="")
    source: Mapped[str] = mapped_column(
        String(30), default="detected"
    )  # detected | llm_priming_fallback | manual
    is_new: Mapped[bool] = mapped_column(Boolean, default=False)  # [NEW] item discovered by AI
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    disposition: Mapped[str | None] = mapped_column(String(20), nullable=True)  # going|staying|scrap|haul|sell|null
    frame_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    image_path: Mapped[str | None] = mapped_column(String(500), nullable=True)  # path to best crop image

    scan: Mapped["Scan"] = relationship(back_populates="inventory_items")


# ── Scan Result (raw pipeline output) ────────────────────────────────────────


class ScanResult(Base):
    """Raw detection data from the pipeline — bounding boxes, tracks, confidences.

    Kept separate from InventoryItem (user-editable final list) so we can
    audit what the model actually saw vs what the user shipped.
    """
    __tablename__ = "scan_results"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    scan_id: Mapped[str] = mapped_column(ForeignKey("scans.id", ondelete="CASCADE"), index=True)
    step: Mapped[str] = mapped_column(String(50))  # scene_detection | llm_priming | gdino | tracking | verification
    frame_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    label: Mapped[str] = mapped_column(String(200), default="")
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    bbox: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # {"x1","y1","x2","y2"}
    track_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    extra_data: Mapped[dict | None] = mapped_column("extra_data", JSON, nullable=True)  # flexible extra data per step
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    scan: Mapped["Scan"] = relationship(back_populates="scan_results")


# ── Pipeline Cost ────────────────────────────────────────────────────────────


class PipelineCost(Base):
    """Per-step cost tracking for investor metrics & cost optimization."""
    __tablename__ = "pipeline_costs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    scan_id: Mapped[str] = mapped_column(ForeignKey("scans.id", ondelete="CASCADE"), index=True)
    step: Mapped[str] = mapped_column(String(50))  # llm_priming | verification | gdino_inference | scene_clip
    provider: Mapped[str] = mapped_column(String(50), default="openai")  # openai | local_gpu | clip
    model: Mapped[str] = mapped_column(String(100), default="")  # gpt-4o | gdino-swint | clip-vit-b32
    tokens_in: Mapped[int] = mapped_column(Integer, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0)
    duration_s: Mapped[float] = mapped_column(Float, default=0.0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    extra_data: Mapped[dict | None] = mapped_column("extra_data", JSON, nullable=True)  # extra: frames processed, batch count, etc.
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    scan: Mapped["Scan"] = relationship(back_populates="pipeline_costs")


# ── Push Subscription ────────────────────────────────────────────────────────


class PushSubscription(Base):
    """Web Push subscription per user/device."""
    __tablename__ = "push_subscriptions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    endpoint: Mapped[str] = mapped_column(Text)
    p256dh: Mapped[str] = mapped_column(String(200))
    auth: Mapped[str] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
