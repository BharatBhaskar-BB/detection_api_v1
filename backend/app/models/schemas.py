"""Pydantic schemas — request/response models for the API."""

from datetime import datetime
from pydantic import BaseModel, EmailStr, Field


# ── Auth ─────────────────────────────────────────────────────────────────────


class GoogleAuthRequest(BaseModel):
    credential: str  # Google ID token or auth code


class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    name: str = Field(min_length=1, max_length=200)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: str
    email: str | None = None
    phone: str | None = None
    name: str
    picture: str | None = None

    model_config = {"from_attributes": True}


# ── Phone Auth (Twilio WhatsApp OTP) ─────────────────────────────────────────


class PhoneSendOTPRequest(BaseModel):
    phone: str = Field(min_length=10, max_length=20, pattern=r"^\+?[0-9]+$")
    channel: str = Field(default="sms", pattern=r"^(sms|whatsapp)$")


class PhoneVerifyOTPRequest(BaseModel):
    phone: str = Field(min_length=10, max_length=20, pattern=r"^\+?[0-9]+$")
    code: str = Field(min_length=4, max_length=6)


class PhoneSetNameRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    email: str | None = Field(default=None, max_length=320)
    password: str | None = Field(default=None, min_length=8, max_length=128)


# ── Scan ─────────────────────────────────────────────────────────────────────


class RoomCreate(BaseModel):
    name: str = Field(max_length=100)
    emoji: str = Field(default="🏠", max_length=10)
    is_custom: bool = False


class ScanCreate(BaseModel):
    scan_mode: str = Field(pattern=r"^(room_by_room|all_at_once)$")
    llm_provider: str = Field(default="gemini", pattern=r"^(openai|gemini)$")
    rooms: list[RoomCreate] = []


class RoomResponse(BaseModel):
    id: str
    name: str
    emoji: str
    is_custom: bool
    has_video: bool
    status: str
    order: int

    model_config = {"from_attributes": True}


class VideoResponse(BaseModel):
    id: str
    room_name: str | None
    filename: str
    size_bytes: int
    duration_s: float | None
    uploaded_at: datetime

    model_config = {"from_attributes": True}


class ScanResponse(BaseModel):
    id: str
    display_id: str
    scan_mode: str
    llm_provider: str
    detector_type: str
    status: str
    progress: float
    progress_message: str
    pipeline_step: str
    total_items: int
    total_rooms: int
    processing_time_s: float | None
    cost_usd: float | None
    created_at: datetime
    completed_at: datetime | None
    move_summary: dict | None = None
    rooms: list[RoomResponse] = []
    videos: list[VideoResponse] = []

    model_config = {"from_attributes": True}


class ScanListResponse(BaseModel):
    """Lightweight scan for list views."""
    id: str
    display_id: str
    scan_mode: str
    status: str
    progress: float
    progress_message: str
    total_items: int
    total_rooms: int
    created_at: datetime
    completed_at: datetime | None
    move_summary: dict | None = None

    model_config = {"from_attributes": True}


# ── Inventory ────────────────────────────────────────────────────────────────


class InventoryItemResponse(BaseModel):
    id: str
    name: str
    count: int
    room_name: str
    source: str
    is_new: bool
    confidence: float
    disposition: str | None = None
    image_url: str | None = None

    model_config = {"from_attributes": True}


class EvidenceAnchor(BaseModel):
    x: float
    y: float


class EvidenceFrameItem(BaseModel):
    name: str
    count: int
    anchors: list[EvidenceAnchor] = []


class EvidenceFrame(BaseModel):
    frame_index: int
    image_url: str
    items: list[EvidenceFrameItem] = []


class InventoryItemUpdate(BaseModel):
    count: int | None = Field(default=None, ge=0)
    name: str | None = Field(default=None, max_length=200)
    room_name: str | None = Field(default=None, max_length=100)
    disposition: str | None = Field(default=None, max_length=20)


class InventoryItemCreate(BaseModel):
    name: str = Field(max_length=200)
    count: int = Field(default=1, ge=1)
    room_name: str = Field(max_length=100)


class InventoryResponse(BaseModel):
    scan_id: str
    display_id: str
    status: str
    items: list[InventoryItemResponse]
    evidence_frames: list[EvidenceFrame] = []
    total_items: int
    total_count: int  # sum of all counts


# ── Scan Results (raw pipeline output) ───────────────────────────────────────


class ScanResultResponse(BaseModel):
    id: str
    step: str
    frame_index: int | None
    label: str
    confidence: float
    bbox: dict | None = None
    track_id: int | None = None
    extra_data: dict | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ScanResultsResponse(BaseModel):
    scan_id: str
    results: list[ScanResultResponse]
    total: int


# ── Pipeline Cost Tracking ───────────────────────────────────────────────────


class PipelineCostResponse(BaseModel):
    id: str
    step: str
    provider: str
    model: str
    tokens_in: int
    tokens_out: int
    duration_s: float
    cost_usd: float
    extra_data: dict | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class PipelineCostSummary(BaseModel):
    scan_id: str
    costs: list[PipelineCostResponse]
    total_cost_usd: float
    total_tokens_in: int
    total_tokens_out: int
    total_duration_s: float


# ── WebSocket Messages ───────────────────────────────────────────────────────


class WSProgress(BaseModel):
    type: str = "progress"
    scan_id: str
    room_name: str = ""
    step: str
    step_number: int
    total_steps: int = 4
    progress: float
    message: str
    items_found: list[str] = []
    elapsed_s: float = 0
    eta_s: float | None = None


class WSItemAdded(BaseModel):
    type: str = "item_added"
    scan_id: str
    room_name: str = ""
    item_name: str


class WSComplete(BaseModel):
    type: str = "complete"
    scan_id: str
    room_name: str = ""
    total_items: int
    total_rooms: int
    processing_time_s: float
    cost_usd: float


class WSError(BaseModel):
    type: str = "error"
    scan_id: str
    room_name: str = ""
    message: str
