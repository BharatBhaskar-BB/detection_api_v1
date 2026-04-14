"""Scan CRUD routes — create scans, upload videos, manage inventory."""

from datetime import datetime, timezone
import json
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.deps import get_current_user
from app.config import get_settings
from app.database import get_db
from app.models.models import InventoryItem, PipelineCost, Room, Scan, ScanResult, User, Video
from app.models.schemas import (
    EvidenceFrame,
    EvidenceFrameItem,
    InventoryItemCreate,
    InventoryItemResponse,
    InventoryItemUpdate,
    InventoryResponse,
    PipelineCostResponse,
    PipelineCostSummary,
    ScanCreate,
    ScanListResponse,
    ScanResponse,
    ScanResultResponse,
    ScanResultsResponse,
)
from app.services.pipeline_client import trigger_pipeline as trigger_pipeline_backend
from app.services.storage import get_storage

router = APIRouter(prefix="/scans", tags=["scans"])
settings = get_settings()

# Testing/debug output directory (mirrors debug_dump.py)
_TESTING_DIR = Path(__file__).resolve().parent.parent.parent.parent / "testing"


def _cleanup_scan_files(scan: Scan) -> None:
    """Best-effort removal of all files associated with a scan."""
    scan_id = scan.id

    # 1. Uploaded video files
    for video in scan.videos:
        try:
            full = settings.UPLOAD_DIR / video.storage_path
            if full.exists():
                full.unlink()
        except Exception:
            pass

    # 2. Upload folder (e.g. uploads/<user_id>/<scan_id>/)
    for video in scan.videos:
        try:
            folder = (settings.UPLOAD_DIR / video.storage_path).parent
            if folder.exists() and not any(folder.iterdir()):
                folder.rmdir()
        except Exception:
            pass

    # 3. Crop images (uploads/crops/<scan_id>/)
    crops_dir = settings.UPLOAD_DIR / "crops" / scan_id
    if crops_dir.exists():
        try:
            shutil.rmtree(crops_dir)
        except Exception:
            pass

    # 4. Evidence frames (uploads/evidence/<scan_id>/)
    evidence_dir = settings.UPLOAD_DIR / "evidence" / scan_id
    if evidence_dir.exists():
        try:
            shutil.rmtree(evidence_dir)
        except Exception:
            pass

    # 5. Report JSON (uploads/reports/<scan_id>/)
    reports_dir = settings.UPLOAD_DIR / "reports" / scan_id
    if reports_dir.exists():
        try:
            shutil.rmtree(reports_dir)
        except Exception:
            pass

    # 6. Debug/testing output (testing/<scan_id>/)
    testing_dir = _TESTING_DIR / scan_id
    if testing_dir.exists():
        try:
            shutil.rmtree(testing_dir)
        except Exception:
            pass


# ── Helpers ──────────────────────────────────────────────────────────────────


async def _generate_display_id(user_id: str, db: AsyncSession) -> str:
    """Generate a user-friendly scan ID like '2026-03-22 Scan-1'."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    result = await db.execute(
        select(func.count())
        .select_from(Scan)
        .where(Scan.user_id == user_id, Scan.display_id.like(f"{today}%"))
    )
    count = result.scalar() or 0
    return f"{today} Scan-{count + 1}"


async def _get_scan_or_404(
    scan_id: str, user_id: str, db: AsyncSession, *, load_relations: bool = False
) -> Scan:
    stmt = select(Scan).where(Scan.id == scan_id, Scan.user_id == user_id)
    if load_relations:
        stmt = stmt.options(
            selectinload(Scan.rooms),
            selectinload(Scan.videos),
            selectinload(Scan.inventory_items),
        )
    result = await db.execute(stmt)
    scan = result.scalar_one_or_none()
    if scan is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scan not found")
    return scan


# ── Scan CRUD ────────────────────────────────────────────────────────────────


@router.post("", response_model=ScanResponse, status_code=status.HTTP_201_CREATED)
async def create_scan(
    body: ScanCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    display_id = await _generate_display_id(user.id, db)
    scan = Scan(
        user_id=user.id,
        display_id=display_id,
        scan_mode=body.scan_mode,
        llm_provider=body.llm_provider,
        total_rooms=len(body.rooms),
    )
    db.add(scan)
    await db.flush()

    for i, room in enumerate(body.rooms):
        db.add(Room(scan_id=scan.id, name=room.name, emoji=room.emoji, is_custom=room.is_custom, order=i))

    await db.flush()
    return await _get_scan_or_404(scan.id, user.id, db, load_relations=True)


@router.get("", response_model=list[ScanListResponse])
async def list_scans(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Scan).where(Scan.user_id == user.id).order_by(Scan.created_at.desc())
    )
    return result.scalars().all()


@router.get("/{scan_id}", response_model=ScanResponse)
async def get_scan(
    scan_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _get_scan_or_404(scan_id, user.id, db, load_relations=True)


@router.delete("/{scan_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_scan(
    scan_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    scan = await _get_scan_or_404(scan_id, user.id, db, load_relations=True)
    _cleanup_scan_files(scan)
    await db.delete(scan)
    await db.commit()


@router.post("/bulk-delete", status_code=status.HTTP_204_NO_CONTENT)
async def bulk_delete_scans(
    body: dict,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete multiple scans by ID list."""
    scan_ids = body.get("scan_ids", [])
    if not scan_ids or not isinstance(scan_ids, list):
        raise HTTPException(status_code=400, detail="scan_ids must be a non-empty list")

    for sid in scan_ids:
        try:
            scan = await _get_scan_or_404(sid, user.id, db, load_relations=True)
            _cleanup_scan_files(scan)
            await db.delete(scan)
        except HTTPException:
            continue  # skip scans that don't exist or don't belong to user

    await db.commit()


# ── Video Upload ─────────────────────────────────────────────────────────────


@router.post("/{scan_id}/videos", status_code=status.HTTP_201_CREATED)
async def upload_video(
    scan_id: str,
    file: UploadFile,
    room_name: str | None = Form(default=None),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    scan = await _get_scan_or_404(scan_id, user.id, db)

    # room_by_room scans accept uploads while processing (other rooms running)
    allowed = ("recording", "submitted")
    if scan.scan_mode == "room_by_room":
        allowed = ("recording", "submitted", "processing")
    if scan.status not in allowed:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Scan is not accepting uploads")

    # Validate content type
    if file.content_type and not file.content_type.startswith("video/"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File must be a video")

    # Read and store (no file-size cap for local dev)
    content = await file.read()
    size = len(content)

    storage = get_storage()
    storage_path = await storage.save_video(scan.user_id, scan.id, file.filename or "video.mp4", content)

    video = Video(
        scan_id=scan.id,
        room_name=room_name,
        filename=file.filename or "video.mp4",
        storage_path=storage_path,
        size_bytes=size,
    )
    db.add(video)

    # Mark room as having video
    room = None
    if room_name:
        result = await db.execute(
            select(Room).where(Room.scan_id == scan.id, Room.name == room_name)
        )
        room = result.scalar_one_or_none()
        if room:
            room.has_video = True

    await db.flush()

    # For room_by_room mode, auto-trigger pipeline for this room
    if room_name and scan.scan_mode == "room_by_room":
        await trigger_pipeline_backend(
            scan.id,
            room_name,
            storage_path=video.storage_path,
            filename=video.filename,
            request_user_id=user.id,
            room_id=(room.id if room else None),
            video_id=video.id,
        )

    return {"id": video.id, "filename": video.filename, "size_bytes": video.size_bytes}


# ── Submit Scan for Processing ───────────────────────────────────────────────


@router.post("/{scan_id}/submit", response_model=ScanResponse)
async def submit_scan(
    scan_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    scan = await _get_scan_or_404(scan_id, user.id, db, load_relations=True)

    if scan.status not in ("recording", "submitted"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Scan cannot be resubmitted")

    if not scan.videos:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Upload at least one video first")

    scan.status = "submitted"
    await db.flush()

    # Trigger async pipeline processing
    latest_video = max(scan.videos, key=lambda v: v.uploaded_at or datetime.min.replace(tzinfo=timezone.utc))
    await trigger_pipeline_backend(
        scan.id,
        storage_path=latest_video.storage_path,
        filename=latest_video.filename,
        request_user_id=user.id,
        video_id=latest_video.id,
    )

    return await _get_scan_or_404(scan.id, user.id, db, load_relations=True)


# ── Item Crop Images ─────────────────────────────────────────────────────────


@router.get("/{scan_id}/crops/{filename}")
async def get_item_crop(scan_id: str, filename: str):
    """Serve a cropped item image (no auth — scan_id is 32-char hex capability URL)."""
    # Sanitize filename to prevent path traversal
    safe = Path(filename).name
    crop_path = settings.UPLOAD_DIR / "crops" / scan_id / safe
    if not crop_path.is_file():
        raise HTTPException(status_code=404, detail="Crop not found")
    return FileResponse(crop_path, media_type="image/jpeg")


@router.get("/{scan_id}/evidence/{filename}")
async def get_evidence_frame(scan_id: str, filename: str):
    """Serve a saved key-frame evidence image for annotation overlays."""
    safe = Path(filename).name
    image_path = settings.UPLOAD_DIR / "evidence" / scan_id / safe
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail="Evidence frame not found")
    return FileResponse(image_path, media_type="image/jpeg")


# ── Report ───────────────────────────────────────────────────────────────────


@router.get("/{scan_id}/report")
async def get_report(
    scan_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get the full survey report JSON for this scan."""
    await _get_scan_or_404(scan_id, user.id, db)
    report_path = settings.UPLOAD_DIR / "reports" / scan_id / "report.json"
    if not report_path.is_file():
        raise HTTPException(status_code=404, detail="Report not generated yet")
    data = json.loads(report_path.read_text(encoding="utf-8"))
    return data


# ── Inventory ────────────────────────────────────────────────────────────────


@router.get("/{scan_id}/inventory", response_model=InventoryResponse)
async def get_inventory(
    scan_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    scan = await _get_scan_or_404(scan_id, user.id, db, load_relations=True)
    items: list[InventoryItemResponse] = []
    for i in scan.inventory_items:
        resp = InventoryItemResponse.model_validate(i)
        if i.image_path:
            resp.image_url = f"/api/v1/scans/{scan_id}/crops/{i.image_path.split('/')[-1]}"
        items.append(resp)

    evidence_frames: list[EvidenceFrame] = []
    manifest = settings.UPLOAD_DIR / "evidence" / scan_id / "manifest.json"
    if manifest.is_file():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            for frame in data.get("frames", []):
                img_name = Path(frame.get("image_path", "")).name
                if not img_name:
                    continue
                frame_items = [
                    EvidenceFrameItem.model_validate(item)
                    for item in frame.get("items", [])
                ]
                evidence_frames.append(
                    EvidenceFrame(
                        frame_index=int(frame.get("frame_index", 0)),
                        image_url=f"/api/v1/scans/{scan_id}/evidence/{img_name}",
                        items=frame_items,
                    )
                )
        except Exception:
            evidence_frames = []

    return InventoryResponse(
        scan_id=scan.id,
        display_id=scan.display_id,
        status=scan.status,
        items=items,
        evidence_frames=evidence_frames,
        total_items=len(items),
        total_count=sum(i.count for i in items),
    )


@router.post("/{scan_id}/inventory", response_model=InventoryItemResponse, status_code=status.HTTP_201_CREATED)
async def add_inventory_item(
    scan_id: str,
    body: InventoryItemCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    scan = await _get_scan_or_404(scan_id, user.id, db)
    item = InventoryItem(
        scan_id=scan.id,
        name=body.name,
        count=body.count,
        room_name=body.room_name,
        source="manual",
    )
    db.add(item)
    await db.flush()
    return item


@router.patch("/{scan_id}/inventory/{item_id}", response_model=InventoryItemResponse)
async def update_inventory_item(
    scan_id: str,
    item_id: str,
    body: InventoryItemUpdate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _get_scan_or_404(scan_id, user.id, db)  # ownership check
    result = await db.execute(
        select(InventoryItem).where(InventoryItem.id == item_id, InventoryItem.scan_id == scan_id)
    )
    item = result.scalar_one_or_none()
    if item is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Item not found")

    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(item, field, value)
    await db.flush()
    return item


@router.delete("/{scan_id}/inventory/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_inventory_item(
    scan_id: str,
    item_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _get_scan_or_404(scan_id, user.id, db)
    result = await db.execute(
        select(InventoryItem).where(InventoryItem.id == item_id, InventoryItem.scan_id == scan_id)
    )
    item = result.scalar_one_or_none()
    if item is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Item not found")
    await db.delete(item)


@router.post("/{scan_id}/inventory/save", response_model=ScanResponse)
async def save_inventory(
    scan_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Finalize inventory — marks scan as completed."""
    scan = await _get_scan_or_404(scan_id, user.id, db, load_relations=True)
    scan.status = "completed"
    scan.completed_at = datetime.now(timezone.utc)
    scan.total_items = len(scan.inventory_items)
    await db.flush()
    return scan


# ── Scan Results (raw pipeline output) ───────────────────────────────────────


@router.get("/{scan_id}/results", response_model=ScanResultsResponse)
async def get_scan_results(
    scan_id: str,
    step: str | None = None,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get raw pipeline results — optionally filtered by step."""
    await _get_scan_or_404(scan_id, user.id, db)
    q = select(ScanResult).where(ScanResult.scan_id == scan_id)
    if step:
        q = q.where(ScanResult.step == step)
    q = q.order_by(ScanResult.created_at)
    result = await db.execute(q)
    results = list(result.scalars().all())
    return ScanResultsResponse(scan_id=scan_id, results=results, total=len(results))


# ── Pipeline Costs ───────────────────────────────────────────────────────────


@router.get("/{scan_id}/costs", response_model=PipelineCostSummary)
async def get_pipeline_costs(
    scan_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get per-step cost breakdown for a scan."""
    await _get_scan_or_404(scan_id, user.id, db)
    result = await db.execute(
        select(PipelineCost).where(PipelineCost.scan_id == scan_id).order_by(PipelineCost.created_at)
    )
    costs = list(result.scalars().all())
    return PipelineCostSummary(
        scan_id=scan_id,
        costs=costs,
        total_cost_usd=sum(c.cost_usd for c in costs),
        total_tokens_in=sum(c.tokens_in for c in costs),
        total_tokens_out=sum(c.tokens_out for c in costs),
        total_duration_s=sum(c.duration_s for c in costs),
    )
