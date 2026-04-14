"""HTTP routes for pipeline backend service."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select

from app.config import get_settings
from app.database import async_session
from app.models.models import InventoryItem, Scan
from app.pipeline.orchestrator import trigger_pipeline as trigger_local_pipeline
from app.pipeline.orchestrator import trigger_room_pipeline as trigger_local_room_pipeline
from app.pipeline_service.auth import authenticate_pipeline_request, verify_scan_access

settings = get_settings()
router = APIRouter(tags=["pipeline"])


class TriggerRequest(BaseModel):
    scan_id: str
    room_name: str | None = None


@router.post("/trigger", status_code=status.HTTP_202_ACCEPTED)
async def trigger_pipeline_job(
    body: TriggerRequest,
    x_pipeline_token: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
):
    user_id = await authenticate_pipeline_request(x_pipeline_token, authorization)
    await verify_scan_access(body.scan_id, user_id)

    if body.room_name:
        await trigger_local_room_pipeline(body.scan_id, body.room_name)
    else:
        await trigger_local_pipeline(body.scan_id)

    return {
        "status": "accepted",
        "job_id": body.scan_id,
        "websocket_url": f"{settings.PIPELINE_API_PREFIX}/ws/{body.scan_id}",
        "result_url": f"{settings.PIPELINE_API_PREFIX}/upload/{body.scan_id}",
    }


@router.get("/inventory/{scan_id}")
async def get_inventory_json(
    scan_id: str,
    x_pipeline_token: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """Return final inventory JSON from pipeline/backend DB state."""
    user_id = await authenticate_pipeline_request(x_pipeline_token, authorization)
    await verify_scan_access(scan_id, user_id)

    async with async_session() as db:
        scan_result = await db.execute(select(Scan).where(Scan.id == scan_id))
        scan = scan_result.scalar_one_or_none()
        if scan is None:
            raise HTTPException(status_code=404, detail="Scan not found")

        items_result = await db.execute(
            select(InventoryItem).where(InventoryItem.scan_id == scan_id)
        )
        items = items_result.scalars().all()

    return {
        "scan_id": scan_id,
        "status": scan.status,
        "progress": scan.progress,
        "total_items": len(items),
        "inventory": [
            {
                "id": item.id,
                "name": item.name,
                "count": item.count,
                "room_name": item.room_name,
                "confidence": item.confidence,
                "disposition": item.disposition,
                "image_path": item.image_path,
            }
            for item in items
        ],
    }
