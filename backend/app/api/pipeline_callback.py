"""Pipeline callback endpoints to persist standalone pipeline_service results."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import delete, select

from app.config import get_settings
from app.database import async_session
from app.models.models import InventoryItem, Scan

router = APIRouter(prefix="/pipeline-callback", tags=["pipeline-callback"])
settings = get_settings()


class PipelineEvent(BaseModel):
    event: str
    job_id: str | None = None
    scan_id: str | None = None
    room_name: str | None = None
    step: str | None = None
    progress: float | None = None
    message: str | None = None
    usage: dict[str, Any] | None = None
    report: dict[str, Any] | None = None
    inventory: list[dict[str, Any]] | None = None
    result: dict[str, Any] | None = None


def _verify_pipeline_token(x_pipeline_token: str | None) -> None:
    expected = (settings.PIPELINE_SHARED_TOKEN or "").strip()
    if not expected:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Pipeline shared token not configured")
    if (x_pipeline_token or "").strip() != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid pipeline token")


async def _get_scan(scan_id: str) -> Scan | None:
    async with async_session() as db:
        res = await db.execute(select(Scan).where(Scan.id == scan_id))
        return res.scalar_one_or_none()


@router.post("/event")
async def callback_event(
    body: PipelineEvent,
    x_pipeline_token: str | None = Header(default=None),
):
    _verify_pipeline_token(x_pipeline_token)

    scan_id = (body.scan_id or body.job_id or "").strip()
    if not scan_id:
        raise HTTPException(status_code=400, detail="scan_id or job_id is required")

    event = (body.event or "").strip().lower()
    if event not in {"progress", "complete", "error"}:
        raise HTTPException(status_code=400, detail="event must be one of: progress, complete, error")

    async with async_session() as db:
        res = await db.execute(select(Scan).where(Scan.id == scan_id))
        scan = res.scalar_one_or_none()
        if scan is None:
            raise HTTPException(status_code=404, detail="Scan not found")

        if event == "progress":
            scan.status = "processing"
            if body.step is not None:
                scan.pipeline_step = body.step
            if body.progress is not None:
                scan.progress = max(0.0, min(1.0, float(body.progress)))
            if body.message:
                scan.progress_message = body.message
            await db.commit()
            return {"status": "ok"}

        if event == "error":
            scan.status = "failed"
            scan.progress = 1.0
            scan.progress_message = (body.message or "Pipeline failed")[:200]
            await db.commit()
            return {"status": "ok"}

        # complete
        result_payload = body.result if isinstance(body.result, dict) else {}
        inventory = result_payload.get("inventory") if isinstance(result_payload.get("inventory"), list) else (body.inventory or [])
        report = result_payload.get("report") if isinstance(result_payload.get("report"), dict) else body.report

        await db.execute(delete(InventoryItem).where(InventoryItem.scan_id == scan_id))

        total_items = 0
        for item in inventory:
            if not isinstance(item, dict):
                continue
            count = int(item.get("count", 1) or 1)
            total_items += max(count, 0)
            db.add(
                InventoryItem(
                    scan_id=scan_id,
                    name=str(item.get("name") or "Unknown item"),
                    count=max(1, count),
                    room_name=str(item.get("room") or body.room_name or ""),
                    confidence=float(item.get("confidence", 1.0) or 1.0),
                    disposition=item.get("disposition"),
                    source="detected",
                )
            )

        scan.status = "completed"
        scan.progress = 1.0
        scan.pipeline_step = "report"
        scan.progress_message = body.message or "Pipeline complete"
        scan.total_items = total_items
        scan.completed_at = datetime.now(timezone.utc)

        if isinstance(report, dict):
            report_dir = settings.UPLOAD_DIR / "reports" / scan_id
            report_dir.mkdir(parents=True, exist_ok=True)
            (report_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

        await db.commit()

    return {"status": "ok"}
