"""HTTP client for triggering the separate pipeline backend service."""

from __future__ import annotations

from pathlib import Path

import httpx
from fastapi import HTTPException
from loguru import logger

from app.config import get_settings

settings = get_settings()


def _headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    if settings.PIPELINE_SHARED_TOKEN:
        headers["X-Pipeline-Token"] = settings.PIPELINE_SHARED_TOKEN
    return headers


async def trigger_pipeline(
    scan_id: str,
    room_name: str | None = None,
    *,
    storage_path: str | None = None,
    filename: str | None = None,
    request_user_id: str | None = None,
    room_id: str | None = None,
    video_id: str | None = None,
) -> None:
    """Trigger processing in the standalone pipeline service via `/process-video`."""
    if not storage_path:
        raise HTTPException(status_code=400, detail="No uploaded video available for pipeline")

    url = f"{settings.PIPELINE_BACKEND_URL.rstrip('/')}{settings.PIPELINE_API_PREFIX}/process-video"
    data = {
        "scan_id": scan_id,
        "room_name": room_name,
        "room_id": room_id,
        "request_user_id": request_user_id,
        "video_id": video_id,
    }
    data = {k: v for k, v in data.items() if v is not None}

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            if settings.STORAGE_BACKEND == "gcs":
                if settings.GCS_BUCKET and not storage_path.startswith(("gs://", "http://", "https://")):
                    gcs_url = f"gs://{settings.GCS_BUCKET}/{storage_path.lstrip('/')}"
                else:
                    gcs_url = storage_path
                resp = await client.post(url, data={**data, "gcs_url": gcs_url}, headers=_headers())
            else:
                full_path = settings.UPLOAD_DIR / storage_path
                if not full_path.is_file():
                    raise HTTPException(status_code=404, detail=f"Video file not found: {storage_path}")
                with open(full_path, "rb") as f:
                    resp = await client.post(
                        url,
                        data=data,
                        files={"file": (filename or Path(storage_path).name, f, "video/mp4")},
                        headers=_headers(),
                    )
            resp.raise_for_status()
    except httpx.HTTPError as exc:
        logger.exception(f"Failed to trigger pipeline service for scan={scan_id}, room={room_name}")
        raise HTTPException(status_code=502, detail="Pipeline service is unavailable") from exc
