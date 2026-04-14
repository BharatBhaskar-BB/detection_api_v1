"""HTTP routes for pipeline backend service."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from urllib.request import urlretrieve

import httpx
from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from loguru import logger
from pydantic import BaseModel

from app.config import get_settings
from app.pipeline.frame_selector import FrameSelector
from app.pipeline.llm_inventory import LLMInventoryDrafter
from app.pipeline.report_generator import generate_report
from app.pipeline.transcriber import Transcriber
from app.utils.video import extract_frames, get_video_info
from pipeline_service.auth import authenticate_pipeline_request, verify_scan_access
from pipeline_service.queue import publish_pipeline_job_async, queue_enabled
from pipeline_service.websocket import manager

settings = get_settings()
router = APIRouter(tags=["pipeline"])
_JOB_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_JOBS: dict[str, dict[str, Any]] = {}
PIPELINE_STEPS = [
    "initiated",
    "video_processing",
    "frame_selection",
    "ai_analysis",
    "report",
]


def _store_db_enabled() -> bool:
    return bool(getattr(settings, "STORE_DB", False))


def _store_result_in_mongo_sync(job_id: str, result_payload: dict[str, Any], meta: dict[str, Any]) -> None:
    mongo_uri = (getattr(settings, "MONGODB_URI", "") or "").strip()
    if not mongo_uri:
        raise RuntimeError("STORE_DB=true but MONGODB_URI is empty")

    # Lazy import to avoid hard dependency when STORE_DB is disabled.
    from pymongo import MongoClient

    db_name = (getattr(settings, "MONGO_DB_NAME", "bundlebox") or "bundlebox").strip()
    collection_name = (
        getattr(settings, "MONGO_RESULTS_COLLECTION", "pipeline_results")
        or "pipeline_results"
    ).strip()

    now = datetime.now(timezone.utc)
    clean_payload = json.loads(json.dumps(result_payload, default=str))
    doc = {
        "job_id": job_id,
        "scan_id": meta.get("scan_id"),
        "room_id": meta.get("room_id"),
        "user_id": meta.get("user_id"),
        "video_id": meta.get("video_id"),
        "room_name": meta.get("room_name"),
        "status": "completed",
        "result": clean_payload,
        "createdAt": now,
        "updatedAt": now,
    }

    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=10000)
    try:
        client.admin.command("ping")
        collection = client[db_name][collection_name]
        collection.insert_one(doc)
    finally:
        client.close()


def _callback_url() -> str:
    return (os.getenv("PIPELINE_CALLBACK_URL") or "").strip()


def _callback_token() -> str:
    return (os.getenv("PIPELINE_CALLBACK_TOKEN") or settings.PIPELINE_SHARED_TOKEN or "").strip()


async def _send_callback(event: str, payload: dict[str, Any]) -> None:
    url = _callback_url()
    if not url:
        return
    headers = {"Content-Type": "application/json"}
    token = _callback_token()
    if token:
        headers["X-Pipeline-Token"] = token
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json={"event": event, **payload}, headers=headers)
            resp.raise_for_status()
    except Exception as exc:
        logger.warning(f"Pipeline callback failed ({event}): {exc}")


class TriggerRequest(BaseModel):
    scan_id: str
    room_name: str | None = None
    room_id: str | None = None
    video_id: str | None = None


def _job_meta(job_id: str) -> dict[str, Any]:
    state = _load_job(job_id) or {}
    return {
        "scan_id": state.get("scan_id"),
        "room_id": state.get("room_id"),
        "user_id": state.get("user_id"),
        "video_id": state.get("video_id"),
        "room_name": state.get("room_name"),
    }


def _accepted_response(job_id: str) -> dict[str, Any]:
    payload = {
        "status": "accepted",
        "job_id": job_id,
        "websocket_url": f"{settings.PIPELINE_API_PREFIX}/ws/{job_id}",
        "result_url": f"{settings.PIPELINE_API_PREFIX}/upload/{job_id}",
    }
    payload.update(_job_meta(job_id))
    return payload


def _upload_dir(scan_id: str) -> Path:
    p = settings.UPLOAD_DIR / "pipeline_jobs" / scan_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def _state_file(job_id: str) -> Path:
    return _upload_dir(job_id) / "state.json"


def _clear_previous_result(job_id: str) -> None:
    base = _upload_dir(job_id)
    result_file = base / "result.json"
    evidence_dir = base / "evidence"

    try:
        if result_file.is_file():
            result_file.unlink()
    except Exception:
        pass

    try:
        if evidence_dir.is_dir():
            shutil.rmtree(evidence_dir)
    except Exception:
        pass


def _load_job(job_id: str) -> dict[str, Any] | None:
    state = _JOBS.get(job_id)
    if state is not None:
        return state

    sf = _state_file(job_id)
    if not sf.is_file():
        return None
    try:
        loaded = json.loads(sf.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            _JOBS[job_id] = loaded
            return loaded
    except Exception:
        return None
    return None


def _persist_job(job_id: str, state: dict[str, Any]) -> None:
    sf = _state_file(job_id)
    sf.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _validate_job_id(job_id: str) -> str:
    job_id = (job_id or "").strip()
    if not _JOB_ID_RE.match(job_id):
        raise HTTPException(
            status_code=400,
            detail="job_id must be 1-128 chars (letters, numbers, '_' or '-')",
        )
    return job_id


async def _set_job(job_id: str, **fields: Any) -> None:
    state = _load_job(job_id) or {"job_id": job_id, "status": "queued"}
    state.update(fields)
    _JOBS[job_id] = state
    _persist_job(job_id, state)


async def _broadcast(job_id: str, payload: dict[str, Any]) -> None:
    await manager.broadcast(job_id, payload)


async def _emit_progress(
    job_id: str,
    started_at: float,
    step: str,
    progress: float,
    message: str,
) -> None:
    step_number = PIPELINE_STEPS.index(step) + 1 if step in PIPELINE_STEPS else 1
    meta = _job_meta(job_id)
    progress_payload = {
        "type": "progress",
        "job_id": job_id,
        "step": step,
        "step_number": step_number,
        "total_steps": len(PIPELINE_STEPS),
        "progress": progress,
        "message": message,
        "elapsed_s": round(max(0.0, time.time() - started_at), 3),
        **meta,
    }
    await _broadcast(
        job_id,
        progress_payload,
    )
    await _send_callback("progress", progress_payload)


def _meta_from_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {}
    return {
        "scan_id": payload.get("scan_id"),
        "room_id": payload.get("room_id"),
        "user_id": payload.get("user_id"),
        "video_id": payload.get("video_id"),
        "room_name": payload.get("room_name"),
    }


def _normalize_optional_text(value: str | None) -> str | None:
    cleaned = (value or "").strip()
    return cleaned or None


def _parse_gcs_url(gcs_url: str) -> tuple[str, str]:
    raw = (gcs_url or "").strip()
    parsed = urlparse(raw)

    bucket = ""
    blob_name = ""

    if parsed.scheme == "gs":
        bucket = parsed.netloc
        blob_name = parsed.path.lstrip("/")
    elif parsed.scheme in {"http", "https"}:
        host = parsed.netloc.lower()
        path = parsed.path.lstrip("/")
        if host == "storage.googleapis.com":
            bucket, _, blob_name = path.partition("/")
        elif host.endswith(".storage.googleapis.com"):
            bucket = host[: -len(".storage.googleapis.com")]
            blob_name = path

    if not bucket or not blob_name:
        raise HTTPException(
            status_code=400,
            detail="gcs_url must be a valid gs:// URL or storage.googleapis.com URL",
        )

    return bucket, unquote(blob_name)


def _download_video_from_gcs_url(job_id: str, gcs_url: str) -> tuple[Path, str]:
    bucket, blob_name = _parse_gcs_url(gcs_url)
    upload_path = _upload_dir(job_id)
    safe_name = Path(blob_name).name or "input.mp4"
    video_path = upload_path / safe_name

    parsed = urlparse((gcs_url or "").strip())
    if parsed.scheme == "gs":
        try:
            from google.cloud import storage
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"google-cloud-storage import failed: {exc}") from exc

        def _build_gcs_client() -> "storage.Client":
            adc_path = (os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or "").strip()
            if adc_path and Path(adc_path).is_file():
                return storage.Client.from_service_account_json(adc_path)

            try:
                return storage.Client()
            except Exception as adc_exc:
                # Last fallback: look for a service-account JSON in pipeline_service directory.
                svc_dir = Path(__file__).resolve().parent
                for candidate in svc_dir.glob("*.json"):
                    if not candidate.is_file():
                        continue
                    try:
                        return storage.Client.from_service_account_json(str(candidate))
                    except Exception:
                        continue
                raise HTTPException(
                    status_code=500,
                    detail=(
                        "GCS credentials not available. Set GOOGLE_APPLICATION_CREDENTIALS "
                        f"or provide workload identity. ({adc_exc})"
                    ),
                ) from adc_exc

        try:
            client = _build_gcs_client()
            client.bucket(bucket).blob(blob_name).download_to_filename(video_path)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to download gcs_url: {exc}") from exc
    else:
        try:
            urlretrieve(gcs_url, video_path)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to download gcs_url: {exc}") from exc

    return video_path, safe_name


def _resolve_video_input(file: UploadFile | None, gcs_url: str | None) -> tuple[UploadFile | None, str | None]:
    normalized_gcs = _normalize_optional_text(gcs_url)
    has_file = file is not None
    has_gcs = normalized_gcs is not None

    if has_file and has_gcs:
        raise HTTPException(status_code=400, detail="Provide either file or gcs_url, not both")
    if not has_file and not has_gcs:
        raise HTTPException(status_code=400, detail="Either file or gcs_url is required")
    return file, normalized_gcs


def _image_url(job_id: str, image_name: str) -> str:
    rel = f"{settings.PIPELINE_API_PREFIX}/upload/{job_id}/images/{image_name}"
    public_base = (os.getenv("PIPELINE_PUBLIC_BASE_URL") or "").strip().rstrip("/")
    return f"{public_base}{rel}" if public_base else rel


def _is_evidence_image_url(value: str) -> bool:
    if value.startswith("http://") or value.startswith("https://"):
        return True
    return value.startswith(f"{settings.PIPELINE_API_PREFIX}/upload/") and "/images/" in value


def _with_public_base_if_relative(url_value: str) -> str:
    public_base = (os.getenv("PIPELINE_PUBLIC_BASE_URL") or "").strip().rstrip("/")
    if not public_base:
        return url_value
    if url_value.startswith("http://") or url_value.startswith("https://"):
        return url_value
    if url_value.startswith("/"):
        return f"{public_base}{url_value}"
    return url_value


def _materialize_report_images(job_id: str, report: dict[str, Any]) -> dict[str, Any]:
    items = report.get("items")
    if not isinstance(items, list):
        return report

    evidence_dir = _upload_dir(job_id) / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue

        raw = item.get("evidence_image")
        if not raw or not isinstance(raw, str):
            continue

        # Already URL style; keep as-is.
        if _is_evidence_image_url(raw):
            item["evidence_image"] = _with_public_base_if_relative(raw)
            continue

        b64 = raw
        if raw.lower().startswith("data:image") and "," in raw:
            b64 = raw.split(",", 1)[1]

        try:
            compact_b64 = "".join(b64.split())
            padding_needed = len(compact_b64) % 4
            if padding_needed:
                compact_b64 += "=" * (4 - padding_needed)
            try:
                image_bytes = base64.b64decode(compact_b64, validate=False)
            except Exception:
                image_bytes = base64.urlsafe_b64decode(compact_b64)
            if not image_bytes:
                continue
        except Exception:
            # Keep original field unchanged if decode fails.
            continue

        image_name = f"item_{idx + 1}.jpg"
        image_path = evidence_dir / image_name
        image_path.write_bytes(image_bytes)
        item["evidence_image"] = _image_url(job_id, image_name)

    return report


def _ensure_payload_report_image_urls(job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    report = payload.get("report")
    if isinstance(report, dict):
        payload["report"] = _materialize_report_images(job_id, report)
    return payload


async def _run_uploaded_video_pipeline(job_id: str, video_path: str, meta: dict[str, Any] | None = None) -> None:
    started_at = time.time()
    effective_meta = _meta_from_payload(meta) or _job_meta(job_id)
    if effective_meta:
        await _set_job(job_id, **effective_meta)
    try:
        await _set_job(job_id, status="processing", progress=0.05, message="Starting pipeline")
        await _emit_progress(job_id, started_at, "video_processing", 0.05, "Extracting frames and transcribing audio")

        info = get_video_info(video_path)
        fps = info.get("fps", 30) or 30
        duration_s = info.get("duration_s", 0) or 0

        all_frames = extract_frames(video_path, stride=settings.VIDEO_STRIDE)
        frame_count = len(all_frames)

        transcriber = Transcriber()
        transcript = transcriber.transcribe(video_path)
        has_audio = transcript.has_speech

        await _set_job(job_id, progress=0.25, message=f"Extracted {frame_count} frames")
        await _emit_progress(job_id, started_at, "video_processing", 0.25, f"Extracted {frame_count} frames")

        await _set_job(job_id, progress=0.35, message="Selecting frames")
        await _emit_progress(job_id, started_at, "frame_selection", 0.35, "Selecting key frames")

        selector = FrameSelector()
        selected_frames = await asyncio.to_thread(
            selector.select,
            all_frames,
            fps=fps,
            stride=settings.VIDEO_STRIDE,
            duration_s=duration_s,
        )

        await _set_job(job_id, progress=0.55, message=f"Selected {len(selected_frames)} frames")
        await _emit_progress(job_id, started_at, "frame_selection", 0.55, f"Selected {len(selected_frames)} key frames")

        await _set_job(job_id, progress=0.70, message="Analyzing inventory with Gemini")
        await _emit_progress(job_id, started_at, "ai_analysis", 0.70, "Analyzing inventory with Gemini")

        drafter = LLMInventoryDrafter()
        inventory = await drafter.draft(
            selected_frames,
            transcript=transcript,
            duration_s=duration_s,
        )

        await _set_job(job_id, progress=0.88, message="Generating report")
        await _emit_progress(job_id, started_at, "report", 0.88, "Generating report")

        report = await generate_report(
            inventory,
            selected_frames,
            duration_s,
            has_audio=has_audio,
        )
        report = _materialize_report_images(job_id, report)

        result_payload: dict[str, Any] = {
            "job_id": job_id,
            "status": "completed",
            **effective_meta,
            "inventory": [
                {
                    "name": item.name,
                    "count": item.count,
                    "room": item.room,
                    "confidence": item.confidence,
                    "disposition": item.disposition,
                }
                for item in inventory.items
            ],
            "usage": inventory.usage,
            "report": report,
        }

        out = _upload_dir(job_id) / "result.json"
        out.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")

        if _store_db_enabled():
            try:
                await asyncio.to_thread(_store_result_in_mongo_sync, job_id, result_payload, effective_meta)
            except Exception as exc:
                logger.warning(f"Mongo result persistence failed for job={job_id}: {exc}")

        await _set_job(
            job_id,
            status="completed",
            progress=1.0,
            message="Pipeline complete",
            result=result_payload,
        )
        complete_payload = {
            "type": "complete",
            "job_id": job_id,
            "progress": 1.0,
            "message": "Pipeline complete",
            "total_item_types": len(result_payload["inventory"]),
            **effective_meta,
        }
        await _broadcast(job_id, complete_payload)
        await _send_callback("complete", {**complete_payload, "result": result_payload})
    except Exception as exc:
        await _set_job(job_id, status="failed", progress=1.0, message=str(exc)[:300])
        error_payload = {
            "type": "error",
            "job_id": job_id,
            "message": f"Pipeline failed: {str(exc)[:300]}",
            **effective_meta,
        }
        await _broadcast(job_id, error_payload)
        await _send_callback("error", error_payload)


async def enqueue_uploaded_video_pipeline(job_id: str, video_path: str) -> None:
    payload = {
        "job_id": job_id,
        "video_path": video_path,
        "meta": _job_meta(job_id),
    }
    if queue_enabled():
        try:
            await publish_pipeline_job_async(payload)
            return
        except Exception as exc:
            logger.warning(f"Queue publish failed for job={job_id}; falling back to local task: {exc}")
            asyncio.create_task(_run_uploaded_video_pipeline(job_id, video_path, meta=payload["meta"]))
        return
    asyncio.create_task(_run_uploaded_video_pipeline(job_id, video_path, meta=payload["meta"]))


async def consume_pipeline_job(payload: dict[str, Any]) -> None:
    job_id = _validate_job_id(str(payload.get("job_id") or ""))
    video_path = str(payload.get("video_path") or "")
    if not video_path:
        raise ValueError("Missing video_path in queued payload")
    meta = _meta_from_payload(payload.get("meta") if isinstance(payload.get("meta"), dict) else None)
    await _run_uploaded_video_pipeline(job_id, video_path, meta=meta)


@router.post("/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_video(
    file: UploadFile | None = File(default=None),
    gcs_url: str | None = Form(default=None),
    job_id: str = Form(...),
    scan_id: str | None = Form(default=None),
    room_id: str | None = Form(default=None),
    request_user_id: str | None = Form(default=None),
    video_id: str | None = Form(default=None),
    room_name: str | None = Form(default=None),
    x_pipeline_token: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """Upload a video or pass `gcs_url`, then start pipeline."""
    auth_user_id = await authenticate_pipeline_request(x_pipeline_token, authorization)
    effective_user_id = auth_user_id or request_user_id
    job_id = _validate_job_id(job_id)
    started_at = time.time()

    # Reusing job_id should start fresh, not expose stale previous output.
    _clear_previous_result(job_id)

    file, normalized_gcs_url = _resolve_video_input(file, gcs_url)

    if file is not None:
        if file.content_type and not file.content_type.startswith("video/"):
            raise HTTPException(status_code=400, detail="file must be a video")

        upload_path = _upload_dir(job_id)
        safe_name = Path(file.filename or "input.mp4").name
        video_path = upload_path / safe_name

        with open(video_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    else:
        video_path, safe_name = await asyncio.to_thread(
            _download_video_from_gcs_url,
            job_id,
            normalized_gcs_url or "",
        )

    await _set_job(
        job_id,
        status="queued",
        progress=0.0,
        message="Input accepted and queued",
        result=None,
        video_filename=safe_name,
        video_path=str(video_path),
        gcs_url=normalized_gcs_url,
        owner_user_id=auth_user_id,
        scan_id=scan_id,
        room_id=room_id,
        user_id=effective_user_id,
        video_id=video_id,
        room_name=room_name,
    )
    await _emit_progress(job_id, started_at, "initiated", 0.01, "Process initiated")
    await enqueue_uploaded_video_pipeline(job_id, str(video_path))

    return _accepted_response(job_id)


@router.post("/process-video", status_code=status.HTTP_202_ACCEPTED)
async def process_video(
    file: UploadFile | None = File(default=None),
    gcs_url: str | None = Form(default=None),
    scan_id: str | None = Form(default=None),
    room_id: str | None = Form(default=None),
    request_user_id: str | None = Form(default=None),
    # video_id: str | None = Form(default=None),
    room_name: str | None = Form(default=None),
    x_pipeline_token: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """One-shot endpoint: upload + queue processing in a single request."""

    job_id = scan_id
    video_id = 1
    return await upload_video(
        file=file,
        gcs_url=gcs_url,
        job_id=job_id,
        scan_id=scan_id,
        room_id=room_id,
        request_user_id=request_user_id,
        video_id=video_id,
        room_name=room_name,
        x_pipeline_token=x_pipeline_token,
        authorization=authorization,
    )


@router.post("/upload/init", status_code=status.HTTP_202_ACCEPTED)
async def init_upload_job(
    job_id: str = Form(...),
    scan_id: str | None = Form(default=None),
    room_id: str | None = Form(default=None),
    request_user_id: str | None = Form(default=None),
    video_id: str | None = Form(default=None),
    room_name: str | None = Form(default=None),
    x_pipeline_token: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """Create a job first and return immediately before any file upload."""
    auth_user_id = await authenticate_pipeline_request(x_pipeline_token, authorization)
    effective_user_id = auth_user_id or request_user_id
    job_id = _validate_job_id(job_id)

    # Re-initializing same job_id should reset prior output artifacts.
    _clear_previous_result(job_id)

    await _set_job(
        job_id,
        status="awaiting_upload",
        progress=0.0,
        message="Job created. Upload file then trigger processing.",
        result=None,
        owner_user_id=auth_user_id,
        scan_id=scan_id,
        room_id=room_id,
        user_id=effective_user_id,
        video_id=video_id,
        room_name=room_name,
    )

    payload = _accepted_response(job_id)
    payload["upload_url"] = f"{settings.PIPELINE_API_PREFIX}/upload/{job_id}/file"
    payload["trigger_url"] = f"{settings.PIPELINE_API_PREFIX}/upload/{job_id}/trigger"
    return payload


@router.post("/upload/{job_id}/file", status_code=status.HTTP_201_CREATED)
async def upload_job_file(
    job_id: str,
    file: UploadFile | None = File(default=None),
    gcs_url: str | None = Form(default=None),
    scan_id: str | None = Form(default=None),
    room_id: str | None = Form(default=None),
    request_user_id: str | None = Form(default=None),
    video_id: str | None = Form(default=None),
    room_name: str | None = Form(default=None),
    x_pipeline_token: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """Upload file only (no processing). Use with `/upload/init` + `/upload/{job_id}/trigger`."""
    auth_user_id = await authenticate_pipeline_request(x_pipeline_token, authorization)
    effective_user_id = auth_user_id or request_user_id
    job_id = _validate_job_id(job_id)

    state = _load_job(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="job not found")
    if auth_user_id is not None:
        owner_id = state.get("owner_user_id")
        if owner_id and owner_id != auth_user_id:
            raise HTTPException(status_code=403, detail="Forbidden for this job")

    # New file for existing job_id should invalidate previous output.
    _clear_previous_result(job_id)

    file, normalized_gcs_url = _resolve_video_input(file, gcs_url)

    if file is not None:
        if file.content_type and not file.content_type.startswith("video/"):
            raise HTTPException(status_code=400, detail="file must be a video")

        upload_path = _upload_dir(job_id)
        safe_name = Path(file.filename or "input.mp4").name
        video_path = upload_path / safe_name

        with open(video_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    else:
        video_path, safe_name = await asyncio.to_thread(
            _download_video_from_gcs_url,
            job_id,
            normalized_gcs_url or "",
        )

    await _set_job(
        job_id,
        status="uploaded",
        progress=0.0,
        message="Input saved. Trigger processing to start.",
        result=None,
        video_filename=safe_name,
        video_path=str(video_path),
        gcs_url=normalized_gcs_url,
        scan_id=(scan_id or state.get("scan_id")),
        room_id=(room_id or state.get("room_id")),
        user_id=(effective_user_id or state.get("user_id")),
        video_id=(video_id or state.get("video_id")),
        room_name=(room_name or state.get("room_name")),
    )

    payload = {
        "status": "uploaded",
        "job_id": job_id,
        "video_filename": safe_name,
    }
    payload.update(_job_meta(job_id))
    return payload


@router.post("/upload/{job_id}/trigger", status_code=status.HTTP_202_ACCEPTED)
async def trigger_uploaded_job(
    job_id: str,
    x_pipeline_token: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """Trigger processing for an already-uploaded file and return immediately."""
    user_id = await authenticate_pipeline_request(x_pipeline_token, authorization)
    job_id = _validate_job_id(job_id)

    state = _load_job(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="job not found")
    if user_id is not None:
        owner_id = state.get("owner_user_id")
        if owner_id and owner_id != user_id:
            raise HTTPException(status_code=403, detail="Forbidden for this job")

    video_path = state.get("video_path")
    if not video_path:
        raise HTTPException(status_code=400, detail="No uploaded file for this job")

    started_at = time.time()
    # Ensure retrigger on same job_id doesn't return an old completed payload.
    _clear_previous_result(job_id)
    await _set_job(job_id, status="queued", progress=0.0, message="Processing queued", result=None)
    await _emit_progress(job_id, started_at, "initiated", 0.01, "Process initiated")
    await enqueue_uploaded_video_pipeline(job_id, video_path)

    return _accepted_response(job_id)


@router.get("/upload/{job_id}")
async def get_upload_result(
    job_id: str,
    x_pipeline_token: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    user_id = await authenticate_pipeline_request(x_pipeline_token, authorization)
    job_id = _validate_job_id(job_id)

    state = _load_job(job_id)
    if state is not None and user_id is not None:
        owner_id = state.get("owner_user_id")
        if owner_id and owner_id != user_id:
            raise HTTPException(status_code=403, detail="Forbidden for this job")

    if state is None:
        result_file = _upload_dir(job_id) / "result.json"
        if result_file.is_file():
            result_payload = json.loads(result_file.read_text(encoding="utf-8"))
            if isinstance(result_payload, dict):
                result_payload = _ensure_payload_report_image_urls(job_id, result_payload)
                result_file.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
            return result_payload
        raise HTTPException(status_code=404, detail="upload result not found")

    if state.get("status") == "completed" and isinstance(state.get("result"), dict):
        state["result"] = _ensure_payload_report_image_urls(job_id, state["result"])
        _persist_job(job_id, state)

    return state


@router.get("/upload/{job_id}/images/{image_name}")
async def get_upload_image(
    job_id: str,
    image_name: str,
    x_pipeline_token: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
):
    user_id = await authenticate_pipeline_request(x_pipeline_token, authorization)
    job_id = _validate_job_id(job_id)

    if Path(image_name).name != image_name:
        raise HTTPException(status_code=400, detail="Invalid image name")

    state = _load_job(job_id)
    if state is not None and user_id is not None:
        owner_id = state.get("owner_user_id")
        if owner_id and owner_id != user_id:
            raise HTTPException(status_code=403, detail="Forbidden for this job")

    image_path = _upload_dir(job_id) / "evidence" / image_name
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail="image not found")

    return FileResponse(path=image_path, media_type="image/jpeg", filename=image_name)


@router.post("/trigger", status_code=status.HTTP_202_ACCEPTED)
async def trigger_pipeline_job(
    body: TriggerRequest,
    x_pipeline_token: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
):
    user_id = await authenticate_pipeline_request(x_pipeline_token, authorization)
    await verify_scan_access(body.scan_id, user_id)

    await _set_job(
        body.scan_id,
        scan_id=body.scan_id,
        room_id=body.room_id,
        user_id=user_id,
        video_id=body.video_id,
        room_name=body.room_name,
    )

    raise HTTPException(
        status_code=410,
        detail="/trigger is not available in standalone mode. Use /process-video or /upload endpoints.",
    )


@router.get("/inventory/{scan_id}")
async def get_inventory_json(
    scan_id: str,
    x_pipeline_token: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """Return final inventory JSON from pipeline local result files."""
    await authenticate_pipeline_request(x_pipeline_token, authorization)
    scan_id = _validate_job_id(scan_id)

    result_file = _upload_dir(scan_id) / "result.json"
    if not result_file.is_file():
        raise HTTPException(status_code=404, detail="Scan result not found")

    payload = json.loads(result_file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise HTTPException(status_code=500, detail="Invalid result payload")
    payload = _ensure_payload_report_image_urls(scan_id, payload)
    return payload
