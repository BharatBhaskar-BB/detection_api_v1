"""Pipeline orchestrator — v2 (LLM-first, no GDINO/SAM3).

4-step pipeline:
  1. Video processing — extract frames + transcribe audio
  2. Frame selection — quality filter + CLIP diversity
  3. AI analysis — LLM inventory draft (Gemini 2.0 Flash)
  4. Report generation — volume/weight lookup + evidence frames

Supports two modes:
  - all_at_once: single pipeline run over all videos
  - room_by_room: per-room pipeline runs (parallel), final merge when all done
"""

import asyncio
import base64
import json
import time
from datetime import datetime, timezone

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import get_settings
from app.database import async_session
from app.models.models import InventoryItem, PipelineCost, Room, Scan, ScanResult, Video
from app.models.schemas import WSComplete, WSError, WSItemAdded, WSProgress
from app.utils.video import extract_frames, get_video_info

settings = get_settings()

# Gemini 2.0 Flash pricing
LLM_COST_PER_1M = {"input": 0.10, "output": 0.40}

# Step definitions for progress tracking
STEPS = [
    ("video_processing", "Processing Video", 0.0, 0.20),
    ("frame_selection", "Selecting Frames", 0.20, 0.30),
    ("ai_analysis", "AI Analysis", 0.30, 0.85),
    ("report", "Generating Report", 0.85, 1.0),
]


def _compute_llm_cost(tokens_in: int, tokens_out: int) -> float:
    return (tokens_in / 1_000_000) * LLM_COST_PER_1M["input"] + \
           (tokens_out / 1_000_000) * LLM_COST_PER_1M["output"]


async def _broadcast(scan_id: str, data: dict) -> None:
    from pipeline_service.websocket import manager
    await manager.broadcast(scan_id, data)


async def _update_scan_progress(
    db: AsyncSession, scan_id: str, step: str, progress: float, message: str
) -> None:
    result = await db.execute(select(Scan).where(Scan.id == scan_id))
    scan = result.scalar_one_or_none()
    if scan:
        scan.status = "processing"
        scan.pipeline_step = step
        scan.progress = progress
        scan.progress_message = message
        await db.commit()


async def _run_pipeline(scan_id: str, room_name: str = "") -> None:
    """Execute the v2 pipeline for the whole scan or a single room.

    When room_name is set, only the video for that room is processed and all
    discovered items are tagged with that room name.
    """
    start_time = time.time()
    is_room = bool(room_name)

    async with async_session() as db:
        result = await db.execute(
            select(Scan).where(Scan.id == scan_id).options(
                selectinload(Scan.videos), selectinload(Scan.rooms)
            )
        )
        scan = result.scalar_one_or_none()
        if not scan:
            logger.error(f"Scan {scan_id} not found")
            return

        scan.status = "processing"

        # For room_by_room, mark this room as processing
        if is_room:
            for r in scan.rooms:
                if r.name == room_name:
                    r.status = "processing"
                    break
        await db.commit()

        try:
            # ════════════════════════════════════════════════════════
            # STEP 1: Video Processing — extract frames + transcribe
            # ════════════════════════════════════════════════════════
            step_name, step_label, p_start, p_end = STEPS[0]
            step_msg = f"Processing video{f' — {room_name}' if is_room else ''}…"
            await _update_scan_progress(db, scan_id, step_name, p_start, step_msg)
            await _broadcast(scan_id, WSProgress(
                scan_id=scan_id, room_name=room_name, step=step_name, step_number=1,
                progress=p_start, message=f"Extracting frames & transcribing audio{f' — {room_name}' if is_room else ''}…",
            ).model_dump())

            # Get video paths — filter by room if needed
            from app.services.storage import get_storage
            storage = get_storage()
            video_paths: list[str] = []
            if is_room:
                for video in scan.videos:
                    if video.room_name == room_name:
                        video_path = await storage.get_video_path(video.storage_path)
                        video_paths.append(video_path)
            else:
                for video in scan.videos:
                    video_path = await storage.get_video_path(video.storage_path)
                    video_paths.append(video_path)

            if not video_paths:
                raise ValueError(f"No video files found{f' for room {room_name}' if is_room else ''}")

            # Extract frames
            all_frames = []
            total_duration_s = 0.0
            for vpath in video_paths:
                frames = extract_frames(vpath, stride=settings.VIDEO_STRIDE)
                all_frames.extend(frames)
                info = get_video_info(vpath)
                total_duration_s += info.get("duration_s", 0)

            frame_count = len(all_frames)

            # Transcribe audio
            from app.pipeline.transcriber import Transcriber
            transcriber = Transcriber()
            transcript = transcriber.transcribe(video_paths[0])  # primary video

            has_audio = transcript.has_speech
            mode_label = "voice-assisted" if has_audio else "visual-only"

            await _update_scan_progress(
                db, scan_id, step_name, p_end,
                f"Extracted {frame_count} frames · {mode_label}",
            )
            await _broadcast(scan_id, WSProgress(
                scan_id=scan_id, room_name=room_name, step=step_name, step_number=1,
                progress=p_end,
                message=f"Extracted {frame_count} frames · {mode_label}",
            ).model_dump())

            # ════════════════════════════════════════════════════════
            # STEP 2: Smart Frame Selection
            # ════════════════════════════════════════════════════════
            step_name, step_label, p_start, p_end = STEPS[1]
            await _update_scan_progress(db, scan_id, step_name, p_start, "Selecting best frames…")
            await _broadcast(scan_id, WSProgress(
                scan_id=scan_id, room_name=room_name, step=step_name, step_number=2,
                progress=p_start, message="Selecting best quality frames…",
            ).model_dump())

            from app.pipeline.frame_selector import FrameSelector
            selector = FrameSelector()
            fps = get_video_info(video_paths[0]).get("fps", 30)
            selected_frames = await asyncio.to_thread(
                selector.select, all_frames,
                fps=fps,
                stride=settings.VIDEO_STRIDE,
                duration_s=total_duration_s,
            )

            await _update_scan_progress(
                db, scan_id, step_name, p_end,
                f"Selected {len(selected_frames)} frames",
            )
            await _broadcast(scan_id, WSProgress(
                scan_id=scan_id, room_name=room_name, step=step_name, step_number=2,
                progress=p_end,
                message=f"Selected {len(selected_frames)} key frames",
            ).model_dump())

            # ════════════════════════════════════════════════════════
            # STEP 3: LLM Inventory Draft
            # ════════════════════════════════════════════════════════
            step_name, step_label, p_start, p_end = STEPS[2]
            await _update_scan_progress(db, scan_id, step_name, p_start, "AI analyzing inventory…")
            await _broadcast(scan_id, WSProgress(
                scan_id=scan_id, room_name=room_name, step=step_name, step_number=3,
                progress=p_start, message=f"AI analyzing items{f' in {room_name}' if is_room else ''}…",
            ).model_dump())

            from app.pipeline.llm_inventory import LLMInventoryDrafter
            drafter = LLMInventoryDrafter()
            inventory = await drafter.draft(
                selected_frames, transcript=transcript, duration_s=total_duration_s
            )

            # Override room names if processing per-room
            if is_room:
                for item in inventory.items:
                    item.room = room_name

            # Save LLM cost
            tokens_in = inventory.usage.get("tokens_in", 0)
            tokens_out = inventory.usage.get("tokens_out", 0)
            llm_cost = _compute_llm_cost(tokens_in, tokens_out)
            db.add(PipelineCost(
                scan_id=scan_id, step=f"ai_analysis{'_' + room_name if is_room else ''}", provider="gemini",
                model=inventory.usage.get("model", "gemini-2.0-flash"),
                tokens_in=tokens_in, tokens_out=tokens_out,
                duration_s=time.time() - start_time, cost_usd=llm_cost,
                extra_data={
                    "frames_analyzed": len(selected_frames),
                    "has_audio": has_audio,
                    "items_found": len(inventory.items),
                    "room_name": room_name,
                },
            ))
            await db.flush()

            # Broadcast items as they're found
            items_found = []
            for item in inventory.items:
                items_found.append(item.name)
                await _broadcast(scan_id, WSItemAdded(
                    scan_id=scan_id, item_name=item.name,
                    room_name=item.room,
                ).model_dump())

            await _update_scan_progress(
                db, scan_id, step_name, p_end,
                f"Found {len(inventory.items)} item types",
            )
            await _broadcast(scan_id, WSProgress(
                scan_id=scan_id, room_name=room_name, step=step_name, step_number=3,
                progress=p_end,
                message=f"Found {len(inventory.items)} item types{f' in {room_name}' if is_room else ''}",
                items_found=items_found,
            ).model_dump())

            # Generate move summary only for all_at_once or when this is the final merge
            move_summary = None
            if not is_room:
                move_summary = await drafter.generate_move_summary(
                    inventory, transcript=transcript
                )

            # ════════════════════════════════════════════════════════
            # STEP 4: Report Generation
            # ════════════════════════════════════════════════════════
            step_name, step_label, p_start, p_end = STEPS[3]
            await _update_scan_progress(db, scan_id, step_name, p_start, "Generating report…")
            await _broadcast(scan_id, WSProgress(
                scan_id=scan_id, room_name=room_name, step=step_name, step_number=4,
                progress=p_start, message=f"Generating report{f' — {room_name}' if is_room else ''}…",
            ).model_dump())

            from app.pipeline.report_generator import generate_report
            report = await generate_report(
                inventory, selected_frames, total_duration_s, has_audio=has_audio
            )

            # Save report JSON to disk
            report_dir = settings.UPLOAD_DIR / "reports" / scan_id
            report_dir.mkdir(parents=True, exist_ok=True)
            if is_room:
                safe_room = "".join(c if c.isalnum() or c in "-_ " else "_" for c in room_name).strip()
                report_path = report_dir / f"report_{safe_room}.json"
            else:
                report_path = report_dir / "report.json"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)

            # ════════════════════════════════════════════════════════
            # Save inventory items to database
            # ════════════════════════════════════════════════════════
            crops_dir = settings.UPLOAD_DIR / "crops" / scan_id
            crops_dir.mkdir(parents=True, exist_ok=True)

            for item in inventory.items:
                # Find evidence image for this item
                evidence_b64 = None
                for r_item in report["items"]:
                    if r_item["name"] == item.name:
                        evidence_b64 = r_item.get("evidence_image")
                        break

                # Save evidence crop to disk
                item_image_path = None
                if evidence_b64:
                    try:
                        img_data = evidence_b64
                        if "," in img_data:
                            img_data = img_data.split(",", 1)[1]
                        raw = base64.b64decode(img_data)
                        safe_name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in item.name).strip()
                        crop_filename = f"{safe_name}.jpg"
                        crop_path = crops_dir / crop_filename
                        crop_path.write_bytes(raw)
                        item_image_path = f"crops/{scan_id}/{crop_filename}"
                    except Exception as e:
                        logger.warning(f"Failed to save crop for {item.name}: {e}")

                db.add(InventoryItem(
                    scan_id=scan_id,
                    name=item.name,
                    count=item.count,
                    room_name=item.room or room_name or "",
                    source="llm_draft",
                    is_new=False,
                    confidence=item.confidence,
                    disposition=item.disposition,
                    image_path=item_image_path,
                ))

            elapsed = time.time() - start_time
            total_cost = llm_cost

            if is_room:
                # ── Per-room completion ──
                # Mark this room as completed
                for r in scan.rooms:
                    if r.name == room_name:
                        r.status = "completed"
                        break

                await db.commit()

                # Broadcast room completion
                await _broadcast(scan_id, WSComplete(
                    scan_id=scan_id, room_name=room_name,
                    total_items=len(inventory.items),
                    total_rooms=1,
                    processing_time_s=elapsed,
                    cost_usd=total_cost,
                ).model_dump())

                logger.info(
                    f"Room pipeline complete: scan={scan_id}, room={room_name}, "
                    f"items={len(inventory.items)}, time={elapsed:.1f}s"
                )

                # Check if all rooms are done
                await _check_all_rooms_complete(scan_id)

            else:
                # ── All-at-once completion ──
                scan.status = "completed"
                scan.progress = 1.0
                scan.progress_message = "Scan complete!"
                scan.pipeline_step = "complete"
                scan.total_items = len(inventory.items)
                scan.processing_time_s = elapsed
                scan.cost_usd = total_cost
                scan.completed_at = datetime.now(timezone.utc)
                scan.move_summary = move_summary
                await db.commit()

                await _broadcast(scan_id, WSComplete(
                    scan_id=scan_id,
                    total_items=len(inventory.items),
                    total_rooms=scan.total_rooms,
                    processing_time_s=elapsed,
                    cost_usd=total_cost,
                ).model_dump())

                from app.services.push import send_push_to_scan_owner
                await send_push_to_scan_owner(
                    db, scan_id,
                    title="Scan Complete!",
                    body=f"Found {len(inventory.items)} items in {elapsed:.1f}s",
                    url=f"/scans/{scan_id}/inventory",
                )

                logger.info(
                    f"Pipeline complete: scan={scan_id}, items={len(inventory.items)}, "
                    f"time={elapsed:.1f}s, cost=${total_cost:.4f}"
                )

        except Exception as e:
            logger.exception(f"Pipeline failed: scan={scan_id}, room={room_name}")

            if is_room:
                for r in scan.rooms:
                    if r.name == room_name:
                        r.status = "failed"
                        break
                await db.commit()

                await _broadcast(scan_id, WSError(
                    scan_id=scan_id, room_name=room_name,
                    message=f"{room_name} failed: {str(e)[:200]}",
                ).model_dump())
            else:
                scan.status = "failed"
                scan.progress_message = str(e)[:200]
                await db.commit()

                await _broadcast(scan_id, WSError(
                    scan_id=scan_id, message=f"Processing failed: {str(e)[:200]}",
                ).model_dump())

                from app.services.push import send_push_to_scan_owner
                await send_push_to_scan_owner(
                    db, scan_id,
                    title="Scan Failed",
                    body=f"Error: {str(e)[:100]}",
                    url=f"/scans/{scan_id}",
                )


async def _check_all_rooms_complete(scan_id: str) -> None:
    """After each room finishes, check if all rooms are done. If so, finalize scan."""
    async with async_session() as db:
        result = await db.execute(
            select(Scan).where(Scan.id == scan_id).options(
                selectinload(Scan.rooms),
                selectinload(Scan.inventory_items),
                selectinload(Scan.pipeline_costs),
            )
        )
        scan = result.scalar_one_or_none()
        if not scan:
            return

        all_rooms = [r for r in scan.rooms]
        completed = [r for r in all_rooms if r.status == "completed"]
        failed = [r for r in all_rooms if r.status == "failed"]

        if len(completed) + len(failed) < len(all_rooms):
            # Still waiting for some rooms
            return

        # All rooms done — merge reports and finalize
        logger.info(f"All {len(all_rooms)} rooms done for scan={scan_id} "
                     f"({len(completed)} ok, {len(failed)} failed)")

        # Merge per-room report JSONs into one combined report
        report_dir = settings.UPLOAD_DIR / "reports" / scan_id
        combined_items = []
        combined_rooms: dict[str, list[int]] = {}
        combined_meta = {}

        for room in sorted(all_rooms, key=lambda r: r.order):
            safe_room = "".join(c if c.isalnum() or c in "-_ " else "_" for c in room.name).strip()
            room_report_path = report_dir / f"report_{safe_room}.json"
            if not room_report_path.exists():
                continue
            with open(room_report_path, encoding="utf-8") as f:
                room_report = json.load(f)

            base_idx = len(combined_items)
            room_item_indices = []
            for i, item in enumerate(room_report.get("items", [])):
                combined_items.append(item)
                room_item_indices.append(base_idx + i)
            combined_rooms[room.name] = room_item_indices
            if not combined_meta:
                combined_meta = {k: v for k, v in room_report.items() if k not in ("items", "rooms")}

        combined_report = {
            **combined_meta,
            "items": combined_items,
            "rooms": combined_rooms,
        }
        with open(report_dir / "report.json", "w", encoding="utf-8") as f:
            json.dump(combined_report, f, indent=2)

        # Generate move summary across all items
        try:
            from app.pipeline.llm_inventory import LLMInventoryDrafter
            drafter = LLMInventoryDrafter()
            # Build a minimal inventory for move summary
            from app.pipeline.llm_inventory import DraftInventory
            from app.pipeline.llm_inventory import InventoryItem as LLMInventoryItem
            all_inv_items = []
            for inv_item in scan.inventory_items:
                all_inv_items.append(LLMInventoryItem(
                    name=inv_item.name, count=inv_item.count,
                    room=inv_item.room_name, confidence=inv_item.confidence or 0.9,
                    disposition=inv_item.disposition,
                ))
            combined_inv = DraftInventory(items=all_inv_items, usage={})
            move_summary = await drafter.generate_move_summary(combined_inv)
        except Exception as e:
            logger.warning(f"Move summary generation failed: {e}")
            move_summary = None

        # Compute totals
        total_items = len(scan.inventory_items)
        total_cost = sum(pc.cost_usd for pc in scan.pipeline_costs)

        scan.status = "completed"
        scan.progress = 1.0
        scan.progress_message = "All rooms complete!"
        scan.pipeline_step = "complete"
        scan.total_items = total_items
        scan.total_rooms = len(completed)
        scan.cost_usd = total_cost
        scan.completed_at = datetime.now(timezone.utc)
        scan.move_summary = move_summary
        await db.commit()

        # Broadcast overall completion
        await _broadcast(scan_id, WSComplete(
            scan_id=scan_id,
            total_items=total_items,
            total_rooms=len(completed),
            processing_time_s=0,  # individual rooms have their own times
            cost_usd=total_cost,
        ).model_dump())

        from app.services.push import send_push_to_scan_owner
        await send_push_to_scan_owner(
            db, scan_id,
            title="All Rooms Complete!",
            body=f"Found {total_items} items across {len(completed)} rooms",
            url=f"/scans/{scan_id}/inventory",
        )


async def trigger_pipeline(scan_id: str) -> None:
    """Launch the pipeline as a background task (all_at_once mode)."""
    asyncio.create_task(_run_pipeline(scan_id))
    logger.info(f"Pipeline triggered for scan={scan_id}")


async def trigger_room_pipeline(scan_id: str, room_name: str) -> None:
    """Launch the pipeline for a single room as a background task."""
    asyncio.create_task(_run_pipeline(scan_id, room_name=room_name))
    logger.info(f"Room pipeline triggered: scan={scan_id}, room={room_name}")
