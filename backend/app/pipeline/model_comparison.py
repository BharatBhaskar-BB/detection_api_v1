"""Background model comparison — run alternative Gemini models and log results to CSV."""

import asyncio
import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from app.config import get_settings
from app.pipeline.frame_selector import SelectedFrame
from app.pipeline.llm_inventory import LLMInventoryDrafter
from app.pipeline.transcriber import TranscriptionResult

# Gemini pricing per 1M tokens
MODEL_PRICING = {
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
    "gemini-2.5-flash-lite": {"input": 0.075, "output": 0.30},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
}

CSV_HEADERS = [
    "timestamp", "scan_id", "model", "items_found", "total_pieces",
    "tokens_in", "tokens_out", "cost_usd", "duration_s",
    "item_names", "items_json",
]


def _compute_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    pricing = MODEL_PRICING.get(model, {"input": 0.15, "output": 0.60})
    return (tokens_in / 1_000_000) * pricing["input"] + \
           (tokens_out / 1_000_000) * pricing["output"]


def _get_csv_path() -> Path:
    settings = get_settings()
    results_dir = Path(settings.RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir / "model_comparison.csv"


def _append_csv_row(row: dict) -> None:
    csv_path = _get_csv_path()
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


async def run_comparison_models(
    scan_id: str,
    selected_frames: list[SelectedFrame],
    transcript: TranscriptionResult | None,
    duration_s: float,
    primary_inventory: dict | None = None,
) -> None:
    """Run comparison models in background and log results to CSV.

    Also logs the primary model result for easy side-by-side comparison.
    """
    settings = get_settings()
    comparison_models_str = settings.GEMINI_COMPARISON_MODELS
    if not comparison_models_str:
        return

    comparison_models = [m.strip() for m in comparison_models_str.split(",") if m.strip()]
    if not comparison_models:
        return

    # Log the primary model result first
    if primary_inventory:
        _append_csv_row({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "scan_id": scan_id,
            "model": primary_inventory.get("model", settings.GEMINI_MODEL),
            "items_found": primary_inventory.get("items_found", 0),
            "total_pieces": primary_inventory.get("total_pieces", 0),
            "tokens_in": primary_inventory.get("tokens_in", 0),
            "tokens_out": primary_inventory.get("tokens_out", 0),
            "cost_usd": f"{primary_inventory.get('cost_usd', 0):.6f}",
            "duration_s": f"{primary_inventory.get('duration_s', 0):.1f}",
            "item_names": primary_inventory.get("item_names", ""),
            "items_json": primary_inventory.get("items_json", "[]"),
        })

    drafter = LLMInventoryDrafter()

    for model_name in comparison_models:
        try:
            logger.info(f"[Comparison] Running {model_name} for scan {scan_id}...")
            start = time.time()

            inventory = await drafter.draft_with_model(
                model_name=model_name,
                selected_frames=selected_frames,
                transcript=transcript,
                duration_s=duration_s,
            )

            elapsed = time.time() - start
            tokens_in = inventory.usage.get("tokens_in", 0)
            tokens_out = inventory.usage.get("tokens_out", 0)
            cost = _compute_cost(model_name, tokens_in, tokens_out)
            total_pieces = sum(item.count for item in inventory.items)
            item_names = ", ".join(sorted(set(item.name for item in inventory.items)))
            items_json = json.dumps([
                {"name": item.name, "count": item.count, "room": item.room,
                 "disposition": item.disposition, "size": item.size, "notes": item.notes}
                for item in inventory.items
            ])

            _append_csv_row({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "scan_id": scan_id,
                "model": model_name,
                "items_found": len(inventory.items),
                "total_pieces": total_pieces,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "cost_usd": f"{cost:.6f}",
                "duration_s": f"{elapsed:.1f}",
                "item_names": item_names,
                "items_json": items_json,
            })

            logger.info(
                f"[Comparison] {model_name}: {len(inventory.items)} items, "
                f"{total_pieces} pieces, ${cost:.4f}, {elapsed:.1f}s"
            )

        except Exception as e:
            logger.error(f"[Comparison] {model_name} failed for scan {scan_id}: {e}")
            _append_csv_row({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "scan_id": scan_id,
                "model": model_name,
                "items_found": 0,
                "total_pieces": 0,
                "tokens_in": 0,
                "tokens_out": 0,
                "cost_usd": "0",
                "duration_s": "0",
                "item_names": f"ERROR: {str(e)[:200]}",
                "items_json": "[]",
            })
