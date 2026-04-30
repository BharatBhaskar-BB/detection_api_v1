#!/usr/bin/env python3
"""Standalone CLI script to process a video through the BundleBox pipeline
and generate a self-contained HTML report with embedded evidence images.

Usage (inside the pipeline Docker container on GCP):
    python /srv/scripts/process_video.py /path/to/video.mp4

Or from the host:
    docker exec bb-pipeline_backend-1 python /srv/scripts/process_video.py /app/data/video.mp4

Output: HTML report written next to the video (or to --output path).
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

# ── Ensure PYTHONPATH covers the backend package ──
_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parent
sys.path.insert(0, str(_repo_root / "backend"))
sys.path.insert(0, str(_repo_root))

from app.config import get_settings
from app.utils.video import extract_frames, get_video_info
from app.pipeline.transcriber import Transcriber
from app.pipeline.frame_selector import FrameSelector
from app.pipeline.llm_inventory import LLMInventoryDrafter
from app.pipeline.report_generator import generate_report


# ═══════════════════════════════════════════════════════════
# HTML report renderer
# ═══════════════════════════════════════════════════════════

def _size_badge(size: str) -> str:
    cls = {
        "small": "size-small",
        "medium": "size-medium",
        "large": "size-large",
        "extra-large": "size-extra-large",
    }.get((size or "").lower(), "size-medium")
    return f'<span class="size-badge {cls}">{size or "—"}</span>'


def render_html_report(report: dict, video_name: str) -> str:
    """Render pipeline report dict into a self-contained HTML page."""
    summary = report.get("summary", {})
    items = report.get("items", [])
    rooms = report.get("rooms", {})

    generated_at = report.get("generated_at", "")
    duration = report.get("video_duration_s", 0)
    n_frames = report.get("num_frames_analyzed", 0)
    survey_mode = report.get("survey_mode", "visual_only")

    total_items = summary.get("total_items", 0)
    total_types = summary.get("total_item_types", 0)
    total_vol = summary.get("total_volume_cuft", 0)
    total_wt = summary.get("total_weight_lbs", 0)
    truck = summary.get("truck_recommendation", "—")
    n_rooms = len(rooms)

    # ── CSS ──
    css = """
    @page { margin: 0.5in; }
    @media print {
      .page-break { page-break-before: always; }
      .no-break { page-break-inside: avoid; }
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
           color: #1a1a1a; line-height: 1.5; padding: 20px; background: #fff; }
    .header { text-align: center; padding: 30px 0; border-bottom: 3px solid #2563eb; margin-bottom: 30px; }
    .header h1 { font-size: 28px; color: #1e3a5f; margin-bottom: 5px; }
    .header .subtitle { color: #6b7280; font-size: 14px; }
    .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin-bottom: 30px; }
    .summary-card { background: #f0f7ff; border: 1px solid #bfdbfe; border-radius: 8px; padding: 15px; text-align: center; }
    .summary-card .value { font-size: 24px; font-weight: 700; color: #1e40af; }
    .summary-card .label { font-size: 12px; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px; }
    .room-section { margin-bottom: 30px; }
    .room-header { background: #1e3a5f; color: white; padding: 10px 20px; border-radius: 6px 6px 0 0;
                   font-size: 18px; font-weight: 600; }
    .room-header .count { float: right; font-weight: 400; font-size: 14px; opacity: 0.8; }
    .items-table { width: 100%; border-collapse: collapse; margin-bottom: 0; }
    .items-table th { background: #e5edff; color: #1e3a5f; padding: 8px 12px; text-align: left;
                      font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; border: 1px solid #cbd5e1; }
    .items-table td { padding: 8px 12px; border: 1px solid #e2e8f0; vertical-align: top; font-size: 13px; }
    .items-table tr:nth-child(even) { background: #f8fafc; }
    .item-row { display: flex; gap: 20px; padding: 15px; border: 1px solid #e2e8f0; border-top: none; background: #fff; }
    .item-row:nth-child(even) { background: #f8fafc; }
    .item-image { width: 400px; min-width: 400px; height: 300px; object-fit: contain; border-radius: 6px;
                  border: 1px solid #e2e8f0; background: #f9f9f9; }
    .item-details { flex: 1; }
    .item-name { font-size: 18px; font-weight: 600; color: #1e3a5f; margin-bottom: 8px; text-transform: capitalize; }
    .detail-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px 20px; }
    .detail-item { display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px dotted #e2e8f0; }
    .detail-label { color: #6b7280; font-size: 12px; font-weight: 500; }
    .detail-value { font-weight: 600; font-size: 13px; }
    .totals-row { background: #1e3a5f !important; color: white; font-weight: 700; }
    .totals-row td { border-color: #1e3a5f; color: white; }
    .footer { text-align: center; margin-top: 40px; padding-top: 20px; border-top: 2px solid #e2e8f0;
              color: #9ca3af; font-size: 11px; }
    .size-badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; }
    .size-small { background: #dcfce7; color: #166534; }
    .size-medium { background: #fef9c3; color: #854d0e; }
    .size-large { background: #fee2e2; color: #991b1b; }
    .size-extra-large { background: #f3e8ff; color: #6b21a8; }
    """

    parts = []
    parts.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>BundleBox Inventory Report — {video_name}</title>
<style>{css}</style>
</head>
<body>

<div class="header">
  <h1>BundleBox Moving Inventory Report</h1>
  <div class="subtitle">Generated {generated_at} &bull; Video: {duration:.0f}s &bull;
    {n_frames} frames analyzed &bull; Mode: {survey_mode} &bull; File: {video_name}</div>
</div>

<div class="summary-grid">
  <div class="summary-card"><div class="value">{total_types}</div><div class="label">Item Types</div></div>
  <div class="summary-card"><div class="value">{total_items:,}</div><div class="label">Total Pieces</div></div>
  <div class="summary-card"><div class="value">{total_vol:,.1f}</div><div class="label">Total Cu Ft</div></div>
  <div class="summary-card"><div class="value">{total_wt:,.0f}</div><div class="label">Total Lbs</div></div>
  <div class="summary-card"><div class="value">{n_rooms}</div><div class="label">Rooms</div></div>
  <div class="summary-card"><div class="value">{truck}</div><div class="label">Truck Size</div></div>
</div>
""")

    # ── Summary table ──
    parts.append("""
<h2 style="margin: 30px 0 15px; color: #1e3a5f; font-size: 20px;">Summary Table</h2>
<table class="items-table">
<thead><tr>
  <th>#</th><th>Item</th><th>Room</th><th>Qty</th><th>Size</th>
  <th>Cu Ft (each)</th><th>Lbs (each)</th><th>Total Cu Ft</th><th>Total Lbs</th><th>Special Handling</th>
</tr></thead><tbody>
""")

    for i, item in enumerate(items, 1):
        special = ", ".join(item.get("special_handling", [])) or "—"
        parts.append(f"""<tr class="no-break">
  <td>{i}</td>
  <td style="font-weight:600;text-transform:capitalize">{item['name']}</td>
  <td>{item.get('room', '—')}</td>
  <td style="text-align:center">{item['count']}</td>
  <td>{_size_badge(item.get('size', ''))}</td>
  <td style="text-align:right">{item.get('volume_cuft', 0)}</td>
  <td style="text-align:right">{item.get('weight_lbs', 0)}</td>
  <td style="text-align:right;font-weight:600">{item.get('total_volume_cuft', 0)}</td>
  <td style="text-align:right;font-weight:600">{item.get('total_weight_lbs', 0)}</td>
  <td style="font-size:11px">{special}</td>
</tr>""")

    parts.append(f"""<tr class="totals-row">
  <td colspan="3">TOTALS</td><td style="text-align:center">{total_items}</td>
  <td></td><td></td><td></td>
  <td style="text-align:right">{total_vol:,.1f}</td>
  <td style="text-align:right">{total_wt:,.0f}</td><td></td>
</tr></tbody></table>
""")

    # ── Detailed inventory with evidence photos ──
    parts.append("""
<div class="page-break"></div>
<h2 style="margin: 30px 0 15px; color: #1e3a5f; font-size: 20px;">Detailed Inventory with Evidence Photos</h2>
""")

    for room_name, item_indices in rooms.items():
        room_items = [items[i] for i in item_indices]
        room_count = sum(it["count"] for it in room_items)
        room_vol = sum(it.get("total_volume_cuft", 0) for it in room_items)
        room_wt = sum(it.get("total_weight_lbs", 0) for it in room_items)

        parts.append(f"""
<div class="room-section">
  <div class="room-header">{room_name.title()}
    <span class="count">{len(room_items)} items &bull; {room_count} pcs &bull;
      {room_vol:,.1f} cu ft &bull; {room_wt:,.0f} lbs</span>
  </div>
""")

        for item in room_items:
            evidence = item.get("evidence_image")
            if evidence:
                img_tag = f'<img class="item-image" src="data:image/jpeg;base64,{evidence}" alt="{item["name"]}">'
            else:
                img_tag = '<div class="item-image" style="display:flex;align-items:center;justify-content:center;color:#999;">No image</div>'

            special_html = ", ".join(item.get("special_handling", [])) or "None"
            notes = item.get("notes", "") or ""
            dims = item.get("dimensions") or {}
            dim_str = f'{dims.get("length", "?")}×{dims.get("width", "?")}×{dims.get("height", "?")}' if dims else "—"

            parts.append(f"""
  <div class="item-row no-break">
    {img_tag}
    <div class="item-details">
      <div class="item-name">{item['name']}</div>
      <div class="detail-grid">
        <div class="detail-item"><span class="detail-label">Quantity</span><span class="detail-value">{item['count']}</span></div>
        <div class="detail-item"><span class="detail-label">Size</span><span class="detail-value">{_size_badge(item.get('size', ''))}</span></div>
        <div class="detail-item"><span class="detail-label">Dimensions</span><span class="detail-value">{dim_str}</span></div>
        <div class="detail-item"><span class="detail-label">Volume (each)</span><span class="detail-value">{item.get('volume_cuft', 0)} cu ft</span></div>
        <div class="detail-item"><span class="detail-label">Weight (each)</span><span class="detail-value">{item.get('weight_lbs', 0)} lbs</span></div>
        <div class="detail-item"><span class="detail-label">Total Volume</span><span class="detail-value">{item.get('total_volume_cuft', 0)} cu ft</span></div>
        <div class="detail-item"><span class="detail-label">Total Weight</span><span class="detail-value">{item.get('total_weight_lbs', 0)} lbs</span></div>
        <div class="detail-item"><span class="detail-label">Special Handling</span><span class="detail-value">{special_html}</span></div>
      </div>
      {"<div style='margin-top:8px;font-size:12px;color:#6b7280'><strong>Notes:</strong> " + notes + "</div>" if notes else ""}
    </div>
  </div>
""")

        parts.append("</div>")  # close room-section

    # ── Footer ──
    usage = report.get("usage", {})
    parts.append(f"""
<div class="footer">
  BundleBox Inventory Report &bull; CLI Pipeline &bull;
  Tokens: {usage.get('total_tokens', '?')} &bull;
  Cost: ${usage.get('total_cost', 0):.4f}
</div>
</body></html>""")

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════

async def run_pipeline(video_path: str, output_path: str | None = None) -> str:
    """Run the full inventory pipeline on a video and write an HTML report."""
    settings = get_settings()
    video = Path(video_path)
    if not video.exists():
        print(f"ERROR: Video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    video_name = video.name
    if output_path is None:
        output_path = str(video.with_suffix(".html"))

    print(f"═══ BundleBox CLI Pipeline ═══")
    print(f"Video:  {video_path}")
    print(f"Output: {output_path}")
    print(f"Stride: {settings.VIDEO_STRIDE}")
    print(f"Model:  {settings.GEMINI_MODEL}")
    print()

    # 1. Extract frames
    t0 = time.time()
    print("▸ Extracting frames…")
    info = get_video_info(video_path)
    fps = info.get("fps", 30) or 30
    duration_s = info.get("duration_s", 0) or 0
    all_frames = extract_frames(video_path, stride=settings.VIDEO_STRIDE)
    t1 = time.time()
    print(f"  {len(all_frames)} frames extracted ({t1-t0:.1f}s)")

    # 2. Transcribe audio
    print("▸ Transcribing audio…")
    transcriber = Transcriber()
    transcript = transcriber.transcribe(video_path)
    has_audio = transcript.has_speech
    t2 = time.time()
    print(f"  Audio: {'yes' if has_audio else 'no'} ({t2-t1:.1f}s)")

    # 3. Select key frames
    print("▸ Selecting key frames…")
    selector = FrameSelector()
    selected_frames = await asyncio.to_thread(
        selector.select,
        all_frames,
        fps=fps,
        stride=settings.VIDEO_STRIDE,
        duration_s=duration_s,
    )
    t3 = time.time()
    print(f"  {len(selected_frames)} frames selected ({t3-t2:.1f}s)")

    # 4. LLM inventory draft
    print("▸ Analyzing inventory with Gemini…")
    drafter = LLMInventoryDrafter()
    inventory = await drafter.draft(
        selected_frames,
        transcript=transcript,
        duration_s=duration_s,
    )
    t4 = time.time()
    print(f"  {len(inventory.items)} items found ({t4-t3:.1f}s)")

    # 5. Generate report (with evidence images + centroids)
    print("▸ Generating report…")
    report = await generate_report(
        inventory,
        selected_frames,
        duration_s,
        has_audio=has_audio,
    )
    t5 = time.time()
    print(f"  Report generated ({t5-t4:.1f}s)")

    # 6. Render HTML
    print("▸ Rendering HTML…")
    html = render_html_report(report, video_name)
    Path(output_path).write_text(html, encoding="utf-8")
    t6 = time.time()

    # 7. Also save raw JSON report
    json_path = str(Path(output_path).with_suffix(".json"))
    # Strip evidence images from JSON to keep it small
    report_no_images = {**report}
    report_no_images["items"] = [
        {k: v for k, v in item.items() if k != "evidence_image"}
        for item in report.get("items", [])
    ]
    Path(json_path).write_text(json.dumps(report_no_images, indent=2), encoding="utf-8")

    print()
    print(f"═══ DONE ═══")
    print(f"Total time: {t6-t0:.1f}s")
    print(f"HTML report: {output_path}")
    print(f"JSON report: {json_path}")
    print(f"Items: {len(inventory.items)}, Usage: {inventory.usage}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="BundleBox CLI — process video and generate HTML report")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("-o", "--output", help="Output HTML path (default: same dir as video, .html extension)")
    args = parser.parse_args()
    asyncio.run(run_pipeline(args.video, args.output))


if __name__ == "__main__":
    main()
