#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════
# run_remote_pipeline.sh — Upload video to GCP, run pipeline,
# download HTML report locally.
#
# Usage:
#   ./scripts/run_remote_pipeline.sh /path/to/local/video.mp4
#   ./scripts/run_remote_pipeline.sh /path/to/video.mp4 ./my_report.html
#
# Prerequisites:
#   - SSH key at ~/.ssh/google_compute_engine
#   - GCP VM running at the IP below with Docker stack up
# ═══════════════════════════════════════════════════════════

set -euo pipefail

GCP_HOST="34.106.90.78"
GCP_USER="bharat_bhaskar"
SSH_KEY="$HOME/.ssh/google_compute_engine"
CONTAINER="bb-pipeline_backend-1"
REMOTE_DIR="/home/${GCP_USER}/pipeline_videos"

VIDEO_LOCAL="$1"
OUTPUT_LOCAL="${2:-$(basename "${VIDEO_LOCAL%.*}")_report.html}"

if [[ ! -f "$VIDEO_LOCAL" ]]; then
  echo "ERROR: Video file not found: $VIDEO_LOCAL"
  exit 1
fi

VIDEO_NAME=$(basename "$VIDEO_LOCAL")
REMOTE_VIDEO="${REMOTE_DIR}/${VIDEO_NAME}"
# The container mounts /home/bharat_bhaskar/results:/app/results
CONTAINER_INPUT="/app/results/${VIDEO_NAME}"
CONTAINER_OUTPUT="/app/results/${VIDEO_NAME%.*}.html"
CONTAINER_JSON="/app/results/${VIDEO_NAME%.*}.json"
REMOTE_OUTPUT="/home/${GCP_USER}/results/${VIDEO_NAME%.*}.html"
REMOTE_JSON="/home/${GCP_USER}/results/${VIDEO_NAME%.*}.json"

SSH_CMD="ssh -i $SSH_KEY -o StrictHostKeyChecking=no ${GCP_USER}@${GCP_HOST}"

echo "═══ BundleBox Remote Pipeline ═══"
echo "Video:  $VIDEO_LOCAL"
echo "Output: $OUTPUT_LOCAL"
echo

# 1. Upload video
echo "▸ Uploading video to GCP…"
$SSH_CMD "mkdir -p $REMOTE_DIR /home/${GCP_USER}/results"
scp -i "$SSH_KEY" -o StrictHostKeyChecking=no "$VIDEO_LOCAL" "${GCP_USER}@${GCP_HOST}:${REMOTE_DIR}/"
# Copy into results dir (mounted in container)
$SSH_CMD "cp '${REMOTE_VIDEO}' '/home/${GCP_USER}/results/'"
echo "  Uploaded."

# 2. Run pipeline in Docker container
echo "▸ Running pipeline in Docker container (this may take a few minutes)…"
$SSH_CMD "docker exec $CONTAINER python /srv/scripts/process_video.py '$CONTAINER_INPUT' -o '$CONTAINER_OUTPUT'"
echo "  Pipeline complete."

# 3. Download results
echo "▸ Downloading HTML report…"
scp -i "$SSH_KEY" -o StrictHostKeyChecking=no "${GCP_USER}@${GCP_HOST}:${REMOTE_OUTPUT}" "$OUTPUT_LOCAL"
echo "  → $OUTPUT_LOCAL"

JSON_LOCAL="${OUTPUT_LOCAL%.*}.json"
echo "▸ Downloading JSON report…"
scp -i "$SSH_KEY" -o StrictHostKeyChecking=no "${GCP_USER}@${GCP_HOST}:${REMOTE_JSON}" "$JSON_LOCAL" 2>/dev/null || echo "  (JSON not available)"
echo "  → $JSON_LOCAL"

# 4. Cleanup remote video copy
$SSH_CMD "rm -f '/home/${GCP_USER}/results/${VIDEO_NAME}'" 2>/dev/null || true

echo
echo "═══ DONE ═══"
echo "Open the report: open '$OUTPUT_LOCAL'"
