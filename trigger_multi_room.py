import json
import os
import sys
from pathlib import Path

import requests


BASE_URL = "http://136.110.86.112:8100/pipeline/v1"
SCAN_ID = "scan_multi_1"
JOB_ID = SCAN_ID  # same socket/job stream for all rooms in this scan
REQUEST_USER_ID = "user_1"

ROOM_VIDEOS = [
    {"file": "video1.mp4", "room_id": "room_1", "room_name": "Kitchen", "video_id": "video_1"},
    {"file": "video2.mp4", "room_id": "room_2", "room_name": "Bedroom", "video_id": "video_2"},
]


def _load_env_value(key: str) -> str:
    env_path = Path(__file__).resolve().parent / "deploy" / ".env.prod"
    if not env_path.is_file():
        return ""
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            if k.strip() == key:
                return v.strip().strip('"').strip("'")
    except Exception:
        return ""
    return ""


def _headers() -> dict[str, str]:
    jwt_token = os.getenv("JWT_TOKEN", "") or os.getenv("BB_TOKEN", "")
    pipeline_token = os.getenv("PIPELINE_TOKEN", "") or _load_env_value("PIPELINE_SHARED_TOKEN")

    if not jwt_token and not pipeline_token:
        print("Set JWT_TOKEN (or BB_TOKEN) or PIPELINE_TOKEN before running this script.")
        print("Or set PIPELINE_SHARED_TOKEN in deploy/.env.prod")
        sys.exit(1)

    h: dict[str, str] = {}
    if jwt_token:
        h["Authorization"] = f"Bearer {jwt_token}"
    if pipeline_token:
        h["X-Pipeline-Token"] = pipeline_token
    return h


def main() -> None:
    headers = _headers()

    print("All room events will stream on:")
    print(f"  {BASE_URL}/ws/{SCAN_ID}")

    for idx, room in enumerate(ROOM_VIDEOS, start=1):
        file_path = Path(room["file"])
        if not file_path.is_file():
            print(f"Missing file: {file_path}")
            sys.exit(1)

        print(f"\nRoom {idx}/{len(ROOM_VIDEOS)} -> {room['room_name']} ({file_path.name})")

        with open(file_path, "rb") as f:
            process_resp = requests.post(
                f"{BASE_URL}/process-video",
                files={"file": (file_path.name, f, "video/mp4")},
                data={
                    "job_id": JOB_ID,
                    "scan_id": SCAN_ID,
                    "room_id": room["room_id"],
                    "video_id": room["video_id"],
                    "room_name": room["room_name"],
                    "request_user_id": REQUEST_USER_ID,
                },
                headers=headers,
                timeout=600,
            )

        print("process-video:", process_resp.status_code)
        if process_resp.status_code not in (200, 202):
            print(process_resp.text)
            sys.exit(1)
        try:
            print(json.dumps(process_resp.json(), indent=2))
        except Exception:
            print(process_resp.text)

    print("\nDone. Both room videos were queued with the same scan_id/job_id (no completion wait).")


if __name__ == "__main__":
    main()
