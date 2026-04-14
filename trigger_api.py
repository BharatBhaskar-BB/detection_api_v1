import requests
import json
import os
import sys
from pathlib import Path

from datetime import datetime
start_time = datetime.now()
base_url = "http://34.106.160.70:8100/api/detection/v1"
scan_id = "scan_multi_3"
job_id = scan_id
room_id = "room_1"
video_id = "video_1"
room_name = "Kitchen"
user_id = "user_1"


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


jwt_token = os.getenv("JWT_TOKEN", "") or os.getenv("BB_TOKEN", "")
pipeline_token = os.getenv("PIPELINE_TOKEN", "") or _load_env_value("PIPELINE_SHARED_TOKEN")

if not jwt_token and not pipeline_token:
    print("Set JWT_TOKEN (or BB_TOKEN) or PIPELINE_TOKEN before running this script.")
    print("Or set PIPELINE_SHARED_TOKEN in deploy/.env.prod")
    sys.exit(1)

headers = {}
if jwt_token:
    headers["Authorization"] = f"Bearer {jwt_token}"
if pipeline_token:
    headers["X-Pipeline-Token"] = pipeline_token

print("Step 1/1: process-video (upload + queue)")
t1 = datetime.now()
with open("video1.mp4", "rb") as f:
    resp = requests.post(
        f"{base_url}/process-video",
        files={"file": ("video1.mp4", f, "video/mp4")},
        data={
            "job_id": job_id,
            "scan_id": scan_id,
            "room_id": room_id,
            "video_id": video_id,
            "room_name": room_name,
            "request_user_id": user_id,
        },
        headers=headers,
        timeout=300,
    )
print("process-video status:", resp.status_code, "time:", str(datetime.now() - t1))

print(resp.status_code)
print(json.dumps(resp.json(), indent=2))


print(str(datetime.now() - start_time))