import asyncio
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlencode

import websockets

BASE_WS_URL = "ws://34.106.160.70:8100/api/detection/v1/ws/scan_multi_7"  # replace 1 with your job_id/scan_id


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

params = {}
if jwt_token:
    params["token"] = jwt_token
if pipeline_token:
    params["pipeline_token"] = pipeline_token
WS_URL = f"{BASE_WS_URL}?{urlencode(params)}" if params else BASE_WS_URL

async def main():
    async with websockets.connect(WS_URL) as ws:
        print("connected")

        # keepalive test # add gs uri
        await ws.send("ping")
        print("sent ping")

        # receive messages
        while True:
            msg = await ws.recv()
            try:
                print("recv:", json.loads(msg))
            except Exception:
                print("recv:", msg)

asyncio.run(main())


