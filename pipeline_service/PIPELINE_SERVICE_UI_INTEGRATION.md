# Pipeline Service API – UI Integration Guide

## 1) Overview

This document explains how UI clients should integrate with the Pipeline Service for:
- Single full-video processing
- Multi-room processing (multiple videos for the same scan)
- Real-time progress over WebSocket
- JWT authentication
- RabbitMQ queue monitoring

Base API URL:
- `http://136.110.86.112:8100/pipeline/v1`

Health check:
- `GET /health`

---

## 2) Runtime Architecture (current)

- FastAPI worker count: **4** (Uvicorn `--workers 4`)
- API trigger/upload requests are accepted quickly (`202`) and processed in background.
- RabbitMQ is used for:
  - Job queueing (`pipeline_jobs` queue)
  - Cross-worker WebSocket event fanout (`pipeline_ws_events` exchange)
- WebSocket events are shared across workers, so UI receives updates even when request handling and processing happen on different workers.

---

## 3) Authentication (JWT)

Pipeline endpoints accept either:
1. `Authorization: Bearer <jwt_token>` (recommended for UI)
2. `X-Pipeline-Token: <shared_service_token>` (service-to-service/internal)

### How JWT works (short)
- UI signs up/logs in through backend auth APIs.
- Backend returns a JWT access token.
- UI includes `Authorization: Bearer <token>` in API requests.
- For WebSocket, token is passed in query string (`?token=...`).

### Signup example
```bash
curl -X POST "http://136.110.86.112:8000/api/v1/auth/signup" \
  -H "Content-Type: application/json" \
  -d '{"email":"your_email@example.com","password":"your_password","name":"your_name"}'
```

### Login example
```bash
curl -X POST "http://136.110.86.112:8000/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"email":"your_email@example.com","password":"your_password"}'
```

Use returned access token as Bearer token.

---

## 4) RabbitMQ Monitoring UI

RabbitMQ management URL:
- `http://136.110.86.112:15672/`

What UI/ops team can monitor:
- `pipeline_jobs` queue:
  - **Ready** = waiting jobs
  - **Unacked** = currently consumed/in-progress
  - **Total** = aggregate
- Publish/Deliver rates over time
- Consumer count and connection/channel health

Notes:
- Queue backlog growth indicates processing bottleneck.
- High unacked count for long duration may indicate slow workers.

---

## 5) Core API Endpoints for UI

### Preferred single-call endpoint

For most UI flows, use a single endpoint:

`POST /process-video`

This combines:
- create/init job
- upload file
- queue trigger

Required multipart fields:
- `file` (video)
- `job_id` (use `scan_id` value)
- `scan_id`

Optional metadata:
- `room_id`, `room_name`, `video_id`, `request_user_id`

Response: `202 Accepted` with `job_id`, `websocket_url`, and `result_url`.

The old 3-step flow (`/upload/init` → `/upload/{job_id}/file` → `/upload/{job_id}/trigger`) is still available for advanced control.

## 5.1 Create Job (init)
`POST /upload/init`

Form fields:
- `job_id` (string) – for UI, typically use scan id
- `scan_id` (string)
- `room_id` (optional)
- `video_id` (optional)
- `room_name` (optional)
- `request_user_id` (optional metadata)

Response: `202 Accepted`
```json
{
  "status": "accepted",
  "job_id": "scan_multi_1",
  "websocket_url": "/pipeline/v1/ws/scan_multi_1",
  "result_url": "/pipeline/v1/upload/scan_multi_1",
  "scan_id": "scan_multi_1",
  "room_id": "room_1",
  "user_id": "user_1",
  "video_id": "video_1",
  "room_name": "Kitchen",
  "upload_url": "/pipeline/v1/upload/scan_multi_1/file",
  "trigger_url": "/pipeline/v1/upload/scan_multi_1/trigger"
}
```

## 5.2 Upload Video File
`POST /upload/{job_id}/file`

Multipart:
- `file`: video file
- plus same metadata fields as needed (`scan_id`, `room_id`, `video_id`, `room_name`, `request_user_id`)

Response: `201 Created`
```json
{
  "status": "uploaded",
  "job_id": "scan_multi_1",
  "video_filename": "video1.mp4",
  "scan_id": "scan_multi_1",
  "room_id": "room_1",
  "user_id": "user_1",
  "video_id": "video_1",
  "room_name": "Kitchen"
}
```

## 5.3 Trigger Processing
`POST /upload/{job_id}/trigger`

Response: `202 Accepted`
```json
{
  "status": "accepted",
  "job_id": "scan_multi_1",
  "websocket_url": "/pipeline/v1/ws/scan_multi_1",
  "result_url": "/pipeline/v1/upload/scan_multi_1",
  "scan_id": "scan_multi_1",
  "room_id": "room_1",
  "user_id": "user_1",
  "video_id": "video_1",
  "room_name": "Kitchen"
}
```

## 5.4 Get Job State / Result
`GET /upload/{job_id}`

- During processing: returns state (`status`, `progress`, etc.)
- On completion: returns final result payload

---

## 6) WebSocket Integration

Endpoint:
- `ws://136.110.86.112:8100/pipeline/v1/ws/{job_id_or_scan_id}`

Auth query params:
- JWT mode: `?token=<jwt_token>`
- Service-token mode: `?pipeline_token=<shared_token>`

You can send:
- `ping` text message

You receive:
- `{"type":"pong"}`

Event types:
- `progress`
- `complete`
- `error`

### Example progress event
```json
{
  "type": "progress",
  "job_id": "scan_multi_1",
  "step": "frame_selection",
  "step_number": 3,
  "total_steps": 5,
  "progress": 0.35,
  "message": "Selecting key frames",
  "elapsed_s": 15.603,
  "scan_id": "scan_multi_1",
  "room_id": "room_1",
  "user_id": "user_1",
  "video_id": "video_1",
  "room_name": "Kitchen"
}
```

Step model:
1. `initiated`
2. `video_processing`
3. `frame_selection`
4. `ai_analysis`
5. `report`

---

## 7) Multiple Videos for Same Scan ID (Room-by-room)

Use this pattern:
1. Call `POST /upload/init` once with `job_id = scan_id`
2. For each room video:
   - `POST /upload/{scan_id}/file` with that room’s metadata
   - `POST /upload/{scan_id}/trigger`
3. Keep one WebSocket connection open on `/ws/{scan_id}`

Why this works:
- Same `scan_id` as `job_id` means one stream/channel for all room updates.
- Payload metadata (`room_id`, `room_name`, `video_id`) lets UI separate room events.

UI recommendation:
- Group incoming events by `room_id` or `room_name`
- Show per-room timeline + overall scan timeline

---

## 8) Frontend Request Pattern (recommended)

- Open WebSocket first (`/ws/{scan_id}`)
- Call `POST /process-video`
- Render progress from socket events
- Fallback poll `GET /upload/{scan_id}` if socket disconnects

---

## 9) Error Handling

Common API errors:
- `401`: missing/invalid auth token
- `403`: scan/job not owned by authenticated user
- `404`: job not found
- `400`: invalid input (e.g., non-video upload)

WebSocket auth failures:
- Connection closes with policy violation if token invalid

---

## 10) Quick Curl Samples

## Init
```bash
curl -X POST "http://136.110.86.112:8100/pipeline/v1/upload/init" \
  -H "Authorization: Bearer <JWT_TOKEN>" \
  -F "job_id=scan_multi_1" \
  -F "scan_id=scan_multi_1" \
  -F "room_id=room_1" \
  -F "video_id=video_1" \
  -F "room_name=Kitchen"
```

## Upload
```bash
curl -X POST "http://136.110.86.112:8100/pipeline/v1/upload/scan_multi_1/file" \
  -H "Authorization: Bearer <JWT_TOKEN>" \
  -F "file=@video1.mp4;type=video/mp4" \
  -F "scan_id=scan_multi_1" \
  -F "room_id=room_1" \
  -F "video_id=video_1" \
  -F "room_name=Kitchen"
```

## Trigger
```bash
curl -X POST "http://136.110.86.112:8100/pipeline/v1/upload/scan_multi_1/trigger" \
  -H "Authorization: Bearer <JWT_TOKEN>"
```

## Single-call process-video (recommended)
```bash
curl -X POST "http://136.110.86.112:8100/pipeline/v1/process-video" \
  -H "Authorization: Bearer <JWT_TOKEN>" \
  -F "file=@video1.mp4;type=video/mp4" \
  -F "job_id=scan_multi_1" \
  -F "scan_id=scan_multi_1" \
  -F "room_id=room_1" \
  -F "video_id=video_1" \
  -F "room_name=Kitchen"
```

---

## 11) Notes for UI Team

- Keep scan id stable and reuse as job id for room-by-room.
- Always pass room metadata on upload for clear event mapping.
- Expect immediate trigger response; processing continues asynchronously.
- Use RabbitMQ UI for queue visibility during load/perf testing.
