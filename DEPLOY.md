# BundleBox — Unified Deployment Guide

## Architecture

```
GCP ALB (dev.bundlebox.ai)
  └─► gateway (nginx :80)
        ├─ /api/detection/*  →  pipeline_backend :8110
        ├─ /api/*            →  backend :8010
        ├─ /pipeline/*       →  pipeline_backend :8110
        ├─ /live-detection/* →  florence :8210
        └─ /*                →  frontend :3010
```

All services run from **one repo** (`detection_api_v1`) on the GCP VM.

## Prerequisites

- GCP VM: `gpu-instance-bundlebox-t4` (34.106.90.78)
- SSH: `ssh -i ~/.ssh/google_compute_engine bharat_bhaskar@34.106.90.78`
- Docker + Docker Compose + NVIDIA Container Toolkit on VM

---

## Step 1: Push code to GitHub

```bash
# Local machine
cd /Users/bharat.bhaskar/detection_api_v1
git add florence/ deploy/
git commit -m "Add florence service + unified deployment config"
git push fork feature/detection-api-v1
```

## Step 2: Pull on VM

```bash
# SSH to VM
ssh -i ~/.ssh/google_compute_engine bharat_bhaskar@34.106.90.78

cd ~/detection_api_v1   # or clone fresh
git pull fork feature/detection-api-v1
```

## Step 3: Create .env files

```bash
cd ~/detection_api_v1/deploy

# SC1 services
cp .env.prod.template .env.prod
# Edit .env.prod — fill in all keys from the existing production values:
#   SECRET_KEY, OPENAI_API_KEY, GEMINI_API_KEY, PIPELINE_SHARED_TOKEN,
#   VAPID keys, Twilio keys, etc.
# Source: /home/bharat.bhaskar/bundlebox/deploy/.env.prod

# Florence
cp .env.florence.template .env.florence
# Edit .env.florence — fill in GEMINI_API_KEY
# Source: /home/bharat.bhaskar/florence/.env
```

## Step 4: Build & start (test mode)

Belgin's containers stay running. Our stack uses different ports.

```bash
cd ~/detection_api_v1

docker compose -p bb \
  -f deploy/docker-compose.unified.yml \
  up -d --build
```

Container names will be prefixed with `bb-` (e.g., `bb-backend-1`).

**Port mapping (test vs production):**

| Service          | Test Port | Prod Port | Belgin's Port |
| ---------------- | --------- | --------- | ------------- |
| backend          | 8010      | 8000      | 8000          |
| pipeline_backend | 8110      | 8100      | 8100          |
| florence         | 8210      | 8200      | 8200          |
| frontend         | 3010      | 3000      | 3000          |
| rabbitmq         | 5682      | 5672      | 5672          |

## Step 5: Verify services

```bash
# Backend health
curl http://localhost:8010/api/v1/health

# Pipeline health
curl http://localhost:8110/api/detection/v1/health

# Florence health
curl http://localhost:8210/api/health

# Frontend (should return HTML)
curl -s http://localhost:3010/ | head -5

# Check all containers
docker ps --filter "name=bb-"
```

## Step 6: Switch gateway

Once all services are verified:

```bash
# Back up current gateway config
cp ~/gateway/nginx.conf ~/gateway/nginx.conf.bak

# Deploy new gateway config (points to test ports)
cp ~/detection_api_v1/deploy/gateway-nginx.conf ~/gateway/nginx.conf

# Reload gateway
docker exec gateway nginx -s reload
```

Now `dev.bundlebox.ai/scanner/sc1` and `dev.bundlebox.ai/scanner/live-detection` hit the new containers.

## Step 7: Test via ALB

```bash
# SC1
curl https://dev.bundlebox.ai/scanner/sc1/api/v1/health

# Live Detection
curl https://dev.bundlebox.ai/scanner/live-detection/api/health

# Frontend
curl -s https://dev.bundlebox.ai/scanner/sc1/ | head -5
```

## Step 8: Stop Belgin's containers

Once everything works through the ALB:

```bash
# Stop Belgin's SC1 stack
cd /home/belgin.vinoj/detection_api_v1
docker compose -f deploy/docker-compose.prod.yml down

# Stop Belgin's pipeline service
cd /home/belgin.vinoj/deployment/detection_api_v1_pub/pipeline_service
docker compose down

# Stop old florence
docker stop bundlebox && docker rm bundlebox
```

## Step 9 (optional): Switch to standard ports

After Belgin's containers are gone, update ports in `docker-compose.unified.yml`:

- 8010 → 8000, 8110 → 8100, 8210 → 8200, 3010 → 3000, 5682 → 5672

Update `gateway-nginx.conf` ports to match, then:

```bash
cd ~/detection_api_v1
docker compose -p bb -f deploy/docker-compose.unified.yml up -d
cp deploy/gateway-nginx.conf ~/gateway/nginx.conf
docker exec gateway nginx -s reload
```

---

## File inventory

```
deploy/
  docker-compose.unified.yml   ← main compose (all services)
  docker-compose.prod.yml      ← old SC1-only compose (reference)
  frontend-nginx.conf          ← nginx config mounted into frontend
  gateway-nginx.conf           ← gateway config for VM's nginx container
  .env.prod.template           ← SC1 env template
  .env.florence.template        ← Florence env template
  .env.prod                    ← (created on VM, not in git)
  .env.florence                ← (created on VM, not in git)
florence/
  server.py, segmentor.py, gdino_detector.py, ...
  Dockerfile
  pwa/
```
