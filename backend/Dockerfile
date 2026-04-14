FROM python:3.12-slim AS base

WORKDIR /app

# System deps for OpenCV, ffmpeg, git (for pip+git installs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA 12.4 support
RUN pip install --no-cache-dir torch torchvision \
    --extra-index-url https://download.pytorch.org/whl/cu124

# Install CLIP from GitHub (no PyPI release)
RUN pip install --no-cache-dir git+https://github.com/openai/CLIP.git

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create data directory for persistent storage
RUN mkdir -p /app/data/uploads /app/data/outputs

COPY . .

# Production
FROM base AS production
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# Development (with hot reload)
FROM base AS development
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
