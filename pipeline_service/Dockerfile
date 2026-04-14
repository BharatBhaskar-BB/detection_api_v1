FROM python:3.12-slim

WORKDIR /srv

# System deps for OpenCV, ffmpeg, git (for pip+git installs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA 12.4 support
RUN pip install --no-cache-dir torch torchvision \
    --extra-index-url https://download.pytorch.org/whl/cu124

# Install CLIP from GitHub (no PyPI release)
RUN pip install --no-cache-dir git+https://github.com/openai/CLIP.git

# Install backend dependencies
COPY backend/requirements.txt /srv/backend/requirements.txt
RUN pip install --no-cache-dir -r /srv/backend/requirements.txt

# Copy backend code (DB models, pipeline logic, config)
COPY backend /srv/backend

# Copy standalone pipeline service package
COPY pipeline_service /srv/pipeline_service

ENV PYTHONPATH="/srv/backend:/srv"

CMD ["sh", "-c", "uvicorn pipeline_service.main:app --host 0.0.0.0 --port ${PORT:-8100} --workers 1"]
