# --- Base ---
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=10000 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# --- System deps ---
# - ffmpeg for audio conversion
# - build-essential (gcc, g++) to compile webrtcvad
# - libsndfile1 for python-soundfile
# - ca-certificates for TLS
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    libsndfile1 \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# --- Python deps ---
COPY requirements.txt /app/requirements.txt
# (Optional) cleanse old openai v0 if present in cache
RUN python -m pip install --upgrade pip && \
    pip uninstall -y openai openai-secret-manager || true && \
    pip install --no-cache-dir -r /app/requirements.txt

# --- App files ---
COPY belt_service.py /app/belt_service.py
COPY web/ /app/web/

# --- Health (optional) ---
EXPOSE 10000

# --- Start ---
# Use uvicorn directly; no start.sh needed
CMD ["uvicorn", "belt_service:app", "--host", "0.0.0.0", "--port", "10000", "--log-level", "info"]
