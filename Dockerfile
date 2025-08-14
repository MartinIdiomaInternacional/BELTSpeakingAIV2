# BELT Speaking Test API - Dockerfile (Render-friendly, includes prompts.json)
FROM python:3.11-slim

# System deps for audio
RUN apt-get update && apt-get install -y --no-install-recommends         ffmpeg         libsndfile1     && rm -rf /var/lib/apt/lists/*

# Keep things light & predictable
ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1     OMP_NUM_THREADS=1     MKL_NUM_THREADS=1     BELT_SAFE_MODE=1     PIP_DISABLE_PIP_VERSION_CHECK=1     PROMPTS_PATH=/app/prompts.json

WORKDIR /app

# Base deps (SAFE)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Optional FULL deps (SSL features)
ARG INCLUDE_TORCH=0
COPY requirements-full.txt /app/requirements-full.txt
RUN if [ "$INCLUDE_TORCH" = "1" ]; then pip install --no-cache-dir -r /app/requirements-full.txt; fi

# Optional extras
ARG INCLUDE_EXTRAS=0
COPY requirements-extras.txt /app/requirements-extras.txt
RUN if [ "$INCLUDE_EXTRAS" = "1" ]; then pip install --no-cache-dir -r /app/requirements-extras.txt; fi

# App code and prompts
COPY main.py /app/main.py
COPY prompts.json /app/prompts.json

# Render provides $PORT; fall back to 8000 for local
EXPOSE 8000
CMD ["sh","-c","uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
