# BELT Speaking Test API - Dockerfile (Safe/Full selectable)
FROM python:3.11-slim

# System deps for audio
RUN apt-get update && apt-get install -y --no-install-recommends \        ffmpeg \        libsndfile1 \    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    BELT_SAFE_MODE=1

WORKDIR /app

# Install base deps (SAFE mode)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Optionally install FULL deps (SSL features)
ARG INCLUDE_TORCH=0
COPY requirements-full.txt /app/requirements-full.txt
RUN if [ "$INCLUDE_TORCH" = "1" ]; then pip install --no-cache-dir -r /app/requirements-full.txt; fi

COPY main.py /app/main.py

EXPOSE 8000
CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000" ]
