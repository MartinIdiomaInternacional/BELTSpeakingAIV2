# Dockerfile for CEFR Speaking Evaluator API (no-librosa build)
FROM python:3.11-slim

# Environment
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (no libatlas-base-dev; include ffmpeg for fallback decoding)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# App
COPY main.py .

EXPOSE 10000

# Start the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
