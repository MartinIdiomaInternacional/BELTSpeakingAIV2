
# BELT Speaking Test API - Dockerfile (Safe/Full modes)
FROM python:3.11-slim

# System deps for audio
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r /app/requirements.txt

COPY main.py /app/main.py

EXPOSE 8000
# Use SAFE mode by default to keep containers responsive
ENV BELT_SAFE_MODE=1
CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000" ]
