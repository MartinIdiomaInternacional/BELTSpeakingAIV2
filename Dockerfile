
# BELT Speaking Test API - Dockerfile (CPU)
FROM python:3.11-slim

# System deps for audio + performance
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Prevent Python from writing .pyc files & enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install Python deps (CPU-only PyTorch/Torchaudio)
# Pin to compatible versions; adjust as needed.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        fastapi \
        uvicorn[standard] \
        numpy \
        soundfile \
        librosa \
        --index-url https://pypi.org/simple \
    && pip install --no-cache-dir \
        torch==2.3.1+cpu \
        torchaudio==2.3.1+cpu \
        --index-url https://download.pytorch.org/whl/cpu

WORKDIR /app
COPY main.py /app/main.py

EXPOSE 8000
CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000" ]
