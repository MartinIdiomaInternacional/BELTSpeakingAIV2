
# -------- BELT Speaking AI — One-Service (Docker) --------
# Base image
FROM python:3.11-slim

# Environment hygiene
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps for audio (libsndfile for soundfile; ffmpeg useful for some inputs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# App directory
WORKDIR /app

# Install Python deps first (better cache)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy app code (belt_service.py, add_frontend.py, web/, etc.)
COPY . /app

# Expose default port (Render will inject $PORT)
EXPOSE 8000

# Default envs (can be overridden in Render)
ENV PORT=8000 \
    ASR_BACKEND=openai \
    RUBRIC_MODEL=gpt-4o-mini \
    WHISPER_MODEL=whisper-1

# Start the service; Render provides $PORT
CMD ["/bin/bash", "-lc", "uvicorn belt_service:app --host 0.0.0.0 --port ${PORT}"]
