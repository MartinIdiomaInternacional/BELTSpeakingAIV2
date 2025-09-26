# ---- Base image ----
FROM python:3.10-slim

# Prevent Python buffering & bytecode files in container
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set a working directory
WORKDIR /app

# ---- System deps ----
# ffmpeg is required to transcode browser audio to 16k mono WAV for Whisper
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# ---- Python deps ----
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# ---- App code ----
COPY belt_service.py /app/belt_service.py
COPY web/ /app/web/

# Expose Render port (actual port comes from $PORT)
EXPOSE 10000

# Default environment (can override on Render dashboard)
ENV ASR_BACKEND=openai \
    WHISPER_MODEL=whisper-1 \
    RUBRIC_MODEL=gpt-4o-mini \
    PASS_AVG_THRESHOLD=0.70 \
    PASS_MIN_THRESHOLD=0.60 \
    RECORD_SECONDS=60

# ---- Startup ----
# Run uvicorn directly, binding to the dynamic PORT provided by Render
CMD ["sh", "-c", "uvicorn belt_service:app --host 0.0.0.0 --port ${PORT:-10000} --proxy-headers"]
