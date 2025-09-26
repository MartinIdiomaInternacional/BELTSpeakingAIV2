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
# belt_service.py and the web/ directory must exist at the repo root.
COPY belt_service.py /app/belt_service.py
COPY web/ /app/web/

# Port Render will hit; we’ll read PORT from env in start.sh
EXPOSE 10000

# Default environment (you can override on Render)
ENV ASR_BACKEND=openai \
    WHISPER_MODEL=whisper-1 \
    RUBRIC_MODEL=gpt-4o-mini \
    PASS_AVG_THRESHOLD=0.70 \
    PASS_MIN_THRESHOLD=0.60 \
    RECORD_SECONDS=60

# ---- Startup ----
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]
