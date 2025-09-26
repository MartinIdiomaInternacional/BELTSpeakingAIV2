# Use slim Python base
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ---- System deps (ffmpeg for audio) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ---- Python deps (purge any old openai first) ----
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip uninstall -y openai openai-secret-manager || true && \
    pip install --no-cache-dir -r /app/requirements.txt

# ---- App code ----
COPY belt_service.py /app/belt_service.py
COPY web/ /app/web/

# ---- Runtime ----
ENV PORT=10000
EXPOSE 10000

# Uvicorn server
CMD ["uvicorn", "belt_service:app", "--host", "0.0.0.0", "--port", "10000", "--proxy-headers", "--forwarded-allow-ips", "*"]
