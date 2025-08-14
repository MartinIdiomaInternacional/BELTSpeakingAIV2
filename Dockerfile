# ---- Base image ----
FROM python:3.11-slim

# ---- System deps ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

# ---- Env defaults (override in Render dashboard if needed) ----
# SAFE mode skips local SSL/ASR (faster start, smaller RAM).
ENV BELT_SAFE_MODE=0 \
    HYBRID_W_AUDIO=0.5 \
    HYBRID_W_TEXT=0.5 \
    HYBRID_W_REL=0.2 \
    T_A2=0.25 \
    T_B1=0.40 \
    T_B2=0.60 \
    T_C1=0.75 \
    USE_OPENAI_ASR=0 \
    OPENAI_ASR_MODEL=whisper-1 \
    USE_OPENAI_EMBED=1 \
    OPENAI_EMBED_MODEL=text-embedding-3-small \
    PROMPTS_PATH=/app/prompts.json \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ---- App dir ----
WORKDIR /app

# ---- Python deps ----
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# ---- App code ----
COPY main.py /app/main.py
# prompts.json is optional here; comment out this line if you manage prompts separately
COPY prompts.json /app/prompts.json

# ---- Runtime ----
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=8000"]
