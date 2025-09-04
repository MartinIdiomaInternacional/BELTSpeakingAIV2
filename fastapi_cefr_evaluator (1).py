
"""
FastAPI CEFR Speaking Level Evaluator (v1.3.0)
------------------------------------------------
Upgrades:
- Health/Readiness: /healthz, /readyz
- Metrics: /metrics (Prometheus)
- VAD trimming: energy-based leading/trailing silence removal
- Optional wav2vec2-based embedding scorer (set USE_WAV2VEC=1 to enable)
- Structured logging preserved
"""
import io
import json
import logging
import os
import time
import uuid
from typing import Optional, Tuple

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

# Audio I/O & DSP
import soundfile as sf
from scipy.signal import resample_poly
import audioread

# Metrics
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

APP_NAME = "cefr-evaluator"
TARGET_SR = 16_000
MAX_FILE_MB = 10
MAX_SAMPLES = TARGET_SR * 60 * 5  # 5 minutes cap

# --------------- Logging Setup ---------------
class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "ts": int(time.time() * 1000),
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
        }
        for key in ("request_id", "path", "method", "remote_addr", "elapsed_ms"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        return json.dumps(payload)

logger = logging.getLogger(APP_NAME)
handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logger.setLevel(logging.INFO)
logger.addHandler(handler)

app = FastAPI(title="CEFR Speaking Evaluator", version="1.3.0")

# --------------- Metrics ---------------
REQ_COUNTER = Counter("cefr_eval_requests_total", "Total requests", ["route", "method"])
REQ_LATENCY = Histogram("cefr_eval_request_latency_seconds", "Request latency", ["route", "method"])
AUDIO_BYTES = Histogram("cefr_eval_audio_upload_bytes", "Uploaded audio size (bytes)")
EVAL_SCORE = Histogram("cefr_eval_score", "Raw evaluation score (0..1)")
READY = Gauge("cefr_eval_ready", "Readiness state (1 ready, 0 not)")
READY.set(1.0)

# --------------- Middleware: Request ID + Access log + Metrics ---------------
class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        start = time.perf_counter()
        route = request.url.path
        method = request.method
        REQ_COUNTER.labels(route=route, method=method).inc()
        try:
            response = await call_next(request)
            return response
        finally:
            elapsed = time.perf_counter() - start
            REQ_LATENCY.labels(route=route, method=method).observe(elapsed)
            rec = logging.LogRecord(name=APP_NAME, level=logging.INFO, pathname=__file__,
                                    lineno=0, msg=f"{method} {route}", args=(), exc_info=None)
            rec.request_id = request_id
            rec.path = route
            rec.method = method
            rec.remote_addr = request.client.host if request.client else None
            rec.elapsed_ms = int(elapsed * 1000)
            logger.handle(rec)

app.add_middleware(RequestContextMiddleware)

# --------------- Models ---------------
class EvalResponse(BaseModel):
    level: str = Field(..., description="Predicted CEFR level")
    score: float = Field(..., ge=0, le=1, description="Normalized score")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in prediction")
    duration_sec: float = Field(..., ge=0)
    sample_rate: int = Field(..., ge=8000)
    warnings: list[str] = Field(default_factory=list)
    details: dict = Field(default_factory=dict)

# --------------- Audio utilities ---------------
def _to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x
    return np.mean(x, axis=1)

def _resample(x: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return x
    from math import gcd
    g = gcd(sr, target_sr)
    up = target_sr // g
    down = sr // g
    return resample_poly(x, up, down)

def load_audio_safe(file_bytes: bytes, filename: str) -> Tuple[np.ndarray, int]:
    # soundfile path
    try:
        with sf.SoundFile(io.BytesIO(file_bytes)) as f:
            sr = f.samplerate
            data = f.read(always_2d=True)
            x = data.astype(np.float32)
            x = _to_mono(x)
    except Exception:
        # audioread fallback
        try:
            with audioread.audio_open(io.BytesIO(file_bytes)) as af:
                sr = af.samplerate
                pcm = bytearray()
                for buf in af:
                    pcm.extend(buf)
                x = np.frombuffer(bytes(pcm), dtype=np.int16).astype(np.float32) / 32768.0
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Unsupported or corrupted audio file: {filename} ({e})")

    if x.size == 0:
        raise HTTPException(status_code=400, detail="Empty audio stream")

    # Cap before resample to avoid heavy load
    if x.shape[0] > MAX_SAMPLES * max(1, sr // TARGET_SR):
        x = x[: MAX_SAMPLES * max(1, sr // TARGET_SR)]

    x = _resample(x, sr, TARGET_SR).astype(np.float32)
    x = _to_mono(x)
    return x, TARGET_SR

# --------------- VAD trimming ---------------
def trim_silence(x: np.ndarray, sr: int, frame: int = 400, hop: int = 160, thresh_db: float = -45.0, pad_ms: int = 100) -> Tuple[np.ndarray, float, float]:
    """
    Simple energy-based VAD trimming.
    Returns trimmed signal and leading/trailing times removed (sec).
    """
    if x.size < frame:
        return x, 0.0, 0.0
    # Frame energies in dB
    n_frames = 1 + (len(x) - frame) // hop
    E = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        seg = x[i*hop: i*hop + frame]
        E[i] = 10 * np.log10(np.mean(seg**2) + 1e-12)
    mask = E > thresh_db
    if not np.any(mask):
        return x, 0.0, 0.0
    first = int(np.argmax(mask))
    last = int(len(mask) - 1 - np.argmax(mask[::-1]))
    # padding
    pad = int((pad_ms/1000.0) * sr)
    start = max(0, first*hop - pad)
    end = min(len(x), last*hop + frame + pad)
    lead_sec = start / sr
    trail_sec = (len(x) - end) / sr
    return x[start:end], float(lead_sec), float(trail_sec)

# --------------- Scoring Heuristic ---------------
CEFR_BANDS = ["A1", "A2", "B1", "B1+", "B2", "B2+", "C1", "C2"]

def map_score_to_cefr(score: float) -> str:
    if score < 0.20: return "A1"
    if score < 0.30: return "A2"
    if score < 0.45: return "B1"
    if score < 0.55: return "B1+"
    if score < 0.68: return "B2"
    if score < 0.78: return "B2+"
    if score < 0.88: return "C1"
    return "C2"

def rms_energy(x: np.ndarray, frame: int = 400, hop: int = 160) -> np.ndarray:
    if x.size < frame:
        return np.array([np.sqrt(np.mean(x**2) + 1e-12)], dtype=np.float32)
    n_frames = 1 + (len(x) - frame) // hop
    E = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        seg = x[i*hop: i*hop + frame]
        E[i] = np.sqrt(np.mean(seg**2) + 1e-12)
    return E

def zero_crossing_rate(x: np.ndarray, frame: int = 400, hop: int = 160) -> np.ndarray:
    if x.size < frame:
        return np.array([0.0], dtype=np.float32)
    n_frames = 1 + (len(x) - frame) // hop
    Z = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        seg = x[i*hop: i*hop + frame]
        Z[i] = (np.mean(np.abs(np.diff(np.sign(seg)))))/2.0
    return Z

def spectral_centroid(x: np.ndarray, sr: int, n_fft: int = 512, hop: int = 160) -> np.ndarray:
    if x.size < n_fft:
        X = np.fft.rfft(x, n=n_fft)
        mag = np.abs(X)
        freqs = np.fft.rfftfreq(n_fft, d=1.0/sr)
        return np.array([float(np.sum(freqs * mag) / (np.sum(mag) + 1e-12))], dtype=np.float32)
    frames = []
    for start in range(0, len(x)-n_fft+1, hop):
        seg = x[start:start+n_fft]
        X = np.fft.rfft(seg * np.hanning(n_fft))
        mag = np.abs(X)
        freqs = np.fft.rfftfreq(n_fft, d=1.0/sr)
        frames.append(float(np.sum(freqs * mag) / (np.sum(mag) + 1e-12)))
    return np.array(frames, dtype=np.float32)

def heuristic_evaluate(x: np.ndarray, sr: int):
    warnings = []
    dur = len(x) / sr
    if dur < 3.0:
        warnings.append("Very short duration (<3s) may reduce reliability.")
    if dur > 120.0:
        warnings.append("Long duration; truncated to first 5 minutes for processing.")

    E = rms_energy(x)
    Z = zero_crossing_rate(x)
    C = spectral_centroid(x, sr)

    speech_frames = (E > (E.mean() * 0.5)).sum()
    vad_ratio = speech_frames / max(1, len(E))

    zcr_var = float(np.var(Z))
    cent_var = float(np.var(C))
    energy_mean = float(np.mean(E))

    vad_score = float(np.clip((vad_ratio - 0.2) / 0.7, 0, 1))
    zcr_score = float(np.clip(1.0 - np.tanh(3*zcr_var), 0, 1))
    cent_score = float(np.clip(1.0 - np.tanh(1e-4*cent_var), 0, 1))
    loud_score = float(np.clip(np.log1p(energy_mean*1000.0), 0, 1))

    raw = 0.35*vad_score + 0.25*zcr_score + 0.25*cent_score + 0.15*loud_score
    conf = float(np.clip(0.4 + 0.3*vad_score + 0.2*(1 - abs(zcr_var - 0.02)) + 0.1*np.clip(dur/30.0, 0, 1), 0, 1))

    details = {
        "duration_sec": dur,
        "features": {
            "vad_ratio": vad_ratio,
            "zcr_var": zcr_var,
            "centroid_var": cent_var,
            "energy_mean": energy_mean,
            "scores": {
                "vad": vad_score, "zcr_stability": zcr_score,
                "centroid_stability": cent_score, "loudness": loud_score
            }
        }
    }
    return float(np.clip(raw, 0, 1)), details, warnings

# --------------- Optional wav2vec2 scorer ---------------
class Wav2Vec2Scorer:
    def __init__(self):
        self.enabled = False
        try:
            if os.getenv("USE_WAV2VEC", "0") not in ("1", "true", "True"):
                return
            # Lazy imports
            from transformers import AutoProcessor, AutoModel
            import torch  # noqa
            self.processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
            self.model = AutoModel.from_pretrained("facebook/wav2vec2-base-960h")
            self.model.eval()
            self.enabled = True
            logger.info("wav2vec2 scorer enabled")
        except Exception as e:
            logger.info(f"wav2vec2 scorer not enabled: {e}")
            self.enabled = False

    def score(self, x: np.ndarray, sr: int) -> Optional[Tuple[float, dict]]:
        if not self.enabled:
            return None
        try:
            import torch
            with torch.no_grad():
                inputs = self.processor(x, sampling_rate=sr, return_tensors="pt")
                out = self.model(**inputs)
                # Use hidden states mean as a proxy embedding (batch, time, dim) -> (dim,)
                if hasattr(out, "last_hidden_state"):
                    emb = out.last_hidden_state.mean(dim=1).squeeze(0)  # (dim,)
                    # Normalize and turn into a 0..1 score via logistic
                    score = float(torch.sigmoid(emb.norm() / 100).item())
                    details = {"embedding_norm": float(emb.norm().item()), "backend": "wav2vec2-base-960h"}
                    return score, details
        except Exception as e:
            logger.info(f"wav2vec2 scoring failed, falling back: {e}")
        return None

WAV2VEC = Wav2Vec2Scorer()

# --------------- Exception handlers ---------------
@app.exception_handler(HTTPException)
async def http_exc_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})

@app.exception_handler(Exception)
async def generic_exc_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": "Internal server error"})

# --------------- Endpoints ---------------
@app.get("/healthz", response_class=PlainTextResponse)
async def healthz():
    return "ok"

@app.get("/readyz", response_class=PlainTextResponse)
async def readyz():
    # If we ever make model loading async, flip READY accordingly
    return "ready" if READY._value.get() == 1.0 else "not-ready"

@app.get("/metrics")
async def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/evaluate-bytes", response_model=EvalResponse)
async def evaluate_bytes(request: Request, file: UploadFile = File(...)):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    AUDIO_BYTES.observe(len(content))
    if size_mb > MAX_FILE_MB:
        raise HTTPException(status_code=413, detail=f"File too large ({size_mb:.2f} MB). Limit is {MAX_FILE_MB} MB.")

    try:
        x, sr = load_audio_safe(content, file.filename)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unable to decode audio: {e}")

    # VAD trimming
    x_trim, lead_sec, trail_sec = trim_silence(x, sr)
    trim_info = {"trim_leading_sec": lead_sec, "trim_trailing_sec": trail_sec}

    # Scoring: try wav2vec2 first if enabled, else heuristic
    score_details = {}
    score_tuple = WAV2VEC.score(x_trim, sr)
    if score_tuple is not None:
        score, w2v_details = score_tuple
        score_details["wav2vec2"] = w2v_details
        # Blend with heuristic for stability
        h_score, h_details, warnings = heuristic_evaluate(x_trim, sr)
        score = float(0.6 * score + 0.4 * h_score)
        details = {**h_details, **score_details, **trim_info}
    else:
        score, details, warnings = heuristic_evaluate(x_trim, sr)
        details.update(trim_info)

    level = map_score_to_cefr(score)
    EVAL_SCORE.observe(score)

    rec = logging.LogRecord(name=APP_NAME, level=logging.INFO, pathname=__file__, lineno=0,
                            msg=f"evaluated {file.filename} -> {level} ({score:.3f})", args=(), exc_info=None)
    rec.request_id = request_id
    logger.handle(rec)

    return EvalResponse(
        level=level,
        score=score,
        confidence=float(np.clip(details.get("features", {}).get("scores", {}).get("vad", 0)*0.5 + score*0.5, 0, 1)),
        duration_sec=float(details.get("duration_sec", len(x_trim)/sr)),
        sample_rate=sr,
        warnings=locals().get("warnings", []),
        details=details,
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_cefr_evaluator:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
