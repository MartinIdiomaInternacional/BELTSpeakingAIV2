
"""
FastAPI CEFR Speaking Level Evaluator (Updated)
------------------------------------------------
- Replaces deprecated librosa.load with a robust loader using soundfile -> audioread fallback.
- Adds structured logging, request IDs, and detailed error handling.
- Includes lightweight scoring logic with confidence and rubric mapping to CEFR bands.
- Endpoint: POST /evaluate-bytes  (multipart/form-data: file=audio)
- Target sample rate: 16_000 Hz, mono
"""
import io
import json
import logging
import time
import uuid
from typing import Optional, Tuple

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

# Audio I/O & DSP (no librosa)
import soundfile as sf
from scipy.signal import resample_poly
import audioread

APP_NAME = "cefr-evaluator"
TARGET_SR = 16_000
MAX_FILE_MB = 10
MAX_SAMPLES = TARGET_SR * 60 * 5  # cap processing at 5 minutes

# --------------- Logging Setup ---------------
class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "ts": int(time.time() * 1000),
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
        }
        # Inject extras if present
        for key in ("request_id", "path", "method", "remote_addr", "elapsed_ms"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        return json.dumps(payload)

logger = logging.getLogger(APP_NAME)
handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logger.setLevel(logging.INFO)
logger.addHandler(handler)

app = FastAPI(title="CEFR Speaking Evaluator", version="1.2.0")

# --------------- Middleware: Request ID + Access log ---------------
class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        start = time.perf_counter()
        try:
            response = await call_next(request)
            return response
        finally:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            rec = logging.LogRecord(name=APP_NAME, level=logging.INFO, pathname=__file__,
                                    lineno=0, msg=f"{request.method} {request.url.path}", args=(), exc_info=None)
            rec.request_id = request_id
            rec.path = request.url.path
            rec.method = request.method
            rec.remote_addr = request.client.host if request.client else None
            rec.elapsed_ms = elapsed_ms
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

# --------------- Utility: Safe audio loading ---------------
def _to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x
    return np.mean(x, axis=1)

def _resample(x: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return x
    # resample with rational factor
    from math import gcd
    g = gcd(sr, target_sr)
    up = target_sr // g
    down = sr // g
    return resample_poly(x, up, down)

def load_audio_safe(file_bytes: bytes, filename: str) -> Tuple[np.ndarray, int]:
    """Load audio into mono float32 at TARGET_SR without librosa.load.
    Tries soundfile first; falls back to audioread if needed.
    """
    # Try soundfile
    try:
        with sf.SoundFile(io.BytesIO(file_bytes)) as f:
            sr = f.samplerate
            data = f.read(always_2d=True)  # (frames, channels)
            x = data.astype(np.float32)
            x = _to_mono(x)
    except Exception:
        # Fallback to audioread (works for mp3 and other codecs via ffmpeg/gstreamer)
        try:
            with audioread.audio_open(io.BytesIO(file_bytes)) as af:
                sr = af.samplerate
                # Collect PCM frames
                pcm = bytearray()
                for buf in af:
                    pcm.extend(buf)
                # Decode according to width if available (assume 16-bit little-endian if unknown)
                # audioread yields raw bytes already decoded to pcm16 typically
                x = np.frombuffer(bytes(pcm), dtype=np.int16).astype(np.float32) / 32768.0
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Unsupported or corrupted audio file: {filename} ({e})")

    if x.size == 0:
        raise HTTPException(status_code=400, detail="Empty audio stream")        
    # Cap max length
    if x.shape[0] > MAX_SAMPLES * max(1, sr // TARGET_SR):
        x = x[: MAX_SAMPLES * max(1, sr // TARGET_SR)]

    # Resample and ensure mono float32
    x = _resample(x, sr, TARGET_SR).astype(np.float32)
    x = _to_mono(x)
    return x, TARGET_SR

# --------------- Lightweight features (no external ML) ---------------
def rms_energy(x: np.ndarray, frame: int = 400, hop: int = 160) -> np.ndarray:
    # ~25ms frames at 16k: 400; hop ~10ms: 160
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
    # simple magnitude spectrum centroid
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

# --------------- Scoring Heuristic ---------------
CEFR_BANDS = ["A1", "A2", "B1", "B1+", "B2", "B2+", "C1", "C2"]

def map_score_to_cefr(score: float) -> str:
    # Tunable thresholds; monotonically increasing bands
    if score < 0.20: return "A1"
    if score < 0.30: return "A2"
    if score < 0.45: return "B1"
    if score < 0.55: return "B1+"
    if score < 0.68: return "B2"
    if score < 0.78: return "B2+"
    if score < 0.88: return "C1"
    return "C2"

def evaluate_signal(x: np.ndarray, sr: int) -> Tuple[float, dict, list[str]]:
    """Compute a normalized 0-1 score from simple paralinguistic features.        This is a placeholder until the embedding model hooks are added.
    """
    warnings = []
    dur = len(x) / sr
    if dur < 3.0:
        warnings.append("Very short duration (<3s) may reduce reliability.")
    if dur > 120.0:
        warnings.append("Long duration; truncated to first 5 minutes for processing.")

    E = rms_energy(x)
    Z = zero_crossing_rate(x)
    C = spectral_centroid(x, sr)

    # Voice activity proxy
    speech_frames = (E > (E.mean() * 0.5)).sum()
    vad_ratio = speech_frames / max(1, len(E))

    # Stability: lower ZCR variance & centroid variance => more stable voicing/prosody
    zcr_var = float(np.var(Z))
    cent_var = float(np.var(C))
    energy_mean = float(np.mean(E))

    # Normalize features (rough min-max or soft transforms)
    vad_score = float(np.clip((vad_ratio - 0.2) / 0.7, 0, 1))       # prefer >20% voiced
    zcr_score = float(np.clip(1.0 - np.tanh(3*zcr_var), 0, 1))      # lower variance better
    cent_score = float(np.clip(1.0 - np.tanh(1e-4*cent_var), 0, 1)) # lower variance better
    loud_score = float(np.clip(np.log1p(energy_mean*1000.0), 0, 1)) # avoid near-silence

    raw = 0.35*vad_score + 0.25*zcr_score + 0.25*cent_score + 0.15*loud_score
    # Confidence boosted by duration and consistency
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

# --------------- Exception handlers ---------------
@app.exception_handler(HTTPException)
async def http_exc_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})

@app.exception_handler(Exception)
async def generic_exc_handler(request: Request, exc: Exception):
    # Avoid leaking internals
    return JSONResponse(status_code=500, content={"error": "Internal server error"})

# --------------- Endpoint ---------------
@app.post("/evaluate-bytes", response_model=EvalResponse)
async def evaluate_bytes(request: Request, file: UploadFile = File(...)):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    # Basic content guard
    if file.spool_max_size and file.spool_max_size > 0:
        pass  # FastAPI uses SpooledTemporaryFile; not a strict size guard
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        raise HTTPException(status_code=413, detail=f"File too large ({size_mb:.2f} MB). Limit is {MAX_FILE_MB} MB.")

    try:
        x, sr = load_audio_safe(content, file.filename)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unable to decode audio: {e}")

    # Scoring
    score, details, warnings = evaluate_signal(x, sr)
    level = map_score_to_cefr(score)

    # Audit log
    rec = logging.LogRecord(name=APP_NAME, level=logging.INFO, pathname=__file__, lineno=0,
                            msg=f"evaluated {file.filename} -> {level} ({score:.3f})", args=(), exc_info=None)
    rec.request_id = request_id
    logger.handle(rec)

    return EvalResponse(
        level=level,
        score=score,
        confidence=float(np.clip(details["features"]["scores"]["vad"]*0.5 + score*0.5, 0, 1)),
        duration_sec=float(details["duration_sec"]),
        sample_rate=sr,
        warnings=warnings,
        details=details,
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_cefr_evaluator:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
