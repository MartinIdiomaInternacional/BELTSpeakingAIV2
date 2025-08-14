
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import tempfile
import uvicorn
import traceback

# Import light deps up front only
import librosa

APP_VERSION = "BELT Speaking Test API v0.5-safe (2025-08-14)"
SAFE_MODE = os.getenv("BELT_SAFE_MODE", "0") == "1"

app = FastAPI(title="BELT Speaking Test API (Safe Mode Ready)", version=APP_VERSION)

# ---------- CORS ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Audio utils (no torch/torchaudio until needed) ----------
def load_audio_to_16k_mono(file_path: str):
    y, sr = librosa.load(file_path, sr=16000, mono=True)
    if y.size == 0:
        return np.zeros((1,), dtype=np.float32), 16000
    y = y.astype(np.float32)
    return y, 16000

def rough_speech_activity_metrics(y: np.ndarray, sr: int = 16000):
    duration = len(y) / sr if len(y) else 0.0
    if len(y) == 0:
        return {
            "duration_sec": 0.0,
            "voiced_ratio": 0.0,
            "num_segments": 0,
            "avg_segment_sec": 0.0,
        }
    intervals = librosa.effects.split(y, top_db=30)
    voiced = sum((end - start) for start, end in intervals) / len(y)
    seg_durs = [ (end - start) / sr for start, end in intervals ] if len(intervals) else []
    avg_seg = float(np.mean(seg_durs)) if seg_durs else 0.0
    return {
        "duration_sec": float(duration),
        "voiced_ratio": float(voiced),
        "num_segments": int(len(intervals)),
        "avg_segment_sec": avg_seg,
    }

# ---------- Optional heavy feature extractor (lazy) ----------
_MODEL = None
_MODEL_NAME = None

def get_ssl_embedding(y_16k: np.ndarray):
    """Lazily import torch/torchaudio and compute an embedding.
    Skipped entirely in SAFE_MODE.
    """
    global _MODEL, _MODEL_NAME
    if SAFE_MODE:
        return None, None, "SAFE_MODE_DISABLED_SSL"
    import torch  # heavy import happens here
    import torchaudio
    waveform = torch.tensor(y_16k, dtype=torch.float32).unsqueeze(0)  # (1, T)
    if _MODEL is None:
        try:
            bundle = torchaudio.pipelines.WAV2VEC2_BASE
            _MODEL = bundle.get_model().eval()
            _MODEL_NAME = "WAV2VEC2_BASE"
        except Exception:
            bundle = torchaudio.pipelines.HUBERT_BASE
            _MODEL = bundle.get_model().eval()
            _MODEL_NAME = "HUBERT_BASE"
    model = _MODEL
    with torch.no_grad():
        try:
            feats_list = model.extract_features(waveform)[0]
            feats = feats_list[-1]
            if feats.dim() == 3:
                feats = feats.squeeze(0)
        except Exception:
            out = model(waveform)
            if isinstance(out, (tuple, list)):
                out = out[0]
            feats = out.squeeze(0)
    emb = feats.mean(dim=0).cpu().numpy()
    var = feats.var(dim=0).mean().item()
    return emb, float(var), _MODEL_NAME

def heuristic_level_from_metrics(variance: float, voiced_ratio: float):
    if variance is None:
        # SAFE MODE fallback purely on voiced ratio
        score = 0.4 * voiced_ratio
    else:
        score = 0.6 * (variance / (variance + 1e-6)) + 0.4 * voiced_ratio
    if score < 0.25:
        return "A1-A2 (provisional)"
    elif score < 0.4:
        return "A2-B1 (provisional)"
    elif score < 0.6:
        return "B1-B2 (provisional)"
    elif score < 0.75:
        return "B2-C1 (provisional)"
    else:
        return "C1-C2 (provisional)"

@app.get("/health")
async def health():
    mode = "SAFE" if SAFE_MODE else "FULL"
    return {"status": "ok", "version": APP_VERSION, "mode": mode}

@app.post("/evaluate")
async def evaluate(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm")):
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload audio (wav/mp3/m4a/flac/ogg/webm).")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[-1]) as tmp:
            data = await file.read()
            tmp.write(data)
            tmp_path = tmp.name

        y, sr = load_audio_to_16k_mono(tmp_path)
        y = y / (np.max(np.abs(y)) + 1e-8)

        # Always compute lightweight metrics
        act = rough_speech_activity_metrics(y, sr=sr)

        # Optionally compute SSL embedding
        emb, variance, model_name = get_ssl_embedding(y)
        level = heuristic_level_from_metrics(variance, act["voiced_ratio"])

        result = {
            "ok": True,
            "mode": "SAFE" if SAFE_MODE else "FULL",
            "model": model_name,
            "duration_sec": act["duration_sec"],
            "voiced_ratio": act["voiced_ratio"],
            "segments": act["num_segments"],
            "avg_segment_sec": act["avg_segment_sec"],
            "embedding_norm": float(np.linalg.norm(emb)) if emb is not None else None,
            "feature_variance": variance,
            "provisional_level": level,
            "notes": "In SAFE mode, SSL features are skipped to keep the service responsive."
        }
        return JSONResponse(content=result)
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}\n{tb}")
    finally:
        try:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass

if __name__ == "__main__":
    # Constrain threads to prevent CPU thrash on shared hosts
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
