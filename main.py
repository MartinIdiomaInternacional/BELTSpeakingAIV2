
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import torch
import torchaudio
import librosa
import tempfile
import uvicorn
import os
import traceback

APP_VERSION = "BELT Speaking Test API v0.4 (2025-08-14)"

app = FastAPI(title="BELT Speaking Test API", version=APP_VERSION)

# ---------- CORS ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Audio utils ----------
def load_audio_to_16k_mono(file_path: str):
    # Try torchaudio first (fast), fall back to librosa
    try:
        waveform, sr = torchaudio.load(file_path)  # shape: (channels, T)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.size(0) > 1:  # mixdown to mono
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        waveform = waveform.float()
        return waveform, 16000
    except Exception:
        y, sr = librosa.load(file_path, sr=16000, mono=True)
        waveform = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        return waveform, 16000

_MODEL = None
_MODEL_NAME = None

def get_feature_model():
    global _MODEL, _MODEL_NAME
    if _MODEL is not None:
        return _MODEL, _MODEL_NAME
    # Lazily load an SSL model that ships with torchaudio
    # Prefer Wav2Vec2 base; fallback to HuBERT base
    try:
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        _MODEL = bundle.get_model()
        _MODEL.eval()
        _MODEL_NAME = "WAV2VEC2_BASE"
        return _MODEL, _MODEL_NAME
    except Exception:
        bundle = torchaudio.pipelines.HUBERT_BASE
        _MODEL = bundle.get_model()
        _MODEL.eval()
        _MODEL_NAME = "HUBERT_BASE"
        return _MODEL, _MODEL_NAME

def extract_embedding(waveform: torch.Tensor):
    """Return a pooled embedding and basic stats from an SSL model.        waveform: (1, T) at 16k float32
    """
    model, model_name = get_feature_model()
    with torch.no_grad():
        try:
            # Some models expose extract_features() (returns list of layers)
            feats_list = model.extract_features(waveform)[0]  # list[Tensor(T, D)]
            feats = feats_list[-1]  # last layer
            if feats.dim() == 3:
                feats = feats.squeeze(0)  # (T, D)
        except Exception:
            # Fallback to forward(); many torchaudio SSL models return (B, T, D)
            out = model(waveform)
            if isinstance(out, (tuple, list)):
                out = out[0]
            feats = out.squeeze(0)  # (T, D)
    # Mean-pool over time
    emb = feats.mean(dim=0)  # (D,)
    # Also compute rough "stability" via variance
    var = feats.var(dim=0).mean().item()
    return emb.cpu().numpy(), float(var), model_name

def rough_speech_activity_metrics(y: np.ndarray, sr: int = 16000):
    """Estimate voiced ratio and speaking rate-ish proxy.        Returns: dict with duration_sec, voiced_ratio, segments
    """
    duration = len(y) / sr if len(y) else 0.0
    if len(y) == 0:
        return {
            "duration_sec": 0.0,
            "voiced_ratio": 0.0,
            "num_segments": 0,
            "avg_segment_sec": 0.0,
        }
    # Split non-silent regions
    intervals = librosa.effects.split(y, top_db=30)  # (N, 2)
    voiced = sum((end - start) for start, end in intervals) / len(y)
    seg_durs = [ (end - start) / sr for start, end in intervals ] if len(intervals) else []
    avg_seg = float(np.mean(seg_durs)) if seg_durs else 0.0
    return {
        "duration_sec": float(duration),
        "voiced_ratio": float(voiced),
        "num_segments": int(len(intervals)),
        "avg_segment_sec": avg_seg,
    }

def heuristic_level_from_metrics(variance: float, voiced_ratio: float):
    """Very rough, placeholder heuristic for a provisional level label.        This is NOT a certified CEFR evaluation.        """
    # Combine and bucket; tune thresholds as you see fit
    score = 0.6 * (variance / (variance + 1e-6)) + 0.4 * voiced_ratio
    # Map to coarse levels
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

# ---------- Routes ----------
@app.get("/health")
async def health():
    return {"status": "ok", "version": APP_VERSION}

@app.post("/evaluate")
async def evaluate(file: UploadFile = File(...)):
    # Basic validation
    if not file.filename.lower().endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm")):
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload audio (wav/mp3/m4a/flac/ogg/webm).")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[-1]) as tmp:
            data = await file.read()
            tmp.write(data)
            tmp_path = tmp.name

        # Load + normalize
        waveform, sr = load_audio_to_16k_mono(tmp_path)  # (1, T)
        y = waveform.squeeze(0).cpu().numpy()
        y = y / (np.max(np.abs(y)) + 1e-8)

        # Extract embedding & metrics
        emb, variance, model_name = extract_embedding(waveform)
        act = rough_speech_activity_metrics(y, sr=16000)
        level = heuristic_level_from_metrics(variance, act["voiced_ratio"])

        # Keep the response small & clear
        result = {
            "ok": True,
            "model": model_name,
            "duration_sec": act["duration_sec"],
            "voiced_ratio": act["voiced_ratio"],
            "segments": act["num_segments"],
            "avg_segment_sec": act["avg_segment_sec"],
            "embedding_norm": float(np.linalg.norm(emb)),
            "feature_variance": variance,
            "provisional_level": level,
            "notes": "This is a heuristic, non-certified estimate based on SSL features and simple activity metrics."
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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
