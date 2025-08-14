import os, json, tempfile, traceback, re
from pathlib import Path
from functools import lru_cache
from typing import Optional, Tuple

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import numpy as np
import librosa

APP_VERSION = "BELT Speaking Test API v0.7-hybrid (2025-08-14)"
SAFE_MODE = os.getenv("BELT_SAFE_MODE", "0") == "1"

# Weights for hybrid scoring (override via env)
HYBRID_W_AUDIO = float(os.getenv("HYBRID_W_AUDIO", "0.5"))
HYBRID_W_TEXT  = float(os.getenv("HYBRID_W_TEXT", "0.5"))

# CEFR band thresholds (0..1 -> bands)
T_A2 = float(os.getenv("T_A2", "0.25"))
T_B1 = float(os.getenv("T_B1", "0.40"))
T_B2 = float(os.getenv("T_B2", "0.60"))
T_C1 = float(os.getenv("T_C1", "0.75"))

# ---------- App ----------
app = FastAPI(title="BELT Speaking Test API", version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Prompts (question bank) ----------
@lru_cache(maxsize=1)
def load_prompts():
    path = Path(os.getenv("PROMPTS_PATH", "prompts.json"))
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {str(k).strip().upper(): v for k, v in data.items()}
    except Exception:
        return {}

@app.get("/prompts/{level}")
async def get_prompts(level: str):
    prompts = load_prompts()
    lvl = level.strip().upper()
    if lvl not in prompts:
        raise HTTPException(status_code=404, detail=f"No prompts found for level '{level}'.")
    return {"level": lvl, "count": len(prompts[lvl]), "questions": prompts[lvl]}

# ---------- Audio utils ----------
def load_audio_to_16k_mono(file_path: str):
    y, sr = librosa.load(file_path, sr=16000, mono=True)
    if y.size == 0:
        return np.zeros((1,), dtype=np.float32), 16000
    y = y.astype(np.float32)
    return y, 16000

def speech_activity_metrics(y: np.ndarray, sr: int = 16000):
    duration = len(y) / sr if len(y) else 0.0
    if len(y) == 0:
        return {
            "duration_sec": 0.0,
            "voiced_ratio": 0.0,
            "num_segments": 0,
            "avg_segment_sec": 0.0,
            "long_pauses": 0,
        }
    intervals = librosa.effects.split(y, top_db=30)
    voiced = sum((end - start) for start, end in intervals) / len(y)
    seg_durs = [ (end - start) / sr for start, end in intervals ] if len(intervals) else []
    avg_seg = float(np.mean(seg_durs)) if seg_durs else 0.0
    # Count long gaps (>700ms) between speech segments
    long_pauses = 0
    if len(intervals) > 1:
        for i in range(1, len(intervals)):
            prev_end = intervals[i-1][1]
            cur_start = intervals[i][0]
            gap_sec = (cur_start - prev_end) / sr
            if gap_sec >= 0.7:
                long_pauses += 1
    return {
        "duration_sec": float(duration),
        "voiced_ratio": float(voiced),
        "num_segments": int(len(intervals)),
        "avg_segment_sec": avg_seg,
        "long_pauses": int(long_pauses),
    }

# ---------- Optional SSL features (FULL mode) ----------
_SSL_MODEL = None
_SSL_MODEL_NAME = None

def get_ssl_embedding(y_16k: np.ndarray):
    # Compute an SSL embedding variance signal. Skipped in SAFE mode.
    global _SSL_MODEL, _SSL_MODEL_NAME
    if SAFE_MODE:
        return None, None, "SAFE_MODE_DISABLED_SSL"
    import torch
    import torchaudio
    waveform = torch.tensor(y_16k, dtype=torch.float32).unsqueeze(0)  # (1, T)
    if _SSL_MODEL is None:
        try:
            bundle = torchaudio.pipelines.WAV2VEC2_BASE
            _SSL_MODEL = bundle.get_model().eval()
            _SSL_MODEL_NAME = "WAV2VEC2_BASE"
        except Exception:
            bundle = torchaudio.pipelines.HUBERT_BASE
            _SSL_MODEL = bundle.get_model().eval()
            _SSL_MODEL_NAME = "HUBERT_BASE"
    model = _SSL_MODEL
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
    var = feats.var(dim=0).mean().item()
    emb = feats.mean(dim=0).cpu().numpy()
    return emb, float(var), _SSL_MODEL_NAME

# ---------- Optional ASR (cloud or local) ----------
def _get_asr_mode():
    # Decide which ASR provider to use: openai / local / none
    use_openai = os.getenv("USE_OPENAI_ASR", "0") == "1" and os.getenv("OPENAI_API_KEY")
    if use_openai:
        return "openai"
    if not SAFE_MODE:
        return "local"  # try torchaudio wav2vec2 CTC (greedy)
    return "none"

_ASR_MODEL = None
_ASR_BUNDLE = None
def _local_asr_transcribe(y_16k: np.ndarray) -> Optional[str]:
    # Greedy CTC decode using torchaudio's wav2vec2 ASR base. Returns lowercased transcript or None.
    global _ASR_MODEL, _ASR_BUNDLE
    try:
        import torch, torchaudio
        if _ASR_MODEL is None:
            _ASR_BUNDLE = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
            _ASR_MODEL = _ASR_BUNDLE.get_model().eval()
        waveform = torch.tensor(y_16k, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            emissions, _ = _ASR_MODEL(waveform)
            indices = emissions[0].argmax(dim=-1).tolist()
        labels = _ASR_BUNDLE.get_labels()
        blank_idx = 0  # by convention in this bundle
        tokens, last = [], None
        for i in indices:
            if i != blank_idx and i != last:
                tokens.append(labels[i])
            last = i
        text = "".join(tokens).replace("|", " ").strip().lower()
        return text or None
    except Exception:
        return None

def _openai_asr_transcribe(tmp_path: str) -> Optional[str]:
    # Use OpenAI Whisper API if configured. Returns lowercased transcript or None.
    try:
        from openai import OpenAI
        client = OpenAI()
        model = os.getenv("OPENAI_ASR_MODEL", "whisper-1")
        with open(tmp_path, "rb") as f:
            resp = client.audio.transcriptions.create(model=model, file=f)
        text = getattr(resp, "text", None) or (resp.get("text") if isinstance(resp, dict) else None)
        return (str(text).strip().lower()) if text else None
    except Exception:
        return None

def get_transcript(y_16k: np.ndarray, tmp_path: str) -> Tuple[Optional[str], str]:
    # Return (transcript, provider).
    mode = _get_asr_mode()
    if mode == "openai":
        t = _openai_asr_transcribe(tmp_path)
        if t: return t, "openai"
        if not SAFE_MODE:
            t = _local_asr_transcribe(y_16k)
            if t: return t, "local-fallback"
        return None, "none"
    elif mode == "local":
        t = _local_asr_transcribe(y_16k)
        return (t, "local") if t else (None, "none")
    else:
        return None, "none"

# ---------- Hybrid features & scoring ----------
FILLERS = {"um","uh","er","eh","hmm","like","you","know","kind","of","sort","of"}

def _normalize(v, lo, hi):
    return max(0.0, min(1.0, (v - lo) / (hi - lo))) if hi > lo else 0.0

def text_features(transcript: str, duration_sec: float):
    if not transcript or duration_sec <= 0:
        return {"words": 0, "wpm": 0.0, "lexical_diversity": 0.0, "filler_ratio": 0.0}
    words = re.findall(r"[a-zA-Z']+", transcript.lower())
    total = len(words)
    uniq = len(set(words))
    fillers = sum(1 for w in words if w in FILLERS)
    minutes = duration_sec / 60.0 if duration_sec > 0 else 1.0
    wpm = total / minutes
    lex_div = (uniq / total) if total else 0.0
    filler_ratio = (fillers / total) if total else 0.0
    return {"words": total, "wpm": float(wpm), "lexical_diversity": float(lex_div), "filler_ratio": float(filler_ratio)}

def audio_score_component(metrics, ssl_var):
    vr = metrics["voiced_ratio"]
    seg = metrics["avg_segment_sec"]
    pauses = metrics["long_pauses"]
    vr_n = _normalize(vr, 0.20, 0.80)            # speak more than silence
    seg_n = _normalize(seg, 0.30, 2.00)          # longer coherent segments
    pause_n = 1.0 - _normalize(pauses, 1, 6)     # fewer long pauses is better
    ssl_n = 0.0 if ssl_var is None else _normalize(ssl_var, 0.05, 0.50)
    return float(0.4*vr_n + 0.2*seg_n + 0.2*pause_n + 0.2*ssl_n)

def text_score_component(tf):
    wpm_n = _normalize(tf["wpm"], 70, 170)
    lex_n = _normalize(tf["lexical_diversity"], 0.30, 0.65)
    filler_penalty = 1.0 - _normalize(tf["filler_ratio"], 0.02, 0.12)
    return float(0.5*wpm_n + 0.3*lex_n + 0.2*filler_penalty)

def map_score_to_level(score):
    if score < T_A2:   return "A1-A2 (provisional)"
    if score < T_B1:   return "A2-B1 (provisional)"
    if score < T_B2:   return "B1-B2 (provisional)"
    if score < T_C1:   return "B2-C1 (provisional)"
    return "C1-C2 (provisional)"

def hybrid_infer(y, metrics, ssl_var, transcript):
    a = audio_score_component(metrics, ssl_var)
    t = text_score_component(text_features(transcript, metrics["duration_sec"])) if transcript else 0.0
    score = HYBRID_W_AUDIO * a + HYBRID_W_TEXT * t
    level = map_score_to_level(score)
    return score, a, t, level

# ---------- Routes ----------
@app.get("/health")
async def health():
    mode = "SAFE" if SAFE_MODE else "FULL"
    prompts = load_prompts()
    levels = sorted(list(prompts.keys()))
    return {"status": "ok", "version": APP_VERSION, "mode": mode, "prompt_levels": levels}

@app.get("/config")
async def config():
    return {
        "mode": "SAFE" if SAFE_MODE else "FULL",
        "weights": {"audio": HYBRID_W_AUDIO, "text": HYBRID_W_TEXT},
        "thresholds": {"A2": T_A2, "B1": T_B1, "B2": T_B2, "C1": T_C1},
        "asr_mode": _get_asr_mode(),
    }

def _evaluate_core(tmp_path: str) -> dict:
    y, sr = load_audio_to_16k_mono(tmp_path)
    y = y / (np.max(np.abs(y)) + 1e-8)
    metrics = speech_activity_metrics(y, sr=sr)
    emb, ssl_var, ssl_name = get_ssl_embedding(y)
    transcript, asr_provider = get_transcript(y, tmp_path)
    score, a_comp, t_comp, level = hybrid_infer(y, metrics, ssl_var, transcript)
    return {
        "ok": True,
        "mode": "SAFE" if SAFE_MODE else "FULL",
        "ssl_model": ssl_name,
        "asr_provider": asr_provider,
        "duration_sec": metrics["duration_sec"],
        "voiced_ratio": metrics["voiced_ratio"],
        "segments": metrics["num_segments"],
        "avg_segment_sec": metrics["avg_segment_sec"],
        "long_pauses": metrics["long_pauses"],
        "embedding_norm": float(np.linalg.norm(emb)) if emb is not None else None,
        "feature_variance": ssl_var,
        "transcript": transcript,
        "scores": {"hybrid": score, "audio": a_comp, "text": t_comp},
        "provisional_level": level,
    }

@app.post("/evaluate")
async def evaluate(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm")):
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload audio (wav/mp3/m4a/flac/ogg/webm).")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[-1]) as tmp:
            data = await file.read()
            tmp.write(data)
            tmp_path = tmp.name
        result = _evaluate_core(tmp_path)
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

@app.post("/evaluate-bytes")
async def evaluate_bytes(request: Request):
    try:
        data = await request.body()
        if not data:
            raise HTTPException(status_code=400, detail="Empty request body")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        result = _evaluate_core(tmp_path)
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
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
