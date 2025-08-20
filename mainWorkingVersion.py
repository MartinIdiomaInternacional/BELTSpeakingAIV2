import os, json, tempfile, traceback, re
from pathlib import Path
from functools import lru_cache
from typing import Optional, Tuple

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import numpy as np
import librosa

APP_VERSION = "BELT Speaking Test API v0.9 (hybrid + relevance + feedback)"
SAFE_MODE = os.getenv("BELT_SAFE_MODE", "0") == "1"

# ----- Weights (override via env) -----
HYBRID_W_AUDIO = float(os.getenv("HYBRID_W_AUDIO", "0.5"))
HYBRID_W_TEXT  = float(os.getenv("HYBRID_W_TEXT",  "0.5"))
HYBRID_W_REL   = float(os.getenv("HYBRID_W_REL",   "0.2"))   # relevance weight

# ----- CEFR thresholds (0..1 -> bands) -----
T_A2 = float(os.getenv("T_A2", "0.25"))
T_B1 = float(os.getenv("T_B1", "0.40"))
T_B2 = float(os.getenv("T_B2", "0.60"))
T_C1 = float(os.getenv("T_C1", "0.75"))

# ----- ASR / Embeddings config -----
USE_OPENAI_ASR      = os.getenv("USE_OPENAI_ASR", "0") == "1"
OPENAI_ASR_MODEL    = os.getenv("OPENAI_ASR_MODEL", "whisper-1")
USE_OPENAI_EMBED    = os.getenv("USE_OPENAI_EMBED", "1") == "1"
OPENAI_EMBED_MODEL  = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

# ----- Optional LLM feedback (off by default) -----
USE_LLM_FEEDBACK        = os.getenv("USE_LLM_FEEDBACK", "0") == "1"
OPENAI_FEEDBACK_MODEL   = os.getenv("OPENAI_FEEDBACK_MODEL", "gpt-4o-mini")

# ---------- App ----------
app = FastAPI(title="BELT Speaking Test API", version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
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
    return y.astype(np.float32), 16000

def speech_activity_metrics(y: np.ndarray, sr: int = 16000):
    duration = len(y) / sr if len(y) else 0.0
    if len(y) == 0:
        return {"duration_sec": 0.0, "voiced_ratio": 0.0, "num_segments": 0,
                "avg_segment_sec": 0.0, "long_pauses": 0}
    intervals = librosa.effects.split(y, top_db=30)
    voiced = sum((end - start) for start, end in intervals) / len(y)
    seg_durs = [ (end - start) / sr for start, end in intervals ] if len(intervals) else []
    avg_seg = float(np.mean(seg_durs)) if seg_durs else 0.0
    long_pauses = 0
    if len(intervals) > 1:
        for i in range(1, len(intervals)):
            gap_sec = (intervals[i][0] - intervals[i-1][1]) / sr
            if gap_sec >= 0.7: long_pauses += 1
    return {"duration_sec": float(duration), "voiced_ratio": float(voiced),
            "num_segments": int(len(intervals)), "avg_segment_sec": avg_seg,
            "long_pauses": int(long_pauses)}

# ---------- Optional SSL features (FULL mode) ----------
_SSL_MODEL = None
_SSL_MODEL_NAME = None
def get_ssl_embedding(y_16k: np.ndarray):
    if SAFE_MODE:
        return None, None, "SAFE_MODE_DISABLED_SSL"
    import torch, torchaudio
    waveform = torch.tensor(y_16k, dtype=torch.float32).unsqueeze(0)
    global _SSL_MODEL, _SSL_MODEL_NAME
    if _SSL_MODEL is None:
        try:
            bundle = torchaudio.pipelines.WAV2VEC2_BASE
            _SSL_MODEL = bundle.get_model().eval(); _SSL_MODEL_NAME = "WAV2VEC2_BASE"
        except Exception:
            bundle = torchaudio.pipelines.HUBERT_BASE
            _SSL_MODEL = bundle.get_model().eval(); _SSL_MODEL_NAME = "HUBERT_BASE"
    model = _SSL_MODEL
    with torch.no_grad():
        try:
            feats = model.extract_features(waveform)[0][-1]
            if feats.dim() == 3: feats = feats.squeeze(0)
        except Exception:
            out = model(waveform); feats = (out[0] if isinstance(out, (tuple, list)) else out).squeeze(0)
    var = feats.var(dim=0).mean().item()
    emb = feats.mean(dim=0).cpu().numpy()
    return emb, float(var), _SSL_MODEL_NAME

# ---------- Optional ASR ----------
def _get_asr_mode():
    use_openai = USE_OPENAI_ASR and os.getenv("OPENAI_API_KEY")
    if use_openai: return "openai"
    if not SAFE_MODE: return "local"
    return "none"

_ASR_MODEL = None
_ASR_BUNDLE = None
def _local_asr_transcribe(y_16k: np.ndarray) -> Optional[str]:
    try:
        import torch, torchaudio
        global _ASR_MODEL, _ASR_BUNDLE
        if _ASR_MODEL is None:
            _ASR_BUNDLE = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
            _ASR_MODEL = _ASR_BUNDLE.get_model().eval()
        emissions, _ = _ASR_MODEL(torch.tensor(y_16k, dtype=torch.float32).unsqueeze(0))
        indices = emissions[0].argmax(dim=-1).tolist()
        labels = _ASR_BUNDLE.get_labels()
        blank_idx = 0
        tokens, last = [], None
        for i in indices:
            if i != blank_idx and i != last: tokens.append(labels[i])
            last = i
        return "".join(tokens).replace("|", " ").strip().lower() or None
    except Exception:
        return None

def _openai_asr_transcribe(tmp_path: str) -> Optional[str]:
    try:
        from openai import OpenAI
        client = OpenAI()
        with open(tmp_path, "rb") as f:
            resp = client.audio.transcriptions.create(model=OPENAI_ASR_MODEL, file=f)
        text = getattr(resp, "text", None) or (resp.get("text") if isinstance(resp, dict) else None)
        return (str(text).strip().lower()) if text else None
    except Exception:
        return None

def get_transcript(y_16k: np.ndarray, tmp_path: str) -> Tuple[Optional[str], str]:
    mode = _get_asr_mode()
    if mode == "openai":
        t = _openai_asr_transcribe(tmp_path)
        if t: return t, "openai"
        if not SAFE_MODE:
            t = _local_asr_transcribe(y_16k)
            if t: return t, "local-fallback"
        return None, "none"
    if mode == "local":
        t = _local_asr_transcribe(y_16k)
        return (t, "local") if t else (None, "none")
    return None, "none"

# ---------- Scoring ----------
FILLERS = {"um","uh","er","eh","hmm","like","you","know","kind","of","sort","of"}
def _normalize(v, lo, hi): return max(0.0, min(1.0, (v - lo) / (hi - lo))) if hi > lo else 0.0

def text_features(transcript: str, duration_sec: float):
    if not transcript or duration_sec <= 0:
        return {"words": 0, "wpm": 0.0, "lexical_diversity": 0.0, "filler_ratio": 0.0}
    words = re.findall(r"[a-zA-Z']+", transcript.lower())
    total = len(words); uniq = len(set(words))
    fillers = sum(1 for w in words if w in FILLERS)
    minutes = duration_sec / 60.0 if duration_sec > 0 else 1.0
    return {
        "words": total,
        "wpm": float(total / minutes),
        "lexical_diversity": float((uniq / total) if total else 0.0),
        "filler_ratio": float((fillers / total) if total else 0.0),
    }

def audio_score_component(m, ssl_var):
    vr_n   = _normalize(m["voiced_ratio"], 0.20, 0.80)
    seg_n  = _normalize(m["avg_segment_sec"], 0.30, 2.00)
    pause_n= 1.0 - _normalize(m["long_pauses"], 1, 6)
    ssl_n  = 0.0 if ssl_var is None else _normalize(ssl_var, 0.05, 0.50)
    return float(0.4*vr_n + 0.2*seg_n + 0.2*pause_n + 0.2*ssl_n)

def text_score_component(tf):
    wpm_n = _normalize(tf["wpm"], 70, 170)
    lex_n = _normalize(tf["lexical_diversity"], 0.30, 0.65)
    filler_penalty = 1.0 - _normalize(tf["filler_ratio"], 0.02, 0.12)
    return float(0.5*wpm_n + 0.3*lex_n + 0.2*filler_penalty)

# ----- Relevance / Coherence -----
STOP = {"the","a","an","to","of","in","on","for","with","and","or","but","is","are","was","were","be","am","as","at","by",
        "it","this","that","i","you","he","she","they","we","my","your","our"}
def _tokens(t: str): return {w for w in re.findall(r"[a-zA-Z']+", (t or "").lower()) if w not in STOP}

def relevance_heuristic(question: str, transcript: str) -> float:
    q = _tokens(question); a = _tokens(transcript)
    if not q or not a: return 0.0
    inter = len(q & a); union = len(q | a)
    jacc = inter / union; recall = inter / len(q)
    return float(0.6*jacc + 0.4*recall)

def relevance_openai(question: str, transcript: str) -> float:
    try:
        from openai import OpenAI
        client = OpenAI()
        m = OPENAI_EMBED_MODEL
        qv = client.embeddings.create(model=m, input=question).data[0].embedding
        av = client.embeddings.create(model=m, input=transcript).data[0].embedding
        import math
        dot = sum(x*y for x,y in zip(qv, av))
        nq = math.sqrt(sum(x*x for x in qv)); na = math.sqrt(sum(x*x for x in av))
        return float(dot/(nq*na + 1e-9))
    except Exception:
        return relevance_heuristic(question, transcript)

def map_score_to_level(score):
    if score < T_A2:   return "A1-A2 (provisional)"
    if score < T_B1:   return "A2-B1 (provisional)"
    if score < T_B2:   return "B1-B2 (provisional)"
    if score < T_C1:   return "B2-C1 (provisional)"
    return "C1-C2 (provisional)"

# ----- Feedback generation -----
def rule_based_feedback(m, tf, rel, transcript, question):
    strengths, areas, tips = [], [], []
    # Duration / volume proxy
    if m["duration_sec"] >= 12 and m["voiced_ratio"] >= 0.55:
        strengths.append("Good amount of speaking without excessive silence.")
    if m["avg_segment_sec"] >= 1.2:
        strengths.append("Utterances formed into coherent multi-word segments.")
    if tf["filler_ratio"] <= 0.03 and tf["words"] >= 40:
        strengths.append("Low filler usage and adequate length.")
    if tf["lexical_diversity"] >= 0.50:
        strengths.append("Varied vocabulary for the level.")
    if tf["wpm"] >= 110 and tf["wpm"] <= 180:
        strengths.append("Natural speaking pace.")

    # Areas for improvement
    if m["duration_sec"] < 8:
        areas.append("Answer length was short; expand with 2–3 supporting details.")
        tips.append("Add an example and a brief explanation to each main point.")
    if m["voiced_ratio"] < 0.35 or m["long_pauses"] >= 3:
        areas.append("Long silences or many pauses reduced fluency.")
        tips.append("Plan your first sentence before speaking; keep ideas flowing.")
    if m["avg_segment_sec"] < 0.6:
        areas.append("Very short phrases made the answer sound choppy.")
        tips.append("Link sentences with connectors (because, however, for example).")
    if tf["wpm"] < 90:
        areas.append("Speaking pace was slow.")
        tips.append("Practice answering within 20–40 seconds to keep momentum.")
    if tf["wpm"] > 190:
        areas.append("Speaking pace was too fast for clear articulation.")
        tips.append("Pause briefly between ideas and articulate key words.")
    if tf["lexical_diversity"] < 0.38:
        areas.append("Limited lexical variety.")
        tips.append("Paraphrase: replace repeated common words with synonyms.")
    if tf["filler_ratio"] > 0.08:
        areas.append("High filler-word rate (um, uh, like).")
        tips.append("Pause silently instead of using fillers; slow down slightly.")
    if rel is not None and rel < 0.40 and transcript:
        areas.append("Answer was partly off-topic or didn’t address the prompt directly.")
        tips.append("Start by restating the question in your first sentence.")
    if transcript and len(re.findall(r"[a-zA-Z']+", transcript)) < 30:
        areas.append("Response was very brief, limiting the evidence of ability.")
        tips.append("Aim for ~80–120 words with clear beginning, middle, and end.")

    # Keep them concise
    strengths = strengths[:5]; areas = areas[:6]; tips = tips[:6]
    return {"mode": "rule", "strengths": strengths, "areas": areas, "tips": tips}

def llm_feedback(question, transcript, tf, m, rel) -> Optional[dict]:
    if not (USE_LLM_FEEDBACK and os.getenv("OPENAI_API_KEY") and transcript):
        return None
    try:
        from openai import OpenAI
        client = OpenAI()
        sys = "You are a CEFR speaking assessor. Give concise, actionable feedback (bullets). Avoid praise fluff."
        user = {
            "question": question, "transcript": transcript,
            "metrics": m, "text_features": tf, "relevance": rel
        }
        msg = client.chat.completions.create(
            model=OPENAI_FEEDBACK_MODEL,
            messages=[{"role":"system","content":sys},
                      {"role":"user","content":json.dumps(user)}],
            temperature=0.2,
            max_tokens=300
        )
        content = msg.choices[0].message.content.strip()
        # Simple split into three sections if model formats lines with bullets
        return {"mode":"openai", "summary": content}
    except Exception:
        return None

# ----- Hybrid inference -----
def hybrid_infer(m, ssl_var, transcript, question=None):
    a = audio_score_component(m, ssl_var)
    tf = text_features(transcript, m["duration_sec"]) if transcript else {"words":0,"wpm":0,"lexical_diversity":0,"filler_ratio":0}
    t = text_score_component(tf) if transcript else 0.0
    r = 0.0
    if transcript and question:
        r = relevance_openai(question, transcript) if (USE_OPENAI_EMBED and os.getenv("OPENAI_API_KEY")) else relevance_heuristic(question, transcript)
    w_sum = HYBRID_W_AUDIO + HYBRID_W_TEXT + HYBRID_W_REL
    score = (HYBRID_W_AUDIO*a + HYBRID_W_TEXT*t + HYBRID_W_REL*r) / (w_sum if w_sum>0 else 1.0)
    level = map_score_to_level(score)
    return score, a, t, r, tf, level

# ---------- Routes ----------
@app.get("/health")
async def health():
    mode = "SAFE" if SAFE_MODE else "FULL"
    prompts = load_prompts()
    levels = sorted(list(prompts.keys()))
    return {"status":"ok","version":APP_VERSION,"mode":mode,"prompt_levels":levels}

@app.get("/config")
async def config():
    return {
        "mode": "SAFE" if SAFE_MODE else "FULL",
        "weights": {"audio": HYBRID_W_AUDIO, "text": HYBRID_W_TEXT, "relevance": HYBRID_W_REL},
        "thresholds": {"A2": T_A2, "B1": T_B1, "B2": T_B2, "C1": T_C1},
        "asr_mode": _get_asr_mode(),
        "embed_mode": "openai" if (USE_OPENAI_EMBED and os.getenv("OPENAI_API_KEY")) else "heuristic",
        "feedback_mode": "openai" if (USE_LLM_FEEDBACK and os.getenv("OPENAI_API_KEY")) else "rule"
    }

def _evaluate_core(tmp_path: str, question: Optional[str] = None) -> dict:
    y, sr = load_audio_to_16k_mono(tmp_path)
    y = y / (np.max(np.abs(y)) + 1e-8)
    m = speech_activity_metrics(y, sr=sr)
    emb, ssl_var, ssl_name = get_ssl_embedding(y)
    transcript, asr_provider = get_transcript(y, tmp_path)
    score, a_comp, t_comp, rel, tf, level = hybrid_infer(m, ssl_var, transcript, question)

    fb = rule_based_feedback(m, tf, rel, transcript, question)
    llm_fb = llm_feedback(question, transcript, tf, m, rel)
    if llm_fb: fb = {**fb, **llm_fb}

    return {
        "ok": True,
        "mode": "SAFE" if SAFE_MODE else "FULL",
        "ssl_model": ssl_name,
        "asr_provider": asr_provider,
        "question_echo": question,
        "duration_sec": m["duration_sec"],
        "voiced_ratio": m["voiced_ratio"],
        "segments": m["num_segments"],
        "avg_segment_sec": m["avg_segment_sec"],
        "long_pauses": m["long_pauses"],
        "feature_variance": ssl_var,
        "transcript": transcript,
        "scores": {"hybrid": score, "audio": a_comp, "text": t_comp, "relevance": rel,
                   "wpm": tf["wpm"], "lexical_diversity": tf["lexical_diversity"], "filler_ratio": tf["filler_ratio"]},
        "provisional_level": level,
        "feedback": fb
    }

@app.post("/evaluate")
async def evaluate(file: UploadFile = File(...), question: Optional[str] = Form(None)):
    if not file.filename.lower().endswith((".wav",".mp3",".m4a",".flac",".ogg",".webm")):
        raise HTTPException(status_code=400, detail="Unsupported file type. Upload wav/mp3/m4a/flac/ogg/webm.")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[-1]) as tmp:
            tmp.write(await file.read()); tmp_path = tmp.name
        return JSONResponse(content=_evaluate_core(tmp_path, question))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}\n{traceback.format_exc()}")
    finally:
        try:
            if 'tmp_path' in locals() and os.path.exists(tmp_path): os.unlink(tmp_path)
        except Exception: pass

@app.post("/evaluate-bytes")
async def evaluate_bytes(request: Request):
    try:
        data = await request.body()
        if not data: raise HTTPException(status_code=400, detail="Empty request body")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tmp:
            tmp.write(data); tmp_path = tmp.name
        question = request.headers.get("x-question")
        return JSONResponse(content=_evaluate_core(tmp_path, question))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}\n{traceback.format_exc()}")
    finally:
        try:
            if 'tmp_path' in locals() and os.path.exists(tmp_path): os.unlink(tmp_path)
        except Exception: pass

if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT","8000")), reload=False)
