# -*- coding: utf-8 -*-
"""
BELT Speaking Evaluator (FastAPI) - v2.5

What is new in v2.5:
- /warmup-check endpoint: quick mic check that runs VAD and SNR estimate
- VAD + SNR analysis reused in /submit-response; payload now includes:
  speech_ratio (0..1), snr (dB), and a simple confidence (0..1)
- All prior features preserved:
  * Adaptive CEFR (A1..C2) with one retry per level
  * Session-aware prompts with prompt_id/question logging
  * Whisper transcription (OpenAI) and CEFR scoring (LLM + fallback)
  * Per-turn recommendations, session recommendations
  * Native-language feedback translation via LLM
  * Static UI from /web and health endpoints

Notes:
- Uses ffmpeg for format conversion.
- Uses webrtcvad (pure-Python wrapper on WebRTC VAD) for speech detection.
- Uses soundfile (libsndfile) + numpy to read wav and compute SNR.
"""

import os
import io
import json
import uuid
import time
import math
import shutil
import logging
import tempfile
import subprocess
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("belt-service")

app = FastAPI(title="BELT Speaking Evaluator", version="2.5")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

WEB_ROOT = os.path.join(os.getcwd(), "web")
if os.path.isdir(WEB_ROOT):
    app.mount("/static", StaticFiles(directory=os.path.join(WEB_ROOT, "static")), name="static")

CEFR_ORDER = ["A1", "A2", "B1", "B1+", "B2", "B2+", "C1", "C2"]
CEFR_CATEGORIES = ["fluency", "grammar", "vocabulary", "pronunciation", "coherence"]

ASR_BACKEND = os.getenv("ASR_BACKEND", "openai").strip().lower()
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-1").strip()
RUBRIC_MODEL = os.getenv("RUBRIC_MODEL", "gpt-4o-mini").strip()

PASS_AVG_THRESHOLD = float(os.getenv("PASS_AVG_THRESHOLD", "0.70"))
PASS_MIN_THRESHOLD  = float(os.getenv("PASS_MIN_THRESHOLD",  "0.60"))
RECORD_SECONDS      = int(float(os.getenv("RECORD_SECONDS", "60")))
DEBUG_RETURN_TRANSCRIPT = os.getenv("DEBUG_RETURN_TRANSCRIPT", "0").strip() in ("1","true","True")

# Warmup OK thresholds
MIN_SPEECH_RATIO = float(os.getenv("MIN_SPEECH_RATIO", "0.50"))  # at least half of the sample is speech
MIN_SNR_DB       = float(os.getenv("MIN_SNR_DB", "15.0"))        # at least 15 dB SNR typical

# Log OpenAI SDK version if available
try:
    import openai as _oa
    _ver = getattr(_oa, "__version__", "unknown")
    log.info(f"OpenAI SDK version: {_ver}")
except Exception:
    log.info("OpenAI SDK not importable at startup")

# Optional deps (webrtcvad, soundfile, numpy)
try:
    import webrtcvad  # type: ignore
except Exception:
    webrtcvad = None  # graceful fallback

try:
    import soundfile as sf  # type: ignore
    import numpy as np      # type: ignore
except Exception:
    sf = None
    np = None

SESSIONS: Dict[str, Dict[str, Any]] = {}

# --------------------------
# Prompt management
# --------------------------
def load_prompts_for_level(level: str) -> List[Dict[str, Any]]:
    path = os.path.join(WEB_ROOT, "prompts", f"{level}.json")
    items: List[Dict[str, Any]] = []
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for i, it in enumerate(data):
                    if isinstance(it, dict):
                        pid = int(it.get("id", i))
                        txt = str(it.get("instructions") or it.get("question") or "").strip()
                    else:
                        pid = i
                        txt = str(it or "").strip()
                    if txt:
                        items.append({"id": pid, "instructions": txt})
        except Exception as e:
            log.warning(f"Failed to read prompts for {level}: {e}")
    if items:
        return items
    # minimal fallback set
    fallback: Dict[str, List[str]] = {
        "A1": ["Introduce yourself.", "Describe your breakfast.", "Talk about your family."],
        "A2": ["Describe your daily routine.", "Explain your last weekend plans.", "Favorite place in your town."],
        "B1": ["Describe a memorable trip.", "An important goal this year?", "A skill you are learning and why."],
        "B1+": ["Study alone or in a group? Explain.", "A challenge you faced and overcame.", "Is social media helpful or harmful?"],
        "B2": ["Ban junk-food ads? Why or why not?", "How to reduce traffic congestion?", "Pros and cons of remote work."],
        "B2+": ["A technology you cannot live without.", "Require community service to graduate?", "Corporate climate responsibilities?"],
        "C1": ["Is learning history important?", "Ethics of data collection?", "Balance free speech and public safety?"],
        "C2": ["Impact of social media on democracy?", "How should AI be regulated?", "Are standardized tests fair?"],
    }
    return [{"id": i, "instructions": txt} for i, txt in enumerate(fallback.get(level, []))]

def choose_prompt(level: str, session_id: str) -> Dict[str, Any]:
    items = load_prompts_for_level(level)
    served = SESSIONS[session_id]["served_prompts"].setdefault(level, set())
    for item in items:
        if item["id"] not in served:
            served.add(item["id"])
            return item
    SESSIONS[session_id]["served_prompts"][level] = set()
    first = items[0] if items else {"id": 0, "instructions": "Speak for about 60 seconds."}
    SESSIONS[session_id]["served_prompts"][level].add(first["id"])
    return first

# --------------------------
# Audio utils
# --------------------------
def ensure_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception:
        raise HTTPException(500, "ffmpeg not found on server")

def to_wav_16k_mono(in_path: str, out_path: str):
    ensure_ffmpeg()
    cmd = ["ffmpeg", "-y", "-i", in_path, "-ac", "1", "-ar", "16000", "-f", "wav", out_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def read_wav_16k_mono(path: str) -> Tuple[Optional["np.ndarray"], int]:
    """Return (float32 waveform in [-1,1], sample_rate) or (None,0) on failure."""
    if sf is None or np is None:
        return None, 0
    try:
        data, sr = sf.read(path, dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data[:, 0]
        return data, sr
    except Exception as e:
        log.warning(f"read_wav failed: {e}")
        return None, 0

def run_vad_and_snr(path: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute speech_ratio (0..1) and SNR (dB) using webrtcvad and energy stats.
    Returns (speech_ratio, snr_db). If dependencies missing, returns (None, None).
    """
    if webrtcvad is None or sf is None or np is None:
        return None, None

    x, sr = read_wav_16k_mono(path)
    if x is None or sr != 16000 or len(x) < sr // 2:
        return None, None

    # Prepare 30ms frames for WebRTC VAD
    frame_ms = 30
    frame_len = int(sr * frame_ms / 1000)  # 480 samples at 16k
    if frame_len <= 0:
        return None, None

    # Convert to 16-bit PCM bytes for VAD
    pcm16 = (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()

    vad = webrtcvad.Vad(2)  # 0..3 (aggressiveness)
    num_frames = len(pcm16) // (frame_len * 2)  # 2 bytes per sample
    if num_frames <= 0:
        return None, None

    speech_flags: List[bool] = []
    for i in range(num_frames):
        start = i * frame_len * 2
        end = start + frame_len * 2
        frame_bytes = pcm16[start:end]
        is_speech = False
        try:
            is_speech = vad.is_speech(frame_bytes, sr)
        except Exception:
            # if webrtcvad raises, mark non-speech for safety
            is_speech = False
        speech_flags.append(is_speech)

    speech_ratio = float(sum(1 for f in speech_flags if f)) / float(len(speech_flags) or 1)

    # SNR estimate: power in speech frames vs power in non-speech frames
    # Compute energies directly from float waveform per frame.
    speech_energy = []
    noise_energy = []
    for i in range(num_frames):
        start = i * frame_len
        end = start + frame_len
        seg = x[start:end]
        if seg.size == 0:
            continue
        p = float(np.mean(seg * seg))
        if speech_flags[i]:
            speech_energy.append(p)
        else:
            noise_energy.append(p)

    if not speech_energy:
        # No speech detected
        return speech_ratio, None

    Ps = float(np.mean(speech_energy))
    if noise_energy:
        Pn = float(np.mean(noise_energy))
    else:
        # No noise frames; use quiet default
        Pn = max(1e-8, 0.1 * Ps)

    snr_db = 10.0 * math.log10(max(1e-9, Ps / max(1e-9, Pn)))
    return speech_ratio, snr_db

# --------------------------
# ASR
# --------------------------
async def transcribe(openai_client, wav_path: str) -> str:
    if ASR_BACKEND != "openai" or openai_client is None:
        return ""
    try:
        with open(wav_path, "rb") as f:
            resp = openai_client.audio.transcriptions.create(
                model=WHISPER_MODEL, file=f, response_format="text", temperature=0
            )
        if isinstance(resp, str):
            return resp.strip()
        return getattr(resp, "text", "") or ""
    except Exception as e:
        log.warning(f"Transcription failed: {e}")
        return ""

# --------------------------
# Scoring and feedback
# --------------------------
RULEBOOK: Dict[str, List[str]] = {
    "fluency": ["Avoid long pauses.", "Practice transitions: first, next, finally.", "Shadow native clips for rhythm."],
    "grammar": ["Watch subject-verb agreement.", "Keep tenses consistent.", "Use linking clauses correctly."],
    "vocabulary": ["Use topic-specific words.", "Paraphrase to avoid repetition.", "Add collocations you know."],
    "pronunciation": ["Stress content words.", "Differentiate similar vowels.", "Record and compare with natives."],
    "coherence": ["Intro -> 2-3 points -> close.", "Use signposts to connect ideas.", "Avoid run-on sentences."],
}

def weak_categories(scores: Dict[str, float]) -> List[str]:
    items = []
    for k in CEFR_CATEGORIES:
        v = float(scores.get(k, 0.0))
        tier = 2 if v < PASS_MIN_THRESHOLD else (1 if v < PASS_AVG_THRESHOLD else 0)
        if tier > 0:
            items.append((k, tier, v))
    items.sort(key=lambda x: (-x[1], x[2]))
    return [k for (k, _, __) in items]

def rule_based_tips(scores: Dict[str, float], max_tips: int = 3) -> List[str]:
    tips = []
    for cat in weak_categories(scores):
        for tip in RULEBOOK.get(cat, []):
            tips.append(f"{cat.capitalize()}: {tip}")
            if len(tips) >= max_tips:
                return tips
    return tips

def try_llm_scores_and_tips(openai_client, transcript: str, level: str) -> Optional[Dict[str, Any]]:
    if not RUBRIC_MODEL or not transcript or openai_client is None:
        return None
    try:
        sys = ("You are a CEFR speaking evaluator. Return strict JSON: "
               "{\"scores\":{\"fluency\":0..1,\"grammar\":0..1,\"vocabulary\":0..1,\"pronunciation\":0..1,\"coherence\":0..1},"
               "\"tips\":[\"...\",\"...\"]}")
        user = f"Assess the following transcript at target level {level}.\nTranscript:\n{transcript}\nJSON only."
        resp = openai_client.chat.completions.create(
            model=RUBRIC_MODEL,
            messages=[{"role":"system","content":sys},{"role":"user","content":user}],
            temperature=0.2,
            max_tokens=300
        )
        content = resp.choices[0].message.content.strip()
        return json.loads(content)
    except Exception as e:
        log.warning(f"LLM rubric failed: {e}")
        return None

def clamp01(x: float) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0

def compute_scores(openai_client, transcript: str, level: str) -> Dict[str, Any]:
    result = {"scores": {k:0.0 for k in CEFR_CATEGORIES}, "average": 0.0, "tips_ai": []}
    if not transcript.strip():
        return result
    data = try_llm_scores_and_tips(openai_client, transcript, level)
    if data and isinstance(data, dict):
        s = data.get("scores", {}) or {}
        scores = {k: clamp01(s.get(k, 0.0)) for k in CEFR_CATEGORIES}
        avg = sum(scores.values()) / len(CEFR_CATEGORIES)
        tips_ai = [t for t in (data.get("tips") or []) if isinstance(t, str) and t.strip()]
        result.update({"scores": scores, "average": avg, "tips_ai": tips_ai})
        return result
    # heuristic fallback (length-based)
    length = len(transcript.split())
    rough = min(1.0, length / 120.0)
    scores = {k: rough for k in CEFR_CATEGORIES}
    result.update({"scores": scores, "average": rough, "tips_ai": []})
    return result

def finalize_session_recommendations(history: List[Dict[str, Any]]) -> List[str]:
    if not history:
        return []
    cnt = Counter()
    for turn in history:
        s = turn.get("scores", {})
        for cat in CEFR_CATEGORIES:
            if float(s.get(cat, 0.0)) < 0.7:
                cnt[cat] += 1
    if not cnt:
        return ["Great overall balance. Keep practicing varied topics."]
    top2 = [c for c,_ in cnt.most_common(2)]
    tips = []
    for cat in top2:
        basics = RULEBOOK.get(cat, [])[:2]
        for b in basics:
            tips.append(f"Focus on {cat}: {b}")
    tips.append("Daily routine: plan 3 points, record, listen back, re-record.")
    return tips[:5]

def decision_from_scores(scores: Dict[str, float], avg: float) -> str:
    if avg >= PASS_AVG_THRESHOLD and all(scores.get(k,0.0) >= PASS_MIN_THRESHOLD for k in CEFR_CATEGORIES):
        return "advance"
    return "stop"

def estimate_confidence(avg: float, snr_db: Optional[float], speech_ratio: Optional[float]) -> float:
    """
    Simple confidence proxy in 0..1: blend score strength, SNR, and speech coverage.
    This is intentionally basic; can be replaced by an ensemble later.
    """
    s = clamp01(avg)
    q = 0.0
    if snr_db is not None:
        # Map 5..30 dB roughly to 0..1
        q = (snr_db - 5.0) / 25.0
    r = speech_ratio if (speech_ratio is not None) else 0.5
    out = 0.5 * s + 0.25 * clamp01(q) + 0.25 * clamp01(r)
    return clamp01(out)

# --------------------------
# Translation (native-language feedback)
# --------------------------
def translate_list(openai_client, texts: List[str], target_language: str) -> List[str]:
    if not texts or not target_language or openai_client is None:
        return texts
    try:
        sys = "You are a precise translator. Return only the translations as a JSON array of strings, order preserved."
        user = f"Translate each of the following strings into {target_language}. Return JSON array only.\n" + json.dumps(texts, ensure_ascii=False)
        resp = openai_client.chat.completions.create(
            model=RUBRIC_MODEL,
            messages=[{"role":"system","content":sys},{"role":"user","content":user}],
            temperature=0.0,
            max_tokens=400
        )
        content = resp.choices[0].message.content.strip()
        out = json.loads(content)
        if isinstance(out, list) and all(isinstance(x,str) for x in out):
            return out
        return texts
    except Exception as e:
        log.warning(f"Translation failed: {e}")
        return texts

# --------------------------
# HTTP endpoints
# --------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = os.path.join(WEB_ROOT, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>BELT Speaking Evaluator</h1>")

@app.get("/config")
async def get_config():
    return {
        "ASR_BACKEND": ASR_BACKEND,
        "WHISPER_MODEL": WHISPER_MODEL,
        "RUBRIC_MODEL": RUBRIC_MODEL,
        "PASS_AVG_THRESHOLD": PASS_AVG_THRESHOLD,
        "PASS_MIN_THRESHOLD": PASS_MIN_THRESHOLD,
        "RECORD_SECONDS": RECORD_SECONDS,
        "DEBUG_RETURN_TRANSCRIPT": DEBUG_RETURN_TRANSCRIPT,
        # client could later read MIN_SPEECH_RATIO, MIN_SNR_DB if desired
    }

@app.post("/start-session")
async def start_session(level: str = Form("A1"), native_language: str = Form("Spanish")):
    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = {
        "level": level,
        "history": [],
        "served_prompts": {},
        "final_level": None,
        "native_language": (native_language or "Spanish").strip(),
    }
    item = choose_prompt(level, session_id)
    log.info(json.dumps({
        "ts": int(time.time()*1000),
        "msg": "start-session",
        "session_id": session_id,
        "level": level,
        "native_language": SESSIONS[session_id]["native_language"],
    }))
    return {"session_id": session_id, "level": level, "prompt": item.get("instructions"), "prompt_id": item.get("id")}

@app.get("/prompts/{level}")
async def get_prompt(level: str, session_id: str):
    if session_id not in SESSIONS:
        raise HTTPException(400, "Unknown session_id")
    item = choose_prompt(level, session_id)
    return {"level": level, "instructions": item.get("instructions"), "prompt_id": item.get("id")}

@app.post("/warmup-check")
async def warmup_check(file: UploadFile = File(...)):
    """
    Accepts a short audio clip (e.g., 3-6 seconds).
    Converts to 16 kHz mono WAV.
    Runs VAD and SNR estimate.
    Returns JSON with metrics and ok flag.
    """
    with tempfile.TemporaryDirectory() as td:
        src_path = os.path.join(td, file.filename or "warmup.bin")
        with open(src_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        wav_path = os.path.join(td, "warmup.wav")
        try:
            to_wav_16k_mono(src_path, wav_path)
        except Exception as e:
            log.warning(f"ffmpeg convert failed in warmup: {e}")
            raise HTTPException(400, "Audio format not supported or ffmpeg error.")
        speech_ratio, snr_db = run_vad_and_snr(wav_path)

    deps_ok = (webrtcvad is not None and sf is not None and np is not None)
    if not deps_ok:
        # If we cannot compute metrics, do not block the user.
        return {"ok": True, "speech_ratio": None, "snr": None, "note": "VAD/SNR unavailable on server."}

    ok = True
    if speech_ratio is not None and speech_ratio < MIN_SPEECH_RATIO:
        ok = False
    if snr_db is not None and snr_db < MIN_SNR_DB:
        ok = False

    return {
        "ok": ok,
        "speech_ratio": speech_ratio,
        "snr": snr_db,
        "thresholds": {"min_speech_ratio": MIN_SPEECH_RATIO, "min_snr_db": MIN_SNR_DB},
    }

@app.post("/submit-response")
async def submit_response(
    session_id: str = Form(...),
    file: UploadFile = File(...),
    prompt_id: Optional[str] = Form(None),
    question: Optional[str] = Form(None),
):
    if session_id not in SESSIONS:
        raise HTTPException(400, "Unknown session_id")
    state = SESSIONS[session_id]
    level = state.get("level", "A1")
    native_language = state.get("native_language", "Spanish")
    attempt = sum(1 for t in state["history"] if t.get("level") == level) + 1

    with tempfile.TemporaryDirectory() as td:
        src_path = os.path.join(td, file.filename or "answer.bin")
        with open(src_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        size_bytes = os.path.getsize(src_path)
        wav_path = os.path.join(td, "answer.wav")
        try:
            to_wav_16k_mono(src_path, wav_path)
        except Exception as e:
            log.warning(f"ffmpeg convert failed: {e}")
            raise HTTPException(400, "Audio format not supported or ffmpeg error.")

        # Mic quality metrics on submitted audio
        speech_ratio, snr_db = run_vad_and_snr(wav_path)

        # Init OpenAI client for ASR and scoring
        openai_client = None
        if ASR_BACKEND == "openai":
            try:
                from openai import OpenAI
                openai_client = OpenAI()
            except Exception as e:
                log.warning(f"OpenAI init failed: {e}")
                openai_client = None

        transcript = await transcribe(openai_client, wav_path)

    # Resolve prompt text for logging
    used_prompt_text, used_prompt_id = None, None
    if prompt_id is not None:
        try:
            used_prompt_id = int(prompt_id)
        except Exception:
            used_prompt_id = None
        for p in load_prompts_for_level(level):
            if p["id"] == used_prompt_id:
                used_prompt_text = p.get("instructions")
                break
    if not used_prompt_text:
        used_prompt_text = question or "Speak for about 60 seconds."

    # Scoring
    pack = compute_scores(openai_client, transcript or "", level)
    scores, avg = pack["scores"], pack["average"]
    decision = decision_from_scores(scores, avg)
    recs = rule_based_tips(scores, max_tips=3)
    if not recs:
        recs = ["Plan 3 key points before speaking.", "Link sentences with signposts (first, next, finally)."]

    # Translate per-turn recommendations if needed
    if native_language and native_language.lower() not in ("english", "en"):
        recs = translate_list(openai_client, recs, native_language)

    confidence = estimate_confidence(avg, snr_db, speech_ratio)

    turn = {
        "level": level,
        "question": used_prompt_text,
        "prompt_id": used_prompt_id or prompt_id,
        "transcript": transcript,
        "scores": scores,
        "average": avg,
        "decision": decision,
        "attempt": attempt,
        "recommendations": recs,
        "native_language": native_language,
        "speech_ratio": speech_ratio,
        "snr": snr_db,
        "confidence": confidence,
    }
    state["history"].append(turn)
    if decision == "stop":
        state["final_level"] = level

    log.info(json.dumps({
        "ts": int(time.time()*1000), "name": "belt-service", "msg": "submit-response",
        "session_id": session_id, "level": level, "attempt": attempt,
        "upload_kb": round(size_bytes/1024, 1), "transcript_len": len(transcript or ""),
        "avg": avg, "decision": decision, "native_language": native_language,
        "speech_ratio": speech_ratio, "snr": snr_db, "confidence": confidence
    }))

    payload: Dict[str, Any] = {
        "scores": scores,
        "average": avg,
        "decision": decision,
        "attempt": attempt,
        "recommendations": recs,
        "speech_ratio": speech_ratio,
        "snr": snr_db,
        "confidence": confidence,
    }
    if DEBUG_RETURN_TRANSCRIPT:
        payload["transcript"] = transcript or ""

    if decision == "advance":
        i = CEFR_ORDER.index(level)
        if i + 1 < len(CEFR_ORDER):
            nxt = CEFR_ORDER[i + 1]
            state["level"] = nxt
            item = choose_prompt(nxt, session_id)
            payload.update({
                "next_level": nxt,
                "next_prompt": item.get("instructions"),
                "next_prompt_id": item.get("id")
            })
    else:
        if attempt >= 2:
            sess_recs = finalize_session_recommendations(state["history"])
            if native_language and native_language.lower() not in ("english", "en"):
                sess_recs = translate_list(openai_client, sess_recs, native_language)
            payload["session_recommendations"] = sess_recs

    return JSONResponse(payload)

@app.get("/report/{session_id}")
async def report(session_id: str):
    if session_id not in SESSIONS:
        raise HTTPException(404, "Unknown session_id")
    state = SESSIONS[session_id]
    native_language = state.get("native_language", "Spanish")
    history = state.get("history", [])
    sess_recs = finalize_session_recommendations(history)
    try:
        from openai import OpenAI
        client = OpenAI()
    except Exception:
        client = None
    if native_language and native_language.lower() not in ("english","en"):
        sess_recs = translate_list(client, sess_recs, native_language)
    return {
        "session_id": session_id,
        "final_level": state.get("final_level"),
        "native_language": native_language,
        "history": history,
        "recommendations": sess_recs,
    }

@app.get("/healthz")
async def healthz():
    return {"ok": True}

@app.head("/")
async def head_root():
    return Response(status_code=200)

@app.get("/favicon.ico")
async def favicon():
    path = os.path.join(WEB_ROOT, "favicon.ico")
    if os.path.isfile(path):
        return FileResponse(path)
    return Response(status_code=204)
