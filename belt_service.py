# -*- coding: utf-8 -*-
"""
BELT Speaking Evaluator (FastAPI) - Full Service with debug hooks

Features:
- Adaptive CEFR evaluation (A1 -> C2), one retry per level
- Session-aware prompts (no repeats), prompt_id/question logging
- Whisper transcription (OpenAI) -> CEFR scoring (LLM + fallback)
- Per-turn recommendations + session-level recommendations
- Static UI served from /web
- Health endpoints, favicon handler
- Debug extras: logs audio size, transcript length; optional transcript echo in JSON (DEBUG_RETURN_TRANSCRIPT=1)
"""

import os
import json
import uuid
import time
import shutil
import logging
import tempfile
import subprocess
from typing import Dict, List, Any, Optional
from collections import Counter

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# ---------- App setup ----------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("belt-service")

app = FastAPI(title="BELT Speaking Evaluator", version="2.3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

WEB_ROOT = os.path.join(os.getcwd(), "web")
if os.path.isdir(WEB_ROOT):
    app.mount("/static", StaticFiles(directory=os.path.join(WEB_ROOT, "static")), name="static")

# ---------- Config / Env ----------

CEFR_ORDER = ["A1", "A2", "B1", "B1+", "B2", "B2+", "C1", "C2"]
CEFR_CATEGORIES = ["fluency", "grammar", "vocabulary", "pronunciation", "coherence"]

ASR_BACKEND = os.getenv("ASR_BACKEND", "openai").strip().lower()
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-1").strip()
RUBRIC_MODEL = os.getenv("RUBRIC_MODEL", "gpt-4o-mini").strip()

PASS_AVG_THRESHOLD = float(os.getenv("PASS_AVG_THRESHOLD", "0.70"))
PASS_MIN_THRESHOLD  = float(os.getenv("PASS_MIN_THRESHOLD",  "0.60"))
RECORD_SECONDS      = int(float(os.getenv("RECORD_SECONDS", "60")))

# Debug flags
DEBUG_RETURN_TRANSCRIPT = os.getenv("DEBUG_RETURN_TRANSCRIPT", "0").strip() in ("1", "true", "True")

# Log OpenAI SDK version (helps catch 0.x vs 1.x issues)
try:
    import openai as _oa_mod
    _oa_ver = getattr(_oa_mod, "__version__", "unknown")
    log.info(f"OpenAI SDK version: {_oa_ver}")
    # Fail loudly if 0.x is installed
    try:
        major = int(_oa_ver.split(".")[0])
        if major < 1:
            raise RuntimeError(f"OpenAI SDK too old: {_oa_ver}. Please use >= 1.0.0")
    except Exception as e:
        log.warning(f"OpenAI SDK version check: {e}")
except Exception:
    log.info("OpenAI SDK not importable at startup (will try later when used)")

# ---------- In-memory session store ----------

SESSIONS: Dict[str, Dict[str, Any]] = {}

# ---------- Prompts ----------

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

    # Minimal fallback
    fallback: Dict[str, List[str]] = {
        "A1": ["Introduce yourself.", "Describe your breakfast.", "Talk about your family."],
        "A2": ["Describe your daily routine.", "Explain your last weekend plans.", "Favorite place in your town."],
        "B1": ["Describe a memorable trip.", "An important goal this year?", "A skill you’re learning and why."],
        "B1+": ["Study alone or in a group? Explain.", "A challenge you faced & overcame.", "Is social media helpful or harmful?"],
        "B2": ["Ban junk-food ads? Why/why not?", "How to reduce traffic congestion?", "Pros and cons of remote work."],
        "B2+": ["A technology you can’t live without.", "Require community service to graduate?", "Corporate climate responsibilities?"],
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
    first = items[0] if items else {"id": 0, "instructions": "Speak for ~60 seconds."}
    SESSIONS[session_id]["served_prompts"][level].add(first["id"])
    return first

# ---------- Audio ----------

def ensure_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception:
        raise HTTPException(500, "ffmpeg not found on server")

def to_wav_16k_mono(in_path: str, out_path: str):
    ensure_ffmpeg()
    cmd = ["ffmpeg", "-y", "-i", in_path, "-ac", "1", "-ar", "16000", "-f", "wav", out_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

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

# ---------- Scoring ----------

RULEBOOK: Dict[str, List[str]] = {
    "fluency": ["Avoid long pauses.", "Shadow native clips daily.", "Use fillers naturally."],
    "grammar": ["Watch subject–verb agreement.", "Keep tenses consistent.", "Use linking words."],
    "vocabulary": ["Add topic words ahead.", "Paraphrase when stuck.", "Learn 2–3 collocations/day."],
    "pronunciation": ["Stress key words.", "Open vowel contrasts.", "Record & compare to native."],
    "coherence": ["Intro -> 2–3 points -> wrap-up.", "Signpost ideas.", "Avoid run-ons."],
}

def weak_categories(scores: Dict[str, float]) -> List[str]:
    items = []
    for k in CEFR_CATEGORIES:
        v = float(scores.get(k, 0.0))
        tier = 2 if v < PASS_MIN_THRESHOLD else (1 if v < PASS_AVG_THRESHOLD else 0)
        if tier > 0: items.append((k, tier, v))
    items.sort(key=lambda x: (-x[1], x[2]))
    return [k for (k, _, __) in items]

def rule_based_tips(scores: Dict[str, float], max_tips: int = 3) -> List[str]:
    tips = []
    for cat in weak_categories(scores):
        for tip in RULEBOOK.get(cat, []):
            tips.append(f"{cat.capitalize()}: {tip}")
            if len(tips) >= max_tips: return tips
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
    try: return max(0.0, min(1.0, float(x)))
    except Exception: return 0.0

def compute_scores(openai_client, transcript: str, level: str) -> Dict[str, Any]:
    result = {"scores": {k:0.0 for k in CEFR_CATEGORIES}, "average": 0.0, "tips_ai": []}
    if not transcript.strip(): return result
    data = try_llm_scores_and_tips(openai_client, transcript, level)
    if data and isinstance(data, dict):
        s = data.get("scores", {}) or {}
        scores = {k: clamp01(s.get(k, 0.0)) for k in CEFR_CATEGORIES}
        avg = sum(scores.values())/len(CEFR_CATEGORIES)
        tips_ai = [t for t in (data.get("tips") or []) if isinstance(t, str) and t.strip()]
        result.update({"scores": scores, "average": avg, "tips_ai": tips_ai})
        return result
    # crude fallback
    length = len(transcript.split())
    rough = min(1.0, length / 120.0)
    scores = {k: rough for k in CEFR_CATEGORIES}
    result.update({"scores": scores, "average": rough, "tips_ai": []})
    return result

def finalize_session_recommendations(history: List[Dict[str, Any]]) -> List[str]:
    if not history: return []
    cnt = Counter()
    for turn in history:
        s = turn.get("scores", {})
        for cat in CEFR_CATEGORIES:
            if float(s.get(cat, 0.0)) < 0.7: cnt[cat] += 1
    if not cnt:
        return ["Great overall balance. Keep practicing varied topics."]
    top2 = [c for c,_ in cnt.most_common(2)]
    tips = []
    for cat in top2:
        basics = RULEBOOK.get(cat, [])[:2]
        for b in basics: tips.append(f"Focus on {cat}: {b}")
    tips.append("Daily routine: 10 minutes — plan 3 points, record, listen back, re-record.")
    return tips[:5]

def decision_from_scores(scores: Dict[str, float], avg: float) -> str:
    if avg >= PASS_AVG_THRESHOLD and all(scores.get(k,0.0) >= PASS_MIN_THRESHOLD for k in CEFR_CATEGORIES):
        return "advance"
    return "stop"

# ---------- Routes ----------

@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = os.path.join(WEB_ROOT, "index.html")
    if os.path.isfile(index_path): return FileResponse(index_path)
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
    }

@app.post("/start-session")
async def start_session(level: str = Form("A1")):
    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = {"level": level, "history": [], "served_prompts": {}, "final_level": None}
    item = choose_prompt(level, session_id)
    log.info(json.dumps({"ts": int(time.time()*1000), "msg": "start-session", "session_id": session_id, "level": level}))
    return {"session_id": session_id, "level": level, "prompt": item.get("instructions"), "prompt_id": item.get("id")}

@app.get("/prompts/{level}")
async def get_prompt(level: str, session_id: str):
    if session_id not in SESSIONS: raise HTTPException(400, "Unknown session_id")
    item = choose_prompt(level, session_id)
    return {"level": level, "instructions": item.get("instructions"), "prompt_id": item.get("id")}

@app.post("/submit-response")
async def submit_response(
    session_id: str = Form(...),
    file: UploadFile = File(...),
    prompt_id: Optional[str] = Form(None),
    question: Optional[str] = Form(None),
):
    if session_id not in SESSIONS: raise HTTPException(400, "Unknown session_id")
    state = SESSIONS[session_id]
    level = state.get("level", "A1")
    attempt = sum(1 for t in state["history"] if t.get("level") == level) + 1

    # Save upload
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
        openai_client = None
        if ASR_BACKEND == "openai":
            try:
                from openai import OpenAI
                openai_client = OpenAI()
            except Exception as e:
                log.warning(f"OpenAI init failed: {e}")
                openai_client = None
        transcript = await transcribe(openai_client, wav_path)

    # Determine used prompt text for logging
    used_prompt_text = None
    used_prompt_id = None
    if prompt_id is not None:
        try: used_prompt_id = int(prompt_id)
        except Exception: used_prompt_id = None
        for p in load_prompts_for_level(level):
            if p["id"] == used_prompt_id:
                used_prompt_text = p.get("instructions"); break
    if not used_prompt_text:
        used_prompt_text = question or "Speak for ~60 seconds."

    # Compute scores
    pack = compute_scores(openai_client, transcript or "", level)
    scores, avg = pack["scores"], pack["average"]
    decision = decision_from_scores(scores, avg)
    recs = rule_based_tips(scores, max_tips=3)
    if not recs:
        recs = ["Plan 3 key points before speaking.", "Link sentences with signposts (first, next, finally)."]

    # Turn log
    turn = {
        "level": level, "question": used_prompt_text, "prompt_id": used_prompt_id or prompt_id,
        "transcript": transcript, "scores": scores, "average": avg, "decision": decision,
        "attempt": attempt, "recommendations": recs,
    }
    state["history"].append(turn)
    if decision == "stop": state["final_level"] = level

    # Debug log line
    log.info(json.dumps({
        "ts": int(time.time()*1000), "name": "belt-service", "msg": "submit-response",
        "session_id": session_id, "level": level, "attempt": attempt,
        "upload_kb": round(size_bytes/1024, 1), "transcript_len": len(transcript or ""),
        "avg": avg, "decision": decision
    }))

    # Response payload
    payload: Dict[str, Any] = {
        "scores": scores, "average": avg, "decision": decision, "attempt": attempt, "recommendations": recs
    }
    if DEBUG_RETURN_TRANSCRIPT:
        payload["transcript"] = transcript or ""

    if decision == "advance":
        i = CEFR_ORDER.index(level)
        if i + 1 < len(CEFR_ORDER):
            nxt = CEFR_ORDER[i + 1]
            state["level"] = nxt
            item = choose_prompt(nxt, session_id)
            payload.update({"next_level": nxt, "next_prompt": item.get("instructions"), "next_prompt_id": item.get("id")})
    else:
        if attempt >= 2:
            payload["session_recommendations"] = finalize_session_recommendations(state["history"])

    return JSONResponse(payload)

@app.get("/report/{session_id}")
async def report(session_id: str):
    if session_id not in SESSIONS: raise HTTPException(404, "Unknown session_id")
    state = SESSIONS[session_id]
    return {
        "session_id": session_id,
        "final_level": state.get("final_level"),
        "history": state.get("history", []),
        "recommendations": finalize_session_recommendations(state.get("history", [])),
    }

# ---------- Health & misc ----------

@app.get("/healthz")
async def healthz(): return {"ok": True}

@app.head("/")
async def head_root(): return Response(status_code=200)

@app.get("/favicon.ico")
async def favicon():
    path = os.path.join(WEB_ROOT, "favicon.ico")
    if os.path.isfile(path): return FileResponse(path)
    return Response(status_code=204)
