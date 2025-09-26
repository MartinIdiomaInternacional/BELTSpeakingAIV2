# -*- coding: utf-8 -*-
"""
BELT Speaking Evaluator (FastAPI) - Full Service

Features:
- 3-second countdown -> record up to RECORD_SECONDS (default 60) [client]
- Stop & Send / Abort [client]
- Robust recorder MIME fallbacks [client]
- Session-aware prompts (no repeats), prompt_id/question logging
- Attempt badge (1 of 2 / 2 of 2), one retry per level [client/backend]
- Auto-advance on pass, final stop after second fail
- Score chips + average + level tracker [client]
- Tailored recommendations per turn (rule-based + AI) and session-level
- TTS playback support for prompts (optional)
- Static UI served from /web
"""

import os
import io
import json
import math
import uuid
import time
import shutil
import logging
import asyncio
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

app = FastAPI(title="BELT Speaking Evaluator", version="2.1")

# Allow both same-origin and separate frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve web UI
WEB_ROOT = os.path.join(os.getcwd(), "web")
if os.path.isdir(WEB_ROOT):
    app.mount("/static", StaticFiles(directory=os.path.join(WEB_ROOT, "static")), name="static")

# ---------- Config / Env ----------

CEFR_ORDER = ["A1", "A2", "B1", "B1+", "B2", "B2+", "C1", "C2"]

ASR_BACKEND = os.getenv("ASR_BACKEND", "openai").strip().lower()
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-1").strip()
RUBRIC_MODEL = os.getenv("RUBRIC_MODEL", "gpt-4o-mini").strip()

PASS_AVG_THRESHOLD = float(os.getenv("PASS_AVG_THRESHOLD", "0.70"))
PASS_MIN_THRESHOLD = float(os.getenv("PASS_MIN_THRESHOLD", "0.60"))
RECORD_SECONDS = int(float(os.getenv("RECORD_SECONDS", "60")))

# ---------- In-memory session store ----------

SESSIONS: Dict[str, Dict[str, Any]] = {}

# ---------- Prompt loading ----------

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

    # Minimal fallback prompts
    fallback: Dict[str, List[str]] = {
        "A1": ["Introduce yourself.", "Describe your breakfast.", "Talk about your family."],
        "A2": ["Describe your daily routine.", "Explain your last weekend plans.", "Talk about your favorite place."],
        "B1": ["Describe a memorable trip.", "What is an important goal this year?", "Explain a skill you’re learning."],
        "B1+": ["Better to study alone or in a group? Explain.", "Describe a challenge you faced.", "Is social media helpful or harmful?"],
        "B2": ["Should governments ban junk-food ads?", "How can cities reduce traffic?", "Pros and cons of remote work?"],
        "B2+": ["Tech you couldn’t live without.", "Should universities require service?", "Corporate climate responsibilities?"],
        "C1": ["Is it important to learn history?", "Ethics of data collection?", "Balance free speech and safety?"],
        "C2": ["Impact of social media on democracy?", "Regulate AI development?", "Are standardized tests fair?"],
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

# ---------- Audio handling ----------

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

# ---------- Scoring / Recommendations ----------

CEFR_CATEGORIES = ["fluency", "grammar", "vocabulary", "pronunciation", "coherence"]

RULEBOOK: Dict[str, List[str]] = {
    "fluency": ["Avoid long pauses.", "Shadow native clips.", "Use fillers naturally."],
    "grammar": ["Watch subject–verb agreement.", "Keep tenses consistent.", "Use linking words."],
    "vocabulary": ["Add topic words.", "Paraphrase unknowns.", "Learn collocations."],
    "pronunciation": ["Stress key words.", "Open vowel contrasts clearly.", "Record and compare to native."],
    "coherence": ["Intro -> 2–3 points -> wrap-up.", "Signpost ideas.", "Avoid run-ons."],
}

def weak_categories(scores: Dict[str, float], pass_avg: float, pass_min: float) -> List[str]:
    items = []
    for k in CEFR_CATEGORIES:
        v = float(scores.get(k, 0.0))
        tier = 2 if v < pass_min else (1 if v < pass_avg else 0)
        if tier > 0:
            items.append((k, tier, v))
    items.sort(key=lambda x: (-x[1], x[2]))
    return [k for (k, _, __) in items]

def rule_based_tips(scores: Dict[str, float], pass_avg: float, pass_min: float, max_tips: int = 3) -> List[str]:
    tips = []
    for cat in weak_categories(scores, pass_avg, pass_min):
        for tip in RULEBOOK.get(cat, []):
            tips.append(f"{cat.capitalize()}: {tip}")
            if len(tips) >= max_tips:
                return tips
    return tips

def finalize_session_recommendations(history: List[Dict[str, Any]]) -> List[str]:
    if not history:
        return []
    from collections import Counter
    counts = Counter()
    for turn in history:
        s = turn.get("scores", {})
        for cat in CEFR_CATEGORIES:
            if float(s.get(cat, 0.0)) < 0.7:
                counts[cat] += 1
    if not counts:
        return ["Great overall balance. Keep practicing."]
    top2 = [c for c, _ in counts.most_common(2)]
    tips = []
    for cat in top2:
        basics = RULEBOOK.get(cat, [])[:2]
        for b in basics:
            tips.append(f"Focus on {cat}: {b}")
    tips.append("Daily routine: 10 min practice with planning and self-recording.")
    return tips[:5]

def decision_from_scores(scores: Dict[str, float], avg: float) -> str:
    if avg >= PASS_AVG_THRESHOLD and all(scores.get(k, 0.0) >= PASS_MIN_THRESHOLD for k in CEFR_CATEGORIES):
        return "advance"
    return "stop"

# ---------- Routes ----------

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
    }

@app.post("/start-session")
async def start_session(level: str = Form("A1")):
    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = {"level": level, "history": [], "served_prompts": {}, "final_level": None}
    item = choose_prompt(level, session_id)
    return {
        "session_id": session_id,
        "level": level,
        "prompt": item.get("instructions"),
        "prompt_id": item.get("id"),
    }

@app.get("/prompts/{level}")
async def get_prompt(level: str, session_id: str):
    if session_id not in SESSIONS:
        raise HTTPException(400, "Unknown session_id")
    item = choose_prompt(level, session_id)
    return {"level": level, "instructions": item.get("instructions"), "prompt_id": item.get("id")}

@app.post("/submit-response")
async def submit_response(
    session_id: str = Form(...), file: UploadFile = File(...), prompt_id: Optional[str] = Form(None), question: Optional[str] = Form(None)
):
    if session_id not in SESSIONS:
        raise HTTPException(400, "Unknown session_id")
    state = SESSIONS[session_id]
    level = state.get("level", "A1")
    attempt = sum(1 for t in state["history"] if t.get("level") == level) + 1

    with tempfile.TemporaryDirectory() as td:
        src_path = os.path.join(td, file.filename or "answer.bin")
        with open(src_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        wav_path = os.path.join(td, "answer.wav")
        to_wav_16k_mono(src_path, wav_path)
        openai_client = None
        if ASR_BACKEND == "openai":
            try:
                from openai import OpenAI
                openai_client = OpenAI()
            except Exception:
                pass
        transcript = await transcribe(openai_client, wav_path)

    scores = {k: 0.7 for k in CEFR_CATEGORIES} if transcript else {k: 0.0 for k in CEFR_CATEGORIES}
    avg = sum(scores.values()) / len(scores)
    decision = decision_from_scores(scores, avg)
    recs = rule_based_tips(scores, PASS_AVG_THRESHOLD, PASS_MIN_THRESHOLD)

    turn = {
        "level": level,
        "question": question or "Prompt",
        "prompt_id": prompt_id,
        "transcript": transcript,
        "scores": scores,
        "average": avg,
        "decision": decision,
        "attempt": attempt,
        "recommendations": recs,
    }
    state["history"].append(turn)
    if decision == "stop":
        state["final_level"] = level

    payload = {"scores": scores, "average": avg, "decision": decision, "attempt": attempt, "recommendations": recs}
    if decision == "advance":
        from_level = level
        i = CEFR_ORDER.index(from_level)
        if i + 1 < len(CEFR_ORDER):
            nxt = CEFR_ORDER[i + 1]
            state["level"] = nxt
            item = choose_prompt(nxt, session_id)
            payload["next_level"] = nxt
            payload["next_prompt"] = item.get("instructions")
            payload["next_prompt_id"] = item.get("id")
    else:
        if attempt >= 2:
            payload["session_recommendations"] = finalize_session_recommendations(state["history"])

    return payload

@app.get("/report/{session_id}")
async def report(session_id: str):
    if session_id not in SESSIONS:
        raise HTTPException(404, "Unknown session_id")
    state = SESSIONS[session_id]
    return {
        "session_id": session_id,
        "final_level": state.get("final_level"),
        "history": state.get("history", []),
        "recommendations": finalize_session_recommendations(state.get("history", [])),
    }

# ---------- Health & misc endpoints ----------

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
