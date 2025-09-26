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

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# ---------- App setup ----------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("belt-service")

app = FastAPI(title="BELT Speaking Evaluator", version="2.0")

# Allow both same-origin and separate frontends (tighten as desired)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve web UI (index.html + /static/*)
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
# session schema:
# {
#   "level": "A1",
#   "history": [ {level, question, prompt_id, transcript, scores, average, decision, attempt, recommendations} ],
#   "served_prompts": { "A1": set(ids), "A2": set(ids), ... },
#   "final_level": Optional[str]
# }

# ---------- Prompt loading ----------

def load_prompts_for_level(level: str) -> List[Dict[str, Any]]:
    """
    Load prompts from web/prompts/<level>.json if present; otherwise use built-in fallback.
    Each prompt: { "id": <int>, "instructions": <str> }
    """
    path = os.path.join(WEB_ROOT, "prompts", f"{level}.json")
    items: List[Dict[str, Any]] = []
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # allow either a list of strings or list of objects
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

    # Fallback minimal prompts per level (you likely have fuller sets in /web/prompts)
    fallback: Dict[str, List[str]] = {
        "A1": [
            "Introduce yourself. Say your name, where you’re from, and one hobby.",
            "Describe your typical breakfast.",
            "Talk about your family."
        ],
        "A2": [
            "Describe your daily routine on weekdays.",
            "Explain your last weekend plans.",
            "Talk about your favorite place in your town."
        ],
        "B1": [
            "Describe a memorable trip. Where did you go and why was it memorable?",
            "What is an important goal you have this year and how will you achieve it?",
            "Explain a skill you’re learning and why it matters."
        ],
        "B1+": [
            "Do you think it’s better to study alone or in a group? Explain.",
            "Describe a challenge you faced and how you overcame it.",
            "Is social media helpful or harmful? Give reasons."
        ],
        "B2": [
            "Do you think governments should ban junk-food advertising? Why or why not?",
            "How can cities reduce traffic congestion? Propose solutions.",
            "What are the pros and cons of remote work?"
        ],
        "B2+": [
            "Tell me about a technology you couldn’t live without and why.",
            "Should universities require community service for graduation? Defend your view.",
            "What responsibilities do large companies have regarding climate change?"
        ],
        "C1": [
            "Is it important to learn history? Defend your position.",
            "What are the ethical implications of data collection by tech firms?",
            "How should societies balance free speech and public safety?"
        ],
        "C2": [
            "What is the impact of social media on democracy?",
            "To what extent should governments regulate AI development?",
            "Are standardized tests a fair measure of ability? Argue your case."
        ],
    }
    return [{"id": i, "instructions": txt} for i, txt in enumerate(fallback.get(level, []))]

def choose_prompt(level: str, session_id: str) -> Dict[str, Any]:
    """
    Session-aware prompt selection without repeats at that level.
    """
    items = load_prompts_for_level(level)
    served = SESSIONS[session_id]["served_prompts"].setdefault(level, set())
    # pick first not served (deterministic); you can randomize if you prefer
    for item in items:
        if item["id"] not in served:
            served.add(item["id"])
            return item
    # if all served, reset and give the first one again
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
    cmd = [
        "ffmpeg", "-y",
        "-i", in_path,
        "-ac", "1",           # mono
        "-ar", "16000",       # 16 kHz
        "-f", "wav",
        out_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

async def transcribe(openai_client, wav_path: str) -> str:
    """
    Transcribe with OpenAI Whisper (if enabled). Fail-safe to empty string if disabled.
    """
    if ASR_BACKEND != "openai":
        return ""
    if openai_client is None:
        return ""
    try:
        with open(wav_path, "rb") as f:
            resp = openai_client.audio.transcriptions.create(
                model=WHISPER_MODEL,
                file=f,
                response_format="text",
                temperature=0
            )
        if isinstance(resp, str):
            return resp.strip()
        text = getattr(resp, "text", "") or ""
        return text.strip()
    except Exception as e:
        log.warning(f"Transcription failed: {e}")
        return ""

# ---------- Scoring / Recommendations ----------

CEFR_CATEGORIES = ["fluency", "grammar", "vocabulary", "pronunciation", "coherence"]

RULEBOOK: Dict[str, List[str]] = {
    "fluency": [
        "Aim for fewer long pauses; speak in phrases of 4–7 words before pausing.",
        "Try shadowing a short native clip daily to build rhythm.",
        "Use fillers naturally (e.g., ‘well’, ‘actually’) instead of silence."
    ],
    "grammar": [
        "Watch subject–verb agreement (he/she/it + verbs ending in -s).",
        "Keep tenses consistent when telling a story (mostly past simple).",
        "Use linking words (because, although, however) to join clauses."
    ],
    "vocabulary": [
        "Add 3–5 topic words before speaking (e.g., travel: itinerary, budget, delay).",
        "Paraphrase when you don’t know a word (use a more general term).",
        "Learn 2–3 collocations per day (e.g., ‘make a decision’, ‘highly likely’)."
    ],
    "pronunciation": [
        "Slow down slightly and stress key words in each sentence.",
        "Open vowel contrasts clearly (/iː/ vs /ɪ/, /ɑː/ vs /æ/).",
        "Record yourself and compare to a native sample for 30 seconds daily."
    ],
    "coherence": [
        "Use a simple structure: intro -> 2–3 points -> short wrap-up.",
        "Signpost ideas (‘firstly’, ‘on the other hand’, ‘finally’).",
        "Keep each sentence to one main idea; avoid run-ons."
    ],
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
    tips: List[str] = []
    for cat in weak_categories(scores, pass_avg, pass_min):
        for tip in RULEBOOK.get(cat, []):
            tips.append(f"{cat.capitalize()}: {tip}")
            if len(tips) >= max_tips:
                return tips
    return tips

def dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set(); out = []
    for it in items:
        key = it.strip().lower()
        if key not in seen:
            seen.add(key)
            out.append(it.strip())
    return out

def try_llm_scores_and_tips(openai_client, transcript: str, level: str) -> Optional[Dict[str, Any]]:
    """
    Ask RUBRIC_MODEL to produce JSON with scores 0..1 and 2–3 tailored tips.
    Fail-safe: return None if anything goes wrong. We clamp values later.
    """
    if not RUBRIC_MODEL or not transcript or openai_client is None:
        return None
    try:
        sys = (
            "You are a CEFR speaking evaluator. "
            "Return strict JSON with keys: "
            "{scores:{fluency,grammar,vocabulary,pronunciation,coherence} in 0..1}, "
            "\"tips\": array of 2-3 short actionable strings. No extra commentary."
        )
        user = (
            f"Assess the following transcript at level target {level}.\n"
            f"Transcript:\n{transcript}\n"
            "JSON only."
        )
        resp = openai_client.chat.completions.create(
            model=RUBRIC_MODEL,
            messages=[{"role": "system", "content": sys},
                      {"role": "user", "content": user}],
            temperature=0.2,
            max_tokens=300,
        )
        content = resp.choices[0].message.content.strip()
        data = json.loads(content)
        return data
    except Exception as e:
        log.warning(f"LLM rubric failed: {e}")
        return None

def clamp01(x: float) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0

def compute_scores(openai_client, transcript: str, level: str) -> Dict[str, Any]:
    """
    Prefer LLM JSON; fallback to naive heuristics if empty transcript.
    """
    result = {
        "scores": {k: 0.0 for k in CEFR_CATEGORIES},
        "average": 0.0,
        "tips_ai": [],
    }
    if not transcript.strip():
        return result

    data = try_llm_scores_and_tips(openai_client, transcript, level)
    if data and isinstance(data, dict):
        s = data.get("scores", {}) or {}
        scores = {k: clamp01(s.get(k, 0.0)) for k in CEFR_CATEGORIES}
        avg = sum(scores.values()) / len(CEFR_CATEGORIES)
        tips_ai = data.get("tips", []) if isinstance(data.get("tips", []), list) else []
        result["scores"] = scores
        result["average"] = avg
        result["tips_ai"] = [t for t in tips_ai if isinstance(t, str) and t.strip()]
        return result

    # Fallback crude heuristic if LLM failed: length-based
    length = len(transcript.split())
    rough = min(1.0, length / 120.0)  # 120+ words ~ 1.0
    scores = {k: rough for k in CEFR_CATEGORIES}
    avg = rough
    result["scores"] = scores
    result["average"] = avg
    result["tips_ai"] = []
    return result

def finalize_session_recommendations(history: List[Dict[str, Any]]) -> List[str]:
    if not history:
        return []
    counts = Counter()
    for turn in history:
        s = turn.get("scores", {})
        for cat in CEFR_CATEGORIES:
            if float(s.get(cat, 0.0)) < 0.7:
                counts[cat] += 1
    if not counts:
        return ["Great overall balance. Keep practicing spontaneous speaking on varied topics."]
    top2 = [c for c, _ in counts.most_common(2)]
    tips = []
    for cat in top2:
        basics = RULEBOOK.get(cat, [])[:2]
        if basics:
            tips.append(f"Focus — {cat.capitalize()}: {basics[0]}")
        if len(basics) > 1:
            tips.append(f"Also for {cat}: {basics[1]}")
    tips.append("Routine: 10 minutes daily — plan 3 points, record once, listen back, and re-record focusing on the tips above.")
    return tips[:5]

# ---------- Helpers ----------

def next_level(level: str) -> Optional[str]:
    try:
        i = CEFR_ORDER.index(level)
        if i + 1 < len(CEFR_ORDER):
            return CEFR_ORDER[i + 1]
    except ValueError:
        pass
    return None

def decision_from_scores(scores: Dict[str, float], avg: float) -> str:
    if avg >= PASS_AVG_THRESHOLD and all(scores.get(k, 0.0) >= PASS_MIN_THRESHOLD for k in CEFR_CATEGORIES):
        return "advance"
    return "stop"

# ---------- Routes ----------

@app.get("/", response_class=HTMLResponse)
async def index():
    if not os.path.isdir(WEB_ROOT):
        return HTMLResponse("<h1>BELT Speaking Evaluator</h1><p>web/ folder not found.</p>")
    index_path = os.path.join(WEB_ROOT, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>BELT Speaking Evaluator</h1><p>index.html not found.</p>")

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
    SESSIONS[session_id] = {
        "level": level,
        "history": [],
        "served_prompts": {},
        "final_level": None,
    }
    item = choose_prompt(level, session_id)
    payload = {
        "session_id": session_id,
        "level": level,
        "prompt": item.get("instructions"),
        "prompt_id": item.get("id"),
        # optional TTS: if you pre-generate and host per prompt, add URL here
        "prompt_tts_url": None,
    }
    log.info(json.dumps({"ts": int(time.time()*1000), "msg": "start-session", "session_id": session_id, "level": level}))
    return JSONResponse(payload)

@app.get("/prompts/{level}")
async def get_prompt(level: str, session_id: str):
    if session_id not in SESSIONS:
        raise HTTPException(400, "Unknown session_id")
    state = SESSIONS[session_id]
    state["level"] = level  # keep session level in sync
    item = choose_prompt(level, session_id)
    return {
        "level": level,
        "instructions": item.get("instructions"),
        "prompt_id": item.get("id"),
        "prompt_tts_url": None,
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

    # Determine attempt number based on history at this level
    prev_same_level = sum(1 for t in state["history"] if t.get("level") == level)
    attempt = prev_same_level + 1

    # Persist upload to temp file
    with tempfile.TemporaryDirectory() as td:
        src_path = os.path.join(td, file.filename or "answer.bin")
        with open(src_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        wav_path = os.path.join(td, "answer.wav")
        try:
            to_wav_16k_mono(src_path, wav_path)
        except Exception as e:
            log.warning(f"ffmpeg convert failed: {e}")
            raise HTTPException(400, "Audio format not supported or ffmpeg error.")

        # OpenAI client (lazy import so service can start without key for static routes)
        openai_client = None
        if ASR_BACKEND == "openai" or RUBRIC_MODEL:
            try:
                from openai import OpenAI
                openai_client = OpenAI()
            except Exception as e:
                log.warning(f"OpenAI client init failed: {e}")
                openai_client = None

        transcript = await transcribe(openai_client, wav_path)

    # Resolve prompt text/id for logging & consistency
    used_prompt_text = None
    used_prompt_id = None
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
        if question:
            used_prompt_text = question
        else:
            last_served_id = None
            served = state["served_prompts"].get(level) or set()
            if served:
                last_served_id = list(served)[-1]
            if last_served_id is not None:
                for p in load_prompts_for_level(level):
                    if p["id"] == last_served_id:
                        used_prompt_text = p.get("instructions"); used_prompt_id = last_served_id
                        break
        used_prompt_text = used_prompt_text or "Speak for ~60 seconds."

    # Compute scores (LLM or heuristic)
    score_pack = compute_scores(openai_client, transcript or "", level)
    scores = score_pack["scores"]
    avg = score_pack["average"]
    ai_tips = score_pack.get("tips_ai", [])

    # Decision
    decision = decision_from_scores(scores, avg)

    # Merge per-turn recommendations: rule-based + AI tips
    rb_recs = rule_based_tips(scores, PASS_AVG_THRESHOLD, PASS_MIN_THRESHOLD, max_tips=3)
    merged_recs = dedupe_keep_order(rb_recs + ai_tips)[:5]
    if not merged_recs:
        merged_recs = [
            "Before speaking, jot 3 key points and 2 useful phrases for the topic.",
            "Speak in short sentences and link them with signposts (first, next, finally).",
        ]

    # Turn log
    turn = {
        "level": level,
        "question": used_prompt_text,
        "prompt_id": used_prompt_id if used_prompt_id is not None else prompt_id,
        "transcript": transcript,
        "scores": scores,
        "average": avg,
        "decision": decision,
        "attempt": attempt,
        "recommendations": merged_recs,
    }
    state["history"].append(turn)
    if decision == "stop":
        state["final_level"] = level

    # Response
    payload: Dict[str, Any] = {
        "scores": scores,
        "average": avg,
        "decision": decision,
        "attempt": attempt,
        "recommendations": merged_recs,
    }

    if decision == "advance":
        nxt = next_level(level)
        if nxt:
            state["level"] = nxt
            # Include next prompt to keep UI smooth
            item = choose_prompt(nxt, session_id)
            payload["next_level"] = nxt
            payload["next_prompt"] = item.get("instructions")
            payload["next_prompt_id"] = item.get("id")
        else:
            payload["next_level"] = None
    else:
        # stop case: if second fail, include session-level recommendations
        if attempt >= 2:
            payload["session_recommendations"] = finalize_session_recommendations(state["history"])

    log.info(json.dumps({
        "ts": int(time.time()*1000),
        "name": "belt-service",
        "msg": "submit-response",
        "session_id": session_id,
        "level": level,
        "attempt": attempt,
        "avg": avg,
        "decision": decision
    }))

    return JSONResponse(payload)

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
