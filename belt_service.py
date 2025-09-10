import os
import io
import json
import time
import uuid
import random
import tempfile
import subprocess
import urllib.parse
from pathlib import Path
from typing import Dict, Optional, List, Set

from fastapi import FastAPI, HTTPException, Request, Query
from fastapi import UploadFile, File, Form
from fastapi.responses import (
    HTMLResponse, JSONResponse, PlainTextResponse, Response
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ---------- Simple logger ----------
import logging
logger = logging.getLogger("belt-service")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# ---------- Config / env ----------
ASR_BACKEND = os.getenv("ASR_BACKEND", "openai")
RUBRIC_MODEL = os.getenv("RUBRIC_MODEL", "gpt-4o-mini")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-1")

PASS_AVG_THRESHOLD = float(os.getenv("PASS_AVG_THRESHOLD", "0.70"))
PASS_MIN_THRESHOLD = float(os.getenv("PASS_MIN_THRESHOLD", "0.60"))
# Borderline passes trigger probe levels within this margin
PROBE_MARGIN = float(os.getenv("PROBE_MARGIN", "0.05"))

RECORD_SECONDS = int(os.getenv("RECORD_SECONDS", "60"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# TTS (optional)
TTS_MODEL = os.getenv("TTS_MODEL", "gpt-4o-mini-tts")
TTS_VOICE = os.getenv("TTS_VOICE", "alloy")

# ---------- LLM/ASR clients (OpenAI style) ----------
try:
    from openai import OpenAI
    openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception:
    logger.warning("OpenAI SDK not available or API key missing; falling back to dummy scoring/TTS.")
    openai_client = None

# ---------- App ----------
app = FastAPI(title="BELT Speaking AI")

# CORS (tighten in production if you host frontend separately)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your domain in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Frontend mounting ----------
WEB_DIR = Path(__file__).parent / "web"
STATIC_DIR = WEB_DIR / "static"
PROMPTS_DIR = WEB_DIR / "prompts"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Root serves index.html
@app.get("/", response_class=HTMLResponse)
async def index():
    idx = WEB_DIR / "index.html"
    if not idx.exists():
        return HTMLResponse("<h1>BELT Speaking AI</h1><p>Frontend missing. Place web/index.html here.</p>", status_code=200)
    return HTMLResponse(idx.read_text(encoding="utf-8"))

# ---------- Health / metrics ----------
@app.get("/healthz")
async def healthz():
    return PlainTextResponse("ok")

@app.get("/metrics")
async def metrics():
    lines = [
        "# HELP belt_requests_total Total requests.",
        "# TYPE belt_requests_total counter",
        'belt_requests_total{endpoint="/"} 1'
    ]
    return PlainTextResponse("\n".join(lines))

# ---------- Live config for UI ----------
@app.get("/config")
async def config():
    return {
        "ASR_BACKEND": ASR_BACKEND,
        "RUBRIC_MODEL": RUBRIC_MODEL,
        "WHISPER_MODEL": WHISPER_MODEL,
        "PASS_AVG_THRESHOLD": str(PASS_AVG_THRESHOLD),
        "PASS_MIN_THRESHOLD": str(PASS_MIN_THRESHOLD),
        "PROBE_MARGIN": str(PROBE_MARGIN),
        "RECORD_SECONDS": RECORD_SECONDS,
        "TTS_MODEL": TTS_MODEL,
        "TTS_VOICE": TTS_VOICE,
    }

# ---------- Session models / state ----------
CEFR_ORDER: List[str] = ["A1", "A2", "B1", "B1+", "B2", "B2+", "C1", "C2"]

class ScoreDict(BaseModel):
    fluency: float = 0.0
    grammar: float = 0.0
    vocabulary: float = 0.0
    pronunciation: float = 0.0
    coherence: float = 0.0

class Turn(BaseModel):
    level: str
    question: str
    transcript: str
    scores: ScoreDict
    average: float
    decision: str  # "advance" or "stop"
    attempt: int   # 1, 2, ...

class FinalReport(BaseModel):
    session_id: str
    final_level: str
    history: List[Turn]
    recommendations: List[str] = Field(default_factory=list)

# SESSIONS holds per-session state, including used question indices per level and current question.
# SESSIONS[session_id] = {
#   "current_level": "A1",
#   "current_question": "....",
#   "current_prompt_id": Optional[int],
#   "history": [Turn, ...],
#   "final_level": Optional[str],
#   "used_questions": { "A1": set(int,...), "A2": set(...) }
# }
SESSIONS: Dict[str, Dict] = {}

# Mapping for probe levels (insert a probe level if borderline pass)
PROBE_MAP: Dict[str, str] = {
    "A2": "B1",
    "B1": "B1+",
    "B2": "B2+",
}

# ---------- Guided prompt loading / selection ----------
def _load_questions_for_level(level: str) -> List[str]:
    """
    Reads web/prompts/<LEVEL>.json.
    If it contains "questions": [...], return that list.
    Else, if it contains single "instructions": "...", return [that].
    Else, return a generic fallback.
    """
    p = PROMPTS_DIR / f"{level}.json"
    if not p.exists():
        return [f"Speak for ~60 seconds about a topic suitable for {level}."]
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return [f"Speak for ~60 seconds about a topic suitable for {level}."]

    if isinstance(data, dict):
        if isinstance(data.get("questions"), list) and data["questions"]:
            return [str(q) for q in data["questions"] if isinstance(q, str) and q.strip()]
        instr = data.get("instructions")
        if isinstance(instr, str) and instr.strip():
            return [instr.strip()]
    return [f"Speak for ~60 seconds about a topic suitable for {level}."]

def _choose_question_index(level: str, used: Set[int]) -> int:
    """
    Choose a question index not in 'used' if possible.
    If all are used, reset (allow repeats) and choose again.
    """
    qs = _load_questions_for_level(level)
    n = len(qs)
    if n == 0:
        return 0
    candidates = [i for i in range(n) if i not in used]
    if not candidates:
        used.clear()
        candidates = list(range(n))
    return random.choice(candidates)

def _get_question_for_session(level: str, session_id: Optional[str]) -> (str, Optional[int]):
    """
    If session_id provided and valid, track used indices per level to avoid repeats.
    Returns (question_text, prompt_id). If stateless, prompt_id may be None.
    """
    questions = _load_questions_for_level(level)
    if not questions:
        return (f"Speak for ~60 seconds about a topic suitable for {level}.", None)

    if session_id and session_id in SESSIONS:
        sess = SESSIONS[session_id]
        used_map = sess.setdefault("used_questions", {})
        used_set: Set[int] = used_map.setdefault(level, set())
        idx = _choose_question_index(level, used_set)
        used_set.add(idx)
        return (questions[idx], idx)

    # No session — stateless random
    q = random.choice(questions)
    try:
        idx = questions.index(q)
    except ValueError:
        idx = None
    return (q, idx)

def _next_level(current: str) -> Optional[str]:
    if current not in CEFR_ORDER:
        return None
    idx = CEFR_ORDER.index(current)
    if idx + 1 < len(CEFR_ORDER):
        return CEFR_ORDER[idx + 1]
    return None

# ---------- Audio helpers (robust upload handling) ----------
def _save_upload_to_temp(upload: UploadFile) -> str:
    """Persist UploadFile to a real temp file and return its path."""
    suffix = ""
    if upload.filename and "." in upload.filename:
        suffix = "." + upload.filename.split(".")[-1].lower()
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    upload.file.seek(0)
    with open(tmp_path, "wb") as f:
        f.write(upload.file.read())
    upload.file.seek(0)
    return tmp_path

def _ffmpeg_convert_to_wav16k_mono(src_path: str) -> str:
    """Use ffmpeg to convert any audio (webm/ogg/m4a/mp3/wav) to 16k mono WAV."""
    out_path = src_path + ".__conv.wav"
    cmd = [
        "ffmpeg", "-y",
        "-i", src_path,
        "-ac", "1",
        "-ar", "16000",
        out_path
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg failed: {err[:500]}")
    return out_path

def _infer_should_convert(content_type: Optional[str], filename: Optional[str]) -> bool:
    """Return True if we should run ffmpeg conversion (most types except raw WAV)."""
    ct = (content_type or "").lower()
    name = (filename or "").lower()
    if "audio/wav" in ct or name.endswith(".wav"):
        return False
    return True

# ---------- Simple rubric prompt ----------
RUBRIC_SYSTEM = """You are a CEFR speaking examiner. Score the response 0.0–1.0 in five categories:
- fluency
- grammar
- vocabulary
- pronunciation
- coherence
Return strict JSON: {"fluency": float, "grammar": float, "vocabulary": float, "pronunciation": float, "coherence": float, "feedback": string}.
"""

def _llm_score_transcript(transcript: str, level: str) -> Dict:
    """
    Calls an LLM to score the transcript per CEFR dimensions.
    Returns dict: {"fluency":..., ..., "feedback": "..."}
    """
    if not openai_client:
        # Fallback heuristic if no API key
        length = len(transcript.split())
        base = min(1.0, max(0.1, length / 120))
        return {
            "fluency": round(base, 2),
            "grammar": round(base * 0.95, 2),
            "vocabulary": round(base * 0.90, 2),
            "pronunciation": round(0.75, 2),
            "coherence": round(base * 0.92, 2),
            "feedback": "Heuristic offline scoring (no API key)."
        }

    user_prompt = f"Level: {level}\nTranscript:\n{transcript}\n"
    resp = openai_client.chat.completions.create(
        model=RUBRIC_MODEL,
        messages=[
            {"role": "system", "content": RUBRIC_SYSTEM},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    try:
        data = json.loads(resp.choices[0].message.content)
        return data
    except Exception:
        logger.exception("Failed to parse rubric model JSON; falling back to heuristic.")
        length = len(transcript.split())
        base = min(1.0, max(0.1, length / 120))
        return {
            "fluency": round(base, 2),
            "grammar": round(base * 0.95, 2),
            "vocabulary": round(base * 0.90, 2),
            "pronunciation": round(0.75, 2),
            "coherence": round(base * 0.92, 2),
            "feedback": "Heuristic fallback due to parse error."
        }

def _decide(scores: ScoreDict) -> bool:
    """
    Returns True if pass (advance), False if stop.
    Rule: avg >= PASS_AVG_THRESHOLD AND all >= PASS_MIN_THRESHOLD.
    """
    vals = [scores.fluency, scores.grammar, scores.vocabulary, scores.pronunciation, scores.coherence]
    avg = sum(vals) / len(vals)
    return (avg >= PASS_AVG_THRESHOLD) and all(v >= PASS_MIN_THRESHOLD for v in vals)

def _borderline_pass(scores: ScoreDict) -> bool:
    """
    'Borderline pass' when avg in [PASS_AVG_THRESHOLD, PASS_AVG_THRESHOLD+PROBE_MARGIN)
    or any category within PROBE_MARGIN of PASS_MIN_THRESHOLD.
    """
    vals = [scores.fluency, scores.grammar, scores.vocabulary, scores.pronunciation, scores.coherence]
    avg = sum(vals) / len(vals)
    near_min = any((v - PASS_MIN_THRESHOLD) < PROBE_MARGIN for v in vals)
    return (avg >= PASS_AVG_THRESHOLD and avg < PASS_AVG_THRESHOLD + PROBE_MARGIN) or near_min

# ---------- Core pipeline ----------
async def _asr_transcribe_file(wav_file_path: str) -> str:
    """
    Transcribe a WAV file (16k mono) to text using OpenAI Whisper if enabled,
    otherwise return a dummy placeholder.
    """
    if ASR_BACKEND.lower() != "openai" or not openai_client:
        return "(no-ASR) Placeholder transcript for testing."

    with open(wav_file_path, "rb") as f:
        trans = openai_client.audio.transcriptions.create(
            model=WHISPER_MODEL,
            file=f,
            response_format="text"
        )
    return trans if isinstance(trans, str) else str(trans)

async def transcribe_and_score(
    session_id: str,
    wav_file_path: str,
    *,
    question: str,
    attempt: int
) -> Dict:
    """
    End-to-end: ASR -> LLM scoring -> adaptive decision & next prompt.
    Updates SESSIONS[session_id]["history"] and returns a response dict for the UI.
    """
    state = SESSIONS.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")
    level = state["current_level"]

    # 1) ASR
    transcript = await _asr_transcribe_file(wav_file_path)

    # 2) Score via LLM
    raw = _llm_score_transcript(transcript, level)
    scores = ScoreDict(
        fluency=float(raw.get("fluency", 0.0)),
        grammar=float(raw.get("grammar", 0.0)),
        vocabulary=float(raw.get("vocabulary", 0.0)),
        pronunciation=float(raw.get("pronunciation", 0.0)),
        coherence=float(raw.get("coherence", 0.0)),
    )
    avg = (scores.fluency + scores.grammar + scores.vocabulary + scores.pronunciation + scores.coherence) / 5.0
    passed = _decide(scores)
    borderline = _borderline_pass(scores) if passed else False

    decision = "advance" if passed else "stop"
    result: Dict = {
        "session_id": session_id,
        "level": level,
        "question": question,
        "transcript": transcript,
        "scores": scores.model_dump(),
        "average": round(avg, 4),
        "decision": decision,
        "attempt": attempt,
        "borderline": borderline,
    }

    # 3) Update session history (store question & attempt)
    state["history"].append(Turn(level=level, question=question, transcript=transcript,
                                 scores=scores, average=avg, decision=decision, attempt=attempt))

    if passed:
        # Decide if we should insert a probe level
        next_lvl = _next_level(level)
        if borderline and level in PROBE_MAP:
            probe = PROBE_MAP[level]
            if probe in CEFR_ORDER and probe != level:
                next_lvl = probe

        if next_lvl:
            state["current_level"] = next_lvl
            next_q, next_id = _get_question_for_session(next_lvl, session_id=session_id)
            state["current_question"] = next_q
            state["current_prompt_id"] = next_id
            result["next_level"] = next_lvl
            result["next_prompt"] = next_q
            result["next_prompt_id"] = next_id
        else:
            state["final_level"] = level
            result["decision"] = "stop"
    else:
        # stop at current level
        state["final_level"] = state.get("final_level") or level

    return result

# ---------- API: start-session / prompts / submit-response / evaluate-bytes / report / tts ----------

@app.post("/start-session")
async def start_session(level: str = Form("A1")):
    if level not in CEFR_ORDER:
        level = "A1"
    session_id = str(uuid.uuid4())
    first_q, first_id = _get_question_for_session(level, session_id=session_id)
    SESSIONS[session_id] = {
        "current_level": level,
        "current_question": first_q,
        "current_prompt_id": first_id,
        "history": [],
        # populated when stop
        # "final_level": str
        # used question indices per level:
        "used_questions": {}
    }
    logger.info("start-session", extra={"session_id": session_id, "level": level})
    tts_url = f"/tts?text={urllib.parse.quote_plus(first_q)}"
    return {
        "session_id": session_id,
        "level": level,
        "prompt": first_q,
        "prompt_id": first_id,
        "prompt_tts_url": tts_url
    }

# Prompts by level (returns ONE guided question + prompt_id).
# If session_id is provided (?session_id=UUID), it avoids repeats and updates session current_question/current_prompt_id.
@app.get("/prompts/{level}")
async def prompt_level(level: str, session_id: Optional[str] = Query(default=None)):
    if level not in CEFR_ORDER:
        raise HTTPException(status_code=404, detail=f"Unknown level: {level}")
    q, qid = _get_question_for_session(level, session_id=session_id)
    if session_id and session_id in SESSIONS:
        SESSIONS[session_id]["current_question"] = q
        SESSIONS[session_id]["current_prompt_id"] = qid
    return JSONResponse({
        "level": level,
        "instructions": q,
        "prompt_id": qid,
        "prompt_tts_url": f"/tts?text={urllib.parse.quote_plus(q)}"
    })

@app.post("/submit-response")
async def submit_response(
    # Accept multiple possible field names; the first non-None will be used.
    file: Optional[UploadFile] = File(default=None),
    audio: Optional[UploadFile] = File(default=None),
    audio_file: Optional[UploadFile] = File(default=None),
    session_id: str = Form(...),
    # New: allow frontend to send exact identifiers for perfect logging
    question: Optional[str] = Form(default=None),
    prompt_id: Optional[str] = Form(default=None),  # may come as string from FormData
):
    """
    Accepts recorded audio from the UI and advances/stops adaptively.
    Robust to audio/webm (MediaRecorder), ogg, mp3, m4a, wav (ffmpeg conversion to 16k mono WAV).
    """
    try:
        if session_id not in SESSIONS:
            raise HTTPException(status_code=404, detail="Session not found")

        upload = file or audio or audio_file
        if upload is None:
            raise HTTPException(status_code=400, detail="No audio file received. Expected 'file' (or 'audio'/'audio_file').")

        # Determine level first
        level = SESSIONS[session_id]["current_level"]

        # Resolve q_text from (prompt_id) > (question text) > (session) > (fresh)
        q_text = None
        # Try prompt_id (index into question bank)
        qid_int: Optional[int] = None
        if prompt_id is not None:
            try:
                qid_int = int(prompt_id)
            except Exception:
                qid_int = None
        if qid_int is not None:
            questions = _load_questions_for_level(level)
            if 0 <= qid_int < len(questions):
                q_text = questions[qid_int]

        # Fallback to provided question text
        if not q_text and question and isinstance(question, str) and question.strip():
            q_text = question.strip()

        # Fallback to session's current question
        if not q_text:
            q_text = SESSIONS[session_id].get("current_question")

        # Last resort: pick one now (and inject into session so it's consistent)
        if not q_text:
            q_text, qid = _get_question_for_session(level, session_id=session_id)
            SESSIONS[session_id]["current_question"] = q_text
            SESSIONS[session_id]["current_prompt_id"] = qid

        # Persist chosen question id if we derived it
        if qid_int is not None:
            SESSIONS[session_id]["current_prompt_id"] = qid_int
        else:
            # Try to back-compute index for consistency
            qs = _load_questions_for_level(level)
            try:
                SESSIONS[session_id]["current_prompt_id"] = qs.index(q_text)
            except Exception:
                SESSIONS[session_id]["current_prompt_id"] = None

        # Compute attempt based on prior history at this level
        prev_turns_same_level = sum(1 for t in SESSIONS[session_id]["history"] if t.level == level)
        attempt = prev_turns_same_level + 1

        src_path = _save_upload_to_temp(upload)
        needs_conv = _infer_should_convert(upload.content_type, upload.filename)
        wav_path = src_path if not needs_conv else _ffmpeg_convert_to_wav16k_mono(src_path)

        try:
            result = await transcribe_and_score(
                session_id=session_id,
                wav_file_path=wav_path,
                question=q_text,
                attempt=attempt
            )
        finally:
            try:
                if os.path.exists(src_path):
                    os.remove(src_path)
            except Exception:
                pass
            try:
                if needs_conv and os.path.exists(wav_path):
                    os.remove(wav_path)
            except Exception:
                pass

        return JSONResponse(result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("submit-response failed", extra={
            "error": str(e),
            "content_type": getattr(upload, "content_type", None) if 'upload' in locals() and upload else None,
            "filename": getattr(upload, "filename", None) if 'upload' in locals() and upload else None
        })
        raise HTTPException(status_code=400, detail=f"submit-response error: {str(e)[:400]}")

@app.post("/evaluate-bytes")
async def evaluate_bytes(file: UploadFile = File(...)):
    """
    Single-shot evaluation (not adaptive). Returns scores/avg for a one-off clip.
    """
    try:
        session_id = str(uuid.uuid4())
        first_q, first_id = _get_question_for_session("A1", session_id=session_id)
        SESSIONS[session_id] = {
            "current_level": "A1",
            "current_question": first_q,
            "current_prompt_id": first_id,
            "history": [],
            "used_questions": {}
        }
        src_path = _save_upload_to_temp(file)
        needs_conv = _infer_should_convert(file.content_type, file.filename)
        wav_path = src_path if not needs_conv else _ffmpeg_convert_to_wav16k_mono(src_path)
        try:
            out = await transcribe_and_score(
                session_id=session_id,
                wav_file_path=wav_path,
                question=first_q,
                attempt=1
            )
            out.pop("next_level", None)
            out.pop("next_prompt", None)
            out.pop("next_prompt_id", None)
            return JSONResponse(out)
        finally:
            try:
                if os.path.exists(src_path):
                    os.remove(src_path)
            except Exception:
                pass
            try:
                if needs_conv and os.path.exists(wav_path):
                    os.remove(wav_path)
            except Exception:
                pass
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"evaluate-bytes error: {str(e)[:400]}")

@app.get("/report/{session_id}", response_model=FinalReport)
async def report(session_id: str):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
    state = SESSIONS[session_id]
    final_level = state.get("final_level") or state["current_level"]

    # Simple recommendations based on last turn’s lowest category
    recs: List[str] = []
    if state["history"]:
        last: Turn = state["history"][-1]
        lows = sorted(
            [("fluency", last.scores.fluency),
             ("grammar", last.scores.grammar),
             ("vocabulary", last.scores.vocabulary),
             ("pronunciation", last.scores.pronunciation),
             ("coherence", last.scores.coherence)],
            key=lambda x: x[1]
        )
        focus = ", ".join([k for k, _ in lows[:2]])
        recs.append(f"Focus more on: {focus}.")

    return FinalReport(session_id=session_id, final_level=final_level, history=state["history"], recommendations=recs)

# ---------- TTS (examiner voice) ----------
# GET /tts?text=...  or POST {"text": "..."} -> returns audio/mpeg
@app.get("/tts")
async def tts_get(text: str = Query(..., description="Text to synthesize")):
    return await _tts_core(text)

@app.post("/tts")
async def tts_post(payload: Dict):
    text = payload.get("text") if isinstance(payload, dict) else None
    if not text or not isinstance(text, str):
        raise HTTPException(status_code=400, detail="Missing 'text' in body.")
    return await _tts_core(text)

async def _tts_core(text: str):
    if not openai_client:
        raise HTTPException(status_code=400, detail="TTS unavailable (no OpenAI client).")
    try:
        speech = openai_client.audio.speech.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            input=text,
            format="mp3"
        )
        audio_bytes = speech.read() if hasattr(speech, "read") else speech
        if isinstance(audio_bytes, str):
            audio_bytes = audio_bytes.encode("latin1")
        return Response(content=audio_bytes, media_type="audio/mpeg")
    except Exception as e:
        logger.exception("TTS failed")
        raise HTTPException(status_code=400, detail=f"TTS error: {str(e)[:300]}")

# ---------- Access log ----------
@app.middleware("http")
async def access_log(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = int((time.time() - start) * 1000)
    logger.info(f"{request.method} {request.url.path}", extra={
        "path": request.url.path, "method": request.method, "elapsed_ms": elapsed, "request_id": str(uuid.uuid4()),
        "remote_addr": request.client.host if request.client else None
    })
    return response
