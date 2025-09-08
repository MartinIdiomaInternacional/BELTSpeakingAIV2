import os
import io
import json
import time
import uuid
import glob
import math
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Optional, List

from fastapi import FastAPI, HTTPException, Request
from fastapi import UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, PlainTextResponse
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
RECORD_SECONDS = int(os.getenv("RECORD_SECONDS", "60"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ---------- LLM/ASR clients (OpenAI style) ----------
# Supports modern openai SDK (client.chat.completions / client.audio.transcriptions)
try:
    from openai import OpenAI
    openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception as e:
    logger.warning("OpenAI SDK not available or API key missing; falling back to dummy scoring.")
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

# Prompts by level
@app.get("/prompts/{level}")
async def prompt_level(level: str):
    # accept B1+ style as filename B1+.json
    p = PROMPTS_DIR / f"{level}.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Prompt for level {level} not found")
    return JSONResponse(json.loads(p.read_text(encoding="utf-8")))

# ---------- Health / metrics ----------
@app.get("/healthz")
async def healthz():
    return PlainTextResponse("ok")

@app.get("/metrics")
async def metrics():
    # Minimal Prometheus-style metrics
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
        "RECORD_SECONDS": RECORD_SECONDS,
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
    transcript: str
    scores: ScoreDict
    average: float
    decision: str  # "advance" or "stop"

class FinalReport(BaseModel):
    session_id: str
    final_level: str
    history: List[Turn]
    recommendations: List[str] = Field(default_factory=list)

SESSIONS: Dict[str, Dict] = {}  # session_id -> { current_level, history: [Turn], final_level? }

def _next_level(current: str) -> Optional[str]:
    if current not in CEFR_ORDER:
        return None
    idx = CEFR_ORDER.indexOf(current) if hasattr(CEFR_ORDER, "indexOf") else CEFR_ORDER.index(current)
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
        # Fallback: naive heuristic scoring for offline sanity
        length = len(transcript.split())
        base = min(1.0, max(0.1, length / 120))
        return {
            "fluency": round(base, 2),
            "grammar": round(base * 0.95, 2),
            "vocabulary": round(base * 0.9, 2),
            "pronunciation": round(0.75, 2),
            "coherence": round(base * 0.92, 2),
            "feedback": "Heuristic offline scoring (no API key)."
        }

    user_prompt = f"""Level: {level}
Transcript:
{transcript}
"""
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
    except Exception as e:
        logger.exception("Failed to parse rubric model JSON; falling back to heuristic.")
        # Heuristic fallback if parse fails
        length = len(transcript.split())
        base = min(1.0, max(0.1, length / 120))
        return {
            "fluency": round(base, 2),
            "grammar": round(base * 0.95, 2),
            "vocabulary": round(base * 0.9, 2),
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

def _load_prompt(level: str) -> str:
    p = PROMPTS_DIR / f"{level}.json"
    if not p.exists():
        return f"Speak for ~60 seconds about a topic suitable for {level}."
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            instr = data.get("instructions") or ""
            return instr or f"Speak for ~60 seconds about a topic suitable for {level}."
        return f"Speak for ~60 seconds about a topic suitable for {level}."
    except Exception:
        return f"Speak for ~60 seconds about a topic suitable for {level}."

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
    # openai==1.x returns string for response_format="text"
    return trans if isinstance(trans, str) else str(trans)

async def transcribe_and_score(session_id: str, wav_file_path: str) -> Dict:
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

    decision = "advance" if passed else "stop"
    result: Dict = {
        "session_id": session_id,
        "level": level,
        "transcript": transcript,
        "scores": scores.model_dump(),
        "average": round(avg, 4),
        "decision": decision,
    }

    # 3) Update session history
    state["history"].append(Turn(level=level, transcript=transcript, scores=scores, average=avg, decision=decision))

    if passed:
        nxt = _next_level(level)
        if nxt:
            state["current_level"] = nxt
            result["next_level"] = nxt
            result["next_prompt"] = _load_prompt(nxt)
        else:
            # already at top level; stop
            state["final_level"] = level
            result["decision"] = "stop"
    else:
        # stop at current level
        state["final_level"] = state.get("final_level") or level

    return result

# ---------- API: start-session / submit-response / evaluate-bytes / report ----------

@app.post("/start-session")
async def start_session(level: str = Form("A1")):
    if level not in CEFR_ORDER:
        level = "A1"
    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = {
        "current_level": level,
        "history": [],
        # final_level populated when stop
    }
    logger.info("start-session", extra={"session_id": session_id, "level": level})
    return {
        "session_id": session_id,
        "level": level,
        "prompt": _load_prompt(level),
    }

@app.post("/submit-response")
async def submit_response(
    # Accept multiple possible field names; the first non-None will be used.
    file: Optional[UploadFile] = File(default=None),
    audio: Optional[UploadFile] = File(default=None),
    audio_file: Optional[UploadFile] = File(default=None),
    session_id: str = Form(...),
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

        src_path = _save_upload_to_temp(upload)
        needs_conv = _infer_should_convert(upload.content_type, upload.filename)
        wav_path = src_path if not needs_conv else _ffmpeg_convert_to_wav16k_mono(src_path)

        try:
            result = await transcribe_and_score(session_id=session_id, wav_file_path=wav_path)
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
        # Use a scratch session for single-shot
        session_id = str(uuid.uuid4())
        SESSIONS[session_id] = {"current_level": "A1", "history": []}
        src_path = _save_upload_to_temp(file)
        needs_conv = _infer_should_convert(file.content_type, file.filename)
        wav_path = src_path if not needs_conv else _ffmpeg_convert_to_wav16k_mono(src_path)
        try:
            out = await transcribe_and_score(session_id=session_id, wav_file_path=wav_path)
            # do not advance; just return the first scoring
            out.pop("next_level", None)
            out.pop("next_prompt", None)
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
    # simple recs
    recs: List[str] = []
    if state["history"]:
        last = state["history"][-1]
        low = min(
            [("fluency", last.scores.fluency),
             ("grammar", last.scores.grammar),
             ("vocabulary", last.scores.vocabulary),
             ("pronunciation", last.scores.pronunciation),
             ("coherence", last.scores.coherence)],
            key=lambda x: x[1]
        )[0]
        recs.append(f"Focus more on {low}.")
    return FinalReport(session_id=session_id, final_level=final_level, history=state["history"], recommendations=recs)

# ---------- Error handlers ----------
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
