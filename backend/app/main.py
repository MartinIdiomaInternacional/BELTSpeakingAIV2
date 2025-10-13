
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .models import (
    StartRequest, StartResponse,
    EvaluateBytesRequest, EvaluateResponse, TurnResult,
    ReportRequest, ReportResponse
)
from .prompts import PROMPTS
from .utils.decode import decode_base64_maybe_webm_to_wav_mono_float, decode_base64_maybe_webm_to_wav
from .scoring.audio_features import compute_basic_features
from .scoring.cefr_scorer import score_from_features, map_score_to_level
from .utils.quality import check_quality
from .utils.asr import transcribe_wav_bytes
from .utils.text_metrics import basic_text_metrics
from .utils.report import render_html
from .version import VERSION
from .db import save_session, save_turn, finalize_session

LEVELS = ["A1","A2","B1","B1+","B2","B2+","C1","C2"]
IDX = {l:i for i,l in enumerate(LEVELS)}

app = FastAPI(title="Speaking AI Eval (Adaptive Pro)", version=VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSIONS = {}

@app.get("/")
def root():
    return {"message":"BELT Speaking AI Eval API is running", "version": VERSION}

@app.get("/health")
def health():
    return {"status":"ok","version":VERSION}

def _pick_prompt_for_level(state: dict, level: str) -> str:
    # Rotate prompts per level to avoid repetition when staying at same level
    cursors = state.setdefault("prompt_cursors", {})
    cursor = cursors.get(level, 0)
    items = PROMPTS.get(level, [])
    if not items:
        return "Please speak for 30s about your work."
    prompt = items[cursor % len(items)]["text"]
    cursors[level] = (cursor + 1) % len(items)
    state["prompt_cursors"] = cursors
    return prompt

def _next_index(current_idx: int, inferred_idx: int, confidence: float) -> int:
    if confidence >= 0.7:
        return inferred_idx
    if confidence >= 0.5:
        if inferred_idx > current_idx: return min(current_idx+1, len(LEVELS)-1)
        if inferred_idx < current_idx: return max(current_idx-1, 0)
        return current_idx
    return current_idx

def _update_streak(state: dict, inferred_idx: int, confidence: float):
    if state["streak_level_idx"] is None or state["streak_level_idx"] != inferred_idx:
        state["streak_level_idx"] = inferred_idx
        state["streak_count"] = 1
        state["streak_conf_sum"] = confidence
    else:
        state["streak_count"] += 1
        state["streak_conf_sum"] += confidence

def _is_finished(state: dict, min_streak: int = 3, min_avg_conf: float = 0.65) -> bool:
    if state.get("streak_count", 0) < min_streak: return False
    avg = state["streak_conf_sum"] / state["streak_count"]
    return avg >= min_avg_conf

@app.post("/start", response_model=StartResponse)
def start(req: StartRequest):
    sid = str(uuid.uuid4())
    SESSIONS[sid] = {
        "candidate_id": req.candidate_id,
        "native_language": (req.native_language or "en").lower(),
        "state": {
            "idx": 0,
            "streak_level_idx": None, "streak_count": 0, "streak_conf_sum": 0.0,
            "history": [],
            "prompt_cursors": {},   # store per-level rotation cursor
        },
        "final": None,
    }
    save_session(sid, req.candidate_id, (req.native_language or "en").lower())
    lvl = LEVELS[0]  # A1
    return StartResponse(session_id=sid, message="Session created",
                         current_level=lvl, prompt=_pick_prompt_for_level(SESSIONS[sid]["state"], lvl))

@app.post("/evaluate-bytes", response_model=EvaluateResponse)
def evaluate_bytes(req: EvaluateBytesRequest):
    if req.session_id not in SESSIONS:
        raise HTTPException(404, detail="Invalid session")
    sess = SESSIONS[req.session_id]
    state = sess["state"]
    asked_idx = state["idx"]
    asked_level = LEVELS[asked_idx]

    wav_bytes = decode_base64_maybe_webm_to_wav(req.wav_base64)
    audio, sr = decode_base64_maybe_webm_to_wav_mono_float(req.wav_base64)
    feats = compute_basic_features(audio, sr)

    q_ok, q_reason = check_quality(feats)
    transcription = None

    if q_ok:
        transcription = transcribe_wav_bytes(wav_bytes)
        if transcription:
            tm = basic_text_metrics(transcription)
            lex_boost = min(0.1, tm["ttr"] * 0.1)
        else:
            lex_boost = 0.0

        score, conf, metrics = score_from_features(feats)
        conf = float(min(0.98, max(0.0, conf + lex_boost)))
        inferred_level = map_score_to_level(score)
        inferred_idx = IDX[inferred_level]

        state["history"].append({"asked_idx": asked_idx, "inferred_idx": inferred_idx, "score": float(score), "conf": float(conf)})
        _update_streak(state, inferred_idx, conf)

        save_turn(req.session_id, len(state["history"]), asked_level, inferred_level, float(score), float(conf), transcription, True, None)

        if _is_finished(state):
            sess["final"] = {"level": inferred_level, "score_0_8": score, "confidence": conf, "metrics": metrics}
            finalize_session(req.session_id, inferred_level, float(score), float(conf))
            return EvaluateResponse(
                session_id=req.session_id,
                turn=TurnResult(asked_level=asked_level, inferred_level=inferred_level, score_0_8=score, confidence=conf, transcription=transcription),
                finished=True,
                final_level=inferred_level, final_score_0_8=score, final_confidence=conf
            )

        next_idx = _next_index(asked_idx, inferred_idx, conf)
        state["idx"] = next_idx
        next_level = LEVELS[next_idx]
        next_prompt = _pick_prompt_for_level(state, next_level)

        return EvaluateResponse(
            session_id=req.session_id,
            turn=TurnResult(asked_level=asked_level, inferred_level=inferred_level, score_0_8=score, confidence=conf, transcription=transcription),
            finished=False,
            next_level=next_level,
            next_prompt=next_prompt
        )
    else:
        save_turn(req.session_id, len(state["history"])+1, asked_level, None, None, None, None, False, q_reason)
        # Even on quality fail, rotate to a *new* prompt at the same level to avoid exact repetition
        next_prompt = _pick_prompt_for_level(state, asked_level)
        return EvaluateResponse(
            session_id=req.session_id,
            turn=TurnResult(asked_level=asked_level, quality_ok=False, quality_reason=q_reason),
            finished=False,
            next_level=asked_level,
            next_prompt=next_prompt
        )

@app.post("/report", response_model=ReportResponse)
def report(req: ReportRequest):
    if req.session_id not in SESSIONS:
        raise HTTPException(404, detail="Invalid session")
    sess = SESSIONS[req.session_id]
    final = sess.get("final")
    if not final:
        raise HTTPException(400, detail="Session not finished yet")
    lang = (req.native_language or sess.get("native_language") or "en").lower()
    r = render_html(req.session_id, final["level"], final["score_0_8"], final["confidence"], lang)
    return ReportResponse(session_id=req.session_id, final=final, feedback=r["feedback"], html=r["html"])
