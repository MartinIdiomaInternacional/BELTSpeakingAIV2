import uuid
import time
import logging
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

from .config import (
    CONF_FAST_JUMP, CONF_STRONG_JUMP, CONF_SOFT_JUMP,
    STREAK_REQUIRED, STREAK_MIN_CONF,
    MAX_TURNS, MAX_RECORDING_MINUTES
)

logger = logging.getLogger("uvicorn.error")

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
    cursors = state.setdefault("prompt_cursors", {})
    cursor = cursors.get(level, 0)
    items = PROMPTS.get(level, [])
    if not items:
        return "Please speak for ~30 seconds about your work."
    prompt = items[cursor % len(items)]["text"]
    cursors[level] = (cursor + 1) % len(items)
    state["prompt_cursors"] = cursors
    return prompt

def _next_index(current_idx: int, inferred_idx: int, confidence: float) -> int:
    if confidence >= CONF_STRONG_JUMP:
        return inferred_idx
    delta = inferred_idx - current_idx
    if confidence >= CONF_FAST_JUMP and abs(delta) >= 2:
        step = 2 if delta > 0 else -2
        return max(0, min(len(LEVELS)-1, current_idx + step))
    if confidence >= CONF_SOFT_JUMP:
        if delta > 0:  return min(current_idx+1, len(LEVELS)-1)
        if delta < 0:  return max(current_idx-1, 0)
        return current_idx
    return current_idx

def _update_streak(state: dict, inferred_idx: int, confidence: float):
    if state.get("streak_level_idx") is None or state["streak_level_idx"] != inferred_idx:
        state["streak_level_idx"] = inferred_idx
        state["streak_count"] = 1
        state["streak_conf_sum"] = float(confidence)
    else:
        state["streak_count"] += 1
        state["streak_conf_sum"] = float(state.get("streak_conf_sum", 0.0) + confidence)

def _is_finished(state: dict) -> bool:
    sc = state.get("streak_count", 0)
    if sc < STREAK_REQUIRED:
        return False
    avg_conf = (state.get("streak_conf_sum", 0.0) / sc) if sc else 0.0
    return avg_conf >= STREAK_MIN_CONF

def _final_from_history(state: dict):
    sc = state.get("streak_count", 0)
    if sc >= STREAK_REQUIRED:
        avg = state.get("streak_conf_sum", 0.0) / sc
        if avg >= STREAK_MIN_CONF:
            return LEVELS[state["streak_level_idx"]]
    hist = state.get("history", [])
    if not hist:
        return LEVELS[state["idx"]]
    weights_sum = 0.0
    acc = 0.0
    n = len(hist)
    for i, h in enumerate(hist):
        ii = h.get("inferred_idx")
        if ii is None: continue
        w = float(h.get("conf", 0.5))
        if i >= n - 3: w *= 2.0
        acc += w * ii
        weights_sum += w
    if weights_sum == 0:
        return LEVELS[state["idx"]]
    idx = int(round(acc / weights_sum))
    idx = max(0, min(idx, len(LEVELS)-1))
    return LEVELS[idx]

def _is_budget_exhausted(sess: dict) -> bool:
    state = sess["state"]
    # Turn cap
    if len(state.get("history", [])) >= MAX_TURNS:
        return True
    # Recording-time cap (only speech duration counted)
    rec_total_sec = float(state.get("recording_sec_total", 0.0))
    return (rec_total_sec / 60.0) >= MAX_RECORDING_MINUTES

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
            "prompt_cursors": {},
            "started_at_ts": time.time(),
            "recording_sec_total": 0.0,   # NEW: cumulative recording seconds
        },
        "final": None,
    }
    save_session(sid, req.candidate_id, (req.native_language or "en").lower())
    lvl = LEVELS[0]
    prompt = _pick_prompt_for_level(SESSIONS[sid]["state"], lvl)
    logger.info(f"[START] sid={sid[:8]} candidate={req.candidate_id} level={lvl}")
    return StartResponse(session_id=sid, message="Session created", current_level=lvl, prompt=prompt)

@app.post("/evaluate-bytes", response_model=EvaluateResponse)
def evaluate_bytes(req: EvaluateBytesRequest):
    if req.session_id not in SESSIONS:
        raise HTTPException(404, detail="Invalid session")
    sess = SESSIONS[req.session_id]
    state = sess["state"]
    asked_idx = state["idx"]
    asked_level = LEVELS[asked_idx]

    # Decode & features
    wav_bytes = decode_base64_maybe_webm_to_wav(req.wav_base64)
    audio, sr = decode_base64_maybe_webm_to_wav_mono_float(req.wav_base64)
    feats = compute_basic_features(audio, sr)

    # Compute precise audio duration (seconds) from decoded signal
    duration_sec = float(len(audio) / float(sr) if sr else 0.0)
    # Accumulate recording time even if quality fails (still spent speaking)
    state["recording_sec_total"] = float(state.get("recording_sec_total", 0.0) + max(0.0, duration_sec))

    q_ok, q_reason = check_quality(feats)
    transcription = None

    if q_ok:
        transcription = transcribe_wav_bytes(wav_bytes)
        if transcription:
            from .utils.text_metrics import basic_text_metrics
            tm = basic_text_metrics(transcription)
            lex_boost = min(0.1, tm.get("ttr", 0.0) * 0.1)
        else:
            lex_boost = 0.0

        score, conf, metrics = score_from_features(feats)
        conf = float(min(0.98, max(0.0, conf + lex_boost)))
        inferred_level = map_score_to_level(score)
        inferred_idx = IDX[inferred_level]

        state["history"].append({
            "asked_idx": asked_idx, "inferred_idx": inferred_idx,
            "score": float(score), "conf": float(conf),
            "duration_sec": duration_sec,
        })
        _update_streak(state, inferred_idx, conf)

        turn_no = len(state["history"])
        logger.info(f"[TURN] sid={req.session_id[:8]} turn={turn_no} asked={asked_level} -> inferred={inferred_level} conf={conf:.2f} rec_total={state['recording_sec_total']:.1f}s")
        save_turn(req.session_id, turn_no, asked_level, inferred_level, float(score), float(conf), transcription, True, None)

        if _is_finished(state) or _is_budget_exhausted(sess):
            final_level = _final_from_history(state)
            sess["final"] = {"level": final_level, "score_0_8": score, "confidence": conf, "metrics": metrics}
            finalize_session(req.session_id, final_level, float(score), float(conf))
            return EvaluateResponse(
                session_id=req.session_id,
                turn=TurnResult(
                    asked_level=asked_level, inferred_level=inferred_level,
                    score_0_8=score, confidence=conf, transcription=transcription,
                    duration_sec=duration_sec
                ),
                finished=True,
                final_level=final_level, final_score_0_8=score, final_confidence=conf,
                total_recording_sec=state["recording_sec_total"],
            )

        next_idx = _next_index(asked_idx, inferred_idx, conf)
        state["idx"] = next_idx
        next_level = LEVELS[next_idx]
        next_prompt = _pick_prompt_for_level(state, next_level)
        return EvaluateResponse(
            session_id=req.session_id,
            turn=TurnResult(
                asked_level=asked_level, inferred_level=inferred_level,
                score_0_8=score, confidence=conf, transcription=transcription,
                duration_sec=duration_sec
            ),
            finished=False,
            next_level=next_level,
            next_prompt=next_prompt,
            total_recording_sec=state["recording_sec_total"],
        )

    else:
        turn_no = len(state.get("history", [])) + 1
        logger.info(f"[QUALITY] sid={req.session_id[:8]} turn={turn_no} asked={asked_level} reason={q_reason} rec_total={state['recording_sec_total']:.1f}s")
        save_turn(req.session_id, turn_no, asked_level, None, None, None, None, False, q_reason)

        if _is_budget_exhausted(sess):
            final_level = _final_from_history(state)
            sess["final"] = {"level": final_level, "score_0_8": None, "confidence": None, "metrics": {}}
            finalize_session(req.session_id, final_level, None, None)
            return EvaluateResponse(
                session_id=req.session_id,
                turn=TurnResult(asked_level=asked_level, quality_ok=False, quality_reason=q_reason, duration_sec=duration_sec),
                finished=True,
                final_level=final_level,
                total_recording_sec=state["recording_sec_total"],
            )

        # Rotate a different prompt at the same level
        next_prompt = _pick_prompt_for_level(state, asked_level)
        return EvaluateResponse(
            session_id=req.session_id,
            turn=TurnResult(asked_level=asked_level, quality_ok=False, quality_reason=q_reason, duration_sec=duration_sec),
            finished=False,
            next_level=asked_level,
            next_prompt=next_prompt,
            total_recording_sec=state["recording_sec_total"],
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
