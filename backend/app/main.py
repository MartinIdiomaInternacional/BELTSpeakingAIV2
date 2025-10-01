import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .models import (
    StartRequest, StartResponse,
    EvaluateBytesRequest, EvaluateResponse,
    ReportRequest, ReportResponse, ScoreDetail
)
from .prompts import PROMPTS
from .utils.decode import decode_base64_maybe_webm_to_wav_mono_float
from .scoring.audio_features import compute_basic_features
from .scoring.cefr_scorer import score_from_features, map_score_to_level, borderline, combine_scores
from .utils.report import render_html
from .version import VERSION

app = FastAPI(title="Working Speaking AI Eval 2.0", version=VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSIONS = {}

@app.get("/health")
def health():
    return {"status":"ok","version":VERSION}

@app.post("/start", response_model=StartResponse)
def start(req: StartRequest):
    sid = str(uuid.uuid4())
    SESSIONS[sid] = {
        "candidate_id": req.candidate_id,
        "native_language": (req.native_language or "en").lower(),
        "target_level": req.target_level,
        "base": None,
        "probe": None,
        "final": None,
    }
    return StartResponse(session_id=sid, message="Session created")

@app.get("/prompts/{level}")
def get_prompts(level: str):
    if level not in PROMPTS:
        raise HTTPException(404, detail="Unknown level")
    return PROMPTS[level]

@app.post("/evaluate-bytes", response_model=EvaluateResponse)
def evaluate_bytes(req: EvaluateBytesRequest):
    if req.session_id not in SESSIONS:
        raise HTTPException(404, detail="Invalid session")

    audio, sr = decode_base64_maybe_webm_to_wav_mono_float(req.wav_base64)
    feats = compute_basic_features(audio, sr)
    score, conf, metrics = score_from_features(feats)
    level = map_score_to_level(score)

    detail = ScoreDetail(level=level, score_0_8=score, confidence=conf, metrics={**feats, **metrics})

    sess = SESSIONS[req.session_id]
    probe_needed = False
    probe_prompt = None

    if sess["base"] is None:
        sess["base"] = detail
        probe_needed = borderline(score, conf)
        if probe_needed:
            tgt = sess.get("target_level") or level
            prompt_list = PROMPTS.get(tgt, PROMPTS[level])
            probe_prompt = prompt_list[0]["text"] if prompt_list else "Provide further details on your last answer."
    else:
        sess["probe"] = detail
        base = sess["base"]
        s, c = combine_scores(base.score_0_8, base.confidence, detail.score_0_8, detail.confidence)
        lvl = map_score_to_level(s)
        sess["final"] = ScoreDetail(level=lvl, score_0_8=s, confidence=c, metrics={"combined": True})

    return EvaluateResponse(session_id=req.session_id, base=detail, needs_probe=probe_needed, probe_prompt=probe_prompt)

@app.post("/report", response_model=ReportResponse)
def report(req: ReportRequest):
    if req.session_id not in SESSIONS:
        raise HTTPException(404, detail="Invalid session")
    sess = SESSIONS[req.session_id]
    final = sess.get("final") or sess.get("probe") or sess.get("base")
    if final is None:
        raise HTTPException(400, detail="No evaluation data yet")
    lang = (req.native_language or sess.get("native_language") or "en").lower()
    r = render_html(req.session_id, final.level, final.score_0_8, final.confidence, lang)
    return ReportResponse(session_id=req.session_id, final=final, feedback=r["feedback"], html=r["html"])
