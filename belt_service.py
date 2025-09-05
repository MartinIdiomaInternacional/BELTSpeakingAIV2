
"""
belt_service.py — One-Service BELT Speaking AI
----------------------------------------------
- /evaluate-bytes : signal analysis + CEFR level heuristic
- /start-session, /submit-response, /report/{id} : adaptive progression with ASR+LLM rubric
- /healthz, /metrics
- Frontend mounted at / (index.html), /static, /prompts/{level}

Run:
    uvicorn belt_service:app --host 0.0.0.0 --port 8000

Env:
    OPENAI_API_KEY=...
    ASR_BACKEND=openai|none        (default: openai; set none for offline dev)
    RUBRIC_MODEL=gpt-4o-mini
    WHISPER_MODEL=whisper-1
    PASS_AVG_THRESHOLD=0.70
    PASS_MIN_THRESHOLD=0.60
"""
import io
import os
import json
import time
import uuid
from typing import Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

# Audio & DSP
import soundfile as sf
from scipy.signal import resample_poly
import audioread

# Metrics
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# ---------- Logging ----------
import logging
APP_NAME = "belt-service"
class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "ts": int(time.time() * 1000),
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
        }
        for key in ("request_id","path","method","remote_addr","elapsed_ms"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        return json.dumps(payload)

logger = logging.getLogger(APP_NAME)
handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# ---------- App & Metrics ----------
app = FastAPI(title="BELT Speaking AI (One Service)", version="1.0.0")
REQ_COUNTER = Counter("belt_requests_total","Total requests",["route","method"])
REQ_LATENCY = Histogram("belt_request_latency_seconds","Request latency",["route","method"])
AUDIO_BYTES = Histogram("belt_audio_upload_bytes","Uploaded audio size (bytes)")
EVAL_SCORE = Histogram("belt_eval_score","Heuristic evaluation score (0..1)")
READY = Gauge("belt_ready","Readiness (1 ready, 0 not)")
READY.set(1.0)

class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        start = time.perf_counter()
        route = request.url.path
        method = request.method
        REQ_COUNTER.labels(route=route, method=method).inc()
        try:
            response = await call_next(request)
            return response
        finally:
            elapsed = time.perf_counter() - start
            REQ_LATENCY.labels(route=route, method=method).observe(elapsed)
            rec = logging.LogRecord(name=APP_NAME, level=logging.INFO, pathname=__file__, lineno=0,
                                    msg=f"{method} {route}", args=(), exc_info=None)
            rec.request_id = request_id
            rec.path = route
            rec.method = method
            rec.remote_addr = request.client.host if request.client else None
            rec.elapsed_ms = int(elapsed * 1000)
            logger.handle(rec)

app.add_middleware(RequestContextMiddleware)

# ---------- Evaluator utils ----------
TARGET_SR = 16_000
MAX_FILE_MB = 10
MAX_SAMPLES = TARGET_SR * 60 * 5

def _to_mono(x: np.ndarray) -> np.ndarray:
    return x if x.ndim == 1 else np.mean(x, axis=1)

def _resample(x: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return x
    from math import gcd
    g = gcd(sr, target_sr)
    up = target_sr // g
    down = sr // g
    return resample_poly(x, up, down)

def load_audio_safe(file_bytes: bytes, filename: str) -> Tuple[np.ndarray, int]:
    try:
        with sf.SoundFile(io.BytesIO(file_bytes)) as f:
            sr = f.samplerate
            data = f.read(always_2d=True)
            x = data.astype(np.float32)
            x = _to_mono(x)
    except Exception:
        try:
            with audioread.audio_open(io.BytesIO(file_bytes)) as af:
                sr = af.samplerate
                pcm = bytearray()
                for buf in af: pcm.extend(buf)
                x = np.frombuffer(bytes(pcm), dtype=np.int16).astype(np.float32) / 32768.0
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Unsupported or corrupted audio file: {filename} ({e})")
    if x.size == 0:
        raise HTTPException(status_code=400, detail="Empty audio stream")
    if x.shape[0] > MAX_SAMPLES * max(1, sr // TARGET_SR):
        x = x[: MAX_SAMPLES * max(1, sr // TARGET_SR)]
    x = _resample(x, sr, TARGET_SR).astype(np.float32)
    x = _to_mono(x)
    return x, TARGET_SR

def trim_silence(x: np.ndarray, sr: int, frame: int = 400, hop: int = 160, thresh_db: float = -45.0, pad_ms: int = 100):
    if x.size < frame: return x, 0.0, 0.0
    n_frames = 1 + (len(x) - frame) // hop
    E = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        seg = x[i*hop: i*hop + frame]
        E[i] = 10*np.log10(np.mean(seg**2)+1e-12)
    mask = E > thresh_db
    if not np.any(mask): return x, 0.0, 0.0
    first = int(np.argmax(mask)); last = int(len(mask)-1-np.argmax(mask[::-1]))
    pad = int((pad_ms/1000.0)*sr)
    start = max(0, first*hop - pad); end = min(len(x), last*hop + frame + pad)
    return x[start:end], start/sr, (len(x)-end)/sr

def rms_energy(x, frame=400, hop=160):
    if x.size < frame: return np.array([np.sqrt(np.mean(x**2)+1e-12)], dtype=np.float32)
    n = 1 + (len(x)-frame)//hop
    E = np.empty(n, dtype=np.float32)
    for i in range(n):
        seg = x[i*hop:i*hop+frame]; E[i]=np.sqrt(np.mean(seg**2)+1e-12)
    return E

def zero_crossing_rate(x, frame=400, hop=160):
    if x.size < frame: return np.array([0.0], dtype=np.float32)
    n = 1 + (len(x)-frame)//hop
    Z = np.empty(n, dtype=np.float32)
    for i in range(n):
        seg = x[i*hop:i*hop+frame]; Z[i]=(np.mean(np.abs(np.diff(np.sign(seg)))))/2.0
    return Z

def spectral_centroid(x, sr, n_fft=512, hop=160):
    if x.size < n_fft:
        X = np.fft.rfft(x, n=n_fft); mag = np.abs(X); freqs=np.fft.rfftfreq(n_fft, d=1.0/sr)
        return np.array([float(np.sum(freqs*mag)/(np.sum(mag)+1e-12))], dtype=np.float32)
    frames=[]
    for start in range(0, len(x)-n_fft+1, hop):
        seg=x[start:start+n_fft]; X=np.fft.rfft(seg*np.hanning(n_fft)); mag=np.abs(X)
        freqs=np.fft.rfftfreq(n_fft, d=1.0/sr); frames.append(float(np.sum(freqs*mag)/(np.sum(mag)+1e-12)))
    return np.array(frames, dtype=np.float32)

def heuristic_evaluate(x: np.ndarray, sr: int):
    warnings=[]
    dur=len(x)/sr
    if dur<3.0: warnings.append("Very short duration (<3s) may reduce reliability.")
    if dur>120.0: warnings.append("Long duration; truncated to 5 minutes.")
    E=rms_energy(x); Z=zero_crossing_rate(x); C=spectral_centroid(x,sr)
    speech_frames=(E>(E.mean()*0.5)).sum(); vad_ratio=speech_frames/max(1,len(E))
    zcr_var=float(np.var(Z)); cent_var=float(np.var(C)); energy_mean=float(np.mean(E))
    vad_score=float(np.clip((vad_ratio-0.2)/0.7,0,1))
    zcr_score=float(np.clip(1.0-np.tanh(3*zcr_var),0,1))
    cent_score=float(np.clip(1.0-np.tanh(1e-4*cent_var),0,1))
    loud_score=float(np.clip(np.log1p(energy_mean*1000.0),0,1))
    raw=0.35*vad_score+0.25*zcr_score+0.25*cent_score+0.15*loud_score
    conf=float(np.clip(0.4+0.3*vad_score+0.2*(1-abs(zcr_var-0.02))+0.1*np.clip(dur/30.0,0,1),0,1))
    details={"duration_sec":dur,"features":{"vad_ratio":vad_ratio,"zcr_var":zcr_var,"centroid_var":cent_var,"energy_mean":energy_mean,
             "scores":{"vad":vad_score,"zcr_stability":zcr_score,"centroid_stability":cent_score,"loudness":loud_score}}}
    return float(np.clip(raw,0,1)), details, warnings

def map_score_to_cefr(score: float) -> str:
    if score<0.20: return "A1"
    if score<0.30: return "A2"
    if score<0.45: return "B1"
    if score<0.55: return "B1+"
    if score<0.68: return "B2"
    if score<0.78: return "B2+"
    if score<0.88: return "C1"
    return "C2"

class EvalResponse(BaseModel):
    level: str
    score: float = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0, le=1)
    duration_sec: float = Field(..., ge=0)
    sample_rate: int = Field(..., ge=8000)
    warnings: List[str] = Field(default_factory=list)
    details: Dict = Field(default_factory=dict)

# ---------- Health & Metrics ----------
@app.get("/healthz", response_class=PlainTextResponse)
async def healthz(): return "ok"

@app.get("/readyz", response_class=PlainTextResponse)
async def readyz(): return "ready" if READY._value.get()==1.0 else "not-ready"

@app.get("/metrics")
async def metrics(): return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# ---------- /evaluate-bytes ----------
@app.post("/evaluate-bytes", response_model=EvalResponse)
async def evaluate_bytes(request: Request, file: UploadFile = File(...)):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    if not file or not file.filename: raise HTTPException(status_code=400, detail="No file uploaded")
    content = await file.read()
    AUDIO_BYTES.observe(len(content))
    size_mb = len(content)/(1024*1024)
    if size_mb>MAX_FILE_MB: raise HTTPException(status_code=413, detail=f"File too large ({size_mb:.2f} MB). Limit {MAX_FILE_MB} MB.")
    try:
        x, sr = load_audio_safe(content, file.filename)
    except HTTPException: raise
    except Exception as e: raise HTTPException(status_code=400, detail=f"Unable to decode audio: {e}")
    x_trim, lead, trail = trim_silence(x, sr)
    score, details, warnings = heuristic_evaluate(x_trim, sr)
    level = map_score_to_cefr(score)
    EVAL_SCORE.observe(score)
    rec = logging.LogRecord(name=APP_NAME, level=logging.INFO, pathname=__file__, lineno=0,
                            msg=f"evaluated {file.filename} -> {level} ({score:.3f})", args=(), exc_info=None)
    rec.request_id = request_id; logger.handle(rec)
    details.update({"trim_leading_sec":lead,"trim_trailing_sec":trail})
    return EvalResponse(level=level, score=score,
                        confidence=float(np.clip(details["features"]["scores"]["vad"]*0.5 + score*0.5, 0, 1)),
                        duration_sec=float(details["duration_sec"]), sample_rate=sr,
                        warnings=warnings, details=details)

# ---------- Adaptive Session Manager (ASR + LLM) ----------
CEFR_ORDER = ["A1","A2","B1","B1+","B2","B2+","C1","C2"]
PASS_AVG_THRESHOLD = float(os.getenv("PASS_AVG_THRESHOLD","0.70"))
PASS_MIN_THRESHOLD = float(os.getenv("PASS_MIN_THRESHOLD","0.60"))
ASR_BACKEND = os.getenv("ASR_BACKEND","openai").lower()
RUBRIC_MODEL = os.getenv("RUBRIC_MODEL","gpt-4o-mini")
WHISPER_MODEL = os.getenv("WHISPER_MODEL","whisper-1")
PROMPTS = {
    "A1":"Introduce yourself: your name, job/studies, and one hobby.",
    "A2":"Describe a recent weekend or holiday you enjoyed. What did you do?",
    "B1":"Discuss advantages and disadvantages of remote work.",
    "B1+":"Talk about a challenge at work/study and how you solved it.",
    "B2":"Compare two approaches to team communication and argue for one.",
    "B2+":"Explain a complex process from your field to a newcomer.",
    "C1":"Evaluate the impact of AI on your industry, including risks and benefits.",
    "C2":"Present a nuanced argument on a controversial topic with counterarguments.",
}
class StartSessionResponse(BaseModel):
    session_id: str; level: str; prompt: str
class SubmitResponse(BaseModel):
    session_id: str; level: str; scores: Dict[str,float]; decision: str
    next_level: Optional[str]=None; next_prompt: Optional[str]=None
    transcript: Optional[str]=None; signal: Dict = Field(default_factory=dict)
class FinalReport(BaseModel):
    session_id: str; final_level: str; history: List[Dict]; recommendations: List[str]=Field(default_factory=list)
SESSIONS: Dict[str, Dict] = {}

def _next_level(level: str) -> Optional[str]:
    if level not in CEFR_ORDER: return None
    i = CEFR_ORDER.index(level); return CEFR_ORDER[i+1] if i+1 < len(CEFR_ORDER) else None
def _pass_fail(scores: Dict[str,float]) -> bool:
    vals=[v for v in scores.values() if isinstance(v,(int,float))]
    return bool(vals) and (np.mean(vals)>=PASS_AVG_THRESHOLD) and (min(vals)>=PASS_MIN_THRESHOLD)

def asr_transcribe_openai(bytes_data: bytes, filename: str) -> str:
    try:
        from openai import OpenAI
        client = OpenAI()
        resp = client.audio.transcriptions.create(
            model=WHISPER_MODEL, file=(filename, io.BytesIO(bytes_data), "application/octet-stream")
        )
        return getattr(resp,"text","") or ""
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI ASR failed: {e}")

def transcribe_audio(x: np.ndarray, sr: int, raw_bytes: bytes, filename: str) -> str:
    if ASR_BACKEND=="none": return "[TRANSCRIPT_PLACEHOLDER]"
    return asr_transcribe_openai(raw_bytes, filename)

RUBRIC_SYS_PROMPT = """
You are a strict CEFR speaking examiner. Score the candidate's response (transcript provided)
on these five categories, output JSON ONLY with floats 0..1:
{"grammar":0.0,"vocabulary":0.0,"pronunciation":0.0,"fluency":0.0,"coherence":0.0}
Anchor points: 0.20≈A1, 0.30≈A2, 0.45≈B1, 0.55≈B1+, 0.68≈B2, 0.78≈B2+, 0.88≈C1, 0.95+≈C2
"""
def rubric_score_llm(transcript: str) -> Dict[str,float]:
    try:
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model=RUBRIC_MODEL,
            messages=[{"role":"system","content":RUBRIC_SYS_PROMPT},
                      {"role":"user","content":f"Transcript:\n{transcript}\nReturn JSON only."}],
            response_format={"type":"json_object"}, temperature=0.2,
        )
        data=json.loads(resp.choices[0].message.content)
        out={}
        for k in ["grammar","vocabulary","pronunciation","fluency","coherence"]:
            v=float(data.get(k,0)); out[k]=float(max(0.0,min(1.0,v)))
        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM rubric scoring failed: {e}")

@app.post("/start-session", response_model=StartSessionResponse)
async def start_session(level: str = Form("A1")):
    if level not in CEFR_ORDER: raise HTTPException(status_code=400, detail="Invalid start level")
    sid=str(uuid.uuid4())
    SESSIONS[sid]={"session_id":sid,"current_level":level,"history":[],"final_level":None,"started_at":int(time.time())}
    return StartSessionResponse(session_id=sid, level=level, prompt=PROMPTS[level])

@app.post("/submit-response", response_model=SubmitResponse)
async def submit_response(session_id: str = Form(...), file: UploadFile = File(...)):
    if session_id not in SESSIONS: raise HTTPException(status_code=404, detail="Session not found")
    state=SESSIONS[session_id]; level=state["current_level"]
    raw=await file.read()
    try:
        x, sr = load_audio_safe(raw, file.filename)
        x_trim, lead_s, trail_s = trim_silence(x, sr)
        sig_score, sig_details, sig_warn = heuristic_evaluate(x_trim, sr)
        signal={"duration_sec":sig_details.get("duration_sec", len(x_trim)/sr),
                "features":sig_details.get("features",{}),"warnings":sig_warn,
                "overall_signal_score":sig_score}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio processing failed: {e}")
    transcript = transcribe_audio(x_trim, sr, raw, file.filename)
    scores = rubric_score_llm(transcript)
    scores["fluency"]=float(0.7*scores["fluency"] + 0.3*min(1.0,max(0.0,sig_score)))
    scores["pronunciation"]=float(0.7*scores["pronunciation"] + 0.3*min(1.0,max(0.0,sig_score)))
    decision = "advance" if _pass_fail(scores) else "stop"
    next_level = _next_level(level) if decision=="advance" else None
    next_prompt = PROMPTS[next_level] if next_level else None
    state["history"].append({"level":level,"scores":scores,"decision":decision,"transcript":transcript})
    if decision=="advance" and next_level:
        state["current_level"]=next_level
    else:
        state["final_level"]=CEFR_ORDER[max(0, CEFR_ORDER.index(level)-1)] if decision=="stop" else level
    return SubmitResponse(session_id=session_id, level=level, scores=scores, decision=decision,
                          next_level=next_level, next_prompt=next_prompt, transcript=transcript, signal=signal)

@app.get("/report/{session_id}", response_model=FinalReport)
async def report(session_id: str):
    if session_id not in SESSIONS: raise HTTPException(status_code=404, detail="Session not found")
    state=SESSIONS[session_id]; final_level=state.get("final_level") or state["current_level"]
    cat_sums={"fluency":0,"grammar":0,"vocabulary":0,"pronunciation":0,"coherence":0}; n=0
    for step in state["history"]:
        for k in cat_sums: cat_sums[k]+=step["scores"].get(k,0)
        n+=1
    recs=[]
    if n>0:
        avgs={k:(v/n) for k,v in cat_sums.items()}
        weak=sorted(avgs.items(), key=lambda x: x[1])[:2]
        for name,val in weak: recs.append(f"Focus on {name} (avg {val:.2f}); targeted tasks to reach next level.")
    return FinalReport(session_id=session_id, final_level=final_level, history=state["history"], recommendations=recs)

# --- Live config endpoint for frontend ---
@app.get("/config")
async def config():
    # Expose runtime config so the UI can fetch thresholds and durations
    return {
        "ASR_BACKEND": os.getenv("ASR_BACKEND", "openai"),
        "RUBRIC_MODEL": os.getenv("RUBRIC_MODEL", "gpt-4o-mini"),
        "WHISPER_MODEL": os.getenv("WHISPER_MODEL", "whisper-1"),
        "PASS_AVG_THRESHOLD": os.getenv("PASS_AVG_THRESHOLD", "0.70"),
        "PASS_MIN_THRESHOLD": os.getenv("PASS_MIN_THRESHOLD", "0.60"),
        # Optionally control UI recording duration from env
        "RECORD_SECONDS": int(os.getenv("RECORD_SECONDS", "60"))
    }

# ---------- Error handlers ----------
@app.exception_handler(HTTPException)
async def http_exc_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})
@app.exception_handler(Exception)
async def generic_exc_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": "Internal server error"})

# ---------- Attach frontend ----------
try:
    from add_frontend import attach_frontend
    attach_frontend(app, web_dir="web")
except Exception as e:
    logger.info(f"Frontend attach skipped: {e}")
