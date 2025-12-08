from datetime import datetime
import tempfile
from typing import Dict

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from app.scoring.audio_features import compute_basic_features
from app.scoring.cefr_scorer import score_from_features, map_score_to_level
from app.db import log_result
from app.version import VERSION


app = FastAPI(
    title="BELT Speaking AI 2.0",
    description="Prototype API for automatic speaking evaluation with CEFR-like output.",
    version=VERSION,
)

# CORS: in production you may want to restrict this to your frontend origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def build_explanation(level: str, feats: Dict[str, float], score_0_8: float, confidence: float) -> str:
    """Generate a short explanation using the classic feature set."""
    dur = feats.get("duration_s", 0.0)
    vr = feats.get("voiced_ratio", 0.0)
    sr = feats.get("speech_rate_proxy", 0.0)
    zc = feats.get("avg_zcr", 0.0)

    # Base text by level
    if level in ("C1", "C2"):
        base = (
            "Your speaking sample suggests an advanced level of fluency and control. "
            "You are able to sustain speech with relatively few pauses and a stable signal."
        )
    elif level in ("B2", "B2+"):
        base = (
            "Your speaking sample suggests an upper-intermediate level. "
            "You can sustain speech for extended periods with generally good flow."
        )
    elif level in ("B1", "B1+"):
        base = (
            "Your speaking sample suggests an intermediate level. "
            "You can communicate your ideas but with more frequent pauses or hesitations."
        )
    else:  # A1, A2
        base = (
            "Your speaking sample suggests a basic level. "
            "Spoken segments are shorter and there are more pauses and silence."
        )

    details = (
        f" You spoke for approximately {dur:.1f} seconds. "
        f"The proportion of voiced speech (parts with clear signal) is about {vr*100:.0f}%. "
        f"The speech-rate proxy (how many voiced segments you produce per second) is around {sr:.1f}, "
        f"and the average zero-crossing rate (signal stability) is {zc:.3f}."
    )

    conf = (
        f" The internal consistency of the acoustic metrics suggests a confidence of about "
        f"{confidence*100:.0f}% in this estimate."
    )

    return base + details + conf


def build_recommendations(level: str, feats: Dict[str, float]) -> str:
    """Generate concrete recommendations based on the acoustic features and level."""
    dur = feats.get("duration_s", 0.0)
    vr = feats.get("voiced_ratio", 0.0)
    sr = feats.get("speech_rate_proxy", 0.0)
    zc = feats.get("avg_zcr", 0.0)

    recs = []

    # Duration-based advice
    if dur < 20:
        recs.append(
            "- Try to give longer answers (aim for 30–45 seconds per task) so you can develop your ideas more fully."
        )
    else:
        recs.append(
            "- Keep giving extended answers, and focus on adding more detail, examples and clear structure."
        )

    # Voiced ratio / pauses
    if vr < 0.4:
        recs.append(
            "- There is a high proportion of silence and pauses. Plan 2–3 key ideas before you start speaking to reduce long gaps."
        )
    elif vr < 0.7:
        recs.append(
            "- Your speech is somewhat continuous, but there is still room to reduce unnecessary pauses by linking ideas more smoothly."
        )
    else:
        recs.append(
            "- Your speech is quite continuous. You can now focus on accuracy, precision and range of vocabulary."
        )

    # Speech rate
    if sr < 5:
        recs.append(
            "- Your speech rate is relatively slow. Practice answering familiar questions with slightly faster delivery while staying clear."
        )
    elif sr > 11:
        recs.append(
            "- Your speech rate is quite fast. Make sure you articulate clearly and avoid running words together."
        )
    else:
        recs.append(
            "- Your speech rate is within a natural range. Keep this pace while focusing on clarity and organization."
        )

    # Stability / noise
    if zc > 0.07:
        recs.append(
            "- The signal seems a bit unstable or noisy. Use a quiet environment and keep the microphone at a consistent distance."
        )

    # Level-based language advice
    if level in ("C1", "C2"):
        recs.extend(
            [
                "- Work on nuance in tone and register depending on the context (formal vs. informal).",
                "- Incorporate more advanced and domain-specific vocabulary to sound even more natural and precise.",
            ]
        )
    elif level in ("B2", "B2+"):
        recs.extend(
            [
                "- Use more precise connectors (however, moreover, on the other hand) to structure arguments.",
                "- Practice discussing abstract topics and giving structured opinions.",
            ]
        )
    elif level in ("B1", "B1+"):
        recs.extend(
            [
                "- Strengthen your sentence structure so you can build longer and clearer ideas (using linking words like because, although, so).",
                "- Expand your range of common phrases and collocations for work, study and daily life.",
            ]
        )
    else:  # A1, A2
        recs.extend(
            [
                "- Focus on very frequent topics (family, work, daily routine) and short, clear sentences.",
                "- Learn and reuse simple patterns like 'I usually...', 'I would like to...', 'In my free time I...' to build confidence.",
            ]
        )

    return "\n".join(recs)


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "BELT Speaking AI 2.0",
        "version": VERSION,
        "message": "API is running.",
    }


@app.post("/evaluate")
async def evaluate(
    audio: UploadFile = File(...),
    task_id: int = Form(...),
):
    """
    Evaluate a single speaking task.

    Expects:
    - audio: uploaded file (webm/wav/etc.); will be decoded to float audio here.
    - task_id: integer identifying the task (1, 2, 3, ...).

    Returns:
    - score: CEFR-like level label (A1–C2, using your classic scoring grid).
    - explanation: short textual explanation of the level.
    - recommendations: concrete suggestions to improve.
    - seconds: approximate speaking duration.
    - task_id: echoed task id.
    """
    # Save the uploaded file to a temporary path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    # --- Classic acoustic feature pipeline ---
    from librosa import load  # convert file -> float array

    y, sr = load(tmp_path, sr=16000)
    feats = compute_basic_features(y, sr)

    score_0_8, confidence, metrics = score_from_features(feats)
    level = map_score_to_level(score_0_8)

    explanation = build_explanation(level, feats, score_0_8, confidence)
    recommendations = build_recommendations(level, feats)
    seconds = float(feats.get("duration_s", 0.0))

    # Log result (CSV-based)
    timestamp = datetime.utcnow().isoformat()
    log_result(
        timestamp_utc=timestamp,
        task_id=task_id,
        seconds=seconds,
        level=level,
        explanation=explanation,
        recommendations=recommendations,
    )

    # Make metrics JSON-friendly (no numpy types)
    metrics_json = {k: float(v) for k, v in metrics.items()}

    return {
        "score": level,
        "explanation": explanation,
        "recommendations": recommendations,
        "seconds": seconds,
        "task_id": task_id,
        "score_0_8": float(score_0_8),
        "confidence": float(confidence),
        "metrics": metrics_json,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
