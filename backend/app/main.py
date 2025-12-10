from datetime import datetime
import tempfile
from typing import Dict, Any, Optional

import numpy as np
import librosa
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from app.scoring.cefr_scorer import score_from_features, map_score_to_level
from app.scoring.text_cefr import score_transcript_cefr
from app.nlp.transcription import transcribe_audio
from app.db import log_result
from app.version import VERSION

# ---------------------------------------------------------------------------
# FastAPI app configuration
# ---------------------------------------------------------------------------

app = FastAPI(
    title="BELT Speaking AI 2.0",
    description=(
        "Prototype API for automatic speaking evaluation combining "
        "acoustic metrics and CEFR-oriented NLP analysis."
    ),
    version=VERSION,
)

# In production you may want to restrict origins to your frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Task descriptions mirrored from the frontend (for better NLP context)
TASK_PROMPTS: Dict[int, str] = {
    1: (
        "Personal introduction: The candidate should talk about their background, "
        "where they live, what they do, and one interesting fact about themselves."
    ),
    2: (
        "Describe a challenging situation they faced recently and how they handled it."
    ),
    3: (
        "Give their opinion on whether technology has improved communication and why."
    ),
}


# ---------------------------------------------------------------------------
# Local acoustic feature computation (self-contained)
# ---------------------------------------------------------------------------


def _safe_log10(x: float) -> float:
    return float(np.log10(max(x, 1e-8)))


def compute_basic_features(y, sr: int) -> Dict[str, float]:
    """
    Compute the core acoustic features used by the BELT Speaking AI scorer.

    Returns a dict with (at least) the following keys:
      - duration_s
      - voiced_ratio
      - voiced_s
      - avg_energy
      - avg_zcr
      - speech_rate_proxy
      - snr_db
    """
    y = np.asarray(y, dtype=np.float32)

    # Duration (seconds)
    duration_s = float(len(y) / float(sr)) if sr > 0 else 0.0

    # Frame settings
    frame_length = 1024
    hop_length = 512

    # RMS energy per frame
    rms = librosa.feature.rms(
        y=y, frame_length=frame_length, hop_length=hop_length
    )[0]  # shape (n_frames,)
    rms = np.asarray(rms, dtype=np.float32)

    if rms.size == 0:
        # Edge case: empty or too short audio
        return {
            "duration_s": 0.0,
            "voiced_ratio": 0.0,
            "voiced_s": 0.0,
            "avg_energy": 0.0,
            "avg_zcr": 0.0,
            "speech_rate_proxy": 0.0,
            "snr_db": 0.0,
        }

    # Threshold for "voiced" frames: fraction of max RMS
    max_rms = float(rms.max())
    energy_threshold = 0.4 * max_rms if max_rms > 0 else 0.0
    voiced_mask = rms > energy_threshold
    voiced_ratio = float(voiced_mask.mean())
    voiced_s = float(voiced_ratio * duration_s)

    # Average energy (RMS)
    avg_energy = float(rms.mean())

    # Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(
        y=y, frame_length=frame_length, hop_length=hop_length
    )[0]
    avg_zcr = float(zcr.mean()) if zcr.size > 0 else 0.0

    # Approximate speech rate: number of voiced frames per second
    frames_per_second = float(sr) / float(hop_length) if hop_length > 0 else 0.0
    voiced_frames = float(voiced_mask.sum())
    if duration_s > 0:
        speech_rate_proxy = voiced_frames / duration_s
    else:
        speech_rate_proxy = 0.0

    # Very rough SNR estimate:
    #   - "signal" = high-energy frames
    #   - "noise"  = low-energy frames
    high_energy = rms[voiced_mask]
    low_energy = rms[~voiced_mask]
    if high_energy.size > 0 and low_energy.size > 0:
        signal_power = float(np.mean(high_energy ** 2))
        noise_power = float(np.mean(low_energy ** 2))
        snr_db = 10.0 * _safe_log10(signal_power / (noise_power + 1e-8))
    else:
        snr_db = 0.0

    return {
        "duration_s": float(duration_s),
        "voiced_ratio": float(voiced_ratio),
        "voiced_s": float(voiced_s),
        "avg_energy": float(avg_energy),
        "avg_zcr": float(avg_zcr),
        "speech_rate_proxy": float(speech_rate_proxy),
        "snr_db": float(snr_db),
    }


# ---------------------------------------------------------------------------
# Helper functions to build explanations & recommendations
# ---------------------------------------------------------------------------


def build_acoustic_explanation(
    level: str,
    feats: Dict[str, float],
    score_0_8: float,
    confidence: float,
) -> str:
    """Generate a short explanation using the classic acoustic feature set."""
    dur = feats.get("duration_s", 0.0)
    vr = feats.get("voiced_ratio", 0.0)
    sr = feats.get("speech_rate_proxy", 0.0)
    zc = feats.get("avg_zcr", 0.0)

    if level in ("C1", "C2"):
        base = (
            "Acoustic analysis suggests an advanced level of fluency and control. "
            "You can sustain speech with relatively few pauses and a stable signal."
        )
    elif level in ("B2", "B2+"):
        base = (
            "Acoustic analysis suggests an upper-intermediate level. "
            "You can sustain speech for extended periods with generally good flow."
        )
    elif level in ("B1", "B1+"):
        base = (
            "Acoustic analysis suggests an intermediate level. "
            "You can communicate your ideas but with more frequent pauses or hesitations."
        )
    else:  # A1, A2
        base = (
            "Acoustic analysis suggests a basic level. "
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


def build_acoustic_recommendations(level: str, feats: Dict[str, float]) -> str:
    """Generate concrete recommendations based on acoustic features and level."""
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

    # Level-based language advice (generic)
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


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


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

    Steps:
    1) Save uploaded audio to a temporary WAV file.
    2) Acoustic scoring (BELT metrics -> CEFR level A1–C2).
    3) ASR transcription (Whisper) to text.
    4) Text-based CEFR analysis with GPT (grammar, vocabulary, coherence, etc.).
    5) Combine into a final level, explanation and recommendations.
    """
    # ------------------------------------------------------------------ #
    # 1) Persist audio temporarily
    # ------------------------------------------------------------------ #
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    # ------------------------------------------------------------------ #
    # 2) Acoustic feature pipeline (BELT logic)
    # ------------------------------------------------------------------ #
    y, sr = librosa.load(tmp_path, sr=16000)
    feats = compute_basic_features(y, sr)

    score_0_8, confidence, metrics = score_from_features(feats)
    acoustic_level = map_score_to_level(score_0_8)
    acoustic_explanation = build_acoustic_explanation(
        acoustic_level, feats, score_0_8, confidence
    )
    acoustic_recs = build_acoustic_recommendations(acoustic_level, feats)
    seconds = float(feats.get("duration_s", 0.0))

    # ------------------------------------------------------------------ #
    # 3) ASR transcription
    # ------------------------------------------------------------------ #
    transcript: Optional[str] = transcribe_audio(tmp_path, language="en")

    # ------------------------------------------------------------------ #
    # 4) Text-based CEFR analysis (if transcription succeeded)
    # ------------------------------------------------------------------ #
    text_eval: Optional[Dict[str, Any]] = None
    if transcript:
        task_instruction = TASK_PROMPTS.get(
            task_id,
            "General speaking sample for a language test. Evaluate CEFR level.",
        )
        try:
            text_eval = score_transcript_cefr(
                transcript=transcript,
                task_instruction=task_instruction,
                target_language="English",
            )
        except Exception as e:
            print(f"[evaluate] Error during text CEFR scoring: {e}")
            text_eval = None

    # ------------------------------------------------------------------ #
    # 5) Combine acoustic + text results
    # ------------------------------------------------------------------ #
    if text_eval is not None:
        text_level = text_eval.get("overall_level") or acoustic_level
        overall_level = text_level

        text_comment = text_eval.get("overall_comment", "")
        text_advice = text_eval.get("improvement_advice", "")

        explanation = (
            text_comment.strip()
            + " "
            + acoustic_explanation.strip()
        ).strip()

        recommendations = (
            (text_advice.strip() + "\n\n" + acoustic_recs.strip())
            if text_advice
            else acoustic_recs
        )
    else:
        overall_level = acoustic_level
        text_level = None
        explanation = acoustic_explanation
        recommendations = acoustic_recs

    # ------------------------------------------------------------------ #
    # 6) Log result (CSV-based) using the combined level and explanation
    # ------------------------------------------------------------------ #
    timestamp = datetime.utcnow().isoformat()
    log_result(
        timestamp_utc=timestamp,
        task_id=task_id,
        seconds=seconds,
        level=overall_level,
        explanation=explanation,
        recommendations=recommendations,
    )

    metrics_json = {k: float(v) for k, v in metrics.items()}

    dimensions = (text_eval or {}).get("dimensions")
    text_overall_comment = (text_eval or {}).get("overall_comment")
    text_improvement = (text_eval or {}).get("improvement_advice")

    return {
        "score": overall_level,
        "explanation": explanation,
        "recommendations": recommendations,
        "seconds": seconds,
        "task_id": task_id,
        "acoustic_level": acoustic_level,
        "score_0_8": float(score_0_8),
        "confidence": float(confidence),
        "metrics": metrics_json,
        "transcript": transcript,
        "text_level": text_level,
        "text_dimensions": dimensions,
        "text_overall_comment": text_overall_comment,
        "text_improvement_advice": text_improvement,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
