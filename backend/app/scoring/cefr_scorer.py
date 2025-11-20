import numpy as np
import torch

from app.scoring.audio_features import (
    load_audio,
    extract_feature_vector,
    extract_debug_metrics,
)


def _rule_based_score(features: torch.Tensor, metrics: dict) -> float:
    """
    Rule-based 'model' that maps acoustic features to a 0–1 proficiency score.

    Intuition:
    - Longer speaking time (up to a point) → more fluent
    - Higher speech activity (less silence) → more fluent
    - Lower ZCR (less noisy / choppy) → more stable speech signal
    """

    duration = float(metrics["duration_sec"])
    speech_ratio = float(metrics["speech_ratio"])
    zcr_mean = float(metrics["zcr_mean"])

    # 1) Fluency: how long the person spoke.
    #    0 sec → 0.0, 30+ sec → ~1.0
    fluency = np.clip(duration / 30.0, 0.0, 1.0)

    # 2) Speech activity: how much of the recording has actual voice.
    #    0.0 → all silence, 1.0 → constant speech
    activity = float(np.clip(speech_ratio, 0.0, 1.0))

    # 3) Stability: lower ZCR suggests smoother, less noisy voice
    #    ZCR ~ 0.02–0.08 typical speech; we invert it into a 0–1 score.
    #    If zcr_mean is high, stability is lower.
    stability = float(np.clip(0.1 - zcr_mean, 0.0, 0.1) / 0.1)

    # Weighted combination
    # You can tune these weights later if needed.
    score = 0.5 * fluency + 0.3 * activity + 0.2 * stability
    return float(np.clip(score, 0.0, 1.0))


def _score_to_cefr(score: float) -> str:
    """Map a 0–1 global score into CEFR-like levels."""
    if score >= 0.8:
        return "C1"
    if score >= 0.6:
        return "B2"
    if score >= 0.4:
        return "B1"
    return "A2"


def _build_explanation(level: str, metrics: dict) -> str:
    """Generate a short explanation using the metrics."""

    duration = metrics["duration_sec"]
    speech_ratio = metrics["speech_ratio"]
    silence_ratio = metrics["silence_ratio"]

    if level == "C1":
        return (
            "Your speaking sample is long and continuous, with few pauses and a stable signal. "
            f"You spoke for approximately {duration:.1f} seconds, with voice present during most "
            "of the recording. This suggests strong fluency and good control when speaking."
        )
    if level == "B2":
        return (
            "Your speaking sample shows generally good fluency and fairly continuous speech. "
            f"You spoke for around {duration:.1f} seconds and maintained voice during a large "
            "portion of the recording, with some pauses or hesitations."
        )
    if level == "B1":
        return (
            "Your speaking sample shows basic fluency, but with more frequent pauses and shorter segments. "
            f"You spoke for about {duration:.1f} seconds and there is a noticeable proportion of silence. "
            "This suggests you can communicate core ideas but may hesitate or stop to think often."
        )
    # A2
    return (
        "Your speaking sample is relatively short with significant silence. "
        f"You spoke for roughly {duration:.1f} seconds and a large part of the recording has no voice, "
        "which suggests limited fluency and short, simple responses."
    )


def _build_recommendations(level: str, metrics: dict) -> str:
    """Generate concrete recommendations based on the level and metrics."""

    duration = metrics["duration_sec"]
    speech_ratio = metrics["speech_ratio"]

    recs = []

    # Generic fluency recommendation
    if duration < 20:
        recs.append(
            "- Practice giving longer answers (aim for 30–45 seconds per question)."
        )
    else:
        recs.append(
            "- Keep giving extended answers and try to include more detail and examples."
        )

    # Silence vs speech
    if speech_ratio < 0.5:
        recs.append(
            "- Reduce long pauses by planning 2–3 key ideas before you start speaking."
        )
    elif speech_ratio < 0.7:
        recs.append(
            "- Continue improving your fluency by connecting ideas with linking phrases."
        )
    else:
        recs.append(
            "- Your speech is quite continuous; focus now on accuracy and range of language."
        )

    # Level-specific advice
    if level == "C1":
        recs.extend(
            [
                "- Work on nuance in tone and register depending on the situation.",
                "- Incorporate more advanced, domain-specific vocabulary in your field.",
            ]
        )
    elif level == "B2":
        recs.extend(
            [
                "- Use more precise vocabulary when explaining complex ideas.",
                "- Practice discussing abstract topics and giving structured opinions.",
            ]
        )
    elif level == "B1":
        recs.extend(
            [
                "- Strengthen your sentence structure so you can build longer, clearer ideas.",
                "- Learn and reuse common phrases and collocations to speak more naturally.",
            ]
        )
    else:  # A2
        recs.extend(
            [
                "- Focus on very common topics (family, work, daily routine) and simple sentences.",
                "- Memorize and practice basic patterns like 'I usually...', 'I would like to...', "
                "to increase confidence.",
            ]
        )

    return "\n".join(recs)


def evaluate_audio(path: str):
    """
    Main entry point used by the FastAPI app.

    1. Load audio & extract features
    2. Compute a 0–1 proficiency score using a rule-based 'model'
    3. Map score → CEFR level
    4. Generate explanation + recommendations
    """
    audio, sr = load_audio(path)
    features = extract_feature_vector(audio, sr)
    metrics = extract_debug_metrics(audio, sr)

    global_score = _rule_based_score(features, metrics)
    level = _score_to_cefr(global_score)
    explanation = _build_explanation(level, metrics)
    recommendations = _build_recommendations(level, metrics)

    return {
        "level": level,
        "explanation": explanation,
        "recommendations": recommendations,
        "seconds": float(metrics["duration_sec"]),
        "score_raw": float(global_score),
        "metrics": metrics,  # keep this if you ever want to inspect stats
    }
