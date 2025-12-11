import numpy as np
import librosa
from typing import Dict, Tuple


# -------------------------------------------------------------
# Minimal helper to satisfy cefr_scorer imports
# -------------------------------------------------------------

def load_audio(path: str, sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Simple wrapper to load audio from a file path using librosa.
    This exists only because cefr_scorer.py imports it.
    """
    y, sr = librosa.load(path, sr=sr)
    return y.astype(np.float32), sr


# -------------------------------------------------------------
# Helper used by the original BELT feature-based model
# -------------------------------------------------------------

def _safe_log10(x: float) -> float:
    return float(np.log10(max(x, 1e-8)))


def compute_basic_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Compute core acoustic features used by the BELT feature model.
    Returns:
      - duration_s
      - voiced_ratio
      - voiced_s
      - avg_energy
      - avg_zcr
      - speech_rate_proxy
      - snr_db
    """
    y = np.asarray(y, dtype=np.float32)

    duration_s = float(len(y) / float(sr)) if sr > 0 else 0.0

    frame_length = 1024
    hop_length = 512

    rms = librosa.feature.rms(
        y=y, frame_length=frame_length, hop_length=hop_length
    )[0]

    if rms.size == 0:
        return {
            "duration_s": 0.0,
            "voiced_ratio": 0.0,
            "voiced_s": 0.0,
            "avg_energy": 0.0,
            "avg_zcr": 0.0,
            "speech_rate_proxy": 0.0,
            "snr_db": 0.0,
        }

    max_rms = float(rms.max())
    energy_threshold = 0.4 * max_rms if max_rms > 0 else 0.0
    voiced_mask = rms > energy_threshold
    voiced_ratio = float(voiced_mask.mean())
    voiced_s = float(voiced_ratio * duration_s)

    avg_energy = float(rms.mean())

    zcr = librosa.feature.zero_crossing_rate(
        y=y, frame_length=frame_length, hop_length=hop_length
    )[0]
    avg_zcr = float(zcr.mean()) if zcr.size > 0 else 0.0

    voiced_frames = float(voiced_mask.sum())
    if duration_s > 0:
        speech_rate_proxy = voiced_frames / duration_s
    else:
        speech_rate_proxy = 0.0

    high_energy = rms[voiced_mask]
    low_energy = rms[~voiced_mask]
    if high_energy.size > 0 and low_energy.size > 0:
        signal_power = float(np.mean(high_energy ** 2))
        noise_power = float(np.mean(low_energy ** 2))
        snr_db = 10.0 * _safe_log10(signal_power / (noise_power + 1e-8))
    else:
        snr_db = 0.0

    return {
        "duration_s": duration_s,
        "voiced_ratio": voiced_ratio,
        "voiced_s": voiced_s,
        "avg_energy": avg_energy,
        "avg_zcr": avg_zcr,
        "speech_rate_proxy": speech_rate_proxy,
        "snr_db": snr_db,
    }
