import numpy as np
import librosa
from typing import Dict


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

    All values are plain Python floats (JSON-friendly).
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
