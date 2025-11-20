import librosa
import numpy as np
import torch


def load_audio(path: str, target_sr: int = 16000):
    """Load mono audio at a fixed sample rate."""
    audio, sr = librosa.load(path, sr=target_sr)
    if audio.ndim > 1:
        audio = librosa.to_mono(audio)
    return audio, sr


def compute_duration(audio: np.ndarray, sr: int) -> float:
    """Duration in seconds."""
    return float(len(audio) / sr)


def compute_rms(audio: np.ndarray, frame_length: int = 1024, hop_length: int = 512):
    """Frame-wise RMS and its mean/std."""
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    return float(rms.mean()), float(rms.std())


def compute_zero_crossing_rate(audio: np.ndarray, frame_length: int = 1024, hop_length: int = 512):
    """Zero-crossing rate (how often signal changes sign)."""
    zcr = librosa.feature.zero_crossing_rate(
        y=audio, frame_length=frame_length, hop_length=hop_length
    )[0]
    return float(zcr.mean()), float(zcr.std())


def compute_spectral_centroid(audio: np.ndarray, sr: int, hop_length: int = 512):
    """Spectral centroid: rough 'brightness' of the sound."""
    sc = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop_length)[0]
    return float(sc.mean()), float(sc.std())


def compute_speech_activity(
    audio: np.ndarray,
    frame_length: int = 1024,
    hop_length: int = 512,
    energy_threshold_ratio: float = 0.4,
):
    """
    Very simple voice activity estimate:
    - compute RMS per frame
    - frames above (threshold_ratio * max_rms) = speech
    Returns: (speech_ratio, silence_ratio)
    """
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    max_rms = float(rms.max()) if rms.size > 0 else 0.0
    if max_rms <= 0:
        return 0.0, 1.0

    threshold = energy_threshold_ratio * max_rms
    speech_frames = np.sum(rms > threshold)
    total_frames = rms.size
    speech_ratio = float(speech_frames / total_frames) if total_frames > 0 else 0.0
    silence_ratio = 1.0 - speech_ratio
    return speech_ratio, silence_ratio


def extract_feature_vector(audio: np.ndarray, sr: int) -> torch.Tensor:
    """
    Build a small numeric feature vector summarizing the audio.
    This is our 'real model input' (even if later we replace
    the rule-based scoring with a trained model).
    """
    duration_sec = compute_duration(audio, sr)
    rms_mean, rms_std = compute_rms(audio)
    zcr_mean, zcr_std = compute_zero_crossing_rate(audio)
    sc_mean, sc_std = compute_spectral_centroid(audio, sr)
    speech_ratio, silence_ratio = compute_speech_activity(audio)

    # Simple log-safe transforms
    def safe_log(x):
        return float(np.log10(x + 1e-6))

    features = np.array(
        [
            duration_sec,
            rms_mean,
            rms_std,
            zcr_mean,
            zcr_std,
            sc_mean,
            sc_std,
            speech_ratio,
            silence_ratio,
            safe_log(duration_sec),
            safe_log(rms_mean),
            safe_log(sc_mean),
        ],
        dtype=np.float32,
    )

    return torch.tensor(features, dtype=torch.float32)


def extract_debug_metrics(audio: np.ndarray, sr: int) -> dict:
    """Return a dict with human-readable metrics (for explanations/debug)."""
    duration_sec = compute_duration(audio, sr)
    rms_mean, rms_std = compute_rms(audio)
    zcr_mean, zcr_std = compute_zero_crossing_rate(audio)
    sc_mean, sc_std = compute_spectral_centroid(audio, sr)
    speech_ratio, silence_ratio = compute_speech_activity(audio)

    return {
        "duration_sec": float(duration_sec),
        "rms_mean": float(rms_mean),
        "rms_std": float(rms_std),
        "zcr_mean": float(zcr_mean),
        "zcr_std": float(zcr_std),
        "spectral_centroid_mean": float(sc_mean),
        "spectral_centroid_std": float(sc_std),
        "speech_ratio": float(speech_ratio),
        "silence_ratio": float(silence_ratio),
    }
