import numpy as np

def compute_basic_features(audio: np.ndarray, sr: int) -> dict:
    frame = int(0.03 * sr)
    hop = int(0.015 * sr)
    n = len(audio)
    if n <= frame:
        return {
            "duration_s": n / sr if sr else 0.0,
            "voiced_ratio": 0.0,
            "avg_energy": 0.0,
            "avg_zcr": 0.0,
            "speech_rate_proxy": 0.0,
        }

    energies, zcrs = [], []
    for start in range(0, n - frame, hop):
        seg = audio[start:start+frame]
        energies.append(float(np.mean(seg ** 2)))
        zcrs.append(float(np.mean(np.abs(np.diff(np.sign(seg))) > 0)))

    energies = np.array(energies) if energies else np.array([0.0])
    zcrs = np.array(zcrs) if zcrs else np.array([0.0])

    voiced_ratio = float((energies > 0.001).mean()))
    avg_energy = float(energies.mean())
    avg_zcr = float(zcrs.mean())

    duration_s = n / sr
    speech_rate_proxy = voiced_ratio * (1.0 / (hop / sr))

    return {
        "duration_s": duration_s,
        "voiced_ratio": voiced_ratio,
        "avg_energy": avg_energy,
        "avg_zcr": avg_zcr,
        "speech_rate_proxy": speech_rate_proxy,
    }
