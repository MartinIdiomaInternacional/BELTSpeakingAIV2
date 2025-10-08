
import os
MIN_VOICED_S = float(os.getenv("QUALITY_MIN_VOICED_S", "5.0"))
MIN_SNR_DB = float(os.getenv("QUALITY_MIN_SNR_DB", "10.0"))
def check_quality(feats: dict) -> (bool, str):
    voiced_s = feats.get("voiced_s", 0.0)
    snr_db = feats.get("snr_db", 0.0)
    if voiced_s < MIN_VOICED_S:
        return False, f"Voiced speech too short ({voiced_s:.1f}s < {MIN_VOICED_S:.1f}s)."
    if snr_db < MIN_SNR_DB:
        return False, f"Audio too noisy (SNR {snr_db:.1f} dB < {MIN_SNR_DB:.1f} dB)."
    return True, ""
