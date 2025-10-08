
from typing import Tuple, Dict
import numpy as np

LEVELS = ["A1","A2","B1","B1+","B2","B2+","C1","C2"]

THRESH = {
    "voiced_ratio":      [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85],
    "speech_rate_proxy": [5,    6,    7,    8,    9,    10,   11,   12],
    "avg_zcr":           [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09],
}
WEIGHTS = {"voiced_ratio":0.45,"speech_rate_proxy":0.35,"avg_zcr":0.20}

def _metric_to_score(value: float, grid: list) -> float:
    score = 0.0
    for i, t in enumerate(grid):
        if value >= t:
            score = i+1
    return float(score)

def score_from_features(feats: Dict[str, float]) -> Tuple[float, float, Dict]:
    s_vr = _metric_to_score(feats["voiced_ratio"], THRESH["voiced_ratio"])
    s_sr = _metric_to_score(feats["speech_rate_proxy"], THRESH["speech_rate_proxy"])
    s_zc = _metric_to_score(feats["avg_zcr"], THRESH["avg_zcr"])
    score = WEIGHTS["voiced_ratio"]*s_vr + WEIGHTS["speech_rate_proxy"]*s_sr + WEIGHTS["avg_zcr"]*s_zc

    spread = np.std([s_vr, s_sr, s_zc])
    confidence = float(max(0.3, 1.0 - spread/4.0))
    metrics = {"s_voiced_ratio": s_vr, "s_speech_rate": s_sr, "s_avg_zcr": s_zc, "confidence_components_spread": spread}
    return float(score), confidence, metrics

def map_score_to_level(score_0_8: float) -> str:
    idx = int(max(0, min(7, round(score_0_8))))
    return LEVELS[idx]
