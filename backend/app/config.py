import os

# ─────────────────────────────────────────────────────────────
# Basic service configuration
# ─────────────────────────────────────────────────────────────
ENV = os.getenv("ENV", "production")
DEBUG = ENV != "production"

# ─────────────────────────────────────────────────────────────
# Adaptive evaluation configuration
# ─────────────────────────────────────────────────────────────

# Confidence thresholds for jumps
CONF_FAST_JUMP = float(os.getenv("CONF_FAST_JUMP", "0.6"))     # step 2 levels if gap ≥2 and conf ≥ 0.6
CONF_STRONG_JUMP = float(os.getenv("CONF_STRONG_JUMP", "0.7")) # direct jump if conf ≥ 0.7
CONF_SOFT_JUMP = float(os.getenv("CONF_SOFT_JUMP", "0.5"))     # step 1 if conf ≥ 0.5

# Streak logic
STREAK_REQUIRED = int(os.getenv("STREAK_REQUIRED", "2"))       # same level N times to finalize
STREAK_MIN_CONF = float(os.getenv("STREAK_MIN_CONF", "0.6"))

# Budget (time + number of questions)
MAX_TURNS = int(os.getenv("MAX_TURNS", "6"))                   # Max number of questions per session
MAX_TOTAL_MINUTES = float(os.getenv("MAX_TOTAL_MINUTES", "7")) # Total time cap (in minutes)

# ─────────────────────────────────────────────────────────────
# Audio / Speech Analysis
# ─────────────────────────────────────────────────────────────
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
MIN_DURATION_SEC = float(os.getenv("MIN_DURATION_SEC", "2.0"))
MAX_DURATION_SEC = float(os.getenv("MAX_DURATION_SEC", "60.0"))
SILENCE_THRESHOLD = float(os.getenv("SILENCE_THRESHOLD", "0.01"))
QUALITY_MIN_CONFIDENCE = float(os.getenv("QUALITY_MIN_CONFIDENCE", "0.5"))

# ─────────────────────────────────────────────────────────────
# File paths and storage
# ─────────────────────────────────────────────────────────────
PROMPTS_DIR = os.getenv("PROMPTS_DIR", "data/prompts")
SESSIONS_DIR = os.getenv("SESSIONS_DIR", "data/sessions")

os.makedirs(PROMPTS_DIR, exist_ok=True)
os.makedirs(SESSIONS_DIR, exist_ok=True)
