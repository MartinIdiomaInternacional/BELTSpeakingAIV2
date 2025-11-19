import os

ENV = os.getenv("ENV", "production")
DEBUG = ENV != "production"

# Adaptive jump thresholds
CONF_FAST_JUMP = float(os.getenv("CONF_FAST_JUMP", "0.6"))
CONF_STRONG_JUMP = float(os.getenv("CONF_STRONG_JUMP", "0.7"))
CONF_SOFT_JUMP = float(os.getenv("CONF_SOFT_JUMP", "0.5"))

# Streak logic
STREAK_REQUIRED = int(os.getenv("STREAK_REQUIRED", "2"))
STREAK_MIN_CONF = float(os.getenv("STREAK_MIN_CONF", "0.6"))

# Budget (turns + **recording time only**)
MAX_TURNS = int(os.getenv("MAX_TURNS", "6"))
# New: count only recorded speech time (in minutes)
MAX_RECORDING_MINUTES = float(os.getenv("MAX_RECORDING_MINUTES", "7"))

# Audio analysis
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
MIN_DURATION_SEC = float(os.getenv("MIN_DURATION_SEC", "2.0"))
MAX_DURATION_SEC = float(os.getenv("MAX_DURATION_SEC", "60.0"))
SILENCE_THRESHOLD = float(os.getenv("SILENCE_THRESHOLD", "0.01"))
QUALITY_MIN_CONFIDENCE = float(os.getenv("QUALITY_MIN_CONFIDENCE", "0.5"))

PROMPTS_DIR = os.getenv("PROMPTS_DIR", "data/prompts")
SESSIONS_DIR = os.getenv("SESSIONS_DIR", "data/sessions")

os.makedirs(PROMPTS_DIR, exist_ok=True)
os.makedirs(SESSIONS_DIR, exist_ok=True)
