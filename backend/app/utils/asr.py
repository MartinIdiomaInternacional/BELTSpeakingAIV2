
import io, os, requests, uuid
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
def transcribe_wav_bytes(wav_bytes: bytes) -> str | None:
    if not OPENAI_API_KEY:
        return None
    url = f"{OPENAI_API_BASE}/audio/transcriptions"
    files = { "file": (f"{uuid.uuid4()}.wav", io.BytesIO(wav_bytes), "audio/wav") }
    data = {"model": "whisper-1", "response_format": "text"}
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    try:
        r = requests.post(url, headers=headers, data=data, files=files, timeout=60)
        r.raise_for_status()
        return r.text.strip()
    except Exception:
        return None
