from typing import Optional, List
from pydantic import BaseModel, Field, root_validator

# ─────────────────────────────────────────────────────────────
# Request/Response Schemas
# ─────────────────────────────────────────────────────────────

# /start
class StartRequest(BaseModel):
    candidate_id: str
    native_language: Optional[str] = None

class StartResponse(BaseModel):
    session_id: str
    message: str
    current_level: str
    prompt: str

# /evaluate-bytes
class EvaluateBytesRequest(BaseModel):
    session_id: str
    # Primary field for the audio payload
    wav_base64: Optional[str] = Field(default=None)
    # Backward/forward-compat aliases
    audio_base64: Optional[str] = Field(default=None)
    webm_base64: Optional[str] = Field(default=None)
    bytes_b64: Optional[str] = Field(default=None)

    @root_validator(pre=True)
    def coalesce_audio(cls, values):
        # Prefer wav_base64 if present; otherwise fall back through known aliases
        for k in ["wav_base64", "audio_base64", "webm_base64", "bytes_b64"]:
            v = values.get(k)
            if v:
                values["wav_base64"] = v
                return values
        # explicit error to produce a clearer 422 message
        raise ValueError("Missing audio base64 payload (wav_base64/audio_base64/webm_base64/bytes_b64)")

class TurnResult(BaseModel):
    asked_level: Optional[str] = None
    inferred_level: Optional[str] = None
    score_0_8: Optional[float] = None
    confidence: Optional[float] = None
    transcription: Optional[str] = None
    quality_ok: Optional[bool] = True
    quality_reason: Optional[str] = None

class EvaluateResponse(BaseModel):
    session_id: str
    turn: Optional[TurnResult] = None
    finished: bool

    # Next-step info (when finished == False)
    next_level: Optional[str] = None
    next_prompt: Optional[str] = None

    # Final results (when finished == True)
    final_level: Optional[str] = None
    final_score_0_8: Optional[float] = None
    final_confidence: Optional[float] = None

# /report
class ReportRequest(BaseModel):
    session_id: str
    native_language: Optional[str] = None

class ReportResponse(BaseModel):
    session_id: str
    final: Optional[dict] = None
    feedback: Optional[str] = None
    html: Optional[str] = None
