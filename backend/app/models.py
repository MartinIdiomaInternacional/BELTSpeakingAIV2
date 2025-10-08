
from pydantic import BaseModel, Field
from typing import Optional, Literal, Dict

CEFRLevel = Literal["A1","A2","B1","B1+","B2","B2+","C1","C2"]

class StartRequest(BaseModel):
    candidate_id: str
    native_language: Optional[str] = Field(default="en")

class StartResponse(BaseModel):
    session_id: str
    message: str
    current_level: CEFRLevel
    prompt: str

class EvaluateBytesRequest(BaseModel):
    session_id: str
    sample_rate: int
    wav_base64: str
    prompt_id: Optional[str] = None

class TurnResult(BaseModel):
    asked_level: CEFRLevel
    inferred_level: Optional[CEFRLevel] = None
    score_0_8: Optional[float] = None
    confidence: Optional[float] = None
    transcription: Optional[str] = None
    quality_ok: bool = True
    quality_reason: Optional[str] = None

class EvaluateResponse(BaseModel):
    session_id: str
    turn: TurnResult
    finished: bool
    next_level: Optional[CEFRLevel] = None
    next_prompt: Optional[str] = None
    final_level: Optional[CEFRLevel] = None
    final_score_0_8: Optional[float] = None
    final_confidence: Optional[float] = None

class ReportRequest(BaseModel):
    session_id: str
    native_language: Optional[str] = "en"

class ReportResponse(BaseModel):
    session_id: str
    final: Dict
    feedback: Dict
    html: str
