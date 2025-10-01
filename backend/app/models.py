from pydantic import BaseModel, Field
from typing import Optional, Literal, Dict

CEFRLevel = Literal["A1","A2","B1","B1+","B2","B2+","C1","C2"]

class StartRequest(BaseModel):
    candidate_id: str
    target_level: Optional[CEFRLevel] = None
    native_language: Optional[str] = Field(default="en")

class StartResponse(BaseModel):
    session_id: str
    message: str

class EvaluateBytesRequest(BaseModel):
    session_id: str
    sample_rate: int
    wav_base64: str  # actually webm base64 is allowed; backend converts
    prompt_id: Optional[str] = None

class ScoreDetail(BaseModel):
    level: CEFRLevel
    score_0_8: float
    confidence: float
    metrics: Dict
    notes: Optional[str] = None

class EvaluateResponse(BaseModel):
    session_id: str
    base: ScoreDetail
    needs_probe: bool
    probe_prompt: Optional[str] = None

class ReportRequest(BaseModel):
    session_id: str
    native_language: Optional[str] = "en"

class ReportResponse(BaseModel):
    session_id: str
    final: ScoreDetail
    feedback: Dict
    html: str
