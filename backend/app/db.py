
import os
from typing import Optional
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
DATABASE_URL = os.getenv("DATABASE_URL")
_engine: Optional[Engine] = None
def get_engine() -> Optional[Engine]:
    global _engine
    if not DATABASE_URL:
        return None
    if _engine is None:
        _engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        _init_schema(_engine)
    return _engine
def _init_schema(engine: Engine):
    with engine.begin() as conn:
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            candidate_id TEXT,
            native_language TEXT,
            created_at TIMESTAMP DEFAULT NOW(),
            final_level TEXT,
            final_score FLOAT,
            final_confidence FLOAT
        );
        """))
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS turns (
            session_id TEXT,
            idx INTEGER,
            asked_level TEXT,
            inferred_level TEXT,
            score FLOAT,
            confidence FLOAT,
            transcription TEXT,
            quality_ok BOOLEAN,
            quality_reason TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        );
        """))
def save_session(session_id: str, candidate_id: str, native_language: str):
    eng = get_engine()
    if not eng: return
    with eng.begin() as conn:
        conn.execute(text("""
            INSERT INTO sessions (id, candidate_id, native_language) 
            VALUES (:id,:cid,:lang) ON CONFLICT (id) DO NOTHING
        """), {"id": session_id, "cid": candidate_id, "lang": native_language})
def save_turn(session_id: str, idx: int, asked: str, inferred: str|None, score: float|None, conf: float|None,
              transcription: str|None, quality_ok: bool, quality_reason: str|None):
    eng = get_engine()
    if not eng: return
    with eng.begin() as conn:
        conn.execute(text("""
            INSERT INTO turns (session_id, idx, asked_level, inferred_level, score, confidence, transcription, quality_ok, quality_reason)
            VALUES (:sid,:idx,:asked,:inferred,:s,:c,:t,:q,:qr)
        """), {"sid":session_id,"idx":idx,"asked":asked,"inferred":inferred,"s":score,"c":conf,"t":transcription,"q":quality_ok,"qr":quality_reason})
def finalize_session(session_id: str, level: str, score: float, conf: float):
    eng = get_engine()
    if not eng: return
    with eng.begin() as conn:
        conn.execute(text("""
            UPDATE sessions SET final_level=:l, final_score=:s, final_confidence=:c WHERE id=:id
        """), {"l":level,"s":score,"c":conf,"id":session_id})
