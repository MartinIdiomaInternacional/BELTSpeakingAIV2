from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import tempfile
import os

from app.scoring.cefr_scorer import evaluate_audio
from app.db import log_result
from app.version import VERSION


# ----------------------------------------------------
# FastAPI App Setup
# ----------------------------------------------------
app = FastAPI(
    title="BELT Speaking AI V2",
    description="AI-powered speaking evaluation API.",
    version=VERSION,
)

# CORS (allow frontend dev environment)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],         # Change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------
# Root Endpoint
# ----------------------------------------------------
@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "BELT Speaking AI V2",
        "version": VERSION,
        "message": "API is running."
    }


# ----------------------------------------------------
# Main Evaluation Endpoint
# ----------------------------------------------------
@app.post("/evaluate")
async def evaluate(
    audio: UploadFile = File(...),
    task_id: int = Form(...),
):
    """
    Receives:
    - audio: WAV/OGG/M4A file
    - task_id: int

    Returns:
    - level (A2–C1)
    - explanation text
    - recommendations
    - duration seconds
    - task_id
    """

    # -------------------------------------------
    # Save temporary audio file to disk
    # -------------------------------------------
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    # -------------------------------------------
    # Run scoring pipeline
    # -------------------------------------------
    result = evaluate_audio(tmp_path)
    level = result["level"]
    explanation = result["explanation"]
    recommendations = result["recommendations"]
    seconds = result["seconds"]

    # -------------------------------------------
    # Log result (CSV-based for now)
    # -------------------------------------------
    timestamp = datetime.utcnow().isoformat()

    log_result(
        timestamp=timestamp,
        task_id=task_id,
        seconds=seconds,
        level=level,
        explanation=explanation,
        recommendations=recommendations,
    )

    # -------------------------------------------
    # Return JSON response
    # -------------------------------------------
    return {
        "score": level,
        "explanation": explanation,
        "recommendations": recommendations,
        "seconds": seconds,
        "task_id": task_id,
    }


# ----------------------------------------------------
# Uvicorn Local Dev (Render uses CMD instead)
# ----------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
