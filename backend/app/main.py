from datetime import datetime
import tempfile

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from app.scoring.cefr_scorer import evaluate_audio
from app.db import log_result
from app.version import VERSION


app = FastAPI(
    title="BELT Speaking AI 2.0",
    description="Prototype API for automatic speaking evaluation with CEFR-like output.",
    version=VERSION,
)

# CORS: in production you may want to restrict this to your frontend origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "BELT Speaking AI 2.0",
        "version": VERSION,
        "message": "API is running.",
    }


@app.post("/evaluate")
async def evaluate(
    audio: UploadFile = File(...),
    task_id: int = Form(...),
):
    """Evaluate a single speaking task.

    Expects:
    - audio: uploaded file (webm/wav/etc.); will be decoded by librosa.
    - task_id: integer identifying the task (1, 2, 3, ...).

    Returns:
    - score: CEFR-like level label (A2–C1, for now).
    - explanation: short textual explanation of the level.
    - recommendations: concrete suggestions to improve.
    - seconds: approximate speaking duration.
    - task_id: echoed task id.
    """
    # Save the uploaded file to a temporary path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    # Run scoring pipeline
    result = evaluate_audio(tmp_path)
    level = result["level"]
    explanation = result["explanation"]
    recommendations = result["recommendations"]
    seconds = float(result["seconds"])

    # Log result (CSV-based)
    timestamp = datetime.utcnow().isoformat()
    log_result(
        timestamp_utc=timestamp,
        task_id=task_id,
        seconds=seconds,
        level=level,
        explanation=explanation,
        recommendations=recommendations,
    )

    return {
        "score": level,
        "explanation": explanation,
        "recommendations": recommendations,
        "seconds": seconds,
        "task_id": task_id,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
