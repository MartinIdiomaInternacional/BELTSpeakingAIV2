from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import tempfile

from app.scoring.cefr_scorer import evaluate_audio
from app.db import log_result
from app.version import VERSION


app = FastAPI(
    title="Speaking Test AI 2.0",
    description="AI-powered speaking evaluation API.",
    version=VERSION,
)

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
        "service": "Speaking Test AI 2.0",
        "version": VERSION,
    }


@app.post("/evaluate")
async def evaluate(
    audio: UploadFile = File(...),
    task_id: int = Form(...),
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    result = evaluate_audio(tmp_path)
    level = result["level"]
    explanation = result["explanation"]
    recommendations = result["recommendations"]
    seconds = result["seconds"]

    timestamp = datetime.utcnow().isoformat()
    log_result(
        timestamp=timestamp,
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
