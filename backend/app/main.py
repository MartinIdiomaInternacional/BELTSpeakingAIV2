from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import tempfile

from app.scoring.cefr_scorer import evaluate_audio
from app.db import log_result
from app.version import VERSION

app = FastAPI(title="Speaking Test AI", version=VERSION)

app.add_middleware(CORSORMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def root():
    return {"status":"ok","version":VERSION}

@app.post("/evaluate")
async def evaluate(audio: UploadFile = File(...), task_id: int = Form(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await audio.read())
        p=tmp.name
    r = evaluate_audio(p)
    log_result(datetime.utcnow().isoformat(),task_id,r["seconds"],r["level"],r["explanation"],r["recommendations"])
    return {"score":r["level"],"explanation":r["explanation"],"recommendations":r["recommendations"],"seconds":r["seconds"],"task_id":task_id}
