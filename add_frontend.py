
# add_frontend.py
from fastapi import APIRouter, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import json

def attach_frontend(app, web_dir: str = "web"):
    wd = Path(web_dir).resolve()
    static_dir = wd / "static"
    prompts_dir = wd / "prompts"

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.mount("/static", StaticFiles(directory=str(static_dir), html=False), name="static")

    router = APIRouter()

    @router.get("/")
    async def index():
        return FileResponse(str(wd / "index.html"))

    @router.get("/prompts/{level}")
    async def get_prompt(level: str):
        p = prompts_dir / f"{level}.json"
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"No prompt for level '{level}'")
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)

    app.include_router(router)
