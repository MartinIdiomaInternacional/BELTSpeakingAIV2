
# Docker Deploy (Render)

## Files
- Dockerfile
- .dockerignore
- render.yaml (optional blueprint)

## Steps
1) Place these files **in the repo root** (same level as `belt_service.py` and `requirements.txt`).
2) Commit & push to GitHub.
3) In Render:
   - New → Web Service → Connect your repo.
   - Runtime will be **Docker** automatically (Render sees the Dockerfile).
   - Add env var: `OPENAI_API_KEY = sk-...`
   - Deploy.

Logs should show Uvicorn binding to `0.0.0.0:$PORT` and `Service is live`.
