
# BELT Speaking AI — One Service

Single FastAPI service that includes:
- /evaluate-bytes (single-shot CEFR estimate)
- Adaptive flow: /start-session, /submit-response, /report/{id}
- Frontend served at / (index.html), /static, /prompts/{level}
- Health: /healthz, /readyz; Metrics: /metrics

## Run locally
```
pip install -r requirements.txt
uvicorn belt_service:app --host 0.0.0.0 --port 8000
```
Open http://localhost:8000

## Deploy on Render
- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn belt_service:app --host 0.0.0.0 --port $PORT`
- Set env vars (see .env.example)

## Notes
- If you want offline dev without OpenAI, set `ASR_BACKEND=none` (transcript placeholder).
- For real scoring, set `OPENAI_API_KEY` and leave defaults.
