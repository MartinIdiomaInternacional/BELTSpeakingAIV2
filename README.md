# Working Speaking AI Eval 2.0

End-to-end demo: prep countdown + auto-record, native-language feedback, progressive probe, React frontend, FastAPI backend, Dockerized.

## Quick start
```bash
docker compose up --build
# Web:   http://localhost:5173
# API:   http://localhost:8000/health
```

## Structure
```
working-speaking-ai-eval-2.0/
├─ backend/
│  ├─ app/
│  │  ├─ main.py
│  │  ├─ models.py
│  │  ├─ prompts.py
│  │  ├─ scoring/
│  │  │  ├─ __init__.py
│  │  │  ├─ audio_features.py
│  │  │  └─ cefr_scorer.py
│  │  ├─ utils/
│  │  │  ├─ decode.py
│  │  │  ├─ language_feedback.py
│  │  │  └─ report.py
│  │  └─ version.py
│  ├─ requirements.txt
│  └─ Dockerfile
├─ frontend/
│  ├─ index.html
│  ├─ src/
│  │  ├─ main.jsx
│  │  ├─ App.jsx
│  │  ├─ components/
│  │  │  ├─ Countdown.jsx
│  │  │  ├─ Recorder.jsx
│  │  │  ├─ WaveformCanvas.jsx
│  │  │  └─ LanguageSelect.jsx
│  │  ├─ lib/api.js
│  │  └─ styles.css
│  ├─ package.json
│  ├─ vite.config.js
│  └─ Dockerfile
├─ docker-compose.yml
├─ render.yaml
└─ .github/workflows/ci.yml
```

## Render Deploy
1. Create a new repo on GitHub and push this folder.
2. In Render, **New +** → **Blueprint** → point to your repo (uses `render.yaml`).
3. After API deploys, update the **web** service environment variable `VITE_API_BASE` to the **public URL** of the API (shown in the API service dashboard), then redeploy the web service.

## CI
GitHub Actions workflow (`.github/workflows/ci.yml`) runs:
- Frontend: install & build
- Backend: install deps & import-check
- Docker: build both images
