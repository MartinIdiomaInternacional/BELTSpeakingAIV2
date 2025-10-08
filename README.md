
# BELTSpeakingAIV2 — Adaptive Pro

- Adaptive start at A1, confidence jumps, finalize on streak.
- Quality gate: min voiced seconds + SNR.
- Optional ASR (OpenAI Whisper v1 via `OPENAI_API_KEY`) to enrich confidence with lexical metrics.
- Postgres persistence (`DATABASE_URL`).

## Local
```bash
docker compose up --build
# web: http://localhost:5173
# api: http://localhost:8000/health
```
