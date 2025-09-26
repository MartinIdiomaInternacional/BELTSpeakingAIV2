#!/usr/bin/env bash
set -e

# Render sets $PORT; fall back to 10000 for local runs
PORT="${PORT:-10000}"

# Small health log
echo "Starting belt-service on 0.0.0.0:${PORT}"

# Run uvicorn
exec uvicorn belt_service:app --host 0.0.0.0 --port "${PORT}" --proxy-headers
