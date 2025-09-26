#!/usr/bin/env bash
set -e

PORT="${PORT:-10000}"
echo "Starting belt-service on 0.0.0.0:${PORT}"
exec uvicorn belt_service:app --host 0.0.0.0 --port "${PORT}" --proxy-headers

