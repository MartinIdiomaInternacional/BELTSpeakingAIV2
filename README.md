
  # BELT Speaking Test API (Safe Mode Ready)

  Two modes:
  - **SAFE mode (default)**: Skips SSL model downloads/loads. Fast startup, great for testing endpoints and keeping UIs responsive.
  - **FULL mode**: Enables torchaudio SSL features. Set `BELT_SAFE_MODE=0`.

  ## Run locally
  ```bash
  uvicorn main:app --reload
  ```

  ## Docker (SAFE mode default)
  ```bash
  docker build -t belt-api .
  docker run -p 8000:8000 -e BELT_SAFE_MODE=1 belt-api
  ```

  ## Docker (FULL mode with SSL features)
  ```bash
  docker run -p 8000:8000 -e BELT_SAFE_MODE=0 belt-api
  ```

  > On first FULL-mode run, the SSL model weights download (~hundreds of MB). This can cause timeouts. Use FULL mode only when needed, or bake weights into the image during build if you prefer.

  ## Optional: pre-bake SSL weights into the image
  Add after installing requirements:
  ```dockerfile
  # Warm the cache so the first request doesn't download models
  ENV TORCH_HOME=/root/.cache/torch
  RUN python - <<'PY'\nimport torchaudio\nprint('Warming SSL model...')\ntry:\n    torchaudio.pipelines.WAV2VEC2_BASE.get_model()\nexcept Exception:\n    torchaudio.pipelines.HUBERT_BASE.get_model()\nprint('Done.')\nPY
  ```

  ## Health check
  GET http://localhost:8000/health

  ## Evaluate
  ```bash
  curl -X POST "http://localhost:8000/evaluate" \
-H "accept: application/json" \
-H "Content-Type: multipart/form-data" \
-F "file=@sample.wav"
  ```

  ## Notes
  - Thread counts are limited to prevent CPU thrash on small machines.
  - If you deploy to Render, use **Starter** or better for adequate CPU/memory. Set `BELT_SAFE_MODE=1` initially, then flip to 0 after first warm-up.
