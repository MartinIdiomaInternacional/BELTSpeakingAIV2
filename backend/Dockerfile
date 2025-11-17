FROM python:3.11-slim

WORKDIR /app

# 1) Copy backend requirements file from the backend folder
COPY backend/requirements /app/requirements
RUN pip install --no-cache-dir -r requirements

# 2) Copy the backend app package
COPY backend/app /app/app

# 3) Start FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
