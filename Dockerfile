FROM python:3.10-slim

WORKDIR /app

# Install OS-level dependencies required for librosa and soundfile
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy project files first
COPY . .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 10000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
