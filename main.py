from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import librosa
import numpy as np
import torch
import torchaudio
from torchaudio.pipelines import WAV2VEC2_BASE, HUBERT_BASE
import uvicorn
import tempfile
import traceback
import openai
import os

# Load OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://martinidiomainternacional.github.io"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
wav2vec_model = WAV2VEC2_BASE.get_model()
hubert_model = HUBERT_BASE.get_model()

# =========================
# Utility Functions
# =========================

def classify_cefr_level(value, thresholds):
    if value < thresholds[0]:
        return "A1"
    elif value < thresholds[1]:
        return "A2"
    elif value < thresholds[2]:
        return "B1"
    elif value < thresholds[3]:
        return "B2"
    elif value < thresholds[4]:
        return "C1"
    else:
        return "C2"

def get_pronunciation_explanation(level):
    explanations = {
        "A1": "Your speech is hesitant and unclear, making understanding difficult.",
        "A2": "Pronunciation is understandable but with frequent pauses and mispronounced sounds.",
        "B1": "Speech is generally clear but still influenced by your first language.",
        "B2": "You speak with good pronunciation and intonation, with occasional mistakes.",
        "C1": "You sound natural and fluent, with minor pronunciation issues.",
        "C2": "Your pronunciation is near-native, clear, and natural throughout."
    }
    return explanations[level]

def extract_basic_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return {
        "mfcc_std": float(np.std(mfcc)),
        "tempo": float(tempo),
    }

def estimate_level_basic(features):
    tempo = features["tempo"]
    mfcc_std = features["mfcc_std"]
    avg_score = (tempo + mfcc_std * 10) / 2
    return classify_cefr_level(avg_score, [70, 85, 100, 115, 130])

def extract_deep_features(waveform, sr, model):
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    with torch.inference_mode():
        output = model(waveform)
        features = None
        if isinstance(output, tuple):
            for item in output:
                if isinstance(item, torch.Tensor):
                    features = item
                    break
        elif hasattr(output, 'extractor_features'):
            features = output.extractor_features
        elif hasattr(output, 'hidden_states'):
            features = output.hidden_states[-1]
        elif hasattr(output, 'last_hidden_state'):
            features = output.last_hidden_state

        if features is None:
            raise ValueError("Model did not return usable extractor features.")

    return features.mean(dim=1).squeeze().numpy()

def estimate_level_embedding(embedding):
    energy = np.linalg.norm(embedding)
    return classify_cefr_level(energy, [85, 100, 115, 130, 145])

# =========================
# Whisper Transcription
# =========================
def transcribe_audio(file_path):
    with open(file_path, "rb") as audio_file:
        transcript = openai.Audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    return transcript.text

# =========================
# Grammar & Vocabulary Analysis
# =========================
def analyze_grammar_vocabulary(transcript):
    prompt = f"""
    You are a certified English CEFR evaluator. Analyze this transcript and evaluate:
    1. Grammar CEFR level (A1-C2)
    2. Vocabulary CEFR level (A1-C2)
    3. Mistakes made in grammar and vocabulary
    4. Personalized feedback on how to improve based on CEFR descriptors

    Transcript:
    \"\"\"{transcript}\"\"\"

    Provide output in JSON:
    {{
        "grammar": {{"level": "B1", "explanation": "..." }},
        "vocabulary": {{"level": "B2", "explanation": "..." }}
    }}
    """
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message["content"]

# =========================
# API Endpoint
# =========================
@app.post("/evaluate")
async def evaluate_audio(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Load audio
        y, sr = librosa.load(tmp_path, sr=16000)
        try:
            waveform, sr_torch = torchaudio.load(tmp_path)
        except Exception:
            waveform = torch.tensor(y).unsqueeze(0)
            sr_torch = sr

        # 1️⃣ Pronunciation evaluation
        basic_features = extract_basic_features(y, sr)
        level_basic = estimate_level_basic(basic_features)
        emb_w2v = extract_deep_features(waveform, sr_torch, wav2vec_model)
        level_w2v = estimate_level_embedding(emb_w2v)
        emb_hubert = extract_deep_features(waveform, sr_torch, hubert_model)
        level_hubert = estimate_level_embedding(emb_hubert)

        # Final pronunciation level (majority vote)
        pronunciation_levels = [level_basic, level_w2v, level_hubert]
        final_pron_level = max(set(pronunciation_levels), key=pronunciation_levels.count)

        # 2️⃣ Grammar & Vocabulary evaluation
        transcript = transcribe_audio(tmp_path)
        gpt_analysis = analyze_grammar_vocabulary(transcript)

        # Combine results
        result = {
            "pronunciation": {"level": final_pron_level, "explanation": get_pronunciation_explanation(final_pron_level)},
            "text_analysis": gpt_analysis,
            "transcript": transcript
        }

        return JSONResponse(content=result)

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
