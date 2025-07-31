from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import librosa
import numpy as np
import torch
import torchaudio
from torchaudio.pipelines import WAV2VEC2_BASE
import uvicorn
import tempfile
import traceback
import os
import random
import uuid
from openai import OpenAI

# ==============================
# Initialize OpenAI Client
# ==============================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust to your GitHub Pages domain for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load speech model
wav2vec_model = WAV2VEC2_BASE.get_model()

# ==============================
# Question bank per CEFR level
# ==============================
questions = {
    "A1": ["What is your name and where are you from?", "Describe your family.", "What do you usually eat for breakfast?",
           "What is your favorite color and why?", "Tell me about your best friend."],
    "A2": ["Can you describe your last holiday?", "What do you usually do when you meet new people?", "Tell me about your favorite TV program.",
           "What kind of music do you enjoy?", "Describe your favorite restaurant."],
    "B1": ["What are the advantages and disadvantages of online learning?", "Tell me about a difficult decision you had to make.",
           "Describe a memorable experience from your school years.", "What are some problems in your town and how can they be solved?",
           "Discuss why people enjoy traveling."],
    "B2": ["Explain how technology has changed the way we communicate.", "Do you think social media influences politics? Why?",
           "What are some common causes of stress, and how can people deal with it?", "Talk about a book or movie that changed your perspective.",
           "Discuss the importance of learning foreign languages today."],
    "C1": ["Discuss the impact of globalization on local cultures.", "Should governments regulate social media platforms more strictly?",
           "What are the challenges of living in a multicultural society?", "Discuss the role of technology in shaping human relationships.",
           "What should schools teach to prepare students for the 21st century?"],
    "C2": ["If you were leading an international organization, how would you solve a global crisis?", "Is democracy the best form of government for all countries?",
           "What are the long-term consequences of technological dependence on society?", "Should advanced AI systems have rights? Why or why not?",
           "To what extent should cultural heritage be preserved in a fast-changing world?"]
}

# Session storage
sessions = {}

# ==============================
# Helper functions
# ==============================
def classify_cefr_level(value, thresholds):
    if value < thresholds[0]: return "A1"
    elif value < thresholds[1]: return "A2"
    elif value < thresholds[2]: return "B1"
    elif value < thresholds[3]: return "B2"
    elif value < thresholds[4]: return "C1"
    else: return "C2"

def estimate_level_embedding(embedding):
    energy = np.linalg.norm(embedding)
    return classify_cefr_level(energy, [85, 100, 115, 130, 145])

def extract_deep_features(waveform, sr, model):
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    with torch.inference_mode():
        output = model(waveform)
        if isinstance(output, tuple):
            for item in output:
                if isinstance(item, torch.Tensor):
                    return item.mean(dim=1).squeeze().numpy()
        elif hasattr(output, 'extractor_features'):
            return output.extractor_features.mean(dim=1).squeeze().numpy()
    return np.array([0.0])

def transcribe_audio(file_path):
    try:
        with open(file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        return transcript.text or ""
    except Exception as e:
        print("Transcription error:", e)
        return ""

def analyze_text(transcript):
    prompt = f"""
    You are an English CEFR examiner. Evaluate this transcript on grammar, vocabulary, and coherence.
    Provide CEFR level for each, and list 1-2 specific mistakes.
    Transcript:
    \"\"\"{transcript}\"\"\"

    Output in JSON strictly like this:
    {{
      "grammar": {{"level":"B1","explanation":"Frequent tense errors."}},
      "vocabulary": {{"level":"B2","explanation":"Good range but some word choice issues."}},
      "coherence": {{"level":"B1","explanation":"Some sentences lack connection."}}
    }}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content
    except Exception as e:
        print("Analysis error:", e)
        return '{"grammar": {"level":"A1","explanation":"Minimal analysis."}}'

def weighted_final_score(pron, grammar, vocab, coherence):
    levels = {"A1":1,"A2":2,"B1":3,"B2":4,"C1":5,"C2":6}
    score = (0.4*levels.get(pron,1) +
             0.2*levels.get(grammar,1) +
             0.2*levels.get(vocab,1) +
             0.2*levels.get(coherence,1))
    if score <= 1.5: return "A1"
    elif score <= 2.5: return "A2"
    elif score <= 3.5: return "B1"
    elif score <= 4.5: return "B2"
    elif score <= 5.5: return "C1"
    else: return "C2"

# ==============================
# API Endpoints
# ==============================
@app.get("/")
async def root():
    return {"message": "CEFR Speaking Evaluator API v4.5 is running"}

@app.post("/start_test")
async def start_test():
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "questions_asked": 0,
        "answers": [],
        "current_level": "A1",
        "asked_questions": set()
    }
    q = random.choice(questions["A1"])
    sessions[session_id]["asked_questions"].add(q)
    return JSONResponse({"session_id": session_id, "question": q})

@app.post("/next_question")
async def next_question(session_id: str = Form(...), file: UploadFile = File(...)):
    try:
        if session_id not in sessions:
            return JSONResponse(status_code=404, content={"error": "Session not found."})

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        y, sr = librosa.load(tmp_path, sr=16000)
        waveform, sr_torch = torchaudio.load(tmp_path)
        emb = extract_deep_features(waveform, sr_torch, wav2vec_model)
        pron_level = estimate_level_embedding(emb)
        transcript = transcribe_audio(tmp_path)
        text_eval = analyze_text(transcript)

        # Parse grammar/vocabulary/coherence levels from text_eval JSON
        try:
            import json
            parsed_eval = json.loads(text_eval)
        except:
            parsed_eval = {"grammar":{"level":"A1"},"vocabulary":{"level":"A1"},"coherence":{"level":"A1"}}

        sessions[session_id]["answers"].append({
            "pronunciation": pron_level,
            "text": parsed_eval
        })
        sessions[session_id]["questions_asked"] += 1

        # Adaptive progression
        current_level = sessions[session_id]["current_level"]
        level_order = ["A1","A2","B1","B2","C1","C2"]
        highest = max([pron_level, parsed_eval["grammar"]["level"], parsed_eval["vocabulary"]["level"]], key=lambda x: level_order.index(x))
        current_idx = level_order.index(current_level)
        highest_idx = level_order.index(highest)
        next_idx = min(max(current_idx, highest_idx), 5)
        next_level = level_order[next_idx]
        sessions[session_id]["current_level"] = next_level

        if sessions[session_id]["questions_asked"] >= 6:
            return JSONResponse({"done": True})

        # Ensure no repeated questions
        available = [q for q in questions[next_level] if q not in sessions[session_id]["asked_questions"]]
        if not available:
            available = questions[next_level]
        next_q = random.choice(available)
        sessions[session_id]["asked_questions"].add(next_q)

        return JSONResponse({"done": False, "question": next_q})

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/final_result")
async def final_result(session_id: str = Form(...)):
    if session_id not in sessions:
        return JSONResponse(status_code=404, content={"error": "Session not found."})
    answers = sessions[session_id]["answers"]
    final_levels = []
    for ans in answers:
        grammar = ans["text"].get("grammar", {}).get("level","A1")
        vocab = ans["text"].get("vocabulary", {}).get("level","A1")
        coherence = ans["text"].get("coherence", {}).get("level","A1")
        final_levels.append(weighted_final_score(ans["pronunciation"], grammar, vocab, coherence))
    final_level = max(set(final_levels), key=final_levels.count) if final_levels else "A1"
    return JSONResponse({"overall_level": final_level, "details": answers})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
