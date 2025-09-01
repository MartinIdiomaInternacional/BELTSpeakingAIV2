from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
import numpy as np
import torch
import torchaudio
from torchaudio.pipelines import WAV2VEC2_BASE, HUBERT_BASE
import uvicorn
import tempfile
import traceback
import subprocess
import os
import random
import uuid
import openai

# ==============================
# OpenAI API key
# ==============================
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your domain(s) in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# Load speech models (SSL)
# ==============================
wav2vec_model = WAV2VEC2_BASE.get_model()
hubert_model = HUBERT_BASE.get_model()

TARGET_SR = 16_000
CEFR_ORDER = ["A1", "A2", "B1", "B2", "C1", "C2"]

# ==============================
# Question bank (20 per level)
# ==============================
questions = {
    "A1": [
        "What is your name and where are you from?", "Describe your family.", "What do you usually eat for breakfast?",
        "What is your favorite color and why?", "Tell me about your best friend.", "Describe your house or apartment.",
        "What do you do on weekends?", "What is your favorite food?", "Describe the clothes you are wearing today.",
        "What’s the weather like today?", "Tell me about your school or workplace.", "What do you usually drink in the morning?",
        "Can you describe your daily routine?", "Tell me about a hobby you like.", "What is your favorite animal?",
        "Who do you spend most of your time with?", "What is your favorite season?", "Tell me about a place you like to visit.",
        "What time do you usually go to bed?", "What do you usually do after dinner?"
    ],
    "A2": [
        "Can you describe your last holiday?", "What do you usually do when you meet new people?", "Tell me about your favorite TV program.",
        "What kind of music do you enjoy?", "Describe your favorite restaurant.", "Talk about a person you admire.",
        "Tell me about a celebration or festival in your country.", "What do you do to stay healthy?", "What kind of sports do you like?",
        "Describe a typical day at work or school.", "Talk about a time you had to make a choice.", "Tell me about your favorite book.",
        "What kind of transport do you use most often?", "Describe a city or town you like.", "What is your dream job and why?",
        "Talk about your favorite holiday tradition.", "Can you describe an object that is important to you?",
        "Tell me about a funny thing that happened recently.", "What do you usually do when you feel tired?",
        "Talk about your favorite weekend activity."
    ],
    "B1": [
        "What are the advantages and disadvantages of online learning?", "Tell me about a difficult decision you had to make.",
        "Describe a memorable experience from your school years.", "What are some problems in your town and how can they be solved?",
        "Discuss why people enjoy traveling.", "Talk about a time you helped someone.", "What are the pros and cons of social media?",
        "Tell me about a new skill you want to learn and why.", "Describe a challenge you overcame.", "Talk about a tradition you think is important.",
        "How has technology changed the way you shop?", "What makes a good friend?", "Do you think it's important to protect the environment? Why?",
        "How do you usually prepare for a trip?", "Talk about a goal you have for the next five years.", "What is a job that is very important in society?",
        "How do you celebrate birthdays in your culture?", "Talk about a recent change in your life.", "Should children use smartphones?",
        "Describe a good teacher you had and what made them special."
    ],
    "B2": [
        "Explain how technology has changed the way we communicate.", "Do you think social media influences politics? Why?",
        "What are some common causes of stress, and how can people deal with it?", "Talk about a book or movie that changed your perspective.",
        "Discuss the importance of learning foreign languages today.", "Should governments spend more on the arts or science? Why?",
        "Do you believe that money can buy happiness? Explain.", "Talk about a time when you had to convince someone of your opinion.",
        "Is it better to live in a big city or a small town? Why?", "How can young people prepare for the future job market?",
        "Discuss the pros and cons of remote work.", "Do advertisements influence people's choices too much?",
        "What makes a good leader in your opinion?", "Do you think climate change can still be stopped? How?",
        "Talk about a difficult cultural difference you have experienced.", "Should public transport be free for everyone?",
        "How do movies and books affect our emotions?", "Explain why people enjoy extreme sports.",
        "Is online dating changing relationships? Explain your view.", "Discuss how tourism affects local cultures."
    ],
    "C1": [
        "Discuss the impact of globalization on local cultures.", "Should governments regulate social media platforms more strictly?",
        "What are the challenges of living in a multicultural society?", "Discuss the role of technology in shaping human relationships.",
        "What should schools teach to prepare students for the 21st century?", "Do celebrities have a responsibility to be good role models?",
        "How important is freedom of speech in modern society?", "Should space exploration be publicly funded? Why?",
        "Discuss the ethics of genetic engineering in humans.", "What role should art play in education?",
        "Talk about a law you believe should be changed and why.", "Is social inequality inevitable in every society?",
        "How does the media shape public opinion about important issues?", "Discuss whether animal testing should be banned completely.",
        "What role do traditions play in modern society?", "Do you think science and religion can coexist?",
        "What are the pros and cons of artificial intelligence?", "How important is emotional intelligence for success?",
        "Talk about a global issue that needs urgent attention.", "Should universities focus more on research or teaching?"
    ],
    "C2": [
        "If you were leading an international organization, how would you solve a global crisis?", "Is democracy the best form of government for all countries?",
        "What are the long-term consequences of technological dependence on society?", "Should advanced AI systems have rights? Why or why not?",
        "To what extent should cultural heritage be preserved in a fast-changing world?", "Can morality exist without religion? Discuss your view.",
        "How should nations address global wealth inequality?", "Is human progress always beneficial? Consider environmental costs.",
        "Should all scientific research be openly accessible to everyone?", "How do philosophical ideas shape political systems?",
        "Discuss whether a universal basic income would solve poverty.", "How can individuals balance personal freedom with social responsibility?",
        "Should medical resources be prioritized for younger patients in crises?", "What responsibilities do powerful nations have toward weaker ones?",
        "How might virtual reality change human interaction in the next 50 years?", "Should history be rewritten to correct past biases?",
        "Do international laws need to change to handle cyber warfare?", "Discuss the ethical dilemmas of cloning humans.",
        "Should space colonization be humanity's top priority?", "To what extent can art influence global political change?"
    ]
}

# ==============================
# Session storage (for legacy routes)
# ==============================
sessions: dict[str, dict] = {}

# ==============================
# Helper functions
# ==============================
def classify_cefr_level(value: float, thresholds: list[float]) -> str:
    if value < thresholds[0]: return "A1"
    elif value < thresholds[1]: return "A2"
    elif value < thresholds[2]: return "B1"
    elif value < thresholds[3]: return "B2"
    elif value < thresholds[4]: return "C1"
    else: return "C2"

def estimate_level_from_emb(emb: torch.Tensor) -> str:
    # L2-normalize then scale so the classic thresholds (85..145) make sense
    emb = torch.nn.functional.normalize(emb, dim=-1)
    energy = float(torch.linalg.vector_norm(emb).item() * 100.0)  # ~100
    return classify_cefr_level(energy, [85, 100, 115, 130, 145])

def ensure_mono_16k(waveform: torch.Tensor, sr: int) -> tuple[torch.Tensor, int]:
    # (C, T) or (T,) -> (1, T) at 16k
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # (1, T)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16_000:
        waveform = torchaudio.functional.resample(waveform, sr, 16_000)
        sr = 16_000
    return waveform.to(torch.float32), sr

def load_audio_any(path: str) -> tuple[torch.Tensor, int]:
    # Try torchaudio first; on failure, ffmpeg-convert to 16k mono WAV
    try:
        waveform, sr = torchaudio.load(path)  # (C, T)
    except Exception:
        tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp_wav_path = tmp_wav.name
        tmp_wav.close()
        try:
            # Convert using ffmpeg
            subprocess.run(
                ["ffmpeg", "-y", "-i", path, "-ac", "1", "-ar", "16000", "-f", "wav", tmp_wav_path],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            waveform, sr = torchaudio.load(tmp_wav_path)
        finally:
            if os.path.exists(tmp_wav_path):
                os.unlink(tmp_wav_path)
    waveform, sr = ensure_mono_16k(waveform, sr)
    return waveform, sr

def ssl_embedding(waveform: torch.Tensor, sr: int, model, device: str = "cpu") -> torch.Tensor:
    # waveform expected (1, T) at 16k; enforce anyway
    waveform, _ = ensure_mono_16k(waveform, sr)
    waveform = waveform.to(device)
    model = model.to(device).eval()
    with torch.inference_mode():
        # Prefer torchaudio's explicit extractor
        if hasattr(model, "extract_features"):
            feats, _ = model.extract_features(waveform)  # may be List[Tensor] or Tensor
            x = feats[-1] if isinstance(feats, (list, tuple)) else feats  # (B, frames, feat_dim)
        else:
            out = model(waveform)
            if isinstance(out, (list, tuple)) and len(out) > 0:
                x = out[0]
            elif hasattr(out, "extractor_features"):
                x = out.extractor_features
            elif hasattr(out, "last_hidden_state"):
                x = out.last_hidden_state
            else:
                x = out
    if x.dim() == 3 and x.shape[1] < x.shape[2]:
        # likely (B, T, C) already
        pass
    elif x.dim() == 3 and x.shape[1] > x.shape[2]:
        # (B, C, T) -> (B, T, C)
        x = x.transpose(1, 2)
    elif x.dim() == 2:
        x = x.unsqueeze(1)
    emb = x.mean(dim=1)  # (B, C)
    return emb.squeeze(0)

def transcribe_audio(file_path: str) -> str:
    try:
        with open(file_path, "rb") as audio_file:
            # openai==0.28 style
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
        return transcript.get("text", "") if isinstance(transcript, dict) else ""
    except Exception as e:
        print("Transcription error:", e)
        return ""

def analyze_text(transcript: str) -> str:
    prompt = f"""
You are an English CEFR examiner. Evaluate this transcript on grammar, vocabulary, and coherence.
Provide CEFR level for each and 1–2 specific mistakes found.

Transcript:
\"\"\"{transcript}\"\"\"

Output in JSON ONLY (no prose):
{{
  "grammar": {{"level":"B1","explanation":"..."}},
  "vocabulary": {{"level":"B2","explanation":"..."}},
  "coherence": {{"level":"B1","explanation":"..."}}
}}
"""
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message["content"]
    except Exception as e:
        print("Analysis error:", e)
        return '{"grammar": {"level":"A1","explanation":"Minimal analysis."}}'

# ==============================
# Basic routes
# ==============================
@app.get("/")
async def root():
    return {"message": "CEFR Speaking Evaluator API (no-librosa build) is running"}

@app.get("/healthz")
async def healthz():
    return {"ok": True}

# Return a random prompt for a given level key (accepts B1+ / B2+ aliases)
@app.get("/prompts/{level}")
async def get_prompt(level: str):
    key = level.upper().replace("%2B", "+").replace(" ", "")
    if key.endswith("+"):
        key = key[:-1]
    if key not in questions:
        return JSONResponse(status_code=404, content={"error": f"Unknown level '{level}'"})
    return {"level": key, "question": random.choice(questions[key])}

# ==============================
# Evaluation routes
# ==============================
@app.post("/evaluate")
async def evaluate(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename or '')[-1] or ".bin") as tmp:
            tmp.write(await file.read())
            path = tmp.name

        waveform, sr = load_audio_any(path)
        emb = ssl_embedding(waveform, sr, wav2vec_model)
        pron_level = estimate_level_from_emb(emb)

        transcript = transcribe_audio(path)
        text_eval = analyze_text(transcript) if transcript else '{"grammar": {"level":"A1","explanation":"No transcript"}}'

        return {
            "pronunciation_level": pron_level,
            "embedding_energy_hint": float(torch.linalg.vector_norm(torch.nn.functional.normalize(emb, dim=-1)).item() * 100.0),
            "transcript": transcript,
            "text_eval": text_eval
        }
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        try:
            if 'path' in locals() and os.path.exists(path):
                os.unlink(path)
        except Exception:
            pass

# Raw-bytes endpoint (no multipart needed)
@app.post("/evaluate-bytes")
async def evaluate_bytes(request: Request):
    try:
        data = await request.body()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tmp:
            tmp.write(data)
            path = tmp.name

        waveform, sr = load_audio_any(path)
        emb = ssl_embedding(waveform, sr, wav2vec_model)
        pron_level = estimate_level_from_emb(emb)

        transcript = transcribe_audio(path)
        text_eval = analyze_text(transcript) if transcript else '{"grammar": {"level":"A1","explanation":"No transcript"}}'

        return {
            "pronunciation_level": pron_level,
            "embedding_energy_hint": float(torch.linalg.vector_norm(torch.nn.functional.normalize(emb, dim=-1)).item() * 100.0),
            "transcript": transcript,
            "text_eval": text_eval
        }
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        try:
            if 'path' in locals() and os.path.exists(path):
                os.unlink(path)
        except Exception:
            pass

# ==============================
# Legacy flows kept for compatibility
# ==============================
@app.post("/start_test")
async def start_test():
    session_id = str(uuid.uuid4())
    sessions[session_id] = {"questions_asked": 0, "answers": [], "current_level": "A1"}
    q = random.choice(questions["A1"])
    return JSONResponse({"session_id": session_id, "question": q})

@app.post("/next_question")
async def next_question(session_id: str = Form(...), file: UploadFile = File(...)):
    try:
        if session_id not in sessions:
            return JSONResponse(status_code=404, content={"error": "Session not found."})

        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename or '')[-1] or ".bin") as tmp:
            tmp.write(await file.read())
            path = tmp.name

        waveform, sr = load_audio_any(path)
        emb = ssl_embedding(waveform, sr, wav2vec_model)
        pron_level = estimate_level_from_emb(emb)

        transcript = transcribe_audio(path)
        text_eval = analyze_text(transcript) if transcript else '{"grammar": {"level":"A1","explanation":"No transcript"}}'

        sessions[session_id]["answers"].append({"pronunciation": pron_level, "text": text_eval})
        sessions[session_id]["questions_asked"] += 1

        current_level = sessions[session_id]["current_level"]
        idx = CEFR_ORDER.index(current_level)
        if pron_level in ["B2","C1","C2"] or "B2" in text_eval or "\"C" in text_eval:
            idx = min(idx + 1, len(CEFR_ORDER) - 1)
        sessions[session_id]["current_level"] = CEFR_ORDER[idx]

        if sessions[session_id]["questions_asked"] >= 6:
            return JSONResponse({"done": True, "progress": 100})

        next_q = random.choice(questions[CEFR_ORDER[idx]])
        progress = int((sessions[session_id]["questions_asked"] / 6) * 100)
        return JSONResponse({"done": False, "question": next_q, "progress": progress})

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        try:
            if 'path' in locals() and os.path.exists(path):
                os.unlink(path)
        except Exception:
            pass

@app.post("/final_result")
async def final_result(session_id: str = Form(...)):
    if session_id not in sessions:
        return JSONResponse(status_code=404, content={"error": "Session not found."})
    answers = sessions[session_id]["answers"]
    if not answers:
        return JSONResponse({"overall_level": "A1", "details": []})
    levels = [ans["pronunciation"] for ans in answers]
    # majority vote
    final_level = max(set(levels), key=levels.count)
    return JSONResponse({"overall_level": final_level, "details": answers})

# ==============================
# Run server (when not launched by uvicorn CLI)
# ==============================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
