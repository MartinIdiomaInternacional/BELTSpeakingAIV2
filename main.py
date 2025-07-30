from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
import librosa
import numpy as np
import torch
import torchaudio
from torchaudio.pipelines import WAV2VEC2_BASE, HUBERT_BASE
import uvicorn
import tempfile
import traceback
import os
import random
import uuid
from gtts import gTTS
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust to your GitHub domain if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load speech models
wav2vec_model = WAV2VEC2_BASE.get_model()
hubert_model = HUBERT_BASE.get_model()

# ==============================
# Question bank per CEFR level
# ==============================
questions = {
        "A1": ["What is your name and where are you from?", "Describe your family.", "What do you usually eat for breakfast?", 
           "What is your favorite color and why?", "Tell me about your best friend.", "Describe your house or apartment.",
           "What do you do on weekends?", "What is your favorite food?", "Describe the clothes you are wearing today.",
           "What’s the weather like today?", "Tell me about your school or workplace.", "What do you usually drink in the morning?",
           "Can you describe your daily routine?", "Tell me about a hobby you like.", "What is your favorite animal?",
           "Who do you spend most of your time with?", "What is your favorite season?", "Tell me about a place you like to visit.",
           "What time do you usually go to bed?", "What do you usually do after dinner?"],
    "A2": ["Can you describe your last holiday?", "What do you usually do when you meet new people?", "Tell me about your favorite TV program.",
           "What kind of music do you enjoy?", "Describe your favorite restaurant.", "Talk about a person you admire.",
           "Tell me about a celebration or festival in your country.", "What do you do to stay healthy?", "What kind of sports do you like?",
           "Describe a typical day at work or school.", "Talk about a time you had to make a choice.", "Tell me about your favorite book.",
           "What kind of transport do you use most often?", "Describe a city or town you like.", "What is your dream job and why?",
           "Talk about your favorite holiday tradition.", "Can you describe an object that is important to you?",
           "Tell me about a funny thing that happened recently.", "What do you usually do when you feel tired?",
           "Talk about your favorite weekend activity."],
    "B1": ["What are the advantages and disadvantages of online learning?", "Tell me about a difficult decision you had to make.",
           "Describe a memorable experience from your school years.", "What are some problems in your town and how can they be solved?",
           "Discuss why people enjoy traveling.", "Talk about a time you helped someone.", "What are the pros and cons of social media?",
           "Tell me about a new skill you want to learn and why.", "Describe a challenge you overcame.", "Talk about a tradition you think is important.",
           "How has technology changed the way you shop?", "What makes a good friend?", "Do you think it's important to protect the environment? Why?",
           "How do you usually prepare for a trip?", "Talk about a goal you have for the next five years.", "What is a job that is very important in society?",
           "How do you celebrate birthdays in your culture?", "Talk about a recent change in your life.", "Should children use smartphones?",
           "Describe a good teacher you had and what made them special."],
    "B2": ["Explain how technology has changed the way we communicate.", "Do you think social media influences politics? Why?",
           "What are some common causes of stress, and how can people deal with it?", "Talk about a book or movie that changed your perspective.",
           "Discuss the importance of learning foreign languages today.", "Should governments spend more on the arts or science? Why?",
           "Do you believe that money can buy happiness? Explain.", "Talk about a time when you had to convince someone of your opinion.",
           "Is it better to live in a big city or a small town? Why?", "How can young people prepare for the future job market?",
           "Discuss the pros and cons of remote work.", "Do advertisements influence people's choices too much?",
           "What makes a good leader in your opinion?", "Do you think climate change can still be stopped? How?",
           "Talk about a difficult cultural difference you have experienced.", "Should public transport be free for everyone?",
           "How do movies and books affect our emotions?", "Explain why people enjoy extreme sports.",
           "Is online dating changing relationships? Explain your view.", "Discuss how tourism affects local cultures."],
    "C1": ["Discuss the impact of globalization on local cultures.", "Should governments regulate social media platforms more strictly?",
           "What are the challenges of living in a multicultural society?", "Discuss the role of technology in shaping human relationships.",
           "What should schools teach to prepare students for the 21st century?", "Do celebrities have a responsibility to be good role models?",
           "How important is freedom of speech in modern society?", "Should space exploration be publicly funded? Why?",
           "Discuss the ethics of genetic engineering in humans.", "What role should art play in education?",
           "Talk about a law you believe should be changed and why.", "Is social inequality inevitable in every society?",
           "How does the media shape public opinion about important issues?", "Discuss whether animal testing should be banned completely.",
           "What role do traditions play in modern society?", "Do you think science and religion can coexist?",
           "What are the pros and cons of artificial intelligence?", "How important is emotional intelligence for success?",
           "Talk about a global issue that needs urgent attention.", "Should universities focus more on research or teaching?"],
    "C2": ["If you were leading an international organization, how would you solve a global crisis?", "Is democracy the best form of government for all countries?",
           "What are the long-term consequences of technological dependence on society?", "Should advanced AI systems have rights? Why or why not?",
           "To what extent should cultural heritage be preserved in a fast-changing world?", "Can morality exist without religion? Discuss your view.",
           "How should nations address global wealth inequality?", "Is human progress always beneficial? Consider environmental costs.",
           "Should all scientific research be openly accessible to everyone?", "How do philosophical ideas shape political systems?",
           "Discuss whether a universal basic income would solve poverty.", "How can individuals balance personal freedom with social responsibility?",
           "Should medical resources be prioritized for younger patients in crises?", "What responsibilities do powerful nations have toward weaker ones?",
           "How might virtual reality change human interaction in the next 50 years?", "Should history be rewritten to correct past biases?",
           "Do international laws need to change to handle cyber warfare?", "Discuss the ethical dilemmas of cloning humans.",
           "Should space colonization be humanity's top priority?", "To what extent can art influence global political change?"]

}

# ==============================
# Session Storage
# ==============================
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
    Provide CEFR level for each and 1-2 specific mistakes found.
    Transcript:
    \"\"\"{transcript}\"\"\"

    Output in JSON:
    {{
      "grammar": {{"level":"B1","explanation":"..."}},
      "vocabulary": {{"level":"B2","explanation":"..."}},
      "coherence": {{"level":"B1","explanation":"..."}}
    }}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        print("Analysis error:", e)
        return '{"grammar": {"level":"A1","explanation":"Minimal analysis."}}'

def generate_tts(text, filename):
    try:
        tts = gTTS(text)
        tts.save(filename)
    except Exception as e:
        print("TTS error:", e)
    return filename

# ==============================
# API Endpoints
# ==============================

@app.get("/")
async def root():
    return {"message": "CEFR Speaking Evaluator API v4.2 (patched) is running"}

@app.post("/start_test")
async def start_test():
    session_id = str(uuid.uuid4())
    sessions[session_id] = {"questions_asked": 0, "answers": [], "current_level": "A1"}
    q = random.choice(questions["A1"])
    audio_file = f"{session_id}_q1.mp3"
    generate_tts(q, audio_file)
    return JSONResponse({"session_id": session_id, "question": q, "audio_url": f"/audio/{audio_file}"})

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

        sessions[session_id]["answers"].append({"pronunciation": pron_level, "text": text_eval})
        sessions[session_id]["questions_asked"] += 1

        current_level = sessions[session_id]["current_level"]
        level_order = ["A1","A2","B1","B2","C1","C2"]
        next_idx = max(level_order.index(current_level), 0)
        if "B2" in text_eval or "C" in text_eval or pron_level in ["B2","C1","C2"]:
            next_idx = min(next_idx+1, 5)
        next_level = level_order[next_idx]
        sessions[session_id]["current_level"] = next_level

        if sessions[session_id]["questions_asked"] >= 6:
            return JSONResponse({"done": True})

        next_q = random.choice(questions.get(next_level, questions["A1"])) or "Please answer this next question."
        audio_file = f"{session_id}_q{sessions[session_id]['questions_asked']+1}.mp3"
        generate_tts(next_q, audio_file)
        return JSONResponse({"done": False, "question": next_q, "audio_url": f"/audio/{audio_file}"})

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/final_result")
async def final_result(session_id: str = Form(...)):
    if session_id not in sessions:
        return JSONResponse(status_code=404, content={"error": "Session not found."})
    answers = sessions[session_id]["answers"]
    levels = [ans["pronunciation"] for ans in answers]
    final_level = max(set(levels), key=levels.count) if levels else "A1"
    feedback_text = f"Your estimated CEFR level is {final_level}. Keep practicing grammar and vocabulary to improve further."
    feedback_audio = f"{session_id}_final.mp3"
    generate_tts(feedback_text, feedback_audio)
    return JSONResponse({"overall_level": final_level, "details": answers, "feedback_audio_url": f"/audio/{feedback_audio}"})

@app.get("/audio/{filename}")
async def get_audio(filename: str):
    return FileResponse(filename, media_type="audio/mpeg")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
