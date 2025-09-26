# -*- coding: utf-8 -*-
"""
BELT Speaking Evaluator (FastAPI)

Features:
- 3-second countdown -> record up to RECORD_SECONDS (default 60)
- Stop & Send / Abort
- Robust recorder MIME fallbacks (handled client-side)
- Session-aware prompts (no repeats), prompt_id/question logging
- Attempt badge (1 of 2 / 2 of 2), one retry per level
- Auto-advance on pass, final stop after second fail
- Score chips + average + level tracker (client)
- Tailored recommendations per turn (rule-based + AI) and session-level
- TTS playback support for prompts (optional)
- Static UI served from /web
"""

import os
import io
import json
import math
import uuid
import time
import shutil
import logging
import asyncio
import tempfile
import subprocess
from typing import Dict, List, Any, Optional
from collections import Counter
const el = {
  btnStart:      document.querySelector("#btnStart"),
  btnAbort:      document.querySelector("#btnAbort"),
  btnStopSend:   document.querySelector("#btnStopSend"),
  status:        document.querySelector("#status"),
  countdown:     document.querySelector("#countdown"),
  timer:         document.querySelector("#timer"),
  progress:      document.querySelector("#progress"),
  levels:        document.querySelector("#levels"),
  prompt:        document.querySelector("#prompt"),
  attemptNote:   document.querySelector("#attemptNote"),
  player:        document.querySelector("#player"),
  chips:         document.querySelector("#chips"),
  avg:           document.querySelector("#avg"),
  recs:          document.querySelector("#recs"),   // ← new (optional) recommendations area
  live:          document.querySelector("#live"),
  result:        document.querySelector("#result"),
};

// ===========================
// App State
// ===========================
let SESSION_ID = null;
let CURRENT_LEVEL = "A1";
let CURRENT_PROMPT = "";
let CURRENT_PROMPT_ID = null;
let RECORD_SECONDS = 60;      // loaded from /config (min 60)
let PASS_AVG_THRESHOLD = 0.7; // from /config
let PASS_MIN_THRESHOLD = 0.6; // from /config

// Recorder state
let mediaStream = null;
let mediaRecorder = null;
let recordedChunks = [];
let isRecording = false;
let hasStoppedRecorder = false;
let timerInterval = null;
let startTimestamp = null;

// ===========================
// Global error -> show in UI
// ===========================
window.addEventListener("error", (e) => {
  appendLive(`JS Error: ${e.message}`, true);
  console.error(e.error || e.message);
});
window.addEventListener("unhandledrejection", (e) => {
  appendLive(`Async Error: ${e.reason}`, true);
  console.error(e.reason);
});

// ===========================
// UI helpers
// ===========================
function show(elm) { if (elm) elm.classList.remove("hidden"); }
function hide(elm) { if (elm) elm.classList.add("hidden"); }
function setText(elm, txt) { if (elm) elm.textContent = txt; }
function setHTML(elm, html) { if (elm) elm.innerHTML = html; }
function setStatus(txt) { if (el.status) { setText(el.status, txt); show(el.status); } }
function appendLive(msg, isErr=false) {
  if (!el.live) return;
  const ts = new Date().toLocaleTimeString();
  const line = `[${ts}] ${msg}\n`;
  el.live.textContent = line + el.live.textContent;
  if (isErr) setStatus("Error");
}
function safeAudioPlay(url) { try { new Audio(url).play().catch(()=>{}); } catch(_){} }

// ===========================
// Config & Levels
// ===========================
async function loadConfig() {
  try {
    const r = await fetch(CONFIG_ENDPOINT, FETCH_OPTS);
    const cfg = await r.json();
    const rec = parseInt(cfg.RECORD_SECONDS ?? RECORD_SECONDS, 10);
    RECORD_SECONDS = Number.isFinite(rec) ? Math.max(60, rec) : 60;
    PASS_AVG_THRESHOLD = parseFloat(cfg.PASS_AVG_THRESHOLD ?? PASS_AVG_THRESHOLD);
    PASS_MIN_THRESHOLD = parseFloat(cfg.PASS_MIN_THRESHOLD ?? PASS_MIN_THRESHOLD);
    appendLive(`Config: RECORD_SECONDS=${RECORD_SECONDS}, PASS_AVG=${PASS_AVG_THRESHOLD}, MIN=${PASS_MIN_THRESHOLD}`);
  } catch (e) {
    RECORD_SECONDS = 60;
    appendLive(`Config load failed; defaults in use. ${e}`);
  }
}
function renderLevels(current) {
  if (!el.levels) return;
  const html = CEFR_LEVELS.map(lvl => {
    let cls = "lvl locked";
    if (lvl === current) cls = "lvl current";
    if (CEFR_LEVELS.indexOf(lvl) < CEFR_LEVELS.indexOf(current)) cls = "lvl passed";
    return `<span class="${cls}">${lvl}</span>`;
  }).join("");
  setHTML(el.levels, html);
}

// ===========================
// Prompt & Scores
// ===========================
function renderPrompt(text, level, attemptLabel = "") {
  if (el.prompt) setText(el.prompt, text || "(none)");
  if (el.attemptNote) setText(el.attemptNote, attemptLabel);
  renderLevels(level || CURRENT_LEVEL);
}
function renderScores(result) {
  if (!result || !result.scores) {
    setHTML(el.chips, "");
    setHTML(el.avg, "");
    renderRecommendations(null);
    return;
  }
  const s = result.scores;
  const avgPct = Math.round((result.average ?? 0) * 100);
  const chip = (label, val) => {
    const pct = Math.round((val ?? 0) * 100);
    let cls = "chip err";
    if (pct >= PASS_MIN_THRESHOLD * 100 && pct < PASS_AVG_THRESHOLD * 100) cls = "chip warn";
    if (pct >= PASS_AVG_THRESHOLD * 100) cls = "chip ok";
    return `<span class="${cls}">${label}: <span class="val">${pct}</span></span>`;
  };
  setHTML(el.chips, [
    chip("Fluency", s.fluency),
    chip("Grammar", s.grammar),
    chip("Vocabulary", s.vocabulary),
    chip("Pronunciation", s.pronunciation),
    chip("Coherence", s.coherence),
  ].join(" "));
  const decision = result.decision ? ` • decision: ${result.decision}` : "";
  const probe = result.borderline ? ` • probe` : "";
  setText(el.avg, `Average: ${avgPct}${decision}${probe}`);

  // NEW: show recommendations (per turn)
  renderRecommendations(result);
}

// Centralized recommendations renderer
function renderRecommendations(result) {
  if (!el.recs) return; // HTML may not have the container yet; safe no-op
  if (!result) { setHTML(el.recs, ""); return; }

  // Accept either array of strings, or single string under common keys
  const recArray =
      (Array.isArray(result.recommendations) && result.recommendations)
   || (Array.isArray(result.suggestions) && result.suggestions)
   || (typeof result.recommendations === "string" && [result.recommendations])
   || (typeof result.suggestions === "string" && [result.suggestions])
   || [];

  if (recArray.length === 0) { setHTML(el.recs, ""); return; }

  // Simple bullet list
  const html = `
    <div style="margin-top: .25rem">
      <strong>Recommendations:</strong>
      <ul style="margin:.25rem 0 0 .9rem; padding:0;">
        ${recArray.map(x => `<li>${escapeHtml(x)}</li>`).join("")}
      </ul>
    </div>
  `;
  setHTML(el.recs, html);
}

function escapeHtml(str) {
  return String(str)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

// ===========================
// Progress & Timer
// ===========================
function resetProgressUI() {
  if (el.progress) el.progress.style.width = "0%";
  setText(el.timer, "00:00");
  hide(el.timer);
  setText(el.countdown, "—");
}
function startTimer() {
  show(el.timer);
  startTimestamp = Date.now();
  timerInterval = setInterval(() => {
    const elapsed = Math.floor((Date.now() - startTimestamp) / 1000);
    const pct = Math.min(100, Math.floor((elapsed / RECORD_SECONDS) * 100));
    if (el.progress) el.progress.style.width = `${pct}%`;
    const mm = String(Math.floor(elapsed / 60)).padStart(2, "0");
    const ss = String(elapsed % 60).padStart(2, "0");
    setText(el.timer, `${mm}:${ss}`);
    if (elapsed >= RECORD_SECONDS) stopRecordingAndSend();
  }, 250);
}
function stopTimer() {
  if (timerInterval) clearInterval(timerInterval);
  timerInterval = null;
}

// ===========================
// Recorder (with fallbacks)
// ===========================
async function startRecording() {
  if (isRecording) return;
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) {
    setStatus("Mic permission denied");
    appendLive(`getUserMedia error: ${e}`, true);
    return;
  }

  recordedChunks = [];
  mediaRecorder = tryCreateRecorder(mediaStream, [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/ogg;codecs=opus",
    "audio/mp4",
    "" // default
  ]);
  if (!mediaRecorder) {
    setStatus("Recorder unsupported");
    appendLive("MediaRecorder not supported with tested MIME types.", true);
    cleanupStream();
    return;
  }

  mediaRecorder.ondataavailable = (e) => {
    if (e.data && e.data.size > 0) recordedChunks.push(e.data);
  };
  mediaRecorder.onstop = handleRecorderStop;

  hasStoppedRecorder = false;
  mediaRecorder.start();  // continuous until stop()
  isRecording = true;

  setStatus("Recording");
  el.btnAbort.disabled = false;
  el.btnStopSend.disabled = false;
  startTimer();
  appendLive(`Recorder started (mime=${mediaRecorder.mimeType || "default"})`);
}
function tryCreateRecorder(stream, mimeList) {
  for (const mt of mimeList) {
    try {
      if (mt && !MediaRecorder.isTypeSupported(mt)) continue;
      return mt ? new MediaRecorder(stream, { mimeType: mt }) : new MediaRecorder(stream);
    } catch (_) { /* try next */ }
  }
  return null;
}
function cleanupStream() {
  try { mediaStream?.getTracks().forEach(t => t.stop()); } catch(_) {}
  mediaStream = null;
}
function handleRecorderStop() {
  stopTimer();
  cleanupStream();
  isRecording = false;
  el.btnAbort.disabled = true;
  el.btnStopSend.disabled = true;

  const mime = (mediaRecorder && mediaRecorder.mimeType) ? mediaRecorder.mimeType : "audio/webm";
  const blob = new Blob(recordedChunks, { type: mime });

  if (el.player) {
    el.player.src = URL.createObjectURL(blob);
    show(el.player);
  }

  setStatus("Sending");
  submitRecording(blob).catch((e) => {
    setStatus("Error");
    appendLive(`submitRecording error: ${e?.message || e}`, true);
  });
}
function stopRecordingAndSend() {
  if (!isRecording || !mediaRecorder) return;
  if (hasStoppedRecorder) return;
  hasStoppedRecorder = true;
  try { mediaRecorder.stop(); } catch (_) {}
}
function abortRecording() {
  if (!isRecording || !mediaRecorder) return;
  hasStoppedRecorder = true;
  try { mediaRecorder.stop(); } catch (_) {}
  recordedChunks = []; // do NOT send
  setStatus("Aborted");
  appendLive("Recording aborted by user.");
}

// ===========================
// Backend calls
// ===========================
async function startSession() {
  const fd = new FormData();
  fd.append("level", CURRENT_LEVEL);
  const r = await fetch(`${API_BASE}/start-session`, { method: "POST", body: fd });
  const data = await r.json();

  SESSION_ID = data.session_id;
  CURRENT_LEVEL = data.level;
  CURRENT_PROMPT = data.prompt || data.instructions || "Speak for ~60 seconds.";
  CURRENT_PROMPT_ID = (typeof data.prompt_id === "number") ? data.prompt_id : null;

  renderPrompt(CURRENT_PROMPT, CURRENT_LEVEL, "Attempt 1 of 2");
  resetProgressUI();
  setStatus("Ready");
  if (data.prompt_tts_url) safeAudioPlay(API_BASE + data.prompt_tts_url);

  // Auto countdown then record
  await countdownThenRecord();
}

async function fetchPromptForLevel(level) {
  const url = `${API_BASE}/prompts/${encodeURIComponent(level)}?session_id=${encodeURIComponent(SESSION_ID)}`;
  const r = await fetch(url, FETCH_OPTS);
  const data = await r.json();

  CURRENT_PROMPT = data.instructions || "Speak for ~60 seconds.";
  CURRENT_PROMPT_ID = (typeof data.prompt_id === "number") ? data.prompt_id : null;

  renderPrompt(CURRENT_PROMPT, level, "Attempt 2 of 2");
  resetProgressUI();
  if (data.prompt_tts_url) safeAudioPlay(API_BASE + data.prompt_tts_url);
}

async function submitRecording(blob) {
  const fd = new FormData();
  fd.append("session_id", SESSION_ID);
  const ext = (blob && blob.type) ? (blob.type.split("/")[1] || "webm") : "webm";
  fd.append("file", blob, `answer.${ext}`);

  if (CURRENT_PROMPT_ID !== null && CURRENT_PROMPT_ID !== undefined) {
    fd.append("prompt_id", String(CURRENT_PROMPT_ID));
  } else {
    fd.append("question", CURRENT_PROMPT);
  }

  const r = await fetch(`${API_BASE}/submit-response`, { method: "POST", body: fd });
  const result = await r.json();
  setHTML(el.result, JSON.stringify(result, null, 2));

  // Reflect backend's attempt back into the badge
  if (typeof result.attempt === "number" && el.attemptNote) {
    el.attemptNote.textContent = `Attempt ${result.attempt} of 2`;
  }

  if (!r.ok) {
    setStatus("Error");
    appendLive(`submit-response failed: ${result?.detail || "unknown error"}`, true);
    // still try to surface any recommendations the backend gave
    renderRecommendations(result);
    return;
  }

  renderScores(result);             // ← includes renderRecommendations(result)

  if (result.decision === "advance" && result.next_level) {
    CURRENT_LEVEL = result.next_level;

    if (result.next_prompt) {
      CURRENT_PROMPT = result.next_prompt;
      CURRENT_PROMPT_ID = (typeof result.next_prompt_id === "number") ? result.next_prompt_id : null;
      renderPrompt(CURRENT_PROMPT, CURRENT_LEVEL, "Attempt 1 of 2"); // reset for new level
      resetScoresUI();
      resetProgressUI();
      // Clear recommendations from last turn until next result
      renderRecommendations(null);
    } else {
      await fetchPromptForLevel(CURRENT_LEVEL);
    }
    setStatus(`Advanced to ${CURRENT_LEVEL}`);
    appendLive(`Advanced to ${CURRENT_LEVEL}`);
    await countdownThenRecord(); // smooth flow into next turn
  } else if (result.decision === "stop") {
    // Either first fail (retry available) or second fail (final stop)
    if (result.attempt >= 2) {
      setStatus("Stopped");
      appendLive("Evaluation stopped. Open /report for details.");
      const url = `${API_BASE}/report/${encodeURIComponent(SESSION_ID)}`;
      setHTML(el.avg, `<a href="${url}" target="_blank" rel="noopener">Open Final Report</a>`);
      // If backend returned session-level recommendations, show them
      renderRecommendations(result);
    } else {
      setStatus("Retry available — you'll get one more question at this level (Attempt 2 of 2).`);
      appendLive("No advance; click Start to retry with another question at this level.");
      // Show per-turn recs right away so they can improve on retry
      renderRecommendations(result);
    }
  }
}

// ===========================
// Countdown → Record
// ===========================
async function countdownThenRecord() {
  const secs = [3,2,1];
  for (const n of secs) {
    setText(el.countdown, String(n));
    setStatus("Get ready");
    await delay(700);
  }
  setText(el.countdown, "Go!");
  await delay(300);
  setText(el.countdown, "—");
  setStatus("Recording");
  el.btnAbort.disabled = false;
  el.btnStopSend.disabled = false;
  await startRecording();
}
function delay(ms) { return new Promise(res => setTimeout(res, ms)); }

// ===========================
// Button Handlers
// ===========================
async function handleStartClick() {
  try {
    if (!SESSION_ID) {             // first time → start session
      await startSession();
      return;
    }
    if (!isRecording) {            // retry → new prompt at same level
      await fetchPromptForLevel(CURRENT_LEVEL);
      await countdownThenRecord();
      return;
    }
  } catch (e) {
    setStatus("Error");
    appendLive(`Start error: ${e?.message || e}`, true);
  }
}
function handleAbortClick() { abortRecording(); }
function handleStopSendClick() { stopRecordingAndSend(); }

// ===========================
// Reset helpers
// ===========================
function resetScoresUI() { setHTML(el.chips, ""); setHTML(el.avg, ""); renderRecommendations(null); }

// ===========================
// Init
// ===========================
async function init() {
  el.btnStart?.addEventListener("click", handleStartClick);
  el.btnAbort?.addEventListener("click", handleAbortClick);
  el.btnStopSend?.addEventListener("click", handleStopSendClick);

  setText(el.prompt, "(none)");
  setText(el.countdown, "—");
  setText(el.timer, "00:00");
  hide(el.timer);
  resetProgressUI();
  setHTML(el.result, "(no response yet)");
  setHTML(el.live, "(no activity)");
  renderRecommendations(null);

  el.btnStart.disabled = false;
  el.btnAbort.disabled = true;
  el.btnStopSend.disabled = true;

  await loadConfig();
  setStatus("Idle");
  renderLevels(CURRENT_LEVEL);
}
document.addEventListener("DOMContentLoaded", init);
import os
import io
import json
import math
import uuid
import time
import shutil
import logging
import asyncio
import tempfile
import subprocess
from typing import Dict, List, Any, Optional
from collections import Counter

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# ---------- App setup ----------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("belt-service")

app = FastAPI(title="BELT Speaking Evaluator", version="2.0")

# Allow both same-origin and separate frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock down if you want specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve web UI (index.html and /static/*)
WEB_ROOT = os.path.join(os.getcwd(), "web")
if os.path.isdir(WEB_ROOT):
    app.mount("/static", StaticFiles(directory=os.path.join(WEB_ROOT, "static")), name="static")


# ---------- Config ----------

CEFR_ORDER = ["A1", "A2", "B1", "B1+", "B2", "B2+", "C1", "C2"]

ASR_BACKEND = os.getenv("ASR_BACKEND", "openai").strip().lower()
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-1").strip()
RUBRIC_MODEL = os.getenv("RUBRIC_MODEL", "gpt-4o-mini").strip()

PASS_AVG_THRESHOLD = float(os.getenv("PASS_AVG_THRESHOLD", "0.70"))
PASS_MIN_THRESHOLD = float(os.getenv("PASS_MIN_THRESHOLD", "0.60"))
RECORD_SECONDS = int(float(os.getenv("RECORD_SECONDS", "60")))

# ---------- In-memory session store ----------

SESSIONS: Dict[str, Dict[str, Any]] = {}
# session schema:
# {
#   "level": "A1",
#   "history": [ {level, question, prompt_id, transcript, scores, average, decision, attempt, recommendations} ],
#   "served_prompts": { "A1": set(ids), "A2": set(ids), ... },
#   "final_level": Optional[str]
# }

# ---------- Prompt loading ----------

def load_prompts_for_level(level: str) -> List[Dict[str, Any]]:
    """
    Load prompts from web/prompts/<level>.json if present; otherwise use built-in fallback.
    Each prompt: { "id": <int>, "instructions": <str> }
    """
    path = os.path.join(WEB_ROOT, "prompts", f"{level}.json")
    items: List[Dict[str, Any]] = []
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # allow either a list of strings or list of objects
            if isinstance(data, list):
                for i, it in enumerate(data):
                    if isinstance(it, dict):
                        pid = int(it.get("id", i))
                        txt = str(it.get("instructions") or it.get("question") or "").strip()
                    else:
                        pid = i
                        txt = str(it or "").strip()
                    if txt:
                        items.append({"id": pid, "instructions": txt})
        except Exception as e:
            log.warning(f"Failed to read prompts for {level}: {e}")

    if items:
        return items

    # Fallback minimal prompts per level (you likely have fuller sets in /web/prompts)
    fallback: Dict[str, List[str]] = {
        "A1": [
            "Introduce yourself. Say your name, where you’re from, and one hobby.",
            "Describe your typical breakfast.",
            "Talk about your family."
        ],
        "A2": [
            "Describe your daily routine on weekdays.",
            "Explain your last weekend plans.",
            "Talk about your favorite place in your town."
        ],
        "B1": [
            "Describe a memorable trip. Where did you go and why was it memorable?",
            "What is an important goal you have this year and how will you achieve it?",
            "Explain a skill you’re learning and why it matters."
        ],
        "B1+": [
            "Do you think it’s better to study alone or in a group? Explain.",
            "Describe a challenge you faced and how you overcame it.",
            "Is social media helpful or harmful? Give reasons."
        ],
        "B2": [
            "Do you think governments should ban junk-food advertising? Why or why not?",
            "How can cities reduce traffic congestion? Propose solutions.",
            "What are the pros and cons of remote work?"
        ],
        "B2+": [
            "Tell me about a technology you couldn’t live without and why.",
            "Should universities require community service for graduation? Defend your view.",
            "What responsibilities do large companies have regarding climate change?"
        ],
        "C1": [
            "Is it important to learn history? Defend your position.",
            "What are the ethical implications of data collection by tech firms?",
            "How should societies balance free speech and public safety?"
        ],
        "C2": [
            "What is the impact of social media on democracy?",
            "To what extent should governments regulate AI development?",
            "Are standardized tests a fair measure of ability? Argue your case."
        ],
    }
    return [{"id": i, "instructions": txt} for i, txt in enumerate(fallback.get(level, []))]

def choose_prompt(level: str, session_id: str) -> Dict[str, Any]:
    """
    Session-aware random-ish prompt selection without repeats at that level.
    """
    items = load_prompts_for_level(level)
    served = SESSIONS[session_id]["served_prompts"].setdefault(level, set())
    # pick first not served (deterministic); you can randomize if you prefer
    for item in items:
        if item["id"] not in served:
            served.add(item["id"])
            return item
    # if all served, reset and give the first one again
    SESSIONS[session_id]["served_prompts"][level] = set()
    first = items[0] if items else {"id": 0, "instructions": "Speak for ~60 seconds."}
    SESSIONS[session_id]["served_prompts"][level].add(first["id"])
    return first


# ---------- Audio handling ----------

def ensure_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception:
        raise HTTPException(500, "ffmpeg not found on server")

def to_wav_16k_mono(in_path: str, out_path: str):
    ensure_ffmpeg()
    cmd = [
        "ffmpeg", "-y",
        "-i", in_path,
        "-ac", "1",           # mono
        "-ar", "16000",       # 16 kHz
        "-f", "wav",
        out_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

async def transcribe(openai_client, wav_path: str) -> str:
    """
    Transcribe with OpenAI Whisper (if enabled). Fail-safe to empty string if disabled.
    """
    if ASR_BACKEND != "openai":
        return ""  # You can implement other backends as needed
    if openai_client is None:
        return ""
    try:
        with open(wav_path, "rb") as f:
            resp = openai_client.audio.transcriptions.create(
                model=WHISPER_MODEL,
                file=f,
                response_format="text",
                temperature=0
            )
        if isinstance(resp, str):
            return resp.strip()
        # v1 SDK returns .text for response_format="text"; keep both just in case
        text = getattr(resp, "text", "") or ""
        return text.strip()
    except Exception as e:
        log.warning(f"Transcription failed: {e}")
        return ""


# ---------- Scoring / Recommendations ----------

CEFR_CATEGORIES = ["fluency", "grammar", "vocabulary", "pronunciation", "coherence"]

RULEBOOK: Dict[str, List[str]] = {
    "fluency": [
        "Aim for fewer long pauses; speak in phrases of 4–7 words before pausing.",
        "Try shadowing a short native clip daily to build rhythm.",
        "Use fillers naturally (e.g., ‘well’, ‘actually’) instead of silence."
    ],
    "grammar": [
        "Watch subject–verb agreement (he/she/it + verbs ending in -s).",
        "Keep tenses consistent when telling a story (mostly past simple).",
        "Use linking words (because, although, however) to join clauses."
    ],
    "vocabulary": [
        "Add 3–5 topic words before speaking (e.g., travel: itinerary, budget, delay).",
        "Paraphrase when you don’t know a word (use a more general term).",
        "Learn 2–3 collocations per day (e.g., ‘make a decision’, ‘highly likely’)."
    ],
    "pronunciation": [
        "Slow down slightly and stress key words in each sentence.",
        "Open vowel contrasts clearly (/iː/ vs /ɪ/, /ɑː/ vs /æ/).",
        "Record yourself and compare to a native sample for 30 seconds daily."
    ],
    "coherence": [
        "Use a simple structure: intro → 2–3 points → short wrap-up.",
        "Signpost ideas (‘firstly’, ‘on the other hand’, ‘finally’).",
        "Keep each sentence to one main idea; avoid run-ons."
    ],
}

def weak_categories(scores: Dict[str, float], pass_avg: float, pass_min: float) -> List[str]:
    items = []
    for k in CEFR_CATEGORIES:
        v = float(scores.get(k, 0.0))
        tier = 2 if v < pass_min else (1 if v < pass_avg else 0)
        if tier > 0:
            items.append((k, tier, v))
    items.sort(key=lambda x: (-x[1], x[2]))
    return [k for (k, _, __) in items]

def rule_based_tips(scores: Dict[str, float], pass_avg: float, pass_min: float, max_tips: int = 3) -> List[str]:
    tips: List[str] = []
    for cat in weak_categories(scores, pass_avg, pass_min):
        for tip in RULEBOOK.get(cat, []):
            tips.append(f"{cat.capitalize()}: {tip}")
            if len(tips) >= max_tips:
                return tips
    return tips

def dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set(); out = []
    for it in items:
        key = it.strip().lower()
        if key not in seen:
            seen.add(key)
            out.append(it.strip())
    return out

def try_llm_scores_and_tips(openai_client, transcript: str, level: str) -> Optional[Dict[str, Any]]:
    """
    Ask RUBRIC_MODEL to produce JSON with scores 0..1 and 2–3 tailored tips.
    Fail-safe: return None if anything goes wrong. We clamp values later.
    """
    if not RUBRIC_MODEL or not transcript or openai_client is None:
        return None
    try:
        sys = (
            "You are a CEFR speaking evaluator. "
            "Return strict JSON with keys: "
            "{scores:{fluency,grammar,vocabulary,pronunciation,coherence} in 0..1}, "
            "\"tips\": array of 2-3 short actionable strings. No extra commentary."
        )
        user = (
            f"Assess the following transcript at level target {level}.\n"
            f"Transcript:\n{transcript}\n"
            "JSON only."
        )
        resp = openai_client.chat.completions.create(
            model=RUBRIC_MODEL,
            messages=[{"role": "system", "content": sys},
                      {"role": "user", "content": user}],
            temperature=0.2,
            max_tokens=300,
        )
        content = resp.choices[0].message.content.strip()
        data = json.loads(content)
        return data
    except Exception as e:
        log.warning(f"LLM rubric failed: {e}")
        return None

def clamp01(x: float) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0

def compute_scores(openai_client, transcript: str, level: str) -> Dict[str, Any]:
    """
    Prefer LLM JSON; fallback to naive heuristics if empty transcript.
    """
    result = {
        "scores": {k: 0.0 for k in CEFR_CATEGORIES},
        "average": 0.0,
        "tips_ai": [],
    }
    if not transcript.strip():
        # If nothing said, everything zero
        return result

    # Try LLM rubric for scores + tips
    data = try_llm_scores_and_tips(openai_client, transcript, level)
    if data and isinstance(data, dict):
        s = data.get("scores", {}) or {}
        scores = {k: clamp01(s.get(k, 0.0)) for k in CEFR_CATEGORIES}
        avg = sum(scores.values()) / len(CEFR_CATEGORIES)
        tips_ai = data.get("tips", []) if isinstance(data.get("tips", []), list) else []
        result["scores"] = scores
        result["average"] = avg
        result["tips_ai"] = [t for t in tips_ai if isinstance(t, str) and t.strip()]
        return result

    # Fallback crude heuristic if LLM failed: length-based
    length = len(transcript.split())
    rough = min(1.0, length / 120.0)  # 120+ words ~ 1.0
    scores = {k: rough for k in CEFR_CATEGORIES}
    avg = rough
    result["scores"] = scores
    result["average"] = avg
    result["tips_ai"] = []
    return result

def generate_recommendations(scores: Dict[str, float], transcript: str, level: str, history: List[Dict[str, Any]]) -> List[str]:
    rb = rule_based_tips(scores, PASS_AVG_THRESHOLD, PASS_MIN_THRESHOLD, max_tips=3)
    ai_extra = []  # added from compute_scores() call (per-turn)
    # We will pass AI tips separately from compute_scores

    merged = dedupe_keep_order(rb + ai_extra)
    if not merged:
        merged = [
            "Before speaking, jot 3 key points and 2 useful phrases for the topic.",
            "Speak in short sentences and link them with signposts (first, next, finally).",
        ]
    return merged[:5]

def finalize_session_recommendations(history: List[Dict[str, Any]]) -> List[str]:
    if not history:
        return []
    counts = Counter()
    for turn in history:
        s = turn.get("scores", {})
        for cat in CEFR_CATEGORIES:
            if float(s.get(cat, 0.0)) < 0.7:
                counts[cat] += 1
    if not counts:
        return ["Great overall balance. Keep practicing spontaneous speaking on varied topics."]
    top2 = [c for c, _ in counts.most_common(2)]
    tips = []
    for cat in top2:
        basics = RULEBOOK.get(cat, [])[:2]
        if basics:
            tips.append(f"Focus — {cat.capitalize()}: {basics[0]}")
        if len(basics) > 1:
            tips.append(f"Also for {cat}: {basics[1]}")
    tips.append("Routine: 10 minutes daily — plan 3 points, record once, listen back, and re-record focusing on the tips above.")
    return tips[:5]


# ---------- Helpers ----------

def next_level(level: str) -> Optional[str]:
    try:
        i = CEFR_ORDER.index(level)
        if i + 1 < len(CEFR_ORDER):
            return CEFR_ORDER[i + 1]
    except ValueError:
        pass
    return None

def decision_from_scores(scores: Dict[str, float], avg: float) -> str:
    if avg >= PASS_AVG_THRESHOLD and all(scores.get(k, 0.0) >= PASS_MIN_THRESHOLD for k in CEFR_CATEGORIES):
        return "advance"
    return "stop"


# ---------- Routes ----------

@app.get("/", response_class=HTMLResponse)
async def index():
    if not os.path.isdir(WEB_ROOT):
        return HTMLResponse("<h1>BELT Speaking Evaluator</h1><p>web/ folder not found.</p>")
    index_path = os.path.join(WEB_ROOT, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>BELT Speaking Evaluator</h1><p>index.html not found.</p>")

@app.get("/config")
async def get_config():
    return {
        "ASR_BACKEND": ASR_BACKEND,
        "WHISPER_MODEL": WHISPER_MODEL,
        "RUBRIC_MODEL": RUBRIC_MODEL,
        "PASS_AVG_THRESHOLD": PASS_AVG_THRESHOLD,
        "PASS_MIN_THRESHOLD": PASS_MIN_THRESHOLD,
        "RECORD_SECONDS": RECORD_SECONDS,
    }

@app.post("/start-session")
async def start_session(level: str = Form("A1")):
    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = {
        "level": level,
        "history": [],
        "served_prompts": {},
        "final_level": None,
    }
    item = choose_prompt(level, session_id)
    payload = {
        "session_id": session_id,
        "level": level,
        "prompt": item.get("instructions"),
        "prompt_id": item.get("id"),
        # optional TTS: if you pre-generate and host per prompt, add URL here
        "prompt_tts_url": None,
    }
    log.info(json.dumps({"ts": int(time.time()*1000), "msg": "start-session", "session_id": session_id, "level": level}))
    return JSONResponse(payload)

@app.get("/prompts/{level}")
async def get_prompt(level: str, session_id: str):
    if session_id not in SESSIONS:
        raise HTTPException(400, "Unknown session_id")
    state = SESSIONS[session_id]
    state["level"] = level  # keep session level in sync
    item = choose_prompt(level, session_id)
    return {
        "level": level,
        "instructions": item.get("instructions"),
        "prompt_id": item.get("id"),
        "prompt_tts_url": None,
    }

@app.post("/submit-response")
async def submit_response(
    session_id: str = Form(...),
    file: UploadFile = File(...),
    prompt_id: Optional[str] = Form(None),
    question: Optional[str] = Form(None),
):
    if session_id not in SESSIONS:
        raise HTTPException(400, "Unknown session_id")
    state = SESSIONS[session_id]
    level = state.get("level", "A1")

    # Determine attempt number based on history at this level
    prev_same_level = sum(1 for t in state["history"] if t.get("level") == level)
    attempt = prev_same_level + 1

    # Persist upload to temp file
    with tempfile.TemporaryDirectory() as td:
        src_path = os.path.join(td, file.filename or "answer.bin")
        with open(src_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        wav_path = os.path.join(td, "answer.wav")
        try:
            to_wav_16k_mono(src_path, wav_path)
        except Exception as e:
            log.warning(f"ffmpeg convert failed: {e}")
            raise HTTPException(400, "Audio format not supported or ffmpeg error.")

        # OpenAI clients are imported lazily to avoid dependency issues if keys unset
        openai_client = None
        if ASR_BACKEND == "openai" or RUBRIC_MODEL:
            try:
                from openai import OpenAI
                openai_client = OpenAI()
            except Exception as e:
                log.warning(f"OpenAI client init failed: {e}")
                openai_client = None

        transcript = await transcribe(openai_client, wav_path)

    # If the UI sent prompt_id, prefer it; else fall back to provided question or last served
    used_prompt_text = None
    used_prompt_id = None
    if prompt_id is not None:
        try:
            used_prompt_id = int(prompt_id)
        except Exception:
            used_prompt_id = None
        # resolve text from stored served prompts if needed
        for p in load_prompts_for_level(level):
            if p["id"] == used_prompt_id:
                used_prompt_text = p.get("instructions")
                break
    if not used_prompt_text:
        # If UI sent 'question', use it; else fall back to most recently served prompt for this level
        if question:
            used_prompt_text = question
        else:
            # last served at this level
            last_served_id = None
            served = state["served_prompts"].get(level) or set()
            if served:
                # grab a deterministic last (since we add in order)
                last_served_id = list(served)[-1]
            if last_served_id is not None:
                for p in load_prompts_for_level(level):
                    if p["id"] == last_served_id:
                        used_prompt_text = p.get("instructions"); used_prompt_id = last_served_id
                        break
        # still None → generic
        used_prompt_text = used_prompt_text or "Speak for ~60 seconds."

    # Compute scores (LLM or heuristic)
    score_pack = compute_scores(openai_client, transcript or "", level)
    scores = score_pack["scores"]
    avg = score_pack["average"]
    ai_tips = score_pack.get("tips_ai", [])

    # Decision
    decision = decision_from_scores(scores, avg)

    # Per-turn recommendations: rule-based + (plus any AI tips we just got)
    rb_recs = rule_based_tips(scores, PASS_AVG_THRESHOLD, PASS_MIN_THRESHOLD, max_tips=3)
    merged_recs = dedupe_keep_order(rb_recs + ai_tips)[:5]
    if not merged_recs:
        merged_recs = [
            "Before speaking, jot 3 key points and 2 useful phrases for the topic.",
            "Speak in short sentences and link them with signposts (first, next, finally).",
        ]

    # Turn log
    turn = {
        "level": level,
        "question": used_prompt_text,
        "prompt_id": used_prompt_id if used_prompt_id is not None else prompt_id,
        "transcript": transcript,
        "scores": scores,
        "average": avg,
        "decision": decision,
        "attempt": attempt,
        "recommendations": merged_recs,
    }
    state["history"].append(turn)
    if decision == "stop":
        state["final_level"] = level

    # Build response
    payload: Dict[str, Any] = {
        "scores": scores,
        "average": avg,
        "decision": decision,
        "attempt": attempt,
        "recommendations": merged_recs,
    }

    if decision == "advance":
        nxt = next_level(level)
        if nxt:
            state["level"] = nxt
            # For smoother UX, include the next prompt now if available
            item = choose_prompt(nxt, session_id)
            payload["next_level"] = nxt
            payload["next_prompt"] = item.get("instructions")
            payload["next_prompt_id"] = item.get("id")
        else:
            # Already at top level; end
            payload["next_level"] = None

    else:
        # stop case: if second fail, include session-level recommendations
        if attempt >= 2:
            payload["session_recommendations"] = finalize_session_recommendations(state["history"])

    log.info(json.dumps({
        "ts": int(time.time()*1000),
        "name": "belt-service",
        "msg": "submit-response",
        "session_id": session_id,
        "level": level,
        "attempt": attempt,
        "avg": avg,
        "decision": decision
    }))

    return JSONResponse(payload)

@app.get("/report/{session_id}")
async def report(session_id: str):
    if session_id not in SESSIONS:
        raise HTTPException(404, "Unknown session_id")
    state = SESSIONS[session_id]
    return {
        "session_id": session_id,
        "final_level": state.get("final_level"),
        "history": state.get("history", []),
        "recommendations": finalize_session_recommendations(state.get("history", [])),
    }
