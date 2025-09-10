/* ===========================
   BELT Voice Evaluator - Main JS
   ===========================
   - Session-aware guided prompts (no repeats)
   - Sends prompt_id/question with submissions
   - Records WebM/Opus via MediaRecorder
   - Progress bar, timer, Stop & Send
   - Renders per-category chips + average
   - TTS playback of prompts (if available)
*/

// ===========================
// CONFIG
// ===========================
const API_BASE = ""; // same-origin backend. If different domain, e.g. "https://beltspeakingaiv2.onrender.com"
const CONFIG_ENDPOINT = `${API_BASE}/config`;

// SELECTORS (change here if your HTML uses different IDs)
const SEL = {
  btnStart: "#btnStart",
  btnRetry: "#btnRetry",
  btnStopSend: "#btnStopSend",
  btnStartSession: "#btnStartSession",
  promptText: "#promptText",
  levelBadge: "#levelBadge",
  timerText: "#timerText",
  progressBar: "#progressBar",
  scoresContainer: "#scoresContainer",
  messages: "#messages",
  finalReport: "#finalReport",
  audioPreview: "#audioPreview", // optional
};

// UI STRINGS
const STR = {
  defaultPrompt: "Speak for ~60 seconds.",
  starting: "Starting recorder…",
  recording: "Recording…",
  processing: "Scoring your answer…",
  stopped: "Stopped. Sending audio…",
  ready: "Ready.",
  error: "Something went wrong.",
};

// COLORS for score chips (light hint only; keep CSS in your stylesheet ideally)
const SCORE_COLORS = {
  good: "#16a34a",
  mid: "#f59e0b",
  low: "#dc2626",
};

// ===========================
// App State
// ===========================
let SESSION_ID = null;
let CURRENT_LEVEL = "A1";
let CURRENT_PROMPT = "";
let CURRENT_PROMPT_ID = null;

let RECORD_SECONDS = 60;
let PASS_AVG_THRESHOLD = 0.7;
let PASS_MIN_THRESHOLD = 0.6;

// Recorder state
let mediaStream = null;
let mediaRecorder = null;
let recordedChunks = [];
let isRecording = false;
let timerInterval = null;
let startTimestamp = null;

// ===========================
// DOM Helpers
// ===========================
function $(sel) {
  return document.querySelector(sel);
}
function setText(sel, text) {
  const el = $(sel);
  if (el) el.textContent = text;
}
function setHTML(sel, html) {
  const el = $(sel);
  if (el) el.innerHTML = html;
}
function setDisabled(sel, disabled) {
  const el = $(sel);
  if (el) el.disabled = disabled;
}
function appendMessage(msg, type = "info") {
  const el = $(SEL.messages);
  if (!el) return;
  const div = document.createElement("div");
  div.className = `msg ${type}`;
  div.textContent = msg;
  el.prepend(div);
}
function safeAudioPlay(url) {
  try {
    const a = new Audio(url);
    a.play().catch(() => {});
  } catch (e) {}
}

// ===========================
// Rendering
// ===========================
function renderPrompt(prompt, level) {
  setText(SEL.levelBadge, level || "");
  setText(SEL.promptText, prompt || STR.defaultPrompt);
}

function renderScores(result) {
  const box = $(SEL.scoresContainer);
  if (!box) return;
  if (!result || !result.scores) {
    setHTML(SEL.scoresContainer, "");
    return;
  }
  const s = result.scores;
  const avg = result.average ?? 0;

  const chip = (label, val) => {
    const v = Math.round((val ?? 0) * 100);
    let color = SCORE_COLORS.low;
    if (v >= PASS_MIN_THRESHOLD * 100 && v < PASS_AVG_THRESHOLD * 100) color = SCORE_COLORS.mid;
    if (v >= PASS_AVG_THRESHOLD * 100) color = SCORE_COLORS.good;
    return `
      <div class="score-chip" style="border:1px solid #e5e7eb;border-radius:999px;padding:6px 10px;display:inline-flex;align-items:center;gap:8px;margin:4px;">
        <span style="font-weight:600">${label}</span>
        <span style="background:${color};color:#fff;border-radius:999px;padding:2px 8px;">${v}</span>
      </div>
    `;
  };

  const html = `
    <div style="margin-top:8px;margin-bottom:8px">
      ${chip("Fluency", s.fluency)}
      ${chip("Grammar", s.grammar)}
      ${chip("Vocabulary", s.vocabulary)}
      ${chip("Pronunciation", s.pronunciation)}
      ${chip("Coherence", s.coherence)}
    </div>
    <div style="margin-top:6px">
      <strong>Average:</strong> ${(avg * 100).toFixed(0)}
      ${result.decision ? `<span style="margin-left:8px;padding:2px 8px;border-radius:6px;background:${result.decision==='advance' ? '#d1fae5' : '#fee2e2'};color:${result.decision==='advance' ? '#064e3b' : '#7f1d1d'}">${result.decision}</span>` : ""}
      ${result.borderline ? `<span style="margin-left:8px;padding:2px 8px;border-radius:6px;background:#fef9c3;color:#854d0e">probe</span>` : ""}
    </div>
  `;
  setHTML(SEL.scoresContainer, html);
}

function renderFinalReportLink(sessionId) {
  const el = $(SEL.finalReport);
  if (!el) return;
  const url = `${API_BASE}/report/${encodeURIComponent(sessionId)}`;
  el.innerHTML = `
    <a href="${url}" target="_blank" rel="noopener">Open Final Report</a>
  `;
}

function resetScoresUI() {
  setHTML(SEL.scoresContainer, "");
}

// ===========================
// Progress Bar & Timer
// ===========================
function resetProgressUI() {
  const bar = $(SEL.progressBar);
  const t = $(SEL.timerText);
  if (bar) bar.style.width = "0%";
  if (t) t.textContent = `0/${RECORD_SECONDS}s`;
}

function startTimer() {
  resetProgressUI();
  startTimestamp = Date.now();
  timerInterval = setInterval(() => {
    const elapsedSec = Math.floor((Date.now() - startTimestamp) / 1000);
    const pct = Math.min(100, Math.floor((elapsedSec / RECORD_SECONDS) * 100));
    const bar = $(SEL.progressBar);
    const t = $(SEL.timerText);
    if (bar) bar.style.width = `${pct}%`;
    if (t) t.textContent = `${Math.min(elapsedSec, RECORD_SECONDS)}/${RECORD_SECONDS}s`;
    if (elapsedSec >= RECORD_SECONDS) {
      stopRecordingAndSend(); // auto-stop at limit
    }
  }, 250);
}

function stopTimer() {
  clearInterval(timerInterval);
  timerInterval = null;
}

// ===========================
// Recorder
// ===========================
async function startRecording() {
  if (isRecording) return;
  appendMessage(STR.starting);
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) {
    appendMessage("Mic permission denied.", "error");
    return;
  }

  recordedChunks = [];
  mediaRecorder = new MediaRecorder(mediaStream, { mimeType: "audio/webm;codecs=opus" });
  mediaRecorder.ondataavailable = (e) => {
    if (e.data && e.data.size > 0) recordedChunks.push(e.data);
  };
  mediaRecorder.onstop = handleRecorderStop;

  mediaRecorder.start();
  isRecording = true;
  setDisabled(SEL.btnStart, true);
  setDisabled(SEL.btnStopSend, false);
  appendMessage(STR.recording);
  startTimer();
}

function handleRecorderStop() {
  stopTimer();
  if (mediaStream) {
    mediaStream.getTracks().forEach((t) => t.stop());
  }
  isRecording = false;
  setDisabled(SEL.btnStart, false);
  setDisabled(SEL.btnStopSend, true);

  const blob = new Blob(recordedChunks, { type: "audio/webm" });

  // optional preview
  const audioEl = $(SEL.audioPreview);
  if (audioEl) {
    audioEl.src = URL.createObjectURL(blob);
  }

  // Send automatically after stop
  submitRecording(blob).catch((e) => {
    appendMessage(`${STR.error} ${e?.message || e}`, "error");
  });
}

function stopRecordingAndSend() {
  if (!isRecording || !mediaRecorder) return;
  appendMessage(STR.stopped);
  mediaRecorder.stop();
}

// ===========================
// Backend Integration
// ===========================
async function loadConfig() {
  try {
    const resp = await fetch(CONFIG_ENDPOINT);
    const cfg = await resp.json();
    RECORD_SECONDS = cfg.RECORD_SECONDS ?? RECORD_SECONDS;
    PASS_AVG_THRESHOLD = parseFloat(cfg.PASS_AVG_THRESHOLD ?? PASS_AVG_THRESHOLD);
    PASS_MIN_THRESHOLD = parseFloat(cfg.PASS_MIN_THRESHOLD ?? PASS_MIN_THRESHOLD);
  } catch (e) {
    // use defaults
  }
}

async function startSession() {
  try {
    const fd = new FormData();
    fd.append("level", CURRENT_LEVEL);
    const resp = await fetch(`${API_BASE}/start-session`, { method: "POST", body: fd });
    const data = await resp.json();

    SESSION_ID = data.session_id;
    CURRENT_LEVEL = data.level;
    CURRENT_PROMPT = data.prompt || data.instructions || STR.defaultPrompt;
    CURRENT_PROMPT_ID = typeof data.prompt_id === "number" ? data.prompt_id : null;

    renderPrompt(CURRENT_PROMPT, CURRENT_LEVEL);
    resetScoresUI();
    resetProgressUI();
    appendMessage("Session started.");

    if (data.prompt_tts_url) {
      safeAudioPlay(API_BASE + data.prompt_tts_url);
    }
    setDisabled(SEL.btnRetry, false);
    setDisabled(SEL.btnStart, false);
    setDisabled(SEL.btnStopSend, true);
  } catch (e) {
    appendMessage(`${STR.error} ${e?.message || e}`, "error");
  }
}

// Session-aware fetch (avoids repeats)
async function fetchPromptForLevel(level) {
  const url = `${API_BASE}/prompts/${encodeURIComponent(level)}?session_id=${encodeURIComponent(SESSION_ID)}`;
  const resp = await fetch(url);
  const data = await resp.json();

  CURRENT_PROMPT = data.instructions || STR.defaultPrompt;
  CURRENT_PROMPT_ID = typeof data.prompt_id === "number" ? data.prompt_id : null;

  renderPrompt(CURRENT_PROMPT, level);
  resetScoresUI();
  resetProgressUI();

  if (data.prompt_tts_url) {
    safeAudioPlay(API_BASE + data.prompt_tts_url);
  }
  return CURRENT_PROMPT;
}

// Always send prompt_id or question for perfect logging
async function submitRecording(blob) {
  try {
    appendMessage(STR.processing);
    const fd = new FormData();
    fd.append("session_id", SESSION_ID);
    fd.append("file", blob, "answer.webm");

    if (CURRENT_PROMPT_ID !== null && CURRENT_PROMPT_ID !== undefined) {
      fd.append("prompt_id", String(CURRENT_PROMPT_ID));
    } else {
      fd.append("question", CURRENT_PROMPT);
    }

    const resp = await fetch(`${API_BASE}/submit-response`, { method: "POST", body: fd });
    const result = await resp.json();

    if (!resp.ok) {
      appendMessage(result?.detail || "Submit failed.", "error");
      return;
    }

    renderScores(result);

    if (result.decision === "advance" && result.next_level) {
      // Move to next level
      CURRENT_LEVEL = result.next_level;

      if (result.next_prompt) {
        CURRENT_PROMPT = result.next_prompt;
        CURRENT_PROMPT_ID = typeof result.next_prompt_id === "number" ? result.next_prompt_id : null;
        renderPrompt(CURRENT_PROMPT, CURRENT_LEVEL);
        resetScoresUI();
        resetProgressUI();
      } else {
        await fetchPromptForLevel(CURRENT_LEVEL);
      }
      appendMessage(`Advanced to ${CURRENT_LEVEL}.`);
    } else if (result.decision === "stop") {
      appendMessage("Evaluation stopped at current level.");
      renderFinalReportLink(SESSION_ID);
    }
  } catch (e) {
    appendMessage(`${STR.error} ${e?.message || e}`, "error");
  }
}

// Second chance at same level
async function retrySameLevel() {
  if (!SESSION_ID) return;
  await fetchPromptForLevel(CURRENT_LEVEL);
  appendMessage(`New question for ${CURRENT_LEVEL}.`);
}

// ===========================
// Wire up UI
// ===========================
function bindUI() {
  const bStart = $(SEL.btnStart);
  const bRetry = $(SEL.btnRetry);
  const bStop = $(SEL.btnStopSend);
  const bStartSession = $(SEL.btnStartSession);

  if (bStartSession) bStartSession.addEventListener("click", startSession);
  if (bStart) bStart.addEventListener("click", startRecording);
  if (bStop) bStop.addEventListener("click", stopRecordingAndSend);
  if (bRetry) bRetry.addEventListener("click", retrySameLevel);
}

async function init() {
  bindUI();
  await loadConfig();

  // Initial UI state
  renderPrompt(STR.defaultPrompt, CURRENT_LEVEL);
  setDisabled(SEL.btnRetry, true);
  setDisabled(SEL.btnStart, true);
  setDisabled(SEL.btnStopSend, true);
  appendMessage(STR.ready);
}

document.addEventListener("DOMContentLoaded", init);
