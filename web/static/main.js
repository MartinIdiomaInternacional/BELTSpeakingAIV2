/* ===========================
   BELT Voice Evaluator - Main JS (for your HTML)
   ===========================
   - Matches IDs in index.html (btnStart, btnAbort, countdown, timer, progress, etc.)
   - Session-aware prompts (no repeats), sends prompt_id/question
   - 3-second countdown → record up to RECORD_SECONDS (default 60)
   - Stop & Send; Abort cancels without sending
   - Score chips + average; level tracker; live logs
   - TTS playback when backend provides prompt_tts_url
*/

// ===========================
// CONFIG
// ===========================
const API_BASE = "";                   // Same-origin backend. If hosted elsewhere, set full origin.
const CONFIG_ENDPOINT = `${API_BASE}/config`;
const FETCH_OPTS = { cache: "no-store" }; // avoid stale config/prompts

// CEFR levels in order (with probes)
const CEFR_LEVELS = ["A1","A2","B1","B1+","B2","B2+","C1","C2"];

// ===========================
// DOM (matches your HTML)
// ===========================
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

let RECORD_SECONDS = 60;      // will be loaded from /config (min 60)
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
// Utilities
// ===========================
function show(elm) { if (elm) elm.classList.remove("hidden"); }
function hide(elm) { if (elm) elm.classList.add("hidden"); }
function setText(elm, txt) { if (elm) elm.textContent = txt; }
function setHTML(elm, html) { if (elm) elm.innerHTML = html; }
function logLive(msg) {
  if (!el.live) return;
  const ts = new Date().toLocaleTimeString();
  const line = `[${ts}] ${msg}\n`;
  el.live.textContent = line + el.live.textContent;
}
function setStatus(txt) {
  if (!el.status) return;
  setText(el.status, txt);
  show(el.status);
}
function safeAudioPlay(url) { try { new Audio(url).play().catch(()=>{}); } catch(_){} }

// ===========================
// Config & Level UI
// ===========================
async function loadConfig() {
  try {
    const r = await fetch(CONFIG_ENDPOINT, FETCH_OPTS);
    const cfg = await r.json();
    const rec = parseInt(cfg.RECORD_SECONDS ?? RECORD_SECONDS, 10);
    RECORD_SECONDS = Number.isFinite(rec) ? Math.max(60, rec) : 60;
    PASS_AVG_THRESHOLD = parseFloat(cfg.PASS_AVG_THRESHOLD ?? PASS_AVG_THRESHOLD);
    PASS_MIN_THRESHOLD = parseFloat(cfg.PASS_MIN_THRESHOLD ?? PASS_MIN_THRESHOLD);
    logLive(`Config loaded: RECORD_SECONDS=${RECORD_SECONDS}, PASS_AVG=${PASS_AVG_THRESHOLD}, MIN=${PASS_MIN_THRESHOLD}`);
  } catch (e) {
    RECORD_SECONDS = 60;
    logLive(`Config load failed; using defaults. ${e}`);
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
// Prompt rendering
// ===========================
function renderPrompt(text, level, attemptLabel = "") {
  if (el.prompt) setText(el.prompt, text || "(none)");
  if (el.attemptNote) setText(el.attemptNote, attemptLabel);
  renderLevels(level || CURRENT_LEVEL);
}

// ===========================
// Scores rendering
// ===========================
function renderScores(result) {
  if (!result || !result.scores) {
    setHTML(el.chips, "");
    setHTML(el.avg, "");
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
}

// ===========================
// Progress & Timers
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
// Recorder (with MIME fallbacks)
// ===========================
async function startRecording() {
  if (isRecording) return;
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) {
    setStatus("Mic permission denied");
    logLive(`getUserMedia error: ${e}`);
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
    logLive("MediaRecorder not supported with the tested MIME types.");
    cleanupStream();
    return;
  }

  mediaRecorder.ondataavailable = (e) => {
    if (e.data && e.data.size > 0) recordedChunks.push(e.data);
  };
  mediaRecorder.onstop = handleRecorderStop;

  hasStoppedRecorder = false;
  mediaRecorder.start(); // continuous until stop()
  isRecording = true;

  setStatus("Recording");
  el.btnAbort.disabled = false;
  el.btnStopSend.disabled = false;
  startTimer();
  logLive(`Recorder started (mime=${mediaRecorder.mimeType || "default"})`);
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

  // preview
  if (el.player) {
    el.player.src = URL.createObjectURL(blob);
    show(el.player);
  }

  setStatus("Sending");
  submitRecording(blob).catch((e) => {
    setStatus("Error");
    logLive(`submitRecording error: ${e?.message || e}`);
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
  // but do NOT send; just clear chunks
  recordedChunks = [];
  setStatus("Aborted");
  logLive("Recording aborted by user.");
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

  renderPrompt(CURRENT_PROMPT, CURRENT_LEVEL, "(attempt 1)");
  resetProgressUI();
  setStatus("Ready");
  if (data.prompt_tts_url) safeAudioPlay(API_BASE + data.prompt_tts_url);

  // Begin countdown then record
  await countdownThenRecord();
}

async function fetchPromptForLevel(level) {
  const url = `${API_BASE}/prompts/${encodeURIComponent(level)}?session_id=${encodeURIComponent(SESSION_ID)}`;
  const r = await fetch(url, FETCH_OPTS);
  const data = await r.json();

  CURRENT_PROMPT = data.instructions || "Speak for ~60 seconds.";
  CURRENT_PROMPT_ID = (typeof data.prompt_id === "number") ? data.prompt_id : null;

  renderPrompt(CURRENT_PROMPT, level, "(attempt 2)");
  resetProgressUI();
  if (data.prompt_tts_url) safeAudioPlay(API_BASE + data.prompt_tts_url);
}

async function submitRecording(blob) {
  const fd = new FormData();
  fd.append("session_id", SESSION_ID);
  // Use blob's type to give a hint; backend converts anyway
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

  if (!r.ok) {
    setStatus("Error");
    logLive(`submit-response failed: ${result?.detail || "unknown error"}`);
    return;
  }

  renderScores(result);

  if (result.decision === "advance" && result.next_level) {
    CURRENT_LEVEL = result.next_level;
    if (result.next_prompt) {
      CURRENT_PROMPT = result.next_prompt;
      CURRENT_PROMPT_ID = (typeof result.next_prompt_id === "number") ? result.next_prompt_id : null;
      renderPrompt(CURRENT_PROMPT, CURRENT_LEVEL, "(attempt 1)");
      resetProgressUI();
    } else {
      await fetchPromptForLevel(CURRENT_LEVEL);
    }
    setStatus(`Advanced to ${CURRENT_LEVEL}`);
    logLive(`Advanced to ${CURRENT_LEVEL}`);
    // Auto-start countdown & recording for the next level
    await countdownThenRecord();
  } else if (result.decision === "stop") {
    setStatus("Stopped");
    logLive("Evaluation stopped. Open /report for details.");
    // Show link to final report
    const url = `${API_BASE}/report/${encodeURIComponent(SESSION_ID)}`;
    setHTML(el.avg, `<a href="${url}" target="_blank" rel="noopener">Open Final Report</a>`);
  } else {
    // Not advanced: user can click Start again to retry same level (we’ll fetch a new prompt)
    setStatus("Retry available");
    logLive("No advance; click Start to retry with another question at this level.");
  }
}

// ===========================
// Countdown → Record flow
// ===========================
async function countdownThenRecord() {
  // 3…2…1…
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
    // If no session yet → start one and immediately record
    if (!SESSION_ID) {
      await startSession();
      return;
    }
    // If already in session and NOT currently recording → fetch new prompt at the same level (retry) and record
    if (!isRecording) {
      await fetchPromptForLevel(CURRENT_LEVEL);
      await countdownThenRecord();
      return;
    }
    // If already recording, ignore click
  } catch (e) {
    setStatus("Error");
    logLive(`Start error: ${e?.message || e}`);
  }
}

function handleAbortClick() { abortRecording(); }
function handleStopSendClick() { stopRecordingAndSend(); }

// ===========================
// Init
// ===========================
async function init() {
  // Wire up buttons
  el.btnStart?.addEventListener("click", handleStartClick);
  el.btnAbort?.addEventListener("click", handleAbortClick);
  el.btnStopSend?.addEventListener("click", handleStopSendClick);

  // Initial UI state
  setText(el.prompt, "(none)");
  setText(el.countdown, "—");
  setText(el.timer, "00:00");
  hide(el.timer);
  resetProgressUI();
  setHTML(el.result, "(no response yet)");
  setHTML(el.live, "(no activity)");

  // Enable Start; others disabled
  el.btnStart.disabled = false;
  el.btnAbort.disabled = true;
  el.btnStopSend.disabled = true;

  await loadConfig();
  setStatus("Idle");
  renderLevels(CURRENT_LEVEL);
}

document.addEventListener("DOMContentLoaded", init);
