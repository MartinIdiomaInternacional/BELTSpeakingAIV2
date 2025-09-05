
const API_BASE = ""; // same origin by default
const COUNTDOWN_SECONDS = 3;   // before each recording
let RECORD_SECONDS = 60;       // duration for each level; can be overridden by backend config
const CEFR_ORDER = ["A1","A2","B1","B1+","B2","B2+","C1","C2"];

// thresholds (display only) — will be overwritten by /config if available
let PASS_AVG = 0.70;
let PASS_MIN = 0.60;

let SESSION_ID = null;
let mediaRecorder = null;
let chunks = [];
let lastBlob = null;
let countdownInterval = null;
let timerInterval = null;
let startTime = null;
let aborted = false;
let currentLevel = "A1";
let highestPassedIndex = -1; // -1 means none passed yet
let failedOnceAtLevel = false;
let recordingActive = false;

// ---- UI helpers ----
function setStatus(text) {
  const el = document.getElementById("status");
  el.textContent = text;
  el.classList.remove("hidden");
  logLive(text);
}

function fmtTime(ms) {
  const s = Math.floor(ms/1000);
  const mm = String(Math.floor(s/60)).padStart(2,'0');
  const ss = String(s%60).padStart(2,'0');
  return mm + ":" + ss;
}

function startTimer() {
  const el = document.getElementById("timer");
  el.classList.remove("hidden");
  startTime = Date.now();
  timerInterval = setInterval(() => {
    el.textContent = fmtTime(Date.now() - startTime);
  }, 200);
}

function stopTimer() {
  const el = document.getElementById("timer");
  clearInterval(timerInterval);
  el.textContent = "00:00";
  el.classList.add("hidden");
}

function logLive(msg, obj) {
  const el = document.getElementById("live");
  if (obj) {
    el.textContent = msg + "\n" + JSON.stringify(obj, null, 2);
  } else {
    el.textContent = msg;
  }
}

function renderLevels() {
  const wrap = document.getElementById("levels");
  wrap.innerHTML = "";
  const idx = CEFR_ORDER.indexOf(currentLevel);
  CEFR_ORDER.forEach((lvl, i) => {
    const span = document.createElement("span");
    span.className = "lvl " + (i < highestPassedIndex+1 ? "passed" : (i === idx ? "current" : "locked"));
    span.textContent = lvl;
    wrap.appendChild(span);
  });
  const prog = document.getElementById("progress");
  const progressIndex = Math.max(0, highestPassedIndex + 1);
  const width = (progressIndex / CEFR_ORDER.length) * 100;
  prog.style.width = width + "%";
}

function renderChips(scores) {
  const chips = document.getElementById("chips");
  const avgEl = document.getElementById("avg");
  chips.innerHTML = "";
  if (!scores) { avgEl.textContent = ""; return; }
  const order = ["fluency","grammar","vocabulary","pronunciation","coherence"];
  let sum = 0, n = 0;
  order.forEach(k => {
    if (!(k in scores)) return;
    const v = Number(scores[k] || 0);
    sum += v; n += 1;
    let cls = "warn";
    if (v >= PASS_AVG) cls = "ok";
    else if (v < PASS_MIN) cls = "err";
    const chip = document.createElement("span");
    chip.className = "chip " + cls;
    chip.innerHTML = `${k}: <span class="val">${v.toFixed(2)}</span>`;
    chips.appendChild(chip);
  });
  if (n > 0) {
    const avg = sum / n;
    const cls = avg >= PASS_AVG ? "ok" : (avg < PASS_MIN ? "err" : "warn");
    avgEl.innerHTML = `Average: <b class="${cls}">${avg.toFixed(2)}</b> (live thresholds: avg≥${PASS_AVG}, min≥${PASS_MIN})`;
  } else {
    avgEl.textContent = "";
  }
}

// ---- Config ----
async function fetchConfig() {
  try {
    const resp = await fetch(`${API_BASE}/config`);
    if (!resp.ok) return;
    const cfg = await resp.json();
    if (cfg.PASS_AVG_THRESHOLD) PASS_AVG = Number(cfg.PASS_AVG_THRESHOLD);
    if (cfg.PASS_MIN_THRESHOLD) PASS_MIN = Number(cfg.PASS_MIN_THRESHOLD);
    if (cfg.RECORD_SECONDS) RECORD_SECONDS = Number(cfg.RECORD_SECONDS);
    setStatus(`Loaded config: PASS_AVG=${PASS_AVG}, PASS_MIN=${PASS_MIN}, RECORD_SECONDS=${RECORD_SECONDS}`);
  } catch {}
}

// ---- Mic / recording ----
async function ensureMic() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
    return stream;
  } catch (e) {
    alert("Microphone access denied or unavailable.");
    throw e;
  }
}

async function startSessionA1() {
  const fd = new FormData();
  fd.append("level", "A1");
  const resp = await fetch(`${API_BASE}/start-session`, { method: "POST", body: fd });
  const data = await resp.json();
  SESSION_ID = data.session_id;
  currentLevel = "A1";
  failedOnceAtLevel = false;
  document.getElementById("prompt").textContent = `${data.level}: ${data.prompt}`;
  document.getElementById("attemptNote").textContent = "Attempt 1 of up to 2 at level " + currentLevel;
  setStatus("Session started at A1");
  renderLevels();
  renderChips(null);
  return data;
}

async function loadAlternatePrompt(level) {
  try {
    const resp = await fetch(`${API_BASE}/prompts/${encodeURIComponent(level)}`);
    const data = await resp.json();
    const alt = `${level}: (New question) ${data.instructions}`;
    document.getElementById("prompt").textContent = alt;
  } catch (e) {
    document.getElementById("prompt").textContent = `${level}: (New question) Please talk about a different example on the same topic.`;
  }
}

async function startRecordingFixed(seconds) {
  const stream = await ensureMic();
  chunks = [];
  const player = document.getElementById("player");
  mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
  mediaRecorder.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };
  recordingActive = true;
  document.getElementById("btnStopSend").disabled = false;
  return new Promise((resolve) => {
    mediaRecorder.onstop = () => {
      lastBlob = new Blob(chunks, { type: 'audio/webm' });
      player.src = URL.createObjectURL(lastBlob);
      player.classList.remove("hidden");
      stream.getTracks().forEach(t => t.stop());
      recordingActive = false;
      document.getElementById("btnStopSend").disabled = true;
      resolve(lastBlob);
    };
    mediaRecorder.start();
    startTimer();
    setTimeout(() => {
      if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
      }
      stopTimer();
    }, seconds * 1000);
  });
}

function stopAndSendEarly() {
  // Early stop; resolve the current recording promise by stopping the recorder now
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }
  stopTimer();
  setStatus("Stopped early — sending");
}

// ---- Submit / report ----
async function submitBlob(blob) {
  const fd = new FormData();
  fd.append("session_id", SESSION_ID);
  fd.append("file", blob, "recording.webm");
  const resp = await fetch(`${API_BASE}/submit-response`, { method: "POST", body: fd });
  const data = await resp.json();
  document.getElementById("result").textContent = JSON.stringify(data, null, 2);
  renderChips(data.scores);
  return data;
}

async function fetchReport() {
  const resp = await fetch(`${API_BASE}/report/${SESSION_ID}`);
  const rj = await resp.json();
  document.getElementById("result").textContent = JSON.stringify(rj, null, 2);
  return rj;
}

async function countdown(seconds) {
  const el = document.getElementById("countdown");
  return new Promise((resolve) => {
    let n = seconds;
    el.textContent = String(n);
    countdownInterval = setInterval(() => {
      n -= 1;
      if (n <= 0) {
        clearInterval(countdownInterval);
        el.textContent = "Speak!";
        resolve();
      } else {
        el.textContent = String(n);
      }
    }, 1000);
  });
}

// ---- Main loop ----
async function runAdaptive() {
  aborted = false;
  highestPassedIndex = -1;
  document.getElementById("btnStart").disabled = true;
  document.getElementById("btnAbort").disabled = false;
  renderLevels();

  await fetchConfig(); // pull live thresholds and recording seconds
  await startSessionA1(); // start at A1

  while (!aborted) {
    document.getElementById("attemptNote").textContent = `Attempt ${failedOnceAtLevel ? 2 : 1} of up to 2 at level ${currentLevel}`;
    await countdown(COUNTDOWN_SECONDS);
    if (aborted) break;

    setStatus("Recording…");
    const blob = await startRecordingFixed(RECORD_SECONDS);
    if (aborted) break;
    setStatus("Submitting…");

    const data = await submitBlob(blob);
    if (data.decision === "advance") {
      const justIdx = CEFR_ORDER.indexOf(currentLevel);
      if (justIdx > highestPassedIndex) highestPassedIndex = justIdx;
      currentLevel = data.next_level;
      document.getElementById("prompt").textContent = `${data.next_level}: ${data.next_prompt}`;
      failedOnceAtLevel = false;
      setStatus("Advanced to " + data.next_level);
      renderLevels();
      continue;
    } else {
      if (!failedOnceAtLevel) {
        failedOnceAtLevel = true;
        setStatus("Below threshold — giving one more question at the same level.");
        await loadAlternatePrompt(currentLevel);
        continue;
      } else {
        setStatus("Session finished");
        await fetchReport();
        break;
      }
    }
  }

  document.getElementById("btnStart").disabled = false;
  document.getElementById("btnAbort").disabled = true;
  document.getElementById("countdown").textContent = "—";
}

// ---- Abort & events ----
function abortAdaptive() {
  aborted = true;
  try {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      mediaRecorder.stop();
    }
  } catch {}
  stopTimer();
  clearInterval(countdownInterval);
  setStatus("Aborted");
  document.getElementById("btnStart").disabled = false;
  document.getElementById("btnAbort").disabled = true;
}

document.getElementById("btnStart").addEventListener("click", runAdaptive);
document.getElementById("btnAbort").addEventListener("click", abortAdaptive);
document.getElementById("btnStopSend").addEventListener("click", stopAndSendEarly);
renderLevels();
