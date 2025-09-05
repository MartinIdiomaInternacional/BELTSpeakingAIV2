
const API_BASE = ""; // same origin by default
let SESSION_ID = null;
let mediaRecorder = null;
let chunks = [];
let lastBlob = null;
let timerInterval = null;
let startTime = null;

function setStatus(text) {
  const el = document.getElementById("status");
  el.textContent = text;
  el.classList.remove("hidden");
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

async function ensureMic() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
    return stream;
  } catch (e) {
    alert("Microphone access denied or unavailable.");
    throw e;
  }
}

async function startRecording() {
  const recBtn = document.getElementById("btnRecord");
  const stopBtn = document.getElementById("btnStop");
  const evalBtn = document.getElementById("btnEvalSingle");
  const submitBtn = document.getElementById("btnSubmit");
  const player = document.getElementById("player");
  const resEl = document.getElementById("result");

  const stream = await ensureMic();
  chunks = [];
  mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
  mediaRecorder.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };
  mediaRecorder.onstop = () => {
    lastBlob = new Blob(chunks, { type: 'audio/webm' });
    player.src = URL.createObjectURL(lastBlob);
    player.classList.remove("hidden");
    evalBtn.disabled = false;
    submitBtn.disabled = false;
    setStatus("Recorded " + Math.round(lastBlob.size/1024) + " KB");
  };

  mediaRecorder.start();
  recBtn.disabled = true;
  stopBtn.disabled = false;
  evalBtn.disabled = true;
  submitBtn.disabled = true;
  resEl.textContent = "(recording...)";
  startTimer();
  setStatus("Recording…");
}

function stopRecording() {
  const recBtn = document.getElementById("btnRecord");
  const stopBtn = document.getElementById("btnStop");
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach(t => t.stop());
  }
  recBtn.disabled = false;
  stopBtn.disabled = true;
  stopTimer();
  setStatus("Stopped");
}

async function startSession() {
  const level = document.getElementById("level").value;
  const promptEl = document.getElementById("prompt");
  const resEl = document.getElementById("result");
  const fd = new FormData();
  fd.append("level", level);
  const resp = await fetch(`${API_BASE}/start-session`, { method: "POST", body: fd });
  const data = await resp.json();
  SESSION_ID = data.session_id;
  promptEl.textContent = `${data.level}: ${data.prompt}`;
  resEl.textContent = "(session started)";
  setStatus("Session started");
}

async function loadPromptOnly() {
  const level = document.getElementById("level").value;
  const promptEl = document.getElementById("prompt");
  const resp = await fetch(`${API_BASE}/prompts/${encodeURIComponent(level)}`);
  const data = await resp.json();
  promptEl.textContent = JSON.stringify(data, null, 2);
}

async function submitResponse() {
  if (!SESSION_ID) { alert("Start a session first."); return; }
  if (!lastBlob) { alert("Record something first."); return; }
  const resEl = document.getElementById("result");
  const fd = new FormData();
  fd.append("session_id", SESSION_ID);
  fd.append("file", lastBlob, "recording.webm");
  const resp = await fetch(`${API_BASE}/submit-response`, { method: "POST", body: fd });
  const data = await resp.json();
  resEl.textContent = JSON.stringify(data, null, 2);
  if (data.decision === "advance") {
    document.getElementById("prompt").textContent = `${data.next_level}: ${data.next_prompt}`;
    setStatus("Advanced to " + data.next_level);
  } else {
    const rep = await fetch(`${API_BASE}/report/${SESSION_ID}`);
    const rj = await rep.json();
    document.getElementById("prompt").textContent = `Final Level: ${rj.final_level}`;
    resEl.textContent = JSON.stringify(rj, null, 2);
    setStatus("Session finished");
  }
}

async function evaluateSingle() {
  if (!lastBlob) { alert("Record something first."); return; }
  const resEl = document.getElementById("result");
  const fd = new FormData();
  fd.append("file", lastBlob, "single_recording.webm");
  const resp = await fetch(`${API_BASE}/evaluate-bytes`, { method: "POST", body: fd });
  const txt = await resp.text();
  try { resEl.textContent = JSON.stringify(JSON.parse(txt), null, 2); }
  catch { resEl.textContent = txt; }
}

document.getElementById("btnRecord").addEventListener("click", startRecording);
document.getElementById("btnStop").addEventListener("click", stopRecording);
document.getElementById("btnStartSession").addEventListener("click", startSession);
document.getElementById("btnLoadPrompt").addEventListener("click", loadPromptOnly);
document.getElementById("btnSubmit").addEventListener("click", submitResponse);
document.getElementById("btnEvalSingle").addEventListener("click", evaluateSingle);
