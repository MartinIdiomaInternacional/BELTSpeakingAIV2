
const API_BASE = ""; // same-origin

let SESSION_ID = null;

async function evaluateSingle() {
  const fileEl = document.getElementById("file");
  const resEl = document.getElementById("result");
  const status = document.getElementById("status");
  if (!fileEl.files[0]) { resEl.textContent = "Please choose an audio file."; return; }
  const fd = new FormData();
  fd.append("file", fileEl.files[0]);
  status.classList.remove("hidden");
  try {
    const resp = await fetch(`${API_BASE}/evaluate-bytes`, { method: "POST", body: fd });
    const txt = await resp.text();
    try { resEl.textContent = JSON.stringify(JSON.parse(txt), null, 2); } catch { resEl.textContent = txt; }
  } catch (e) {
    resEl.textContent = "Request failed: " + e.message;
  } finally { status.classList.add("hidden"); }
}

async function startSession() {
  const level = document.getElementById("level").value;
  const promptEl = document.getElementById("prompt");
  const resEl = document.getElementById("result");
  const fd = new FormData(); fd.append("level", level);
  const resp = await fetch(`${API_BASE}/start-session`, { method: "POST", body: fd });
  const data = await resp.json();
  SESSION_ID = data.session_id;
  promptEl.textContent = `${data.level}: ${data.prompt}`;
  resEl.textContent = "(session started)";
}

async function submitResponse() {
  const fileEl = document.getElementById("file");
  const promptEl = document.getElementById("prompt");
  const resEl = document.getElementById("result");
  if (!SESSION_ID) { resEl.textContent = "Start a session first."; return; }
  if (!fileEl.files[0]) { resEl.textContent = "Choose an audio file."; return; }
  const fd = new FormData();
  fd.append("session_id", SESSION_ID);
  fd.append("file", fileEl.files[0]);
  const resp = await fetch(`${API_BASE}/submit-response`, { method: "POST", body: fd });
  const data = await resp.json();
  resEl.textContent = JSON.stringify(data, null, 2);
  if (data.decision === "advance") {
    promptEl.textContent = `${data.next_level}: ${data.next_prompt}`;
  } else {
    const rep = await fetch(`${API_BASE}/report/${SESSION_ID}`);
    const rj = await rep.json();
    promptEl.textContent = `Final Level: ${rj.final_level}`;
    resEl.textContent = JSON.stringify(rj, null, 2);
  }
}

document.getElementById("btnEval").addEventListener("click", evaluateSingle);
document.getElementById("btnStart").addEventListener("click", startSession);
document.getElementById("btnSubmit").addEventListener("click", submitResponse);
