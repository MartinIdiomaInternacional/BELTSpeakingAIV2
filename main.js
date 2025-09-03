
const API_BASE = ""; // same origin by default; set to your server URL if hosting separately

async function evaluate() {
  const fileEl = document.getElementById("file");
  const resEl = document.getElementById("result");
  const status = document.getElementById("status");
  if (!fileEl.files[0]) {
    resEl.textContent = "Please choose an audio file first.";
    return;
  }
  const fd = new FormData();
  fd.append("file", fileEl.files[0]);
  status.classList.remove("hidden");
  try {
    const resp = await fetch(`${API_BASE}/evaluate-bytes`, { method: "POST", body: fd });
    const txt = await resp.text();
    try {
      const json = JSON.parse(txt);
      resEl.textContent = JSON.stringify(json, null, 2);
    } catch {
      resEl.textContent = txt;
    }
  } catch (e) {
    resEl.textContent = "Request failed: " + e.message;
  } finally {
    status.classList.add("hidden");
  }
}

async function loadPrompt() {
  const level = document.getElementById("level").value;
  const promptEl = document.getElementById("prompt");
  try {
    const resp = await fetch(`${API_BASE}/prompts/${encodeURIComponent(level)}`);
    if (!resp.ok) {
      promptEl.textContent = `Error ${resp.status}: ${await resp.text()}`;
      return;
    }
    const data = await resp.json();
    promptEl.textContent = JSON.stringify(data, null, 2);
  } catch (e) {
    promptEl.textContent = "Request failed: " + e.message;
  }
}

document.getElementById("btnEval").addEventListener("click", evaluate);
document.getElementById("btnPrompt").addEventListener("click", loadPrompt);
