const BASE = import.meta.env.VITE_API_BASE || '/api';

export async function startSession(candidateId, nativeLanguage, targetLevel) {
  const res = await fetch(`${BASE}/start`, {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ candidate_id: candidateId, native_language: nativeLanguage, target_level: targetLevel })
  });
  if(!res.ok) throw new Error('Failed to start session');
  return res.json();
}

export async function getPrompts(level) {
  const res = await fetch(`${BASE}/prompts/${encodeURIComponent(level)}`);
  if(!res.ok) throw new Error('Failed to fetch prompts');
  return res.json();
}

export async function evaluateBytes(sessionId, sampleRate, wav_base64, promptId) {
  const res = await fetch(`${BASE}/evaluate-bytes`, {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ session_id: sessionId, sample_rate: sampleRate, wav_base64, prompt_id: promptId })
  });
  if(!res.ok) throw new Error('Failed to evaluate audio');
  return res.json();
}

export async function getReport(sessionId, nativeLanguage) {
  const res = await fetch(`${BASE}/report`, {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ session_id: sessionId, native_language: nativeLanguage })
  });
  if(!res.ok) throw new Error('Failed to build report');
  return res.json();
}
