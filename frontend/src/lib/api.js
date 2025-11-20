// Simple client for the BELT Speaking AI 2.0 backend.
// We assume Nginx (in production) proxies /api/* to the FastAPI service.

const BASE = import.meta.env.VITE_API_BASE || "/api";

export async function evaluateSpeaking({ audioBlob, taskId }) {
  const formData = new FormData();
  formData.append("audio", audioBlob, `task${taskId}.webm`);
  formData.append("task_id", String(taskId));

  const res = await fetch(`${BASE}/evaluate`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`API error (${res.status}): ${text || "Unknown error"}`);
  }

  return res.json(); // { score, explanation, recommendations, seconds, task_id }
}
