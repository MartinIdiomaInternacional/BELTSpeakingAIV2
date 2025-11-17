const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export async function evaluateSpeaking({ audioBlob, taskId }) {
  const formData = new FormData();
  formData.append("audio", audioBlob, `task${taskId}.wav`);
  formData.append("task_id", String(taskId));

  const res = await fetch(`${API_URL}/evaluate`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`API error (${res.status}): ${text || "Unknown error"}`);
  }

  return res.json();
}
