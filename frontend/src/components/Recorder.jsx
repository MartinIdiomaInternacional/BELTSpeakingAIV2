import React, { useEffect, useRef, useState } from "react";
import { evaluateSpeaking } from "../lib/api";

function pickSupportedMimeType() {
  const preferred = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/ogg;codecs=opus",
    "audio/ogg",
  ];
  for (const t of preferred) {
    if (window.MediaRecorder && MediaRecorder.isTypeSupported(t)) return t;
  }
  return "";
}

export default function Recorder({ taskId, task, onFinished, showFeedback = true }) {
  const [recording, setRecording] = useState(false);
  const [status, setStatus] = useState("");
  const [err, setErr] = useState("");
  const [result, setResult] = useState(null);
  const [timeLeft, setTimeLeft] = useState(task.maxSeconds);

  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const streamRef = useRef(null);
  const timerRef = useRef(null);

  useEffect(() => {
    setRecording(false);
    setStatus("");
    setErr("");
    setResult(null);
    setTimeLeft(task.maxSeconds);

    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = null;

    try {
      if (mediaRecorderRef.current?.state !== "inactive") {
        mediaRecorderRef.current.stop();
      }
    } catch {}

    try {
      streamRef.current?.getTracks().forEach((t) => t.stop());
    } catch {}

    streamRef.current = null;
    mediaRecorderRef.current = null;
    chunksRef.current = [];
  }, [taskId, task.maxSeconds]);

  async function ensureMic() {
    if (streamRef.current) return streamRef.current;
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    streamRef.current = stream;
    return stream;
  }

  async function start() {
    setErr("");
    setResult(null);
    setStatus("");

    try {
      const stream = await ensureMic();
      const mimeType = pickSupportedMimeType();
      const mr = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);

      chunksRef.current = [];
      mr.ondataavailable = (e) => e.data.size && chunksRef.current.push(e.data);

      mr.onstop = async () => {
        try {
          setStatus("Evaluating...");
          const blob = new Blob(chunksRef.current, {
            type: mr.mimeType || "audio/webm",
          });

          const data = await evaluateSpeaking({
            audioBlob: blob,
            taskId,
            taskTitle: task.title,
            promptText: task.text,
          });

          setResult(data);
          onFinished?.(taskId, data);
          setStatus("Done");
        } catch (e) {
          setErr("Evaluation failed.");
          setStatus("");
        }
      };

      mediaRecorderRef.current = mr;
      mr.start();
      setRecording(true);
      setStatus("Recording...");

      const startMs = Date.now();
      timerRef.current = setInterval(() => {
        const elapsed = (Date.now() - startMs) / 1000;
        const left = Math.max(0, Math.ceil(task.maxSeconds - elapsed));
        setTimeLeft(left);
        if (left <= 0) stop();
      }, 250);
    } catch {
      setErr("Microphone permission denied.");
    }
  }

  function stop() {
    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = null;
    try {
      mediaRecorderRef.current?.stop();
    } catch {}
    setRecording(false);
  }

  const feedback = result?.feedback;

  return (
    <div className="recorder">
      <h2>{task.title}</h2>
      <p className="task-text">{task.text}</p>

      <div className="recorder-controls">
        <button className="btn primary" onClick={start} disabled={recording}>Start</button>
        <button className="btn secondary" onClick={stop} disabled={!recording}>Stop</button>
        <div className="countdown">{timeLeft}s</div>
      </div>

      {status && <div className="recorder-status">{status}</div>}
      {err && <div className="recorder-status error">{err}</div>}

      {showFeedback && feedback && (
        <div className="recorder-result">
          <h3>Feedback</h3>

          <p>{feedback.summary}</p>

          {feedback.strengths?.length > 0 && (
            <>
              <h4>Strengths</h4>
              <ul>{feedback.strengths.map((s) => <li key={s}>{s}</li>)}</ul>
            </>
          )}

          {feedback.priorities?.length > 0 && (
            <>
              <h4>Areas to improve</h4>
              <ul>{feedback.priorities.map((p) => <li key={p}>{p}</li>)}</ul>
            </>
          )}

          <h4>Detailed observations</h4>
          <ul>
            {Object.entries(feedback.dimension_feedback || {}).map(([k, v]) => (
              <li key={k}><strong>{k.replace(/_/g, " ")}:</strong> {v}</li>
            ))}
          </ul>

          {feedback.task_tip && (
            <>
              <h4>Task tip</h4>
              <p>{feedback.task_tip}</p>
            </>
          )}
        </div>
      )}
    </div>
  );
}
