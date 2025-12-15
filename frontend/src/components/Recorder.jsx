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

export default function Recorder({
  taskId,
  task,
  onFinished,
  showFeedback = true,
}) {
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
    // reset per task
    setRecording(false);
    setStatus("");
    setErr("");
    setResult(null);
    setTimeLeft(task.maxSeconds);

    // cleanup any old timers
    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = null;

    // stop any existing recorder
    try {
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
        mediaRecorderRef.current.stop();
      }
    } catch {}

    // stop any existing stream tracks
    try {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
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
      mr.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) chunksRef.current.push(e.data);
      };

      mr.onstop = async () => {
        try {
          setStatus("Evaluating...");
          const blob = new Blob(chunksRef.current, {
            type: mr.mimeType || "audio/webm",
          });

          const data = await evaluateSpeaking({ audioBlob: blob, taskId });

          setResult(data);
          onFinished?.(taskId, data);
          setStatus("Done");
        } catch (e) {
          console.error(e);
          setErr(e?.message || "Evaluation failed");
          setStatus("");
        }
      };

      mediaRecorderRef.current = mr;
      mr.start();

      setRecording(true);
      setStatus("Recording...");

      // countdown timer + auto-stop at maxSeconds
      const startMs = Date.now();
      setTimeLeft(task.maxSeconds);

      timerRef.current = setInterval(() => {
        const elapsed = (Date.now() - startMs) / 1000;
        const left = Math.max(0, Math.ceil(task.maxSeconds - elapsed));
        setTimeLeft(left);

        if (left <= 0) {
          stop();
        }
      }, 250);
    } catch (e) {
      console.error(e);
      setErr("Microphone permission denied or not available.");
    }
  }

  function stop() {
    try {
      if (timerRef.current) clearInterval(timerRef.current);
      timerRef.current = null;
    } catch {}

    try {
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
        mediaRecorderRef.current.stop();
      }
    } catch {}

    setRecording(false);
  }

  // Helpers to display either old or new backend payloads
  const uiLevel = result?.text_level || result?.level || result?.score || null;
  const uiTotal =
    typeof result?.text_total_score === "number"
      ? result.text_total_score
      : typeof result?.score === "number"
      ? result.score
      : null;

  const uiRecs =
    result?.text_recommendations ||
    result?.recommendations ||
    result?.explanation ||
    "";

  return (
    <div className="recorder">
      <div className="recorder-header">
        <h2>{task.title}</h2>
        <p className="task-text">{task.text}</p>
      </div>

      <div className="recorder-controls">
        <button className="btn primary" onClick={start} disabled={recording}>
          Start
        </button>
        <button className="btn secondary" onClick={stop} disabled={!recording}>
          Stop
        </button>

        <div className="countdown">
          <div className="countdown-time">{timeLeft}s</div>
        </div>
      </div>

      {status && <div className="recorder-status">{status}</div>}
      {err && <div className="recorder-status" style={{ color: "#b91c1c" }}>{err}</div>}

      {/* ✅ Only show feedback when showFeedback is true (Task 3) */}
      {showFeedback && result && (
        <div className="recorder-result">
          <h3>Result</h3>

          {uiLevel != null && (
            <p>
              <strong>Level:</strong> {String(uiLevel)}
            </p>
          )}

          {uiTotal != null && (
            <p>
              <strong>Total score:</strong> {uiTotal.toFixed(2)} / 6
            </p>
          )}

          {uiRecs ? (
            <>
              <h4>Feedback</h4>
              <div className="recommendations">{String(uiRecs)}</div>
            </>
          ) : null}
        </div>
      )}
    </div>
  );
}
