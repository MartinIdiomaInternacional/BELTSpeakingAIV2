import React, { useEffect, useRef, useState } from "react";
import Countdown from "./Countdown";
import WaveformCanvas from "./WaveformCanvas";
import { evaluateSpeaking } from "../lib/api";

export default function Recorder({ taskId, task, onFinished }) {
  const [isRecording, setIsRecording] = useState(false);
  const [secondsLeft, setSecondsLeft] = useState(task.maxSeconds);
  const [status, setStatus] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const timerRef = useRef(null);

  useEffect(() => {
    if (!isRecording) {
      clearInterval(timerRef.current);
      return;
    }

    setSecondsLeft(task.maxSeconds);
    timerRef.current = setInterval(() => {
      setSecondsLeft((prev) => {
        if (prev <= 1) {
          clearInterval(timerRef.current);
          stopRecording();
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(timerRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isRecording, task.maxSeconds]);

  const startRecording = async () => {
    try {
      setResult(null);
      setStatus("Requesting microphone…");
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      mediaRecorder.onstop = async () => {
        setStatus("Processing audio…");
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        await sendToApi(blob);
        stream.getTracks().forEach((t) => t.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
      setStatus("Recording…");
    } catch (err) {
      console.error(err);
      setStatus("Microphone access blocked or unavailable.");
    }
  };

  const stopRecording = () => {
    if (!mediaRecorderRef.current) return;
    if (mediaRecorderRef.current.state === "recording") {
      mediaRecorderRef.current.stop();
    }
    setIsRecording(false);
  };

  const sendToApi = async (audioBlob) => {
    setLoading(true);
    try {
      const data = await evaluateSpeaking({
        audioBlob,
        taskId,
      });
      setResult(data);
      setStatus("Result received.");
      if (onFinished) {
        onFinished(taskId, data);
      }
    } catch (err) {
      console.error(err);
      setStatus(`Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="recorder">
      <div className="recorder-header">
        <h2>{task.title}</h2>
        <p className="task-text">{task.text}</p>
      </div>

      <WaveformCanvas isRecording={isRecording} />

      <div className="recorder-controls">
        <button
          onClick={startRecording}
          disabled={isRecording || loading}
          className="btn primary"
        >
          {isRecording ? "Recording…" : "Start recording"}
        </button>

        <button
          onClick={stopRecording}
          disabled={!isRecording}
          className="btn secondary"
        >
          Stop
        </button>

        <Countdown seconds={secondsLeft} isRunning={isRecording} />
      </div>

      <div className="recorder-status">
        {status && <p>{status}</p>}
        {loading && <p>Sending audio to evaluation service…</p>}
      </div>

      {result && (
        <div className="recorder-result">
          <h3>Result</h3>
          <p>
            <strong>Level:</strong> {result.score}
          </p>
          <p>
            <strong>Explanation:</strong> {result.explanation}
          </p>
          <p>
            <strong>Recommendations:</strong>
          </p>
          <p className="recommendations">
            {result.recommendations.split("\n").map((line, idx) => (
              <span key={idx}>
                {line}
                <br />
              </span>
            ))}
          </p>
          <p>
            <strong>Duration (approx.):</strong>{" "}
            {result.seconds ? result.seconds.toFixed(1) : "—"} seconds
          </p>
        </div>
      )}
    </div>
  );
}
