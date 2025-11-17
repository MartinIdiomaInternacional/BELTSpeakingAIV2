import React from "react";

function formatTime(totalSeconds) {
  const s = Math.max(0, Math.floor(totalSeconds));
  const m = String(Math.floor(s / 60)).padStart(2, "0");
  const r = String(s % 60).padStart(2, "0");
  return `${m}:${r}`;
}

export default function Countdown({ seconds, isRunning }) {
  return (
    <div className="countdown">
      <span className={`countdown-time ${isRunning ? "running" : ""}`}>
        {formatTime(seconds)}
      </span>
    </div>
  );
}
