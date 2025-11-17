import React, { useEffect, useRef } from "react";

export default function WaveformCanvas({ isRecording }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "#444";
    ctx.lineWidth = 2;
    ctx.beginPath();

    const mid = canvas.height / 2;
    ctx.moveTo(0, mid);

    const segments = 32;
    for (let i = 0; i <= segments; i++) {
      const x = (canvas.width / segments) * i;
      const amp = isRecording ? 10 : 1;
      const y = mid + Math.sin(i * 0.8) * amp;
      ctx.lineTo(x, y);
    }

    ctx.stroke();
  }, [isRecording]);

  return (
    <canvas
      ref={canvasRef}
      width={400}
      height={80}
      className="waveform-canvas"
    />
  );
}
