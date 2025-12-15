import React from "react";
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Tooltip,
} from "recharts";

function numericToLevel(score) {
  if (score == null || isNaN(score)) return "N/A";
  if (score < 0.5) return "A1";
  if (score < 1.5) return "A2";
  if (score < 2.5) return "B1";
  if (score < 3.5) return "B1+";
  if (score < 4.5) return "B2";
  if (score < 5.5) return "B2+";
  return "C1/C2";
}

function prettyDim(name) {
  // Optional: make labels nicer (feel free to tweak)
  return name
    .replace(/_/g, " ")
    .replace(/\bcefr\b/i, "CEFR")
    .replace(/\bai\b/i, "AI");
}

// Tooltip content
function RadarTooltip({ active, payload }) {
  if (!active || !payload || !payload.length) return null;

  const p = payload[0]?.payload;
  const score = typeof p?.score === "number" ? p.score : null;

  return (
    <div
      style={{
        background: "white",
        border: "1px solid #e5e7eb",
        borderRadius: 10,
        padding: 10,
        boxShadow: "0 10px 25px rgba(15, 23, 42, 0.08)",
      }}
    >
      <div style={{ fontWeight: 700, marginBottom: 6 }}>
        {p?.dimension || "Dimension"}
      </div>
      <div style={{ fontSize: 14 }}>
        <strong>Score:</strong> {score == null ? "—" : score.toFixed(2)} / 6
      </div>
      <div style={{ fontSize: 14 }}>
        <strong>CEFR:</strong> {score == null ? "—" : numericToLevel(score)}
      </div>
    </div>
  );
}

export default function DimensionRadar({ dimensionAverages }) {
  if (!dimensionAverages) return null;

  const data = Object.entries(dimensionAverages).map(([key, value]) => ({
    dimension: prettyDim(key),
    score: Number(value.toFixed(2)),
    fullMark: 6,
  }));

  return (
    <div style={{ width: "100%", height: 350 }}>
      <ResponsiveContainer>
        <RadarChart data={data}>
          <PolarGrid />

          <PolarAngleAxis dataKey="dimension" />

          {/* CEFR-style rings */}
          <PolarRadiusAxis
            domain={[0, 6]}
            ticks={[1, 2, 3, 4, 5, 6]}
            tickFormatter={(v) => {
              // Mapping rings to CEFR-ish anchors
              if (v === 1) return "A2";
              if (v === 2) return "B1";
              if (v === 3) return "B1+";
              if (v === 4) return "B2";
              if (v === 5) return "B2+";
              if (v === 6) return "C1";
              return v;
            }}
          />

          <Tooltip content={<RadarTooltip />} />

          <Radar dataKey="score" fillOpacity={0.5} />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}
