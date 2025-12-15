import React from "react";
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
} from "recharts";

export default function DimensionRadar({ dimensionAverages }) {
  if (!dimensionAverages) return null;

  const data = Object.entries(dimensionAverages).map(([key, value]) => ({
    dimension: key.replace(/_/g, " "),
    score: Number(value.toFixed(2)),
    fullMark: 6,
  }));

  return (
    <div style={{ width: "100%", height: 360 }}>
      <ResponsiveContainer>
        <RadarChart data={data} outerRadius="70%">
          <PolarGrid />
          <PolarAngleAxis dataKey="dimension" />
          <PolarRadiusAxis domain={[0, 6]} tickCount={7} />
          <Radar
            dataKey="score"
            stroke="#2563eb"
            fill="#60a5fa"
            fillOpacity={0.35}
            dot
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}
