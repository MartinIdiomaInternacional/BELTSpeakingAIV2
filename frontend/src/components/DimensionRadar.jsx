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

  const data = Object.entries(dimensionAverages).map(
    ([key, value]) => ({
      dimension: key.replace(/_/g, " "),
      score: Number(value.toFixed(2)),
      fullMark: 6,
    })
  );

  return (
    <div style={{ width: "100%", height: 350 }}>
      <ResponsiveContainer>
        <RadarChart data={data}>
          <PolarGrid />
          <PolarAngleAxis dataKey="dimension" />
          <PolarRadiusAxis domain={[0, 6]} />
          <Radar
            name="CEFR Dimensions"
            dataKey="score"
            stroke="#2563eb"
            fill="#3b82f6"
            fillOpacity={0.5}
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}
