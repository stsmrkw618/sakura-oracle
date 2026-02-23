"use client";

import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ResponsiveContainer,
} from "recharts";

interface RadarData {
  speed: number;
  stamina: number;
  instant: number;
  pedigree: number;
  jockey: number;
  course_fit: number;
}

const LABELS: Record<string, string> = {
  speed: "スピード",
  stamina: "スタミナ",
  instant: "瞬発力",
  pedigree: "血統",
  jockey: "騎手",
  course_fit: "コース適性",
};

export default function HorseRadarChart({ radar }: { radar: RadarData }) {
  const data = Object.entries(radar).map(([key, value]) => ({
    subject: LABELS[key] || key,
    value,
    fullMark: 100,
  }));

  return (
    <ResponsiveContainer width="100%" height={280}>
      <RadarChart data={data} cx="50%" cy="50%" outerRadius="70%">
        <PolarGrid stroke="#1A1A2E" strokeWidth={1} />
        <PolarAngleAxis
          dataKey="subject"
          tick={{ fill: "#A0A0B0", fontSize: 11 }}
        />
        <PolarRadiusAxis
          angle={90}
          domain={[0, 100]}
          tick={false}
          axisLine={false}
        />
        <Radar
          name="能力"
          dataKey="value"
          stroke="#E8879C"
          fill="#E8879C"
          fillOpacity={0.3}
          strokeWidth={2}
          animationBegin={0}
          animationDuration={1200}
        />
      </RadarChart>
    </ResponsiveContainer>
  );
}
