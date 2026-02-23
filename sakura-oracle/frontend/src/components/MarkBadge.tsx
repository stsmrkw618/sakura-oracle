"use client";

const markConfig: Record<string, { label: string; bg: string; text: string }> = {
  "◎": { label: "本命", bg: "bg-[#FFD700]", text: "text-black" },
  "○": { label: "対抗", bg: "bg-[#E8879C]", text: "text-white" },
  "▲": { label: "単穴", bg: "bg-[#FF8C00]", text: "text-white" },
  "△": { label: "連下", bg: "bg-[#6B7280]", text: "text-white" },
  "×": { label: "消し", bg: "bg-[#374151]", text: "text-gray-300" },
};

interface MarkBadgeProps {
  mark: string;
  size?: "sm" | "md" | "lg";
}

export default function MarkBadge({ mark, size = "md" }: MarkBadgeProps) {
  const config = markConfig[mark] || markConfig["×"];
  const sizeClasses = {
    sm: "text-xs px-1.5 py-0.5",
    md: "text-sm px-2 py-1",
    lg: "text-base px-3 py-1.5",
  };

  return (
    <span
      className={`inline-flex items-center gap-1 rounded-md font-bold ${config.bg} ${config.text} ${sizeClasses[size]}`}
    >
      <span>{mark}</span>
      <span className="text-[0.7em]">{config.label}</span>
    </span>
  );
}
