"use client";

import { useRace } from "@/context/RaceContext";

export default function RaceSelector() {
  const { races, selectedRaceId, selectRace } = useRace();

  if (races.length <= 1) return null;

  return (
    <select
      value={selectedRaceId ?? ""}
      onChange={(e) => selectRace(e.target.value)}
      className="w-full bg-navy border border-white/10 rounded-lg px-3 py-2 text-sm text-white appearance-none cursor-pointer focus:outline-none focus:border-sakura-pink/50"
      style={{
        backgroundImage:
          "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%23A0A0B0' d='M2 4l4 4 4-4'/%3E%3C/svg%3E\")",
        backgroundRepeat: "no-repeat",
        backgroundPosition: "right 12px center",
      }}
    >
      {races.map((r) => (
        <option key={r.id} value={r.id}>
          {r.label} — {r.course}
        </option>
      ))}
    </select>
  );
}
