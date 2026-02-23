// 競馬正式枠色 (1-8枠)
export const FRAME_COLORS: Record<number, { bg: string; text: string; border: string }> = {
  1: { bg: "bg-white", text: "text-black", border: "border-white" },
  2: { bg: "bg-black", text: "text-white", border: "border-gray-700" },
  3: { bg: "bg-red-600", text: "text-white", border: "border-red-600" },
  4: { bg: "bg-blue-600", text: "text-white", border: "border-blue-600" },
  5: { bg: "bg-yellow-400", text: "text-black", border: "border-yellow-400" },
  6: { bg: "bg-green-600", text: "text-white", border: "border-green-600" },
  7: { bg: "bg-orange-500", text: "text-white", border: "border-orange-500" },
  8: { bg: "bg-pink-400", text: "text-white", border: "border-pink-400" },
};

// 印の並び順
export const MARK_ORDER: Record<string, number> = {
  "◎": 0,
  "○": 1,
  "▲": 2,
  "△": 3,
  "×": 4,
};
