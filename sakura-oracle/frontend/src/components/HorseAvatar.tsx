"use client";

import Image from "next/image";
import { FRAME_COLORS } from "@/lib/constants";

const FRAME_HEX: Record<number, string> = {
  1: "#FFFFFF",
  2: "#374151",
  3: "#DC2626",
  4: "#2563EB",
  5: "#EAB308",
  6: "#16A34A",
  7: "#F97316",
  8: "#F472B6",
};

interface HorseAvatarProps {
  imageUrl?: string;
  horseNumber: number;
  horseName: string;
  frameNumber: number;
}

export default function HorseAvatar({
  imageUrl,
  horseNumber,
  horseName,
  frameNumber,
}: HorseAvatarProps) {
  const frameColor = FRAME_HEX[frameNumber] || "#FFFFFF";
  const frame = FRAME_COLORS[frameNumber] || FRAME_COLORS[1];

  if (imageUrl) {
    return (
      <div className="relative w-full aspect-[4/3] rounded-xl overflow-hidden border-2 border-white/10">
        <Image
          src={imageUrl}
          alt={horseName}
          fill
          className="object-cover"
          sizes="(max-width: 430px) 100vw, 430px"
        />
        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/70 to-transparent p-3">
          <div className="flex items-center gap-2">
            <span
              className={`w-8 h-8 rounded-md flex items-center justify-center text-sm font-bold border border-white/20 ${frame.bg} ${frame.text}`}
            >
              {horseNumber}
            </span>
            <span className="text-white font-bold text-lg">{horseName}</span>
          </div>
        </div>
      </div>
    );
  }

  // Stylized placeholder with horse silhouette SVG
  return (
    <div
      className="relative w-full aspect-[4/3] rounded-xl overflow-hidden border-2 border-white/10"
      style={{
        background: `linear-gradient(135deg, #0F0F1A 0%, ${frameColor}22 50%, #0F0F1A 100%)`,
      }}
    >
      {/* Horse Silhouette SVG */}
      <div className="absolute inset-0 flex items-center justify-center opacity-20">
        <svg
          viewBox="0 0 200 180"
          className="w-48 h-48"
          fill={frameColor}
        >
          <path d="M160,45 C155,30 145,20 135,18 C130,17 125,20 122,25 C118,15 110,8 100,5 C95,3 90,5 88,10 L85,18 C80,15 72,14 65,16 C58,18 52,24 50,32 C45,30 38,32 35,38 C32,44 34,52 38,56 L42,58 C38,65 36,75 38,85 L35,90 C30,95 28,102 30,110 L32,118 C28,125 26,135 30,142 C34,150 42,155 50,155 L55,155 C58,158 62,160 68,160 L72,160 C75,158 78,155 80,150 L82,142 C90,148 100,152 110,152 L115,155 C118,158 122,160 128,160 L132,160 C135,158 138,155 140,150 L142,142 C148,145 155,142 158,136 C162,128 160,118 155,112 L152,108 C155,100 156,90 154,82 L158,75 C162,68 164,58 160,48 Z" />
        </svg>
      </div>

      {/* Frame color accent ring */}
      <div
        className="absolute top-4 right-4 w-16 h-16 rounded-full flex items-center justify-center border-3"
        style={{
          borderColor: frameColor,
          background: `${frameColor}30`,
        }}
      >
        <span
          className="text-2xl font-bold font-mono"
          style={{ color: frameColor }}
        >
          {horseNumber}
        </span>
      </div>

      {/* Horse name at bottom */}
      <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/60 to-transparent p-3">
        <div className="flex items-center gap-2">
          <span
            className={`w-8 h-8 rounded-md flex items-center justify-center text-sm font-bold border border-white/20 ${frame.bg} ${frame.text}`}
          >
            {horseNumber}
          </span>
          <span className="text-white font-bold text-lg">{horseName}</span>
        </div>
      </div>
    </div>
  );
}
