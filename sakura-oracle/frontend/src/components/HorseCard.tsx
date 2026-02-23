"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import MarkBadge from "./MarkBadge";
import { FRAME_COLORS } from "@/lib/constants";

interface HorseCardProps {
  horse: {
    horse_number: number;
    horse_name: string;
    mark: string;
    win_prob: number;
    show_prob: number;
    ev_win: number;
    speed_index: number;
    last3_results: string[];
    frame_number: number;
    jockey: string;
  };
  index: number;
  liveEv?: number;
  liveMark?: string;
  oddsChanged?: boolean;
}

export default function HorseCard({
  horse,
  index,
  liveEv,
  liveMark,
  oddsChanged,
}: HorseCardProps) {
  const frame = FRAME_COLORS[horse.frame_number] || FRAME_COLORS[1];
  const displayEv = liveEv ?? horse.ev_win;
  const displayMark = liveMark ?? horse.mark;

  return (
    <motion.div
      initial={{ opacity: 0, x: -30 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.4, delay: index * 0.05 }}
    >
      <Link href={`/horse/${horse.horse_number}`}>
        <div className="bg-card rounded-xl p-4 border border-white/5 hover:border-sakura-pink/30 transition-all duration-200 active:scale-[0.98]">
          {/* Header: Mark + Frame + Number + Name + LIVE badge */}
          <div className="flex items-center gap-3 mb-3">
            <MarkBadge mark={displayMark} />
            <span
              className={`w-7 h-7 rounded-sm flex items-center justify-center text-xs font-bold ${frame.bg} ${frame.text}`}
            >
              {horse.horse_number}
            </span>
            <span className="font-bold text-lg flex-1 truncate">
              {horse.horse_name}
            </span>
            {oddsChanged && (
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-sakura-pink/20 text-sakura-pink font-bold">
                LIVE
              </span>
            )}
          </div>

          {/* Probability Bars */}
          <div className="space-y-2 mb-3">
            <div className="flex items-center gap-2">
              <span className="text-xs text-muted-foreground w-8">勝率</span>
              <div className="flex-1 bg-white/5 rounded-full h-3 overflow-hidden">
                <div
                  className="h-full bg-sakura-pink rounded-full transition-all duration-500"
                  style={{ width: `${Math.min(horse.win_prob * 100, 100)}%` }}
                />
              </div>
              <span className="font-mono text-sm text-sakura-pink w-10 text-right">
                {(horse.win_prob * 100).toFixed(0)}%
              </span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs text-muted-foreground w-8">複勝</span>
              <div className="flex-1 bg-white/5 rounded-full h-3 overflow-hidden">
                <div
                  className="h-full bg-sakura-pink/60 rounded-full transition-all duration-500"
                  style={{ width: `${Math.min(horse.show_prob * 100, 100)}%` }}
                />
              </div>
              <span className="font-mono text-sm text-sakura-light w-10 text-right">
                {(horse.show_prob * 100).toFixed(0)}%
              </span>
            </div>
          </div>

          {/* Stats Row */}
          <div className="flex items-center gap-4 text-xs text-muted-foreground">
            <span>
              🏇 前走: {horse.last3_results[0]}
            </span>
            <span>
              ⚡ SI: {horse.speed_index}
            </span>
            <span className={displayEv >= 1.5 ? "text-gold font-bold" : ""}>
              {displayEv >= 1.5 && "🔥 "}
              💰 EV: {displayEv.toFixed(2)}
            </span>
          </div>

          {/* Detail Link */}
          <div className="text-right mt-2">
            <span className="text-xs text-sakura-pink">詳細を見る →</span>
          </div>
        </div>
      </Link>
    </motion.div>
  );
}
