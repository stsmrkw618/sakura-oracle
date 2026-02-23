"use client";

import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import HorseCard from "@/components/HorseCard";
import Navbar from "@/components/Navbar";
import OddsInput from "@/components/OddsInput";
import { MARK_ORDER } from "@/lib/constants";
import { useOdds } from "@/context/OddsContext";
import { useRace } from "@/context/RaceContext";

const goingColors: Record<string, string> = {
  "良": "bg-green-600",
  "稍重": "bg-yellow-600",
  "重": "bg-orange-600",
  "不良": "bg-red-600",
};

export default function PredictionPage() {
  const { predictions } = useRace();
  const { liveHorses, oddsMap, updateOdds, resetOdds } = useOdds();
  const [oddsMode, setOddsMode] = useState(false);

  const sortedHorses = useMemo(() => {
    // Merge static data with live EV/mark/marketProb
    const merged = predictions.predictions.map((h) => {
      const live = liveHorses.find((l) => l.horse_number === h.horse_number);
      return {
        ...h,
        ev_win: live?.ev_win ?? h.ev_win,
        ev_show: live?.ev_show ?? h.ev_show,
        mark: live?.mark ?? h.mark,
        oddsChanged: live?.oddsChanged ?? false,
        marketProb: live?.market_prob ?? 0,
      };
    });

    return merged.sort((a, b) => {
      const orderA = MARK_ORDER[a.mark] ?? 99;
      const orderB = MARK_ORDER[b.mark] ?? 99;
      if (orderA !== orderB) return orderA - orderB;
      return b.win_prob - a.win_prob;
    });
  }, [liveHorses, predictions]);

  const raceInfo = predictions.race_info;
  const goingColor = goingColors[raceInfo.going] || "bg-gray-600";

  const hasAnyChange = liveHorses.some((h) => h.oddsChanged);

  return (
    <div className="min-h-screen bg-navy-dark pb-20">
      {/* Header */}
      <motion.header
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="sticky top-0 z-40 bg-navy-dark/95 backdrop-blur-md border-b border-white/5 px-4 py-3"
      >
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-lg font-bold">{raceInfo.name}</h1>
            <p className="text-xs text-muted-foreground">{raceInfo.course}</p>
          </div>
          <div className="flex items-center gap-2">
            <span className={`text-xs px-2 py-0.5 rounded-full text-white ${goingColor}`}>
              {raceInfo.going}
            </span>
          </div>
        </div>
        <div className="flex items-center justify-between mt-1">
          <p className="text-[10px] text-muted-foreground">
            最終更新: {new Date(raceInfo.updated_at).toLocaleString("ja-JP")}
          </p>
          <button
            onClick={() => setOddsMode((v) => !v)}
            className={`text-[10px] px-2 py-0.5 rounded-full border transition-colors ${
              oddsMode
                ? "bg-sakura-pink/20 border-sakura-pink text-sakura-pink"
                : "border-white/20 text-muted-foreground"
            }`}
          >
            {oddsMode ? "オッズ入力中" : "オッズ更新"}
          </button>
        </div>
        {oddsMode && hasAnyChange && (
          <button
            onClick={resetOdds}
            className="mt-1 text-[10px] text-red-400 underline"
          >
            オッズをリセット
          </button>
        )}
      </motion.header>

      {/* Horse Cards */}
      <main className="px-4 py-4 space-y-3">
        {sortedHorses.map((horse, index) => (
          <div key={horse.horse_number}>
            <HorseCard
              horse={horse}
              index={index}
              liveEv={horse.ev_win}
              liveMark={horse.mark}
              oddsChanged={horse.oddsChanged}
              marketProb={horse.marketProb}
            />
            {oddsMode && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                className="bg-navy/50 rounded-b-xl border border-t-0 border-white/5 px-4 py-2 -mt-1 flex items-center gap-2"
              >
                <span className="text-[10px] text-muted-foreground w-8">単勝</span>
                <OddsInput
                  value={oddsMap[horse.horse_number]?.win ?? horse.odds.win}
                  onChange={(v) => {
                    const current = oddsMap[horse.horse_number];
                    updateOdds(horse.horse_number, v, current?.show ?? horse.odds.show);
                  }}
                  className="w-16 bg-navy border border-white/10 rounded px-2 py-1 text-xs font-mono text-right"
                />
                <span className="text-[10px] text-muted-foreground w-8">複勝</span>
                <OddsInput
                  value={oddsMap[horse.horse_number]?.show ?? horse.odds.show}
                  onChange={(v) => {
                    const current = oddsMap[horse.horse_number];
                    updateOdds(horse.horse_number, current?.win ?? horse.odds.win, v);
                  }}
                  className="w-16 bg-navy border border-white/10 rounded px-2 py-1 text-xs font-mono text-right"
                />
                <span className={`text-xs font-mono ml-auto ${horse.ev_win >= 1.0 ? "text-green-400" : "text-red-400"}`}>
                  EV {horse.ev_win.toFixed(2)}
                </span>
              </motion.div>
            )}
          </div>
        ))}
      </main>

      {/* Glossary */}
      <section className="px-4 pb-4">
        <div className="bg-card rounded-xl p-4 border border-white/5">
          <h3 className="text-xs font-bold text-muted-foreground mb-3">用語解説</h3>
          <dl className="space-y-2 text-xs">
            <div>
              <dt className="text-white font-bold inline">◎ 本命</dt>
              <dd className="text-muted-foreground inline ml-1">— AIが最も勝つ可能性が高いと判断した馬</dd>
            </div>
            <div>
              <dt className="text-white font-bold inline">○ 対抗</dt>
              <dd className="text-muted-foreground inline ml-1">— 本命に次ぐ有力馬。本命を負かす可能性あり</dd>
            </div>
            <div>
              <dt className="text-white font-bold inline">▲ 単穴</dt>
              <dd className="text-muted-foreground inline ml-1">— 人気薄だが一発あり得る穴馬。単勝で狙う価値あり</dd>
            </div>
            <div>
              <dt className="text-white font-bold inline">△ 連下</dt>
              <dd className="text-muted-foreground inline ml-1">— 1着は厳しいが2〜3着に来る可能性がある馬</dd>
            </div>
            <div>
              <dt className="text-white font-bold inline">× 消し</dt>
              <dd className="text-muted-foreground inline ml-1">— 好走の可能性が低いとAIが判断した馬</dd>
            </div>
            <hr className="border-white/5 my-2" />
            <div>
              <dt className="text-white font-bold inline">勝率</dt>
              <dd className="text-muted-foreground inline ml-1">— AIが算出した1着になる確率</dd>
            </div>
            <div>
              <dt className="text-white font-bold inline">複勝</dt>
              <dd className="text-muted-foreground inline ml-1">— AIが算出した3着以内に入る確率</dd>
            </div>
            <div>
              <dt className="text-white font-bold inline">EV（期待値）</dt>
              <dd className="text-muted-foreground inline ml-1">— 100円賭けた時の期待リターン。1.0超で割安、1.5超はお買い得</dd>
            </div>
            <div>
              <dt className="text-white font-bold inline">SI（スピード指数）</dt>
              <dd className="text-muted-foreground inline ml-1">— 過去レースの走破タイムを距離・馬場で補正した能力指標</dd>
            </div>
          </dl>
        </div>
      </section>

      <Navbar />
    </div>
  );
}
