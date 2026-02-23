"use client";

import { useParams, useRouter } from "next/navigation";
import { useMemo } from "react";
import { motion } from "framer-motion";
import MarkBadge from "@/components/MarkBadge";
import HorseAvatar from "@/components/HorseAvatar";
import HorseRadarChart from "@/components/HorseRadarChart";
import Navbar from "@/components/Navbar";
import OddsInput from "@/components/OddsInput";
import { FRAME_COLORS } from "@/lib/constants";
import { useOdds } from "@/context/OddsContext";
import { useRace } from "@/context/RaceContext";

const fadeIn = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
};

export default function HorseDetailClient() {
  const params = useParams();
  const router = useRouter();
  const horseNumber = Number(params.id);
  const { predictions } = useRace();
  const { liveHorses, oddsMap, updateOdds } = useOdds();

  const horse = useMemo(
    () => predictions.predictions.find((h) => h.horse_number === horseNumber),
    [horseNumber, predictions]
  );

  const live = useMemo(
    () => liveHorses.find((h) => h.horse_number === horseNumber),
    [liveHorses, horseNumber]
  );

  if (!horse) {
    return (
      <div className="min-h-screen bg-navy-dark flex items-center justify-center">
        <p className="text-muted-foreground">馬が見つかりません</p>
      </div>
    );
  }

  const frame = FRAME_COLORS[horse.frame_number] || FRAME_COLORS[1];
  const displayMark = live?.mark ?? horse.mark;
  const displayEvWin = live?.ev_win ?? horse.ev_win;
  const displayEvShow = live?.ev_show ?? horse.ev_show;
  const oddsChanged = live?.oddsChanged ?? false;
  const currentOdds = oddsMap[horse.horse_number] ?? horse.odds;

  const resultColor = (result: string) => {
    if (result.startsWith("1")) return "text-gold border-gold bg-gold/10";
    if (result.startsWith("2") || result.startsWith("3"))
      return "text-gray-300 border-gray-400 bg-gray-400/10";
    return "text-gray-500 border-gray-600 bg-gray-600/10";
  };

  return (
    <div className="min-h-screen bg-navy-dark pb-20">
      {/* Header */}
      <header className="sticky top-0 z-40 bg-navy-dark/95 backdrop-blur-md border-b border-white/5 px-4 py-3">
        <button
          onClick={() => router.back()}
          className="text-sakura-pink text-sm mb-1"
        >
          ← 戻る
        </button>
        <div className="flex items-center gap-3">
          <MarkBadge mark={displayMark} size="lg" />
          <h1 className="text-2xl font-bold">{horse.horse_name}</h1>
          {oddsChanged && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-sakura-pink/20 text-sakura-pink font-bold">
              LIVE
            </span>
          )}
        </div>
      </header>

      <main className="px-4 py-4 space-y-5">
        {/* Horse Image */}
        <motion.section {...fadeIn} transition={{ delay: 0.05 }}>
          <HorseAvatar
            imageUrl={(horse as Record<string, unknown>).image as string | undefined}
            horseNumber={horse.horse_number}
            horseName={horse.horse_name}
            frameNumber={horse.frame_number}
          />
        </motion.section>

        {/* Radar Chart */}
        <motion.section {...fadeIn} transition={{ delay: 0.1 }}>
          <div className="bg-card rounded-xl p-4 border border-white/5">
            <h2 className="text-sm font-bold text-muted-foreground mb-2">
              能力チャート
            </h2>
            <HorseRadarChart radar={horse.radar} />
          </div>
        </motion.section>

        {/* Basic Info */}
        <motion.section {...fadeIn} transition={{ delay: 0.2 }}>
          <div className="bg-card rounded-xl p-4 border border-white/5">
            <h2 className="text-sm font-bold text-muted-foreground mb-3">
              基本情報
            </h2>
            <div className="grid grid-cols-2 gap-3 text-sm">
              <div className="flex items-center gap-2">
                <span className="text-muted-foreground">馬番</span>
                <span
                  className={`w-6 h-6 rounded-sm flex items-center justify-center text-xs font-bold ${frame.bg} ${frame.text}`}
                >
                  {horse.horse_number}
                </span>
              </div>
              <div>
                <span className="text-muted-foreground">枠番</span>
                <span className="ml-2 font-bold">{horse.frame_number}枠</span>
              </div>
              <div>
                <span className="text-muted-foreground">騎手</span>
                <span className="ml-2">{horse.jockey}</span>
              </div>
              <div>
                <span className="text-muted-foreground">父</span>
                <span className="ml-2">{horse.sire}</span>
              </div>
            </div>
          </div>
        </motion.section>

        {/* AI Comment */}
        <motion.section {...fadeIn} transition={{ delay: 0.3 }}>
          <div className="bg-card rounded-xl p-4 border border-white/5">
            <h2 className="text-sm font-bold text-sakura-pink mb-2">
              💡 AI分析
            </h2>
            <div className="bg-navy/50 rounded-lg p-3 border-l-4 border-sakura-pink">
              <p className="text-sm leading-relaxed">{horse.comment}</p>
            </div>
          </div>
        </motion.section>

        {/* Risk */}
        <motion.section {...fadeIn} transition={{ delay: 0.4 }}>
          <div className="bg-red-950/30 rounded-xl p-4 border border-red-500/20">
            <h2 className="text-sm font-bold text-red-400 mb-2">
              ⚠️ リスク要因
            </h2>
            <p className="text-sm text-red-300">{horse.risk}</p>
          </div>
        </motion.section>

        {/* Recent Results Timeline */}
        <motion.section {...fadeIn} transition={{ delay: 0.5 }}>
          <div className="bg-card rounded-xl p-4 border border-white/5">
            <h2 className="text-sm font-bold text-muted-foreground mb-3">
              近走成績
            </h2>
            <div className="flex items-center justify-center gap-4">
              {[...horse.last3_results].reverse().map((result, i) => (
                <div key={i} className="flex flex-col items-center gap-1">
                  <span className="text-[10px] text-muted-foreground">
                    {i === 0 ? "3走前" : i === 1 ? "2走前" : "前走"}
                  </span>
                  <span
                    className={`w-12 h-12 rounded-full border-2 flex items-center justify-center text-sm font-bold ${resultColor(result)}`}
                  >
                    {result}
                  </span>
                  {i < 2 && (
                    <span className="text-muted-foreground text-xs absolute translate-x-8">
                      →
                    </span>
                  )}
                </div>
              ))}
            </div>
          </div>
        </motion.section>

        {/* Score Details */}
        <motion.section {...fadeIn} transition={{ delay: 0.6 }}>
          <div className="bg-card rounded-xl p-4 border border-white/5">
            <h2 className="text-sm font-bold text-muted-foreground mb-3">
              スコア詳細
            </h2>

            {/* Speed Index */}
            <div className="text-center mb-4">
              <p className="text-xs text-muted-foreground">スピード指数</p>
              <p className="text-4xl font-mono font-bold text-sakura-pink">
                {horse.speed_index}
              </p>
            </div>

            {/* Win/Show probability bars */}
            <div className="space-y-3 mb-4">
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-muted-foreground">勝率</span>
                  <span className="font-mono text-sakura-pink">
                    {(horse.win_prob * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="bg-white/5 rounded-full h-3 overflow-hidden">
                  <div
                    className="h-full bg-sakura-pink rounded-full"
                    style={{ width: `${horse.win_prob * 100}%` }}
                  />
                </div>
              </div>
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-muted-foreground">複勝率</span>
                  <span className="font-mono text-sakura-light">
                    {(horse.show_prob * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="bg-white/5 rounded-full h-3 overflow-hidden">
                  <div
                    className="h-full bg-sakura-pink/60 rounded-full"
                    style={{ width: `${horse.show_prob * 100}%` }}
                  />
                </div>
              </div>
            </div>

            {/* Odds Input */}
            <div className="bg-navy/50 rounded-lg p-3 mb-4 border border-white/10">
              <h3 className="text-xs font-bold text-muted-foreground mb-2">
                🔧 オッズ入力
              </h3>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-[10px] text-muted-foreground">単勝オッズ</label>
                  <OddsInput
                    value={currentOdds.win}
                    onChange={(v) => updateOdds(horse.horse_number, v, currentOdds.show)}
                    className="w-full bg-navy border border-white/10 rounded px-2 py-1.5 text-sm font-mono text-right mt-1"
                  />
                </div>
                <div>
                  <label className="text-[10px] text-muted-foreground">複勝オッズ</label>
                  <OddsInput
                    value={currentOdds.show}
                    onChange={(v) => updateOdds(horse.horse_number, currentOdds.win, v)}
                    className="w-full bg-navy border border-white/10 rounded px-2 py-1.5 text-sm font-mono text-right mt-1"
                  />
                </div>
              </div>
            </div>

            {/* Edge Analysis */}
            {live && (
              <div className="bg-navy/50 rounded-lg p-3 mb-4 border border-white/10">
                <h3 className="text-xs font-bold text-muted-foreground mb-2">
                  📊 エッジ分析
                </h3>
                <div className="space-y-2">
                  <div>
                    <div className="flex justify-between text-[10px] mb-1">
                      <span className="text-sakura-pink">AI確率</span>
                      <span className="font-mono text-sakura-pink">
                        {(horse.win_prob * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="bg-white/5 rounded-full h-2.5 overflow-hidden">
                      <div
                        className="h-full bg-sakura-pink rounded-full"
                        style={{ width: `${Math.min(horse.win_prob * 100, 100)}%` }}
                      />
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between text-[10px] mb-1">
                      <span className="text-blue-400">市場確率</span>
                      <span className="font-mono text-blue-400">
                        {(live.market_prob * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="bg-white/5 rounded-full h-2.5 overflow-hidden">
                      <div
                        className="h-full bg-blue-400 rounded-full"
                        style={{ width: `${Math.min(live.market_prob * 100, 100)}%` }}
                      />
                    </div>
                  </div>
                  <div className="text-center pt-1">
                    {(() => {
                      const edge = horse.win_prob - live.market_prob;
                      const edgePct = (edge * 100).toFixed(1);
                      return edge > 0 ? (
                        <span className="text-green-400 font-mono text-sm font-bold">
                          +{edgePct}% エッジあり
                        </span>
                      ) : (
                        <span className="text-red-400 font-mono text-sm">
                          {edgePct}% 市場が上
                        </span>
                      );
                    })()}
                  </div>
                </div>
              </div>
            )}

            {/* EV */}
            <div className="grid grid-cols-2 gap-3">
              <div className="bg-navy/50 rounded-lg p-3 text-center">
                <p className="text-xs text-muted-foreground mb-1">
                  単勝期待値
                </p>
                <p
                  className={`text-2xl font-mono font-bold ${
                    displayEvWin >= 1.0 ? "text-green-400" : "text-red-400"
                  }`}
                >
                  {displayEvWin.toFixed(2)}
                </p>
              </div>
              <div className="bg-navy/50 rounded-lg p-3 text-center">
                <p className="text-xs text-muted-foreground mb-1">
                  複勝期待値
                </p>
                <p
                  className={`text-2xl font-mono font-bold ${
                    displayEvShow >= 1.0 ? "text-green-400" : "text-red-400"
                  }`}
                >
                  {displayEvShow.toFixed(2)}
                </p>
              </div>
            </div>
          </div>
        </motion.section>
      </main>

      <Navbar />
    </div>
  );
}
