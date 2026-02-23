"use client";

import { useState, useMemo } from "react";
import { motion } from "framer-motion";
import predictions from "@/data/predictions.json";
import Navbar from "@/components/Navbar";
import OddsInput from "@/components/OddsInput";
import { useOdds } from "@/context/OddsContext";

const fadeIn = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
};

export default function BetGuidePage() {
  const { liveHorses, liveBets, oddsMap, updateOdds, resetOdds } = useOdds();
  const [budget, setBudget] = useState(3000);
  const [glossaryOpen, setGlossaryOpen] = useState<string | null>(null);

  const hasAnyChange = liveHorses.some((h) => h.oddsChanged);

  const totalEv = liveBets.reduce((s, b) => s + b.ev, 0);

  const scaledBets = useMemo(() => {
    return liveBets.map((bet) => {
      const ratio = totalEv > 0 ? bet.ev / totalEv : 0;
      const amount = Math.round((budget * ratio) / 100) * 100;
      return { ...bet, scaledAmount: Math.max(amount, 100) };
    });
  }, [budget, liveBets, totalEv]);

  const totalInvestment = scaledBets.reduce((s, b) => s + b.scaledAmount, 0);
  const expectedReturn = scaledBets.reduce(
    (s, b) => s + b.scaledAmount * b.ev,
    0
  );
  const roi = totalInvestment > 0 ? (expectedReturn / totalInvestment - 1) * 100 : 0;

  const toggleGlossary = (key: string) => {
    setGlossaryOpen(glossaryOpen === key ? null : key);
  };

  // Horses that originally have marks ◎○▲△ (use original marks so they don't vanish during editing)
  const targetHorses = useMemo(() => {
    return predictions.predictions.filter((h) =>
      ["◎", "○", "▲", "△"].includes(h.mark)
    );
  }, []);

  return (
    <div className="min-h-screen bg-navy-dark pb-20">
      <motion.header
        {...fadeIn}
        className="sticky top-0 z-40 bg-navy-dark/95 backdrop-blur-md border-b border-white/5 px-4 py-3"
      >
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-lg font-bold">💰 買い目ガイド</h1>
            <p className="text-xs text-muted-foreground">
              {predictions.recommendations.headline}
            </p>
          </div>
          {hasAnyChange && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-sakura-pink/20 text-sakura-pink font-bold">
              LIVE
            </span>
          )}
        </div>
      </motion.header>

      <main className="px-4 py-4 space-y-5">
        {/* Recommended Bets */}
        <motion.section {...fadeIn} transition={{ delay: 0.1 }}>
          <h2 className="text-sm font-bold text-muted-foreground mb-3">
            推奨買い目
          </h2>
          <div className="space-y-3">
            {scaledBets.map((bet, i) => (
              <div
                key={i}
                className="bg-card rounded-xl p-4 border border-white/5"
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-bold text-sakura-pink">
                    {bet.type}
                  </span>
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-[10px] text-muted-foreground">
                      Kelly {((bet as Record<string, unknown>).kelly as number
                        ? ((bet as Record<string, unknown>).kelly as number * 100).toFixed(1)
                        : "—")}%
                    </span>
                    <span className="font-mono text-sm">
                      {bet.ev >= 1.5 && "🔥 "}
                      <span
                        className={
                          bet.ev >= 1.5 ? "text-gold font-bold" : "text-white"
                        }
                      >
                        EV {bet.ev.toFixed(2)}
                      </span>
                    </span>
                  </div>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">
                    対象: <span className="text-white">{bet.targets}</span>
                  </span>
                  <span className="font-mono text-gold">
                    ¥{bet.scaledAmount.toLocaleString()}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </motion.section>

        {/* Investment Simulator */}
        <motion.section {...fadeIn} transition={{ delay: 0.2 }}>
          <div className="bg-card rounded-xl p-4 border border-white/5">
            <h2 className="text-sm font-bold text-muted-foreground mb-3">
              投資シミュレーター
            </h2>

            <div className="mb-4">
              <div className="flex justify-between text-sm mb-2">
                <span className="text-muted-foreground">予算</span>
                <span className="font-mono text-gold text-lg font-bold">
                  ¥{budget.toLocaleString()}
                </span>
              </div>
              <input
                type="range"
                min={1000}
                max={10000}
                step={500}
                value={budget}
                onChange={(e) => setBudget(Number(e.target.value))}
                className="w-full h-2 bg-white/10 rounded-full appearance-none cursor-pointer accent-sakura-pink"
              />
              <div className="flex justify-between text-[10px] text-muted-foreground mt-1">
                <span>¥1,000</span>
                <span>¥10,000</span>
              </div>
            </div>

            <div className="grid grid-cols-3 gap-3 text-center">
              <div className="bg-navy/50 rounded-lg p-3">
                <p className="text-[10px] text-muted-foreground mb-1">合計投資</p>
                <p className="font-mono text-sm font-bold">
                  ¥{totalInvestment.toLocaleString()}
                </p>
              </div>
              <div className="bg-navy/50 rounded-lg p-3">
                <p className="text-[10px] text-muted-foreground mb-1">期待リターン</p>
                <p className="font-mono text-sm font-bold text-gold">
                  ¥{Math.round(expectedReturn).toLocaleString()}
                </p>
              </div>
              <div className="bg-navy/50 rounded-lg p-3">
                <p className="text-[10px] text-muted-foreground mb-1">期待ROI</p>
                <p
                  className={`font-mono text-lg font-bold ${
                    roi >= 0 ? "text-green-400" : "text-red-400"
                  }`}
                >
                  {roi >= 0 ? "+" : ""}
                  {roi.toFixed(0)}%
                </p>
              </div>
            </div>
          </div>
        </motion.section>

        {/* Odds Manual Update */}
        <motion.section {...fadeIn} transition={{ delay: 0.3 }}>
          <div className="bg-card rounded-xl p-4 border border-white/5">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-sm font-bold text-muted-foreground">
                🔧 オッズ手動更新
              </h2>
              {hasAnyChange && (
                <button
                  onClick={resetOdds}
                  className="text-[10px] text-red-400 underline"
                >
                  リセット
                </button>
              )}
            </div>
            <p className="text-xs text-muted-foreground mb-3">
              当日のオッズを入力すると全ページの期待値・印・買い目が自動更新されます
            </p>

            {/* Column Headers */}
            <div className="flex items-center gap-2 mb-1 text-[10px] text-muted-foreground">
              <span className="w-6"></span>
              <span className="flex-1"></span>
              <span className="w-16 text-center">単勝</span>
              <span className="w-16 text-center">複勝</span>
              <span className="w-12 text-right">EV</span>
            </div>

            <div className="space-y-2 max-h-60 overflow-y-auto">
              {targetHorses.map((h) => {
                const live = liveHorses.find(
                  (l) => l.horse_number === h.horse_number
                );
                const currentWin = oddsMap[h.horse_number]?.win ?? h.odds.win;
                const currentShow = oddsMap[h.horse_number]?.show ?? h.odds.show;
                return (
                  <div key={h.horse_number} className="flex items-center gap-2">
                    <span className="text-xs text-muted-foreground w-6">
                      {h.horse_number}
                    </span>
                    <span className="text-xs flex-1 truncate">
                      {h.horse_name}
                    </span>
                    <OddsInput
                      value={currentWin}
                      onChange={(v) => updateOdds(h.horse_number, v, currentShow)}
                      className="w-16 bg-navy/50 border border-white/10 rounded px-2 py-1 text-xs font-mono text-right"
                    />
                    <OddsInput
                      value={currentShow}
                      onChange={(v) => updateOdds(h.horse_number, currentWin, v)}
                      className="w-16 bg-navy/50 border border-white/10 rounded px-2 py-1 text-xs font-mono text-right"
                    />
                    <span
                      className={`text-xs font-mono w-12 text-right ${
                        (live?.ev_win ?? h.ev_win) >= 1.0
                          ? "text-green-400"
                          : "text-red-400"
                      }`}
                    >
                      {(live?.ev_win ?? h.ev_win).toFixed(2)}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        </motion.section>

        {/* Glossary */}
        <motion.section {...fadeIn} transition={{ delay: 0.4 }}>
          <div className="bg-card rounded-xl p-4 border border-white/5">
            <h2 className="text-sm font-bold text-muted-foreground mb-3">
              📖 初心者向け用語解説
            </h2>

            {[
              {
                key: "kelly",
                title: "Kelly基準とは？",
                content:
                  "最適な賭け金の割合を数学的に決める手法。エッジ（優位性）が大きいほど多く、小さいほど少なく賭ける。当システムでは安全な1/4 Kellyを採用し、モデルの過信を防止。",
              },
              {
                key: "ev",
                title: "期待値とは？",
                content:
                  "期待値 = 予測確率 × オッズ。1.0を超えればプラス期待値（長期的に利益が出る賭け）。例: 勝率20%で8倍なら 0.20 × 8.0 = 1.6（強く推奨）。",
              },
              {
                key: "umaren",
                title: "馬連BOXとは？",
                content:
                  "選んだ馬の中から、1着と2着の組み合わせを全通り買う方式。3頭BOXなら3通り（A-B, A-C, B-C）。順番は関係なし。",
              },
              {
                key: "sanrenpuku",
                title: "三連複とは？",
                content:
                  "1着・2着・3着に入る3頭の組み合わせを当てる馬券。順番は不問。5頭BOXなら10通り。高配当が狙える。",
              },
            ].map((item) => (
              <div key={item.key} className="mb-2">
                <button
                  onClick={() => toggleGlossary(item.key)}
                  className="w-full text-left text-sm py-2 flex items-center justify-between"
                >
                  <span>{item.title}</span>
                  <span className="text-muted-foreground">
                    {glossaryOpen === item.key ? "▲" : "▼"}
                  </span>
                </button>
                {glossaryOpen === item.key && (
                  <motion.p
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    className="text-xs text-muted-foreground pb-2 leading-relaxed"
                  >
                    {item.content}
                  </motion.p>
                )}
              </div>
            ))}
          </div>
        </motion.section>
      </main>

      <Navbar />
    </div>
  );
}
