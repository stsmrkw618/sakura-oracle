"use client";

import { useState, useMemo, useEffect } from "react";
import { motion } from "framer-motion";
import Navbar from "@/components/Navbar";
import OddsInput from "@/components/OddsInput";
import { useOdds } from "@/context/OddsContext";
import { useRace } from "@/context/RaceContext";

const fadeIn = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
};

/** 組合せオッズ入力用コンポーネント */
function ComboOddsInput({
  comboKey,
  comboOddsMap,
  updateComboOdds,
}: {
  comboKey: string;
  comboOddsMap: Record<string, number>;
  updateComboOdds: (key: string, odds: number) => void;
}) {
  const currentValue = comboOddsMap[comboKey];
  const [localValue, setLocalValue] = useState(
    currentValue != null ? String(currentValue) : ""
  );

  // 外部からの変更（モード切替時など）を同期
  useEffect(() => {
    if (currentValue != null) {
      setLocalValue(String(currentValue));
    }
  }, [currentValue]);

  return (
    <input
      type="number"
      step="0.1"
      min="1"
      value={localValue}
      placeholder="--"
      onChange={(e) => {
        setLocalValue(e.target.value);
        const v = parseFloat(e.target.value);
        if (!isNaN(v) && v > 0) {
          updateComboOdds(comboKey, v);
        }
      }}
      className="w-16 bg-navy/50 border border-white/10 rounded px-2 py-1 text-xs font-mono text-right"
    />
  );
}

/** BT実績データ（v10, 50レース） */
const BT_STATS = {
  box: {
    trio: { label: "三連複BOX(5)", hit: 34, roi: 474, cost: "10通り×¥100", sharpe: "0.57" },
    quinella: { label: "馬連BOX(3)", hit: 32, roi: 550, cost: "3通り×¥100", sharpe: "0.56" },
    wide: { label: "ワイド(◎-○)", hit: 30, roi: 465, cost: "1通り×¥100", sharpe: "--" },
  },
  nagashi: {
    trio: { label: "三連複◎軸流し(6)", hit: 28, roi: 589, cost: "6通り×¥100", sharpe: "0.52" },
    quinella: { label: "馬連◎軸流し(4)", hit: 36, roi: 452, cost: "4通り×¥100", sharpe: "0.58" },
    wide: { label: "ワイド(◎-○)", hit: 30, roi: 465, cost: "1通り×¥100", sharpe: "--" },
  },
} as const;

export default function BetGuidePage() {
  const { predictions } = useRace();
  const {
    liveHorses,
    liveBets,
    oddsMap,
    updateOdds,
    resetOdds,
    comboOddsMap,
    updateComboOdds,
    resetComboOdds,
    comboMode,
    setComboMode,
    strategyMode,
    setStrategyMode,
  } = useOdds();
  const [budget, setBudget] = useState(3000);
  const [glossaryOpen, setGlossaryOpen] = useState<string | null>(null);
  const [detailOpen, setDetailOpen] = useState(false);

  // localStorage から予算を復元
  useEffect(() => {
    try {
      const stored = localStorage.getItem("sakura-oracle-budget");
      if (stored) setBudget(Number(stored));
    } catch { /* ignore */ }
  }, []);

  const hasAnyChange = liveHorses.some((h) => h.oddsChanged);
  const hasComboOdds = Object.keys(comboOddsMap).length > 0;

  // 予算に応じたスケーリング
  const scaledBets = useMemo(() => {
    if (liveBets.length === 0) return [];

    // Kelly重み配分: BT事前重みでオッズ未入力時も推定配分
    const weights = liveBets.map((b) => {
      if (b.kelly <= 0) return 0;
      // EV確定 & EV < 1.0 → 見送り
      if (b.evReliable && b.ev < 1.0) return 0;
      // evReliable=true → Kelly / evReliable=false → Kelly × BT事前重み
      const backtestPrior = b.backtestRoi / 10;
      return b.evReliable ? b.kelly : b.kelly * backtestPrior;
    });
    const totalWeight = weights.reduce((s, w) => s + w, 0);

    if (totalWeight === 0) {
      return liveBets.map((b) => ({ ...b, scaledAmount: 0 }));
    }

    // 重みに比例して予算配分（¥100単位）
    const amounts = weights.map((w) => {
      if (w === 0) return 0;
      return Math.max(100, Math.round((w / totalWeight) * budget / 100) * 100);
    });

    // 合計 = 予算に調整
    let total = amounts.reduce((s, a) => s + a, 0);
    const maxIdx = weights.indexOf(Math.max(...weights));
    if (total < budget) {
      amounts[maxIdx] += budget - total;
      total = budget;
    }
    while (total > budget) {
      let reduced = false;
      const ascending = weights
        .map((w, i) => ({ w, i }))
        .filter((x) => x.w > 0)
        .sort((a, b) => a.w - b.w);
      for (const { i } of ascending) {
        if (amounts[i] > 100) {
          amounts[i] -= 100;
          total -= 100;
          reduced = true;
          break;
        }
      }
      if (!reduced) break;
    }

    return liveBets.map((bet, i) => ({ ...bet, scaledAmount: amounts[i] }));
  }, [budget, liveBets]);

  const totalInvestment = scaledBets.reduce((s, b) => s + b.scaledAmount, 0);

  // 期待リターン
  const expectedReturn = scaledBets.reduce((s, b) => {
    if (b.scaledAmount === 0) return s;
    const effectiveEv = b.evReliable ? b.ev : b.backtestRoi;
    return s + b.scaledAmount * effectiveEv;
  }, 0);
  const roi = totalInvestment > 0 ? (expectedReturn / totalInvestment - 1) * 100 : 0;
  const allReliable = scaledBets.filter((b) => b.scaledAmount > 0).every((b) => b.evReliable);

  // 的中時リターン: 各組合せ馬券が的中した場合の最大払戻
  const maxHitReturn = useMemo(() => {
    let best = 0;
    for (const b of scaledBets) {
      if (b.scaledAmount === 0) continue;
      if (b.odds && b.odds > 0) {
        const ret = b.scaledAmount * b.odds;
        if (ret > best) best = ret;
      }
    }
    return best;
  }, [scaledBets]);

  // 馬券タイプ別
  const activeBets = scaledBets.filter((b) => b.scaledAmount > 0);
  const skippedBets = scaledBets.filter((b) => b.scaledAmount === 0 && b.evReliable && b.ev < 1.0);

  const trioBets = scaledBets.filter((b) => b.type === "三連複");
  const quinellaBets = scaledBets.filter((b) => b.type === "馬連");
  const wideBets = scaledBets.filter((b) => b.type === "ワイド");
  const winBets = scaledBets.filter((b) => b.type === "単勝");

  // 対象馬（オッズ入力用）
  const targetHorses = useMemo(() => {
    return predictions.predictions.filter((h) =>
      ["◎", "○", "▲", "△"].includes(h.mark)
    );
  }, [predictions]);

  const stats = BT_STATS[comboMode];

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
          {(hasAnyChange || hasComboOdds) && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-sakura-pink/20 text-sakura-pink font-bold">
              LIVE
            </span>
          )}
        </div>
      </motion.header>

      <main className="px-4 py-4 space-y-5">
        {/* 予算 + 戦略モード */}
        <motion.section {...fadeIn} transition={{ delay: 0.05 }}>
          <div className="bg-card rounded-xl p-4 border border-white/5">
            {/* 予算スライダー */}
            <div className="mb-4">
              <div className="flex justify-between text-sm mb-2">
                <span className="text-muted-foreground font-bold">予算</span>
                <span className="font-mono text-gold text-xl font-bold">
                  ¥{budget.toLocaleString()}
                </span>
              </div>
              <input
                type="range"
                min={1000}
                max={30000}
                step={500}
                value={budget}
                onChange={(e) => {
                  const v = Number(e.target.value);
                  setBudget(v);
                  try { localStorage.setItem("sakura-oracle-budget", String(v)); } catch { /* ignore */ }
                }}
                className="w-full h-2 bg-white/10 rounded-full appearance-none cursor-pointer accent-gold"
              />
              <div className="flex justify-between text-[10px] text-muted-foreground mt-1">
                <span>¥1,000</span>
                <span>¥30,000</span>
              </div>
            </div>

            {/* 戦略モード切替 */}
            <div className="grid grid-cols-2 gap-2 mb-2">
              <button
                onClick={() => setStrategyMode("aggressive")}
                className={`rounded-lg p-3 text-left border transition-all ${
                  strategyMode === "aggressive"
                    ? "border-gold bg-gold/10"
                    : "border-white/10 bg-navy/50"
                }`}
              >
                <span className="text-sm font-bold block mb-1">
                  {strategyMode === "aggressive" ? "● " : "○ "}強気
                </span>
                <span className="text-[10px] text-muted-foreground leading-relaxed block">
                  穴馬で高配当を狙う
                </span>
              </button>
              <button
                onClick={() => setStrategyMode("stable")}
                className={`rounded-lg p-3 text-left border transition-all ${
                  strategyMode === "stable"
                    ? "border-sakura-pink bg-sakura-pink/10"
                    : "border-white/10 bg-navy/50"
                }`}
              >
                <span className="text-sm font-bold block mb-1">
                  {strategyMode === "stable" ? "● " : "○ "}安定
                </span>
                <span className="text-[10px] text-muted-foreground leading-relaxed block">
                  人気馬を軸に手堅く
                </span>
              </button>
            </div>

            {/* 買い方切替 */}
            <div className="grid grid-cols-2 gap-2">
              <button
                onClick={() => setComboMode("box")}
                className={`rounded-lg p-3 text-left border transition-all ${
                  comboMode === "box"
                    ? "border-sakura-pink bg-sakura-pink/10"
                    : "border-white/10 bg-navy/50"
                }`}
              >
                <span className="text-sm font-bold block mb-1">
                  {comboMode === "box" ? "● " : "○ "}BOX
                </span>
                <span className="text-[10px] text-muted-foreground leading-relaxed block">
                  Sharpe最高。安定重視
                </span>
              </button>
              <button
                onClick={() => setComboMode("nagashi")}
                className={`rounded-lg p-3 text-left border transition-all ${
                  comboMode === "nagashi"
                    ? "border-gold bg-gold/10"
                    : "border-white/10 bg-navy/50"
                }`}
              >
                <span className="text-sm font-bold block mb-1">
                  {comboMode === "nagashi" ? "● " : "○ "}◎軸流し
                </span>
                <span className="text-[10px] text-muted-foreground leading-relaxed block">
                  投資効率最高。少額高配当
                </span>
              </button>
            </div>

            {/* BT実績バー（コンパクト） */}
            <div className="mt-3 space-y-1.5">
              {[stats.trio, stats.quinella, stats.wide].map((s) => (
                <div key={s.label} className="flex items-center gap-2">
                  <span className="text-[10px] text-muted-foreground w-32 shrink-0 truncate">
                    {s.label}
                  </span>
                  <div className="flex-1 h-1.5 bg-white/5 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full bg-gold"
                      style={{ width: `${Math.min(100, s.roi / 10)}%` }}
                    />
                  </div>
                  <span className="text-[10px] font-mono text-gold w-10 text-right shrink-0">
                    {s.roi}%
                  </span>
                  <span className="text-[10px] text-muted-foreground w-8 text-right shrink-0">
                    的中{s.hit}%
                  </span>
                </div>
              ))}
            </div>
            <p className="text-[9px] text-muted-foreground mt-2">
              ※ BT実績: 50レース実配当ベース回収率(v10)。強気モード基準
            </p>
          </div>
        </motion.section>

        {/* 購入リスト + KPI */}
        {activeBets.length > 0 && (
          <motion.section {...fadeIn} transition={{ delay: 0.1 }}>
            <div className="bg-card rounded-xl p-4 border border-white/5">
              <h2 className="text-sm font-bold mb-3">
                📋 購入リスト（{activeBets.length}点）
              </h2>

              {/* 買い目一覧 */}
              <div className="space-y-1.5 mb-4">
                {activeBets.map((bet, i) => {
                  const hitReturn = bet.odds && bet.odds > 0
                    ? Math.round(bet.scaledAmount * bet.odds)
                    : null;
                  return (
                    <div
                      key={`${bet.type}-${bet.targets}-${i}`}
                      className="flex items-center justify-between text-sm"
                    >
                      <div className="flex items-center gap-1.5 flex-1 min-w-0">
                        <span className="text-sakura-pink font-bold text-xs shrink-0 w-10">
                          {bet.type}
                        </span>
                        <span className="text-white font-mono text-xs shrink-0">
                          {bet.targets}
                        </span>
                        {bet.evReliable && (
                          <span
                            className={`text-[9px] font-mono shrink-0 ${
                              bet.ev >= 1.5
                                ? "text-gold font-bold"
                                : bet.ev >= 1.0
                                  ? "text-green-400"
                                  : "text-red-400"
                            }`}
                          >
                            EV{bet.ev.toFixed(1)}
                          </span>
                        )}
                      </div>
                      <div className="flex items-center gap-2 shrink-0">
                        {hitReturn != null && (
                          <span className="text-[9px] text-green-400/60 font-mono">
                            的中¥{hitReturn.toLocaleString()}
                          </span>
                        )}
                        <span className="font-mono text-gold font-bold w-16 text-right">
                          ¥{bet.scaledAmount.toLocaleString()}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* 見送り馬券 */}
              {skippedBets.length > 0 && (
                <div className="mb-4">
                  <p className="text-[10px] text-muted-foreground mb-1">
                    EV &lt; 1.0 で見送り（{skippedBets.length}点）:
                  </p>
                  <div className="space-y-0.5">
                    {skippedBets.map((bet, i) => (
                      <div
                        key={`skip-${bet.type}-${bet.targets}-${i}`}
                        className="flex items-center text-[10px] text-muted-foreground/50 line-through"
                      >
                        <span className="w-10 shrink-0">{bet.type}</span>
                        <span className="font-mono">{bet.targets}</span>
                        <span className="ml-2 font-mono">EV{bet.ev.toFixed(2)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* KPI */}
              <div className="grid grid-cols-3 gap-2 text-center border-t border-white/10 pt-3">
                <div>
                  <p className="text-[10px] text-muted-foreground mb-0.5">合計投資</p>
                  <p className="font-mono text-sm font-bold">
                    ¥{totalInvestment.toLocaleString()}
                  </p>
                </div>
                <div>
                  <p className="text-[10px] text-muted-foreground mb-0.5">
                    期待リターン{!allReliable && <span className="text-orange-400">*</span>}
                  </p>
                  <p className="font-mono text-sm font-bold text-gold">
                    ¥{Math.round(expectedReturn).toLocaleString()}
                  </p>
                </div>
                <div>
                  <p className="text-[10px] text-muted-foreground mb-0.5">期待ROI</p>
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

              {maxHitReturn > 0 && (
                <p className="text-[10px] text-green-400/70 mt-2 text-center">
                  最高的中時 ¥{Math.round(maxHitReturn).toLocaleString()} 回収
                </p>
              )}

              <p className="text-[9px] text-muted-foreground mt-2">
                {allReliable
                  ? "全馬券のオッズ入力済み — 実EVベース配分"
                  : "※ BT実績ベースの推定配分。組合せオッズ入力で実EV確定値に切替わります"}
              </p>
            </div>
          </motion.section>
        )}

        {/* 組合せ馬券詳細（折りたたみ） */}
        <motion.section {...fadeIn} transition={{ delay: 0.15 }}>
          <button
            onClick={() => setDetailOpen(!detailOpen)}
            className="w-full bg-card rounded-xl p-4 border border-white/5 text-left"
          >
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-bold">
                🎯 組合せ馬券の詳細・オッズ入力
              </h2>
              <span className="text-muted-foreground text-xs">
                {detailOpen ? "▲ 閉じる" : "▼ 開く"}
              </span>
            </div>
            <p className="text-[10px] text-muted-foreground mt-1">
              JRAオッズを入力するとEV計算 → 配分が自動最適化されます
            </p>
          </button>
        </motion.section>

        {detailOpen && (
          <>
            {/* 三連複 */}
            {trioBets.length > 0 && (
              <motion.section {...fadeIn}>
                <h3 className="text-xs font-bold text-muted-foreground mb-2">
                  {stats.trio.label}（{trioBets.length}通り）
                  <span className="font-normal ml-2">
                    BT: 的中{stats.trio.hit}% / 回収{stats.trio.roi}%
                  </span>
                </h3>
                <div className="space-y-2">
                  {trioBets.map((bet) => (
                    <ComboBetCard
                      key={bet.comboKey || `trio-${bet.targets}`}
                      bet={bet}
                      comboOddsMap={comboOddsMap}
                      updateComboOdds={updateComboOdds}
                    />
                  ))}
                </div>
              </motion.section>
            )}

            {/* 馬連 */}
            {quinellaBets.length > 0 && (
              <motion.section {...fadeIn}>
                <h3 className="text-xs font-bold text-muted-foreground mb-2">
                  {stats.quinella.label}（{quinellaBets.length}通り）
                  <span className="font-normal ml-2">
                    BT: 的中{stats.quinella.hit}% / 回収{stats.quinella.roi}%
                  </span>
                </h3>
                <div className="space-y-2">
                  {quinellaBets.map((bet) => (
                    <ComboBetCard
                      key={bet.comboKey || `quinella-${bet.targets}`}
                      bet={bet}
                      comboOddsMap={comboOddsMap}
                      updateComboOdds={updateComboOdds}
                    />
                  ))}
                </div>
              </motion.section>
            )}

            {/* ワイド */}
            {wideBets.length > 0 && (
              <motion.section {...fadeIn}>
                <h3 className="text-xs font-bold text-muted-foreground mb-2">
                  ワイド(◎-○)
                  <span className="font-normal ml-2">
                    BT: 的中{stats.wide.hit}% / 回収{stats.wide.roi}%
                  </span>
                </h3>
                <div className="space-y-2">
                  {wideBets.map((bet) => (
                    <ComboBetCard
                      key={bet.comboKey || `wide-${bet.targets}`}
                      bet={bet}
                      comboOddsMap={comboOddsMap}
                      updateComboOdds={updateComboOdds}
                    />
                  ))}
                </div>
              </motion.section>
            )}

            {/* 単勝 */}
            {winBets.length > 0 && (
              <motion.section {...fadeIn}>
                <h3 className="text-xs font-bold text-muted-foreground mb-2">
                  単勝（{winBets.length}点）
                  <span className="font-normal ml-2">
                    BT: 的中46% / 回収265%
                  </span>
                </h3>
                <div className="space-y-2">
                  {winBets.map((bet, i) => (
                    <div
                      key={`win-${i}`}
                      className="bg-card rounded-xl p-3 border border-white/5"
                    >
                      <div className="flex items-center justify-between">
                        <div>
                          <span className="text-xs font-bold text-sakura-pink">
                            単勝
                          </span>
                          <span className="text-xs text-muted-foreground ml-2">
                            {bet.targets}
                          </span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span
                            className={`text-xs font-mono ${
                              bet.ev >= 1.5
                                ? "text-gold font-bold"
                                : bet.ev >= 1.0
                                  ? "text-green-400"
                                  : "text-muted-foreground"
                            }`}
                          >
                            EV {bet.ev.toFixed(2)}
                          </span>
                          <span className="font-mono text-gold font-bold text-sm">
                            ¥{bet.scaledAmount.toLocaleString()}
                          </span>
                        </div>
                      </div>
                      {bet.odds && bet.scaledAmount > 0 && (
                        <p className="text-[10px] text-green-400/70 mt-1">
                          的中時 ¥{Math.round(bet.odds * bet.scaledAmount).toLocaleString()}
                          （{bet.odds}倍 × ¥{bet.scaledAmount.toLocaleString()}）
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              </motion.section>
            )}

            {hasComboOdds && (
              <div className="flex justify-end">
                <button
                  onClick={resetComboOdds}
                  className="text-[10px] text-orange-400 underline"
                >
                  組合せオッズをリセット
                </button>
              </div>
            )}
          </>
        )}

        {/* 単勝・複勝オッズ更新 */}
        <motion.section {...fadeIn} transition={{ delay: 0.2 }}>
          <div className="bg-card rounded-xl p-4 border border-white/5">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-sm font-bold text-muted-foreground">
                🔧 単勝・複勝オッズ更新
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
              当日のオッズを入力すると全ページの期待値・印・買い目が即時連動します
            </p>

            {/* ヘッダー */}
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

        {/* 用語解説 */}
        <motion.section {...fadeIn} transition={{ delay: 0.25 }}>
          <div className="bg-card rounded-xl p-4 border border-white/5">
            <h2 className="text-sm font-bold text-muted-foreground mb-3">
              📖 初心者向け用語解説
            </h2>

            {[
              {
                key: "ev",
                title: "期待値（EV）とは？",
                content:
                  "期待値 = AI予測確率 × オッズ。1.0を超えればプラス期待値（長期的に利益が出る賭け）。例: 勝率20%で8倍なら 0.20 × 8.0 = 1.6。EV 1.6は「100円賭けると平均160円返ってくる」という意味。",
              },
              {
                key: "kelly",
                title: "Kelly基準とは？",
                content:
                  "最適な賭け金の割合を数学的に決める手法。エッジ（優位性）が大きいほど多く、小さいほど少なく賭ける。当システムでは安全な1/4 Kellyを採用。",
              },
              {
                key: "combo-ev",
                title: "組合せ馬券のEV計算とは？",
                content:
                  "AIが各馬の勝率を予測 → Harvilleモデルで2頭・3頭の同時入着確率を算出 → JRAオッズを掛けてEVを計算。EV > 1.0なら「買い」判定。",
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
              {
                key: "nagashi",
                title: "軸流しとは？",
                content:
                  "1頭を「軸」として固定し、残りの相手馬との組み合わせを買う方式。BOXより点数を絞れるため、1点あたりの投資額を増やせる。当システムではAI本命◎を軸に設定。",
              },
            ].map((item) => (
              <div key={item.key} className="mb-2">
                <button
                  onClick={() =>
                    setGlossaryOpen(glossaryOpen === item.key ? null : item.key)
                  }
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

/** 組合せ馬券のカードコンポーネント（コンパクト版） */
function ComboBetCard({
  bet,
  comboOddsMap,
  updateComboOdds,
}: {
  bet: {
    type: string;
    targets: string;
    description: string;
    scaledAmount: number;
    ev: number;
    evReliable: boolean;
    odds: number | null;
    kelly: number;
    backtestRoi: number;
    comboProb?: number;
    comboKey?: string;
  };
  comboOddsMap: Record<string, number>;
  updateComboOdds: (key: string, odds: number) => void;
}) {
  const hasOdds = bet.comboKey ? comboOddsMap[bet.comboKey] != null : false;
  const isSkipped = bet.evReliable && bet.ev < 1.0;

  return (
    <div
      className={`bg-card rounded-xl p-3 border ${
        isSkipped
          ? "border-red-500/20 opacity-50"
          : "border-white/5"
      }`}
    >
      <div className="flex items-center justify-between mb-1">
        <div className="flex items-center gap-2">
          <span className="text-xs font-bold text-sakura-pink">
            {bet.type} {bet.targets}
          </span>
          {bet.comboProb != null && (
            <span className="text-[9px] text-muted-foreground font-mono">
              P={( bet.comboProb * 100).toFixed(1)}%
            </span>
          )}
        </div>
        <span className={`font-mono text-sm ${
          isSkipped
            ? "text-muted-foreground line-through"
            : bet.scaledAmount > 0
              ? "text-gold font-bold"
              : "text-muted-foreground"
        }`}>
          {isSkipped
            ? "見送り"
            : bet.scaledAmount > 0
              ? `¥${bet.scaledAmount.toLocaleString()}`
              : "---"}
        </span>
      </div>

      <p className="text-[10px] text-muted-foreground mb-2">
        {bet.description}
      </p>

      {/* オッズ入力 + EV */}
      {bet.comboKey && (
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-muted-foreground">オッズ:</span>
          <ComboOddsInput
            comboKey={bet.comboKey}
            comboOddsMap={comboOddsMap}
            updateComboOdds={updateComboOdds}
          />
          {hasOdds ? (
            <span
              className={`text-xs font-mono font-bold ${
                bet.ev >= 1.5
                  ? "text-gold"
                  : bet.ev >= 1.0
                    ? "text-green-400"
                    : "text-red-400"
              }`}
            >
              EV {bet.ev.toFixed(2)}
              {bet.ev < 1.0 && " 見送り"}
            </span>
          ) : (
            <span className="text-[10px] text-muted-foreground">
              未入力
            </span>
          )}
        </div>
      )}
    </div>
  );
}
