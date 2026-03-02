"use client";

import { useState, useMemo, useEffect } from "react";
import { motion } from "framer-motion";
import Navbar from "@/components/Navbar";
import OddsInput from "@/components/OddsInput";
import { useOdds } from "@/context/OddsContext";
import { useRace } from "@/context/RaceContext";
import { quinellaProb, wideProb, trioProb } from "@/lib/harville";

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

/** ポートフォリオ全体のBT実績ROI（64レース, ¥3,000/R, 実配当ベース） */
const BT_PORTFOLIO_ROI: Record<string, Record<string, { roi: number; hitRate: number }>> = {
  aggressive: {
    box:    { roi: 4.41, hitRate: 0.30 },
    nagashi: { roi: 5.45, hitRate: 0.22 },
  },
  stable: {
    box:    { roi: 2.45, hitRate: 0.30 },
    nagashi: { roi: 2.60, hitRate: 0.25 },
  },
};

/** BT実績データ（v15, 64レース） */
const BT_STATS = {
  box: {
    trio: { label: "三連複BOX(5)", hit: 30, roi: 441, cost: "10通り×¥100", sharpe: "0.57" },
    quinella: { label: "馬連BOX(3)", hit: 19, roi: 91, cost: "3通り×¥100", sharpe: "0.56" },
    wide: { label: "ワイド(◎-○)", hit: 30, roi: 245, cost: "1通り×¥100", sharpe: "--" },
  },
  nagashi: {
    trio: { label: "三連複◎軸流し(6)", hit: 22, roi: 545, cost: "6通り×¥100", sharpe: "0.52" },
    quinella: { label: "馬連◎軸流し(4)", hit: 25, roi: 260, cost: "4通り×¥100", sharpe: "0.58" },
    wide: { label: "ワイド(◎-○)", hit: 30, roi: 245, cost: "1通り×¥100", sharpe: "--" },
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
    normProbs,
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

    // 単勝キャップ: 予算の30%を上限とし、超過分を組合せ馬券に再配分
    const WIN_CAP_RATIO = 0.3;
    const winCap = Math.round(budget * WIN_CAP_RATIO / 100) * 100;
    const winTotal = liveBets.reduce((s, b, i) => s + (b.type === "単勝" ? amounts[i] : 0), 0);
    if (winTotal > winCap && winCap > 0) {
      const scale = winCap / winTotal;
      // 単勝を圧縮
      for (let i = 0; i < liveBets.length; i++) {
        if (liveBets[i].type === "単勝" && amounts[i] > 0) {
          amounts[i] = Math.max(100, Math.round(amounts[i] * scale / 100) * 100);
        }
      }
      // 超過分を組合せ馬券へ均等加算
      const newWinTotal = liveBets.reduce((s, b, i) => s + (b.type === "単勝" ? amounts[i] : 0), 0);
      const excess = winTotal - newWinTotal;
      const comboIndices = liveBets.map((b, i) => ({ i, w: weights[i] }))
        .filter((x) => liveBets[x.i].type !== "単勝" && x.w > 0);
      if (comboIndices.length > 0 && excess > 0) {
        const comboWeight = comboIndices.reduce((s, x) => s + x.w, 0);
        for (const { i, w } of comboIndices) {
          amounts[i] += Math.round((excess * w / comboWeight) / 100) * 100;
        }
      }
    }

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

  // 理論EV（AI確率×オッズ — 穴馬で過大評価になりやすい参考値）
  const expectedReturn = scaledBets.reduce((s, b) => {
    if (b.scaledAmount === 0) return s;
    const effectiveEv = b.evReliable ? b.ev : b.backtestRoi;
    return s + b.scaledAmount * effectiveEv;
  }, 0);
  const allReliable = scaledBets.filter((b) => b.scaledAmount > 0).every((b) => b.evReliable);

  // BT実績ベース（64レース実配当から算出 — より信頼性の高い指標）
  const btStats = BT_PORTFOLIO_ROI[strategyMode]?.[comboMode] ?? { roi: 1.0, hitRate: 0 };
  const btExpectedReturn = totalInvestment * btStats.roi;
  const btRoi = totalInvestment > 0 ? (btExpectedReturn / totalInvestment - 1) * 100 : 0;

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

  // 対象馬（オッズ入力用） — 馬番順で固定（入力中に順番が変わらないように）
  const targetHorses = useMemo(() => {
    return predictions.predictions
      .filter((h) => ["◎", "○", "▲", "△"].includes(h.mark))
      .sort((a, b) => a.horse_number - b.horse_number);
  }, [predictions]);

  // 大穴狙い: メイン買い目外のEV>1.0ハイリターン候補（各¥100固定、予算外）
  const longshotBets = useMemo(() => {
    if (liveHorses.length === 0 || normProbs.size === 0) return [];

    type LshotBet = {
      type: string;
      targets: string;
      description: string;
      odds: number;
      ev: number;
      hitReturn: number;
    };
    const candidates: LshotBet[] = [];

    // メイン買い目の馬券キーを除外用に収集
    const mainKeys = new Set(scaledBets.map((b) => `${b.type}-${b.targets}`));

    // 1. 単勝: EV >= 1.0 でメインリストにない馬
    for (const h of liveHorses) {
      if (h.ev_win < 1.0) continue;
      const targets = `${h.horse_number}番 ${h.horse_name}`;
      if (mainKeys.has(`単勝-${targets}`)) continue;
      candidates.push({
        type: "単勝",
        targets,
        description: `AI勝率${(h.win_prob * 100).toFixed(1)}%`,
        odds: h.odds_win,
        ev: h.ev_win,
        hitReturn: Math.round(100 * h.odds_win),
      });
    }

    // 全馬番リスト
    const nums = liveHorses.map((h) => h.horse_number);
    const nameOf = (n: number) =>
      liveHorses.find((h) => h.horse_number === n)?.horse_name ?? `${n}番`;

    // 2. 馬連: comboOddsあり、EV >= 1.0、メインにない
    for (let i = 0; i < nums.length; i++) {
      for (let j = i + 1; j < nums.length; j++) {
        const pair = [nums[i], nums[j]].sort((a, b) => a - b);
        const key = `quinella-${pair[0]}-${pair[1]}`;
        const odds = comboOddsMap[key];
        if (!odds) continue;
        const tgt = `${pair[0]}-${pair[1]}`;
        if (mainKeys.has(`馬連-${tgt}`)) continue;
        const prob = quinellaProb(normProbs, pair[0], pair[1]);
        const ev = Math.round(prob * odds * 100) / 100;
        if (ev < 1.0) continue;
        candidates.push({
          type: "馬連",
          targets: tgt,
          description: `${nameOf(pair[0])}×${nameOf(pair[1])}`,
          odds,
          ev,
          hitReturn: Math.round(100 * odds),
        });
      }
    }

    // 3. ワイド: comboOddsあり、EV >= 1.0、メインにない
    for (let i = 0; i < nums.length; i++) {
      for (let j = i + 1; j < nums.length; j++) {
        const pair = [nums[i], nums[j]].sort((a, b) => a - b);
        const key = `wide-${pair[0]}-${pair[1]}`;
        const odds = comboOddsMap[key];
        if (!odds) continue;
        const tgt = `${pair[0]}-${pair[1]}`;
        if (mainKeys.has(`ワイド-${tgt}`)) continue;
        const prob = wideProb(normProbs, pair[0], pair[1]);
        const ev = Math.round(prob * odds * 100) / 100;
        if (ev < 1.0) continue;
        candidates.push({
          type: "ワイド",
          targets: tgt,
          description: `${nameOf(pair[0])}×${nameOf(pair[1])}`,
          odds,
          ev,
          hitReturn: Math.round(100 * odds),
        });
      }
    }

    // 4. 三連複: comboOddsあり、EV >= 1.0、メインにない
    for (let i = 0; i < nums.length; i++) {
      for (let j = i + 1; j < nums.length; j++) {
        for (let k = j + 1; k < nums.length; k++) {
          const tri = [nums[i], nums[j], nums[k]].sort((a, b) => a - b);
          const key = `trio-${tri[0]}-${tri[1]}-${tri[2]}`;
          const odds = comboOddsMap[key];
          if (!odds) continue;
          const tgt = `${tri[0]}-${tri[1]}-${tri[2]}`;
          if (mainKeys.has(`三連複-${tgt}`)) continue;
          const prob = trioProb(normProbs, tri[0], tri[1], tri[2]);
          const ev = Math.round(prob * odds * 100) / 100;
          if (ev < 1.0) continue;
          candidates.push({
            type: "三連複",
            targets: tgt,
            description: `${nameOf(tri[0])}・${nameOf(tri[1])}・${nameOf(tri[2])}`,
            odds,
            ev,
            hitReturn: Math.round(100 * odds),
          });
        }
      }
    }

    // 的中時リターン降順（盛り上がり度優先）→ EV降順
    candidates.sort((a, b) => b.hitReturn - a.hitReturn || b.ev - a.ev);
    return candidates.slice(0, 5);
  }, [liveHorses, normProbs, comboOddsMap, scaledBets]);

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
              ※ BT実績: 64レース実配当ベース回収率(v15)
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

              {/* KPI — BT実績ベース */}
              <div className="grid grid-cols-3 gap-2 text-center border-t border-white/10 pt-3">
                <div>
                  <p className="text-[10px] text-muted-foreground mb-0.5">合計投資</p>
                  <p className="font-mono text-sm font-bold">
                    ¥{totalInvestment.toLocaleString()}
                  </p>
                </div>
                <div>
                  <p className="text-[10px] text-muted-foreground mb-0.5">
                    BT実績参考
                  </p>
                  <p className="font-mono text-sm font-bold text-gold">
                    ¥{Math.round(btExpectedReturn).toLocaleString()}
                  </p>
                </div>
                <div>
                  <p className="text-[10px] text-muted-foreground mb-0.5">BT実績ROI</p>
                  <p
                    className={`font-mono text-lg font-bold ${
                      btRoi >= 0 ? "text-green-400" : "text-red-400"
                    }`}
                  >
                    {btRoi >= 0 ? "+" : ""}
                    {btRoi.toFixed(0)}%
                  </p>
                </div>
              </div>

              {/* 当選率 + 最高的中時 */}
              <div className="flex justify-between items-center mt-2">
                <span className="text-[10px] text-muted-foreground">
                  BT当選率: <span className="font-mono font-bold text-white">{Math.round(btStats.hitRate * 100)}%</span>
                  <span className="text-muted-foreground/50">（64R中 何か当たる確率）</span>
                </span>
                {maxHitReturn > 0 && (
                  <span className="text-[10px] text-green-400/70 font-mono">
                    最高的中¥{Math.round(maxHitReturn).toLocaleString()}
                  </span>
                )}
              </div>

              {/* 理論EV（参考） */}
              <div className="mt-2 pt-2 border-t border-white/5">
                <div className="flex items-center justify-between">
                  <span className="text-[9px] text-muted-foreground/60">
                    理論EV: ¥{Math.round(expectedReturn).toLocaleString()}
                    {!allReliable && <span className="text-orange-400">*</span>}
                  </span>
                  <span className="text-[9px] text-muted-foreground/60">
                    （AI確率×オッズ。穴馬で過大評価になりやすい）
                  </span>
                </div>
              </div>

              <p className="text-[9px] text-muted-foreground mt-1">
                ※ BT実績: 64レース実配当ベース({strategyMode === "stable" ? "安定" : "強気"}モード)
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
                    BT: 的中30% / 回収225%
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

        {/* 大穴狙い（おまけ） */}
        {longshotBets.length > 0 && (
          <motion.section {...fadeIn} transition={{ delay: 0.18 }}>
            <div className="bg-card rounded-xl p-4 border border-gold/20">
              <div className="flex items-center justify-between mb-1">
                <h2 className="text-sm font-bold">🎰 大穴狙い（おまけ）</h2>
                <span className="text-[10px] text-muted-foreground font-mono">
                  各¥100固定
                </span>
              </div>
              <p className="text-[10px] text-muted-foreground mb-3">
                当たれば盛り上がる＋EV 1.0超の穴馬券。予算外の追加投資
              </p>

              <div className="space-y-2">
                {longshotBets.map((bet, i) => (
                  <div
                    key={`ls-${bet.type}-${bet.targets}`}
                    className="flex items-start justify-between gap-2"
                  >
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-1.5">
                        <span className="text-xs font-bold text-gold shrink-0">
                          #{i + 1}
                        </span>
                        <span className="text-sakura-pink font-bold text-xs shrink-0">
                          {bet.type}
                        </span>
                        <span className="text-white font-mono text-xs">
                          {bet.targets}
                        </span>
                        <span
                          className={`text-[9px] font-mono shrink-0 ${
                            bet.ev >= 1.5 ? "text-gold font-bold" : "text-green-400"
                          }`}
                        >
                          EV{bet.ev.toFixed(1)}
                        </span>
                      </div>
                      <p className="text-[9px] text-muted-foreground mt-0.5 truncate">
                        {bet.description}
                      </p>
                    </div>
                    <div className="text-right shrink-0">
                      <p className="text-xs font-mono text-green-400 font-bold">
                        ¥{bet.hitReturn.toLocaleString()}
                      </p>
                      <p className="text-[9px] text-muted-foreground">
                        {bet.odds.toFixed(1)}倍
                      </p>
                    </div>
                  </div>
                ))}
              </div>

              <div className="mt-3 pt-2 border-t border-white/10 flex justify-between items-center">
                <span className="text-[10px] text-muted-foreground">
                  追加投資: <span className="font-mono font-bold text-white">¥{(longshotBets.length * 100).toLocaleString()}</span>
                  （{longshotBets.length}点×¥100）
                </span>
                <span className="text-[10px] text-gold/70 font-mono">
                  最高¥{Math.max(...longshotBets.map((b) => b.hitReturn)).toLocaleString()}
                </span>
              </div>
            </div>
          </motion.section>
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
