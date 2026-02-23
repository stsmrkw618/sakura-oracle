"use client";

import { useState, useMemo } from "react";
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
  } = useOdds();
  const [budget, setBudget] = useState(3000);
  const [glossaryOpen, setGlossaryOpen] = useState<string | null>(null);

  const hasAnyChange = liveHorses.some((h) => h.oddsChanged);
  const hasComboOdds = Object.keys(comboOddsMap).length > 0;

  const scaledBets = useMemo(() => {
    if (liveBets.length === 0) return [];

    // --- Kelly重み配分: BT事前重みでオッズ未入力時も推定配分 ---
    const weights = liveBets.map((b) => {
      if (b.kelly <= 0) return 0;
      // EV確定 & EV < 1.0 → 見送り（予算を他に再配分）
      if (b.evReliable && b.ev < 1.0) return 0;
      // evReliable=true（オッズ入力済み）→ Kelly そのまま
      // evReliable=false（未入力）→ Kelly × BT事前重み（backtestRoi/10）
      const backtestPrior = b.backtestRoi / 10;
      return b.evReliable ? b.kelly : b.kelly * backtestPrior;
    });
    const totalWeight = weights.reduce((s, w) => s + w, 0);

    if (totalWeight === 0) {
      return liveBets.map((b) => ({ ...b, scaledAmount: 0 }));
    }

    // 第1パス: 重みに比例して予算配分（¥100単位に丸め）
    const amounts = weights.map((w) => {
      if (w === 0) return 0;
      return Math.max(100, Math.round((w / totalWeight) * budget / 100) * 100);
    });

    // 第2パス: 合計 = 予算 になるよう最大重みの馬券で調整
    let total = amounts.reduce((s, a) => s + a, 0);

    // 不足分 → 最大重みの馬券に加算
    const maxIdx = weights.indexOf(Math.max(...weights));
    if (total < budget) {
      amounts[maxIdx] += budget - total;
      total = budget;
    }
    // 超過分 → 最小重みの非ゼロ馬券から¥100ずつ削減
    while (total > budget) {
      let reduced = false;
      // 重み昇順で走査
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

  // 期待リターン: 全馬券を含む（オッズ未入力はBT実績ROIで推定）
  const expectedReturn = scaledBets.reduce((s, b) => {
    if (b.scaledAmount === 0) return s;
    // オッズ入力済み → 実EV、未入力 → BT実績ROIで推定
    const effectiveEv = b.evReliable ? b.ev : b.backtestRoi;
    return s + b.scaledAmount * effectiveEv;
  }, 0);
  const roi = totalInvestment > 0 ? (expectedReturn / totalInvestment - 1) * 100 : 0;
  const allReliable = scaledBets.filter((b) => b.scaledAmount > 0).every((b) => b.evReliable);

  const toggleGlossary = (key: string) => {
    setGlossaryOpen(glossaryOpen === key ? null : key);
  };

  // Horses that originally have marks ◎○▲△ (use original marks so they don't vanish during editing)
  const targetHorses = useMemo(() => {
    return predictions.predictions.filter((h) =>
      ["◎", "○", "▲", "△"].includes(h.mark)
    );
  }, [predictions]);

  // 馬券タイプ別にグルーピング
  const winBets = scaledBets.filter((b) => b.type === "単勝");
  const quinellaBets = scaledBets.filter((b) => b.type === "馬連");
  const wideBets = scaledBets.filter((b) => b.type === "ワイド");
  const trioBets = scaledBets.filter((b) => b.type === "三連複");

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
        {/* Investment Simulator */}
        <motion.section {...fadeIn} transition={{ delay: 0.1 }}>
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
                <p className="text-[10px] text-muted-foreground mb-1">
                  期待リターン{!allReliable && <span className="text-orange-400">*</span>}
                </p>
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
            <p className="text-[10px] text-muted-foreground mt-2">
              ※ 期待リターン = Σ(賭け金 × EV)。{allReliable
                ? "全馬券のオッズが入力済みです"
                : "オッズ未入力の組合せ馬券はBT実績ROIで推定(*印)。オッズ入力で確定値に切替わります"}
            </p>
          </div>
        </motion.section>

        {/* 戦略サマリー — BT実績ROI */}
        <motion.section {...fadeIn} transition={{ delay: 0.12 }}>
          <div className="bg-card rounded-xl p-4 border border-white/5">
            <h2 className="text-sm font-bold text-muted-foreground mb-3">
              BT実績に基づく推奨配分
            </h2>
            <div className="space-y-2">
              {[
                { label: "三連複BOX(5)", roi: 850, hit: 34, color: "bg-gold" },
                { label: "馬連BOX(3)", roi: 507, hit: 32, color: "bg-sakura-pink" },
                { label: "ワイド(◎-○)", roi: 423, hit: 30, color: "bg-orange-400" },
                { label: "単勝", roi: 245, hit: null, color: "bg-blue-400" },
              ].map((item) => (
                <div key={item.label}>
                  <div className="flex items-center justify-between text-xs mb-1">
                    <span className="text-muted-foreground">{item.label}</span>
                    <span className="font-mono font-bold text-gold">
                      {item.roi}%
                      {item.hit != null && (
                        <span className="text-muted-foreground font-normal ml-1">
                          (的中{item.hit}%)
                        </span>
                      )}
                    </span>
                  </div>
                  <div className="w-full h-2 bg-white/5 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full ${item.color}`}
                      style={{ width: `${Math.min(100, item.roi / 10)}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
            <p className="text-[10px] text-muted-foreground mt-3">
              ※ 50レースのバックテスト実績（v9）。オッズ入力で配分が自動最適化されます
            </p>
          </div>
        </motion.section>

        {/* 購入リスト */}
        {scaledBets.some((b) => b.scaledAmount > 0) && (
          <motion.section {...fadeIn} transition={{ delay: 0.15 }}>
            <div className="bg-card rounded-xl p-4 border border-white/5">
              <h2 className="text-sm font-bold text-muted-foreground mb-3">
                📋 購入リスト
              </h2>
              <div className="space-y-1.5">
                {scaledBets
                  .filter((b) => b.scaledAmount > 0)
                  .map((bet, i) => (
                    <div
                      key={`summary-${i}`}
                      className="flex items-center justify-between text-sm"
                    >
                      <div className="flex items-center gap-2 flex-1 min-w-0">
                        <span className="text-sakura-pink font-bold shrink-0">
                          {bet.type}
                        </span>
                        <span className="text-muted-foreground truncate text-xs">
                          {bet.targets}
                        </span>
                      </div>
                      <div className="flex items-center gap-2 shrink-0">
                        <span
                          className={`text-[10px] px-1 py-0.5 rounded font-bold ${
                            bet.evReliable
                              ? "bg-green-500/20 text-green-400"
                              : "bg-white/10 text-muted-foreground"
                          }`}
                        >
                          {bet.evReliable ? "確定" : "推定"}
                        </span>
                        <span
                          className={`text-[10px] font-mono ${
                            bet.ev >= 1.5
                              ? "text-gold"
                              : bet.ev >= 1.0
                                ? "text-green-400"
                                : "text-muted-foreground"
                          }`}
                        >
                          {bet.evReliable ? `EV ${bet.ev.toFixed(2)}` : ""}
                        </span>
                        <span className="font-mono text-gold font-bold w-16 text-right">
                          ¥{bet.scaledAmount.toLocaleString()}
                        </span>
                      </div>
                    </div>
                  ))}
              </div>
              <div className="border-t border-white/10 mt-3 pt-2 flex items-center justify-between">
                <span className="text-xs text-muted-foreground">合計</span>
                <span className="font-mono text-gold font-bold">
                  ¥{totalInvestment.toLocaleString()}
                </span>
              </div>
              {scaledBets.some((b) => !b.evReliable && b.scaledAmount > 0) && (
                <p className="text-[10px] text-muted-foreground mt-2">
                  ※「推定」はBT実績ベースの事前配分。JRAオッズ入力で「確定」に切替わります
                </p>
              )}
            </div>
          </motion.section>
        )}

        {/* 三連複BOX(5) — BT ROI最高 */}
        {trioBets.length > 0 && (
          <motion.section {...fadeIn} transition={{ delay: 0.2 }}>
            <h2 className="text-sm font-bold text-muted-foreground mb-2">
              三連複BOX(5)（{trioBets.length}通り）
            </h2>
            <p className="text-[10px] text-muted-foreground mb-3">
              BT実績: 的中34% / 回収850% — AI上位5頭から全10通り
            </p>
            <div className="space-y-3">
              {trioBets.map((bet, i) => (
                <ComboBetCard
                  key={`trio-${i}`}
                  bet={bet}
                  comboOddsMap={comboOddsMap}
                  updateComboOdds={updateComboOdds}
                />
              ))}
            </div>
          </motion.section>
        )}

        {/* 馬連BOX(3) */}
        {quinellaBets.length > 0 && (
          <motion.section {...fadeIn} transition={{ delay: 0.25 }}>
            <h2 className="text-sm font-bold text-muted-foreground mb-2">
              馬連BOX(3)（{quinellaBets.length}通り）
            </h2>
            <p className="text-[10px] text-muted-foreground mb-3">
              BT実績: 的中32% / 回収507%
            </p>
            <div className="space-y-3">
              {quinellaBets.map((bet, i) => (
                <ComboBetCard
                  key={`quinella-${i}`}
                  bet={bet}
                  comboOddsMap={comboOddsMap}
                  updateComboOdds={updateComboOdds}
                />
              ))}
            </div>
          </motion.section>
        )}

        {/* ワイド(◎-○) */}
        {wideBets.length > 0 && (
          <motion.section {...fadeIn} transition={{ delay: 0.3 }}>
            <h2 className="text-sm font-bold text-muted-foreground mb-2">
              ワイド(◎-○)（{wideBets.length}点）
            </h2>
            <p className="text-[10px] text-muted-foreground mb-3">
              BT実績: 的中30% / 回収423%
            </p>
            <div className="space-y-3">
              {wideBets.map((bet, i) => (
                <ComboBetCard
                  key={`wide-${i}`}
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
          <motion.section {...fadeIn} transition={{ delay: 0.35 }}>
            <h2 className="text-sm font-bold text-muted-foreground mb-3">
              単勝（{winBets.length}点）
            </h2>
            <div className="space-y-3">
              {winBets.map((bet, i) => (
                <div
                  key={`win-${i}`}
                  className="bg-card rounded-xl p-4 border border-white/5"
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-bold text-sakura-pink">
                      {bet.type}
                    </span>
                    <div className="flex items-center gap-2">
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
                  <p className="text-xs text-muted-foreground mb-1">
                    {bet.targets}
                  </p>
                  <p className="text-xs text-muted-foreground mb-2 leading-relaxed">
                    {bet.description}
                  </p>
                  {bet.odds && (
                    <p className="text-xs text-green-400 mb-2">
                      的中時 ¥{Math.round(bet.odds * bet.scaledAmount).toLocaleString()}（オッズ {bet.odds}倍 × ¥{bet.scaledAmount.toLocaleString()}）
                    </p>
                  )}
                  <div className="flex items-center justify-between text-sm">
                    <span className="font-mono text-[10px] text-muted-foreground">
                      Kelly {(bet.kelly * 100).toFixed(1)}%
                    </span>
                    <span className="font-mono text-gold">
                      ¥{bet.scaledAmount.toLocaleString()}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </motion.section>
        )}

        {/* Odds Manual Update */}
        <motion.section {...fadeIn} transition={{ delay: 0.4 }}>
          <div className="bg-card rounded-xl p-4 border border-white/5">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-sm font-bold text-muted-foreground">
                🔧 単勝・複勝オッズ更新
              </h2>
              <div className="flex gap-2">
                {hasComboOdds && (
                  <button
                    onClick={resetComboOdds}
                    className="text-[10px] text-orange-400 underline"
                  >
                    組合せリセット
                  </button>
                )}
                {hasAnyChange && (
                  <button
                    onClick={resetOdds}
                    className="text-[10px] text-red-400 underline"
                  >
                    リセット
                  </button>
                )}
              </div>
            </div>
            <p className="text-xs text-muted-foreground mb-3">
              当日のオッズを入力すると「予測」タブ含め全ページの期待値・印・買い目が即時連動します
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
        <motion.section {...fadeIn} transition={{ delay: 0.5 }}>
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
                title: "期待値（EV）とは？",
                content:
                  "期待値 = AI予測確率 × オッズ。1.0を超えればプラス期待値（長期的に利益が出る賭け）。例: 勝率20%で8倍なら 0.20 × 8.0 = 1.6。EV 1.6は「100円賭けると平均160円返ってくる」という意味。オッズそのものではありません。",
              },
              {
                key: "combo-ev",
                title: "組合せ馬券のEV計算とは？",
                content:
                  "AIが各馬の勝率を予測 → Harvilleモデルで2頭・3頭の同時入着確率を算出 → JRAオッズを掛けてEVを計算。EV > 1.0なら「買い」判定。オッズは当日JRAサイトから手入力してください。",
              },
              {
                key: "umaren",
                title: "馬連BOXとは？",
                content:
                  "選んだ馬の中から、1着と2着の組み合わせを全通り買う方式。3頭BOXなら3通り（A-B, A-C, B-C）。順番は関係なし。JRAでは1通り最低¥100。",
              },
              {
                key: "sanrenpuku",
                title: "三連複とは？",
                content:
                  "1着・2着・3着に入る3頭の組み合わせを当てる馬券。順番は不問。5頭BOXなら10通り。高配当が狙える。JRAでは1通り最低¥100。",
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

/** 組合せ馬券のカードコンポーネント */
function ComboBetCard({
  bet,
  comboOddsMap,
  updateComboOdds,
}: {
  bet: ReturnType<typeof Object> & {
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

  return (
    <div className="bg-card rounded-xl p-4 border border-white/5">
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-bold text-sakura-pink">
          {bet.type} {bet.targets}
        </span>
        {bet.comboProb != null && (
          <span className="text-[10px] text-muted-foreground font-mono">
            的中率 {(bet.comboProb * 100).toFixed(1)}%
          </span>
        )}
      </div>
      <p className="text-xs text-muted-foreground mb-2 leading-relaxed">
        {bet.description}
      </p>

      {/* オッズ入力 + EV判定 */}
      {bet.comboKey && (
        <div className="flex items-center gap-2 mb-2">
          <span className="text-[10px] text-muted-foreground">JRAオッズ:</span>
          <ComboOddsInput
            comboKey={bet.comboKey}
            comboOddsMap={comboOddsMap}
            updateComboOdds={updateComboOdds}
          />
          {hasOdds ? (
            <div className="flex items-center gap-1">
              <span className="text-xs font-mono">
                EV {bet.ev.toFixed(2)}
              </span>
              <span
                className={`text-[10px] px-1.5 py-0.5 rounded font-bold ${
                  bet.ev >= 1.0
                    ? "bg-green-500/20 text-green-400"
                    : "bg-red-500/20 text-red-400"
                }`}
              >
                {bet.ev >= 1.0 ? "買い" : "見送り"}
              </span>
            </div>
          ) : (
            <span className="text-[10px] text-muted-foreground">
              オッズ未入力
            </span>
          )}
        </div>
      )}

      <div className="flex items-center justify-between text-sm">
        <span className="font-mono text-[10px] text-muted-foreground">
          Kelly {(bet.kelly * 100).toFixed(1)}%
        </span>
        <span className={`font-mono ${bet.scaledAmount > 0 ? "text-gold" : "text-muted-foreground"}`}>
          {bet.scaledAmount > 0
            ? `¥${bet.scaledAmount.toLocaleString()}`
            : "---"}
        </span>
      </div>
    </div>
  );
}
