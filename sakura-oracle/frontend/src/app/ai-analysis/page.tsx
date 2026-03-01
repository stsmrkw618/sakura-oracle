"use client";

import { useMemo } from "react";
import { motion } from "framer-motion";
import Navbar from "@/components/Navbar";
import MarkBadge from "@/components/MarkBadge";
import { useOdds } from "@/context/OddsContext";
import { useRace } from "@/context/RaceContext";

const fadeIn = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
};

/** 予測パターン分類 */
type PredType = "穴狙い型" | "本命型" | "混戦型";

export default function AiAnalysisPage() {
  const { predictions } = useRace();
  const { liveHorses, strategyMode } = useOdds();

  const analysis = useMemo(() => {
    if (liveHorses.length === 0) return null;

    // ◎馬（本命）
    const honmei = liveHorses.find((h) => h.mark === "◎");

    // 市場人気TOP3（単勝オッズ昇順）
    const marketTop3 = [...liveHorses]
      .sort((a, b) => a.odds_win - b.odds_win)
      .slice(0, 3);

    // AI勝率TOP3
    const aiTop3 = [...liveHorses]
      .sort((a, b) => b.win_prob - a.win_prob)
      .slice(0, 3);

    // 予測パターン分類
    let predType: PredType = "混戦型";
    if (honmei) {
      if (honmei.odds_win > 10) predType = "穴狙い型";
      else if (honmei.odds_win < 5) predType = "本命型";
    }

    // 1番人気
    const favorite = marketTop3[0];

    // 注目馬: EV高い馬（AI隠れ推し）— EV > 1.5 かつ市場人気4位以下
    const marketTop3Nums = new Set(marketTop3.map((h) => h.horse_number));
    const hiddenGems = [...liveHorses]
      .filter((h) => h.ev_win >= 1.5 && !marketTop3Nums.has(h.horse_number))
      .sort((a, b) => b.ev_win - a.ev_win)
      .slice(0, 3);

    // 市場過剰評価: 人気TOP3のうちEV < 1.0の馬
    const overrated = marketTop3.filter((h) => h.ev_win < 1.0);

    // リスク指標
    const maxKelly = Math.max(...liveHorses.map((h) => h.kelly_win));
    const plusEvCount = liveHorses.filter((h) => h.ev_win >= 1.0).length;
    const plusEvRate = Math.round((plusEvCount / liveHorses.length) * 100);

    // 人気馬 vs AI の乖離チェック
    const comparisons = marketTop3.map((mh) => {
      const aiRank = [...liveHorses]
        .sort((a, b) => b.win_prob - a.win_prob)
        .findIndex((h) => h.horse_number === mh.horse_number) + 1;
      const marketRank = [...liveHorses]
        .sort((a, b) => a.odds_win - b.odds_win)
        .findIndex((h) => h.horse_number === mh.horse_number) + 1;
      const gap = Math.abs(aiRank - marketRank);
      return { ...mh, aiRank, marketRank, gap, divergent: gap >= 3 };
    });

    return {
      honmei,
      marketTop3,
      aiTop3,
      predType,
      favorite,
      hiddenGems,
      overrated,
      maxKelly,
      plusEvRate,
      plusEvCount,
      comparisons,
    };
  }, [liveHorses]);

  if (!analysis) {
    return (
      <div className="min-h-screen bg-navy-dark pb-20 flex items-center justify-center">
        <p className="text-muted-foreground">データを読み込み中...</p>
        <Navbar />
      </div>
    );
  }

  const {
    honmei, predType, favorite, hiddenGems, overrated,
    maxKelly, plusEvRate, plusEvCount, comparisons,
  } = analysis;

  // パターン別アイコン
  const predTypeIcon = predType === "穴狙い型" ? "🎯" : predType === "本命型" ? "👑" : "🌀";
  const predTypeColor = predType === "穴狙い型" ? "text-gold" : predType === "本命型" ? "text-sakura-pink" : "text-blue-400";

  return (
    <div className="min-h-screen bg-navy-dark pb-20">
      {/* ヘッダー */}
      <motion.header
        {...fadeIn}
        className="sticky top-0 z-40 bg-navy-dark/95 backdrop-blur-md border-b border-white/5 px-4 py-3"
      >
        <h1 className="text-lg font-bold">🧠 AI プロ解説</h1>
        <p className="text-xs text-muted-foreground">
          {predictions.race_info?.name || "レース"}
        </p>
      </motion.header>

      <main className="px-4 py-4 space-y-5">
        {/* 総合評価 */}
        <motion.section {...fadeIn} transition={{ delay: 0.05 }}>
          <div className="bg-card rounded-xl p-4 border border-white/5">
            <h2 className="text-sm font-bold mb-3">📋 総合評価</h2>

            {/* パターン分類 */}
            <div className="flex items-center gap-3 mb-3">
              <span className="text-2xl">{predTypeIcon}</span>
              <div>
                <span className={`text-lg font-bold ${predTypeColor}`}>
                  {predType}
                </span>
                <p className="text-xs text-muted-foreground mt-0.5">
                  {predType === "穴狙い型" && "AIは市場が見落としている穴馬を本命視"}
                  {predType === "本命型" && "AIと市場の見解が一致 — 堅い決着を予測"}
                  {predType === "混戦型" && "実力拮抗 — 展開次第で波乱も"}
                </p>
              </div>
            </div>

            {/* ◎ vs 1番人気 */}
            {honmei && favorite && (
              <div className="grid grid-cols-2 gap-3 mt-3">
                <div className="bg-navy/50 rounded-lg p-3 border border-gold/20">
                  <p className="text-[10px] text-muted-foreground mb-1">AI本命 ◎</p>
                  <p className="text-sm font-bold truncate">{honmei.horse_name}</p>
                  <div className="flex items-center gap-2 mt-1">
                    <span className="text-xs font-mono text-gold">
                      {honmei.odds_win}倍
                    </span>
                    <span className="text-xs font-mono text-green-400">
                      AI {(honmei.win_prob * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
                <div className="bg-navy/50 rounded-lg p-3 border border-white/10">
                  <p className="text-[10px] text-muted-foreground mb-1">1番人気</p>
                  <p className="text-sm font-bold truncate">{favorite.horse_name}</p>
                  <div className="flex items-center gap-2 mt-1">
                    <span className="text-xs font-mono text-sakura-pink">
                      {favorite.odds_win}倍
                    </span>
                    <span className="text-xs font-mono text-green-400">
                      AI {(favorite.win_prob * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>
            )}

            {honmei && favorite && honmei.horse_number !== favorite.horse_number && (
              <p className="text-xs text-orange-400 mt-2">
                ⚠ AI本命と1番人気が異なる — 市場との見解に乖離あり
              </p>
            )}
          </div>
        </motion.section>

        {/* 人気馬 vs AI判定 */}
        <motion.section {...fadeIn} transition={{ delay: 0.1 }}>
          <div className="bg-card rounded-xl p-4 border border-white/5">
            <h2 className="text-sm font-bold mb-3">🔍 人気馬 vs AI判定</h2>
            <div className="space-y-2">
              {comparisons.map((c) => (
                <div
                  key={c.horse_number}
                  className={`flex items-center justify-between p-2 rounded-lg ${
                    c.divergent ? "bg-orange-500/10 border border-orange-500/20" : "bg-navy/30"
                  }`}
                >
                  <div className="flex items-center gap-2 flex-1 min-w-0">
                    <span className="text-xs text-muted-foreground w-5 shrink-0">
                      {c.marketRank}位
                    </span>
                    <span className="text-sm font-bold truncate">
                      {c.horse_name}
                    </span>
                    <MarkBadge mark={c.mark} size="sm" />
                  </div>
                  <div className="flex items-center gap-2 shrink-0">
                    <span className="text-xs font-mono text-muted-foreground">
                      {c.odds_win}倍
                    </span>
                    <span className="text-xs font-mono text-green-400">
                      AI{c.aiRank}位
                    </span>
                    {c.divergent && (
                      <span className="text-[9px] px-1.5 py-0.5 rounded bg-orange-500/20 text-orange-400 font-bold">
                        乖離
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
            <p className="text-[9px] text-muted-foreground mt-2">
              ※ 市場人気(オッズ順)とAI勝率順の比較。乖離3位以上で「乖離」フラグ
            </p>
          </div>
        </motion.section>

        {/* 注目馬ピックアップ */}
        <motion.section {...fadeIn} transition={{ delay: 0.15 }}>
          <div className="bg-card rounded-xl p-4 border border-white/5">
            <h2 className="text-sm font-bold mb-3">✨ 注目馬ピックアップ</h2>

            {/* AI隠れ推し */}
            {hiddenGems.length > 0 && (
              <div className="mb-3">
                <p className="text-xs text-gold font-bold mb-2">
                  🎯 AI隠れ推し（市場が見落としている高EV馬）
                </p>
                <div className="space-y-1.5">
                  {hiddenGems.map((h) => (
                    <div key={h.horse_number} className="flex items-center justify-between bg-gold/5 rounded-lg p-2">
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-muted-foreground">{h.horse_number}番</span>
                        <span className="text-sm font-bold">{h.horse_name}</span>
                        <MarkBadge mark={h.mark} size="sm" />
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-mono">{h.odds_win}倍</span>
                        <span className="text-xs font-mono text-gold font-bold">
                          EV{h.ev_win.toFixed(2)}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
            {hiddenGems.length === 0 && (
              <p className="text-xs text-muted-foreground mb-3">
                人気薄のEV1.5超えなし — 市場は概ね妥当な評価
              </p>
            )}

            {/* 市場過剰評価 */}
            {overrated.length > 0 && (
              <div>
                <p className="text-xs text-red-400 font-bold mb-2">
                  ⚠ 市場過剰評価（人気先行でEV割れ）
                </p>
                <div className="space-y-1.5">
                  {overrated.map((h) => (
                    <div key={h.horse_number} className="flex items-center justify-between bg-red-500/5 rounded-lg p-2">
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-muted-foreground">{h.horse_number}番</span>
                        <span className="text-sm font-bold">{h.horse_name}</span>
                        <MarkBadge mark={h.mark} size="sm" />
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-mono">{h.odds_win}倍</span>
                        <span className="text-xs font-mono text-red-400 font-bold">
                          EV{h.ev_win.toFixed(2)}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
            {overrated.length === 0 && (
              <p className="text-xs text-green-400">
                人気TOP3は全てEV1.0以上 — 過剰評価なし
              </p>
            )}
          </div>
        </motion.section>

        {/* リスク要因 */}
        <motion.section {...fadeIn} transition={{ delay: 0.2 }}>
          <div className="bg-card rounded-xl p-4 border border-white/5">
            <h2 className="text-sm font-bold mb-3">⚡ リスク分析</h2>
            <div className="space-y-2">
              {/* ◎馬のリスク */}
              {honmei && (
                <div className="flex items-center justify-between">
                  <span className="text-xs text-muted-foreground">◎の単勝オッズ</span>
                  <span className={`text-sm font-mono font-bold ${
                    honmei.odds_win > 10 ? "text-orange-400" : "text-green-400"
                  }`}>
                    {honmei.odds_win}倍
                    {honmei.odds_win > 10 && " (高リスク)"}
                    {honmei.odds_win <= 5 && " (堅実)"}
                  </span>
                </div>
              )}

              {/* Kelly最大値 */}
              <div className="flex items-center justify-between">
                <span className="text-xs text-muted-foreground">Kelly最大値</span>
                <span className={`text-sm font-mono font-bold ${
                  maxKelly > 0.05 ? "text-gold" : maxKelly > 0.02 ? "text-green-400" : "text-muted-foreground"
                }`}>
                  {(maxKelly * 100).toFixed(1)}%
                </span>
              </div>

              {/* プラスEV率 */}
              <div className="flex items-center justify-between">
                <span className="text-xs text-muted-foreground">EV1.0以上の馬</span>
                <span className={`text-sm font-mono font-bold ${
                  plusEvRate >= 30 ? "text-gold" : plusEvRate >= 15 ? "text-green-400" : "text-red-400"
                }`}>
                  {plusEvCount}頭 ({plusEvRate}%)
                </span>
              </div>

              {/* リスク総合判定 */}
              <div className="mt-2 pt-2 border-t border-white/10">
                {honmei && honmei.odds_win > 15 ? (
                  <p className="text-xs text-orange-400">
                    🔥 ◎が高オッズ — 的中率は低いが的中時は高配当。資金管理を慎重に
                  </p>
                ) : maxKelly > 0.05 ? (
                  <p className="text-xs text-gold">
                    💎 強いエッジあり — Kelly値が高く、AIが自信を持つレース
                  </p>
                ) : (
                  <p className="text-xs text-muted-foreground">
                    📊 標準的なリスク水準 — 通常の資金管理で問題なし
                  </p>
                )}
              </div>
            </div>
          </div>
        </motion.section>

        {/* 戦略アドバイス */}
        <motion.section {...fadeIn} transition={{ delay: 0.25 }}>
          <div className="bg-card rounded-xl p-4 border border-white/5">
            <h2 className="text-sm font-bold mb-3">🗺️ 戦略アドバイス</h2>
            <div className="space-y-3">
              {/* 強気モード */}
              <div className={`rounded-lg p-3 border ${
                strategyMode === "aggressive"
                  ? "border-gold bg-gold/10"
                  : "border-white/10 bg-navy/30"
              }`}>
                <p className="text-xs font-bold text-gold mb-1">
                  🔥 強気モード（現在{strategyMode === "aggressive" ? "選択中" : "未選択"}）
                </p>
                <p className="text-xs text-muted-foreground leading-relaxed">
                  {predType === "穴狙い型"
                    ? "AIが見出した穴馬を軸に高配当を狙う。このレースはAI本命が人気薄のため、強気モードで期待値を最大化できる可能性が高い。"
                    : predType === "本命型"
                      ? "本命が堅い分、穴馬との組合せで万馬券を狙える。ただし本命決着なら回収率は低め。"
                      : "混戦レースでは穴が出やすい。強気モードで手広く拾うのが有効。"}
                </p>
              </div>

              {/* 安定モード */}
              <div className={`rounded-lg p-3 border ${
                strategyMode === "stable"
                  ? "border-sakura-pink bg-sakura-pink/10"
                  : "border-white/10 bg-navy/30"
              }`}>
                <p className="text-xs font-bold text-sakura-pink mb-1">
                  🛡️ 安定モード（現在{strategyMode === "stable" ? "選択中" : "未選択"}）
                </p>
                <p className="text-xs text-muted-foreground leading-relaxed">
                  {predType === "穴狙い型"
                    ? "AIの穴馬推しに不安がある場合、人気馬軸で手堅く。的中率は上がるが配当は控えめ。"
                    : predType === "本命型"
                      ? "人気馬を軸にした堅実な馬券。的中率重視で安定した回収を目指す。このレースの構図に合った選択。"
                      : "混戦なので人気馬を軸にして的中率を確保。取りこぼしを防ぐ守りの戦略。"}
                </p>
              </div>

              {/* 買い目タブへの誘導 */}
              <a
                href="/bet-guide"
                className="block text-center text-sm font-bold text-sakura-pink bg-sakura-pink/10 rounded-lg py-3 border border-sakura-pink/20 transition-all hover:bg-sakura-pink/20"
              >
                💰 買い目ガイドで馬券を組む →
              </a>
            </div>
          </div>
        </motion.section>
      </main>

      <Navbar />
    </div>
  );
}
