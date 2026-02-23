"use client";

import { motion } from "framer-motion";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  Cell,
  ScatterChart,
  Scatter,
  ZAxis,
  AreaChart,
  Area,
  ReferenceLine,
  Tooltip,
} from "recharts";
import Navbar from "@/components/Navbar";
import backtestAll from "@/data/backtest_all.json";

const fadeIn = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
};

// Feature importance — JSON動的データ優先、なければフォールバック
const featureImportanceFromJson = (backtestAll as Record<string, unknown>).feature_importance as
  | { name: string; key: string; value: number }[]
  | undefined;

const featureImportance = featureImportanceFromJson
  ? featureImportanceFromJson.map((f) => ({ name: f.name, value: f.value }))
  : [
      { name: "スピード指数", value: 0.12 },
      { name: "馬体重", value: 0.12 },
      { name: "場の強さ", value: 0.12 },
      { name: "オッズ", value: 0.11 },
      { name: "馬番", value: 0.09 },
      { name: "枠番", value: 0.06 },
      { name: "馬体重増減", value: 0.06 },
      { name: "騎手勝率", value: 0.05 },
      { name: "人気", value: 0.04 },
      { name: "前走スタート位置", value: 0.03 },
    ];

// Frame win rate — JSON動的データ優先
const frameWinRateFromJson = (backtestAll as Record<string, unknown>).frame_win_rate as
  | { frame: string; rate: number; n: number }[]
  | undefined;
const frameWinRate = frameWinRateFromJson ?? [
  { frame: "1枠", rate: 3.5 },
  { frame: "2枠", rate: 5.4 },
  { frame: "3枠", rate: 8.5 },
  { frame: "4枠", rate: 6.2 },
  { frame: "5枠", rate: 10.9 },
  { frame: "6枠", rate: 11.5 },
  { frame: "7枠", rate: 3.9 },
  { frame: "8枠", rate: 3.8 },
];

// Popularity hit rate — JSON動的データ優先
const popularityRateFromJson = (backtestAll as Record<string, unknown>).popularity_show_rate as
  | { pop: string; rate: number; n: number }[]
  | undefined;
const popularityRate = popularityRateFromJson ?? [
  { pop: "1人気", rate: 51.9 },
  { pop: "2人気", rate: 59.3 },
  { pop: "3人気", rate: 37.0 },
  { pop: "4人気", rate: 27.8 },
  { pop: "5人気", rate: 20.4 },
  { pop: "6人気", rate: 20.4 },
  { pop: "7人気", rate: 25.9 },
  { pop: "8人気", rate: 11.1 },
  { pop: "9人気", rate: 7.5 },
  { pop: "10人気", rate: 11.1 },
];

// Bloodline win rate — JSON動的データ優先
const bloodlineFromJson = (backtestAll as Record<string, unknown>).bloodline_win_rate as
  | { name: string; rate: number; n: number }[]
  | undefined;
const bloodlineData = bloodlineFromJson ?? [
  { name: "Kingman系", rate: 10.9 },
  { name: "キングカメハメハ系", rate: 9.5 },
  { name: "American Pharoah系", rate: 8.7 },
  { name: "Saxon Warrior系", rate: 6.0 },
  { name: "ゴールドアクター系", rate: 3.8 },
  { name: "No Nay Never系", rate: 3.3 },
];

// Backtest data from JSON
const summary = backtestAll.summary;
const byYear = backtestAll.by_year as Record<string, { n: number; win_rate: number; show_rate: number }>;
const backtestYears = Object.entries(byYear)
  .sort(([a], [b]) => a.localeCompare(b))
  .slice(-5)
  .map(([year, data]) => ({
    year,
    hit: Math.round(data.win_rate * data.n),
    total: data.n,
    showRate: data.show_rate,
  }));

// Combo hit rates from backtest (may not exist in older JSON)
const comboHitRates = (backtestAll as Record<string, unknown>).combo_hit_rates as
  | {
      quinella_box3: number; wide_top2: number; trio_box3: number; trio_box5: number;
      quinella_box3_roi?: number; wide_top2_roi?: number; trio_box3_roi?: number; trio_box5_roi?: number;
    }
  | undefined;

// Confidence intervals (may not exist in older JSON)
const confidence = (summary as Record<string, unknown>).confidence as
  | {
      win_hit_rate_ci: [number, number];
      win_roi_ci: [number, number];
      show_hit_rate_ci: [number, number];
      show_roi_ci: [number, number];
      win_roi_pvalue: number;
    }
  | undefined;

// Calibration data (may not exist in older JSON)
const calibration = (backtestAll as Record<string, unknown>).calibration as
  | {
      win: { bin_center: number; predicted: number; observed: number; count: number }[];
      show: { bin_center: number; predicted: number; observed: number; count: number }[];
    }
  | undefined;

// Simulation data (may not exist in older JSON)
const simulation = (backtestAll as Record<string, unknown>).simulation as
  | {
      initial_bankroll: number;
      n_races: number;
      paths: Record<string, number[]>;
      max_drawdown: { median: number; p95: number };
      final_bankroll: { median: number; p5: number; p95: number };
    }
  | undefined;

// Build bankroll chart data from simulation paths
function buildBankrollData() {
  if (!simulation?.paths) return [];
  const p50 = simulation.paths.p50 || [];
  const data = [];
  for (let i = 0; i < p50.length; i++) {
    data.push({
      race: i,
      p5: simulation.paths.p5?.[i] ?? 0,
      p25: simulation.paths.p25?.[i] ?? 0,
      p50: simulation.paths.p50?.[i] ?? 0,
      p75: simulation.paths.p75?.[i] ?? 0,
      p95: simulation.paths.p95?.[i] ?? 0,
    });
  }
  return data;
}

export default function AnalysisPage() {
  const bankrollData = buildBankrollData();

  return (
    <div className="min-h-screen bg-navy-dark pb-20">
      <motion.header
        {...fadeIn}
        className="sticky top-0 z-40 bg-navy-dark/95 backdrop-blur-md border-b border-white/5 px-4 py-3"
      >
        <h1 className="text-lg font-bold">📈 分析</h1>
        <p className="text-xs text-muted-foreground">
          AIモデルの実力と3歳牝馬重賞の傾向
        </p>
      </motion.header>

      <main className="px-4 py-4 space-y-5">
        {/* Model Accuracy with Confidence Intervals */}
        <motion.section {...fadeIn} transition={{ delay: 0.1 }}>
          <div className="bg-card rounded-xl p-4 border border-white/5">
            <h2 className="text-sm font-bold mb-3">🏆 このAIの実力</h2>
            <p className="text-xs text-muted-foreground mb-3">
              過去{summary.n_races}レースのWalk-Forwardバックテスト
            </p>

            <div className="grid grid-cols-2 gap-3 mb-4">
              <div className="bg-navy/50 rounded-lg p-3 text-center">
                <p className="text-[10px] text-muted-foreground mb-1">1着的中率</p>
                <p className="font-mono text-lg font-bold text-gold">
                  {(summary.win_hit_rate * 100).toFixed(0)}%
                </p>
                {confidence && (
                  <p className="text-[9px] text-muted-foreground font-mono">
                    95%CI: [{(confidence.win_hit_rate_ci[0] * 100).toFixed(0)}–{(confidence.win_hit_rate_ci[1] * 100).toFixed(0)}%]
                  </p>
                )}
              </div>
              <div className="bg-navy/50 rounded-lg p-3 text-center">
                <p className="text-[10px] text-muted-foreground mb-1">単勝回収率</p>
                <p className="font-mono text-lg font-bold text-green-400">
                  {(summary.win_roi * 100).toFixed(0)}%
                </p>
                {confidence && (
                  <>
                    <p className="text-[9px] text-muted-foreground font-mono">
                      95%CI: [{(confidence.win_roi_ci[0] * 100).toFixed(0)}–{(confidence.win_roi_ci[1] * 100).toFixed(0)}%]
                    </p>
                    <p className={`text-[9px] font-mono ${confidence.win_roi_pvalue < 0.05 ? "text-green-400" : "text-muted-foreground"}`}>
                      p={confidence.win_roi_pvalue.toFixed(3)}{confidence.win_roi_pvalue < 0.05 ? " *" : ""}
                    </p>
                  </>
                )}
              </div>
              <div className="bg-navy/50 rounded-lg p-3 text-center">
                <p className="text-[10px] text-muted-foreground mb-1">複勝的中率</p>
                <p className="font-mono text-lg font-bold">
                  {(summary.show_hit_rate * 100).toFixed(0)}%
                </p>
                {confidence && (
                  <p className="text-[9px] text-muted-foreground font-mono">
                    95%CI: [{(confidence.show_hit_rate_ci[0] * 100).toFixed(0)}–{(confidence.show_hit_rate_ci[1] * 100).toFixed(0)}%]
                  </p>
                )}
              </div>
              <div className="bg-navy/50 rounded-lg p-3 text-center">
                <p className="text-[10px] text-muted-foreground mb-1">複勝回収率</p>
                <p className="font-mono text-lg font-bold text-green-400">
                  {(summary.show_roi * 100).toFixed(0)}%
                </p>
                {confidence && (
                  <p className="text-[9px] text-muted-foreground font-mono">
                    95%CI: [{(confidence.show_roi_ci[0] * 100).toFixed(0)}–{(confidence.show_roi_ci[1] * 100).toFixed(0)}%]
                  </p>
                )}
              </div>
            </div>

            <div className="space-y-3">
              {backtestYears.map((d) => (
                <div key={d.year}>
                  <div className="flex justify-between text-xs mb-1">
                    <span>{d.year}年</span>
                    <span className="font-mono">
                      {d.hit}/{d.total}レース的中
                    </span>
                  </div>
                  <div className="bg-white/5 rounded-full h-2.5 overflow-hidden">
                    <div
                      className="h-full bg-sakura-pink rounded-full transition-all duration-700"
                      style={{ width: `${(d.hit / Math.max(d.total, 1)) * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </motion.section>

        {/* Calibration Curve */}
        {calibration && calibration.win.length > 0 && (
          <motion.section {...fadeIn} transition={{ delay: 0.12 }}>
            <div className="bg-card rounded-xl p-4 border border-white/5">
              <h2 className="text-sm font-bold mb-3">🎯 確率キャリブレーション</h2>
              <p className="text-xs text-muted-foreground mb-3">
                AI予測確率 vs 実際的中率（対角線 = 完全校正）
              </p>
              <ResponsiveContainer width="100%" height={250}>
                <ScatterChart margin={{ left: 0, right: 10, top: 5, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1A1A2E" />
                  <XAxis
                    dataKey="predicted"
                    type="number"
                    domain={[0, "auto"]}
                    tick={{ fill: "#A0A0B0", fontSize: 10 }}
                    label={{ value: "AI予測確率", position: "bottom", fill: "#A0A0B0", fontSize: 10, offset: -5 }}
                  />
                  <YAxis
                    dataKey="observed"
                    type="number"
                    domain={[0, "auto"]}
                    tick={{ fill: "#A0A0B0", fontSize: 10 }}
                    label={{ value: "実際的中率", angle: -90, position: "insideLeft", fill: "#A0A0B0", fontSize: 10 }}
                  />
                  <ZAxis dataKey="count" range={[30, 300]} />
                  <Tooltip
                    content={({ payload }) => {
                      if (!payload || payload.length === 0) return null;
                      const d = payload[0].payload as { predicted: number; observed: number; count: number };
                      return (
                        <div className="bg-navy border border-white/10 rounded p-2 text-xs">
                          <p>予測: {(d.predicted * 100).toFixed(1)}%</p>
                          <p>実績: {(d.observed * 100).toFixed(1)}%</p>
                          <p>n={d.count}</p>
                        </div>
                      );
                    }}
                  />
                  <ReferenceLine
                    segment={[{ x: 0, y: 0 }, { x: 0.5, y: 0.5 }]}
                    stroke="#666"
                    strokeDasharray="5 5"
                  />
                  <Scatter
                    name="単勝"
                    data={calibration.win}
                    fill="#E8879C"
                  />
                  <Scatter
                    name="複勝"
                    data={calibration.show}
                    fill="#FFD700"
                  />
                </ScatterChart>
              </ResponsiveContainer>
              <div className="flex justify-center gap-4 mt-1">
                <span className="text-[10px] text-sakura-pink">● 単勝</span>
                <span className="text-[10px] text-gold">● 複勝</span>
                <span className="text-[10px] text-muted-foreground">--- 完全校正</span>
              </div>
            </div>
          </motion.section>
        )}

        {/* Combo Hit Rates */}
        {comboHitRates && (
          <motion.section {...fadeIn} transition={{ delay: 0.15 }}>
            <div className="bg-card rounded-xl p-4 border border-white/5">
              <h2 className="text-sm font-bold mb-3">🎯 組合せ馬券 過去的中率</h2>
              <p className="text-xs text-muted-foreground mb-3">
                {summary.n_races}レースのバックテスト（AI上位予測馬での的中率）
              </p>

              <div className="space-y-3">
                {[
                  { label: "馬連BOX(3)", rate: comboHitRates.quinella_box3, roi: comboHitRates.quinella_box3_roi, desc: "上位3頭のうち2頭が1-2着", cost: "3通り×100円" },
                  { label: "ワイド(◎-○)", rate: comboHitRates.wide_top2, roi: comboHitRates.wide_top2_roi, desc: "上位2頭が両方3着以内", cost: "1通り×100円" },
                  { label: "三連複BOX(3)", rate: comboHitRates.trio_box3, roi: comboHitRates.trio_box3_roi, desc: "上位3頭が全員3着以内", cost: "1通り×100円" },
                  { label: "三連複BOX(5)", rate: comboHitRates.trio_box5, roi: comboHitRates.trio_box5_roi, desc: "上位5頭のうち3頭が3着以内", cost: "10通り×100円" },
                ].map((item) => (
                  <div key={item.label}>
                    <div className="flex justify-between text-xs mb-1">
                      <span>{item.label}</span>
                      <div className="flex gap-3">
                        <span className="font-mono text-gold">
                          的中{(item.rate * 100).toFixed(0)}%
                        </span>
                        {item.roi != null && (
                          <span className={`font-mono ${item.roi >= 1 ? "text-green-400" : "text-red-400"}`}>
                            回収{(item.roi * 100).toFixed(0)}%
                          </span>
                        )}
                      </div>
                    </div>
                    <p className="text-[10px] text-muted-foreground mb-1">{item.desc}（{item.cost}）</p>
                    <div className="bg-white/5 rounded-full h-2 overflow-hidden">
                      <div
                        className="h-full bg-gold rounded-full transition-all duration-700"
                        style={{ width: `${Math.min(item.rate * 100, 100)}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>

              <p className="text-[10px] text-muted-foreground mt-3">
                ※ 回収率は過去{summary.n_races}レースの実配当ベース。100%超でプラス収支
              </p>
            </div>
          </motion.section>
        )}

        {/* Bankroll Simulation */}
        {simulation && bankrollData.length > 0 && (
          <motion.section {...fadeIn} transition={{ delay: 0.18 }}>
            <div className="bg-card rounded-xl p-4 border border-white/5">
              <h2 className="text-sm font-bold mb-3">💰 バンクロールシミュレーション</h2>
              <p className="text-xs text-muted-foreground mb-3">
                1/4 Kelly戦略 × 1,000パス Monte Carlo（初期資金¥{simulation.initial_bankroll.toLocaleString()}）
              </p>
              <ResponsiveContainer width="100%" height={250}>
                <AreaChart data={bankrollData} margin={{ left: 10, right: 10, top: 5, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1A1A2E" />
                  <XAxis
                    dataKey="race"
                    tick={{ fill: "#A0A0B0", fontSize: 10 }}
                    label={{ value: "レース数", position: "bottom", fill: "#A0A0B0", fontSize: 10, offset: -5 }}
                  />
                  <YAxis
                    tick={{ fill: "#A0A0B0", fontSize: 10 }}
                    tickFormatter={(v: number) => `¥${(v / 1000).toFixed(0)}k`}
                  />
                  <Tooltip
                    content={({ payload, label }) => {
                      if (!payload || payload.length === 0) return null;
                      const d = payload[0].payload as Record<string, number>;
                      return (
                        <div className="bg-navy border border-white/10 rounded p-2 text-xs">
                          <p>レース #{label}</p>
                          <p>95%tile: ¥{d.p95?.toLocaleString()}</p>
                          <p>中央値: ¥{d.p50?.toLocaleString()}</p>
                          <p>5%tile: ¥{d.p5?.toLocaleString()}</p>
                        </div>
                      );
                    }}
                  />
                  <ReferenceLine y={simulation.initial_bankroll} stroke="#666" strokeDasharray="5 5" />
                  {/* 5%-95% light fill */}
                  <Area type="monotone" dataKey="p95" stroke="none" fill="#E8879C" fillOpacity={0.1} />
                  <Area type="monotone" dataKey="p5" stroke="none" fill="#0F0F1A" fillOpacity={1} />
                  {/* 25%-75% darker fill */}
                  <Area type="monotone" dataKey="p75" stroke="none" fill="#E8879C" fillOpacity={0.2} />
                  <Area type="monotone" dataKey="p25" stroke="none" fill="#0F0F1A" fillOpacity={1} />
                  {/* 50% median line */}
                  <Area type="monotone" dataKey="p50" stroke="#E8879C" strokeWidth={2} fill="none" />
                </AreaChart>
              </ResponsiveContainer>

              {/* KPIs */}
              <div className="grid grid-cols-3 gap-2 mt-3">
                <div className="bg-navy/50 rounded-lg p-2 text-center">
                  <p className="text-[9px] text-muted-foreground">中央値リターン</p>
                  <p className={`font-mono text-sm font-bold ${simulation.final_bankroll.median > simulation.initial_bankroll ? "text-green-400" : "text-red-400"}`}>
                    ¥{simulation.final_bankroll.median.toLocaleString()}
                  </p>
                </div>
                <div className="bg-navy/50 rounded-lg p-2 text-center">
                  <p className="text-[9px] text-muted-foreground">最大DD(中央)</p>
                  <p className="font-mono text-sm font-bold text-orange-400">
                    {(simulation.max_drawdown.median * 100).toFixed(0)}%
                  </p>
                </div>
                <div className="bg-navy/50 rounded-lg p-2 text-center">
                  <p className="text-[9px] text-muted-foreground">5%tile最終</p>
                  <p className={`font-mono text-sm font-bold ${simulation.final_bankroll.p5 > simulation.initial_bankroll ? "text-green-400" : "text-red-400"}`}>
                    ¥{simulation.final_bankroll.p5.toLocaleString()}
                  </p>
                </div>
              </div>
            </div>
          </motion.section>
        )}

        {/* Feature Importance */}
        <motion.section {...fadeIn} transition={{ delay: 0.2 }}>
          <div className="bg-card rounded-xl p-4 border border-white/5">
            <h2 className="text-sm font-bold mb-3">📊 特徴量重要度 Top10</h2>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart
                data={featureImportance}
                layout="vertical"
                margin={{ left: 10, right: 20, top: 5, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#1A1A2E" />
                <XAxis type="number" tick={{ fill: "#A0A0B0", fontSize: 10 }} />
                <YAxis
                  dataKey="name"
                  type="category"
                  width={90}
                  tick={{ fill: "#A0A0B0", fontSize: 10 }}
                />
                <Bar dataKey="value" radius={[0, 4, 4, 0]} animationDuration={1500}>
                  {featureImportance.map((_, i) => (
                    <Cell key={i} fill={i === 0 ? "#FFD700" : "#E8879C"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </motion.section>

        {/* Frame Win Rate */}
        <motion.section {...fadeIn} transition={{ delay: 0.3 }}>
          <div className="bg-card rounded-xl p-4 border border-white/5">
            <h2 className="text-sm font-bold mb-3">🏇 枠順別勝率（{summary.n_races}レース）</h2>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={frameWinRate} margin={{ left: -10, right: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1A1A2E" />
                <XAxis dataKey="frame" tick={{ fill: "#A0A0B0", fontSize: 10 }} />
                <YAxis tick={{ fill: "#A0A0B0", fontSize: 10 }} unit="%" />
                <Bar dataKey="rate" fill="#E8879C" radius={[4, 4, 0, 0]} animationDuration={1200} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </motion.section>

        {/* Popularity Hit Rate */}
        <motion.section {...fadeIn} transition={{ delay: 0.4 }}>
          <div className="bg-card rounded-xl p-4 border border-white/5">
            <h2 className="text-sm font-bold mb-3">
              🎯 人気別3着内率（{summary.n_races}レース）
            </h2>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={popularityRate} margin={{ left: -10, right: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1A1A2E" />
                <XAxis dataKey="pop" tick={{ fill: "#A0A0B0", fontSize: 9 }} />
                <YAxis tick={{ fill: "#A0A0B0", fontSize: 10 }} unit="%" />
                <Bar dataKey="rate" fill="#FFD700" radius={[4, 4, 0, 0]} animationDuration={1200} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </motion.section>

        {/* Bloodline Win Rate */}
        <motion.section {...fadeIn} transition={{ delay: 0.5 }}>
          <div className="bg-card rounded-xl p-4 border border-white/5">
            <h2 className="text-sm font-bold mb-3">🧬 血統カテゴリ別勝率（{summary.n_races}レース）</h2>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart
                data={bloodlineData}
                layout="vertical"
                margin={{ left: 20, right: 20 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#1A1A2E" />
                <XAxis type="number" tick={{ fill: "#A0A0B0", fontSize: 10 }} unit="%" />
                <YAxis
                  dataKey="name"
                  type="category"
                  width={120}
                  tick={{ fill: "#A0A0B0", fontSize: 10 }}
                />
                <Bar dataKey="rate" fill="#E8879C" radius={[0, 4, 4, 0]} animationDuration={1200} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </motion.section>

        {/* AI Reading */}
        <motion.section {...fadeIn} transition={{ delay: 0.6 }}>
          <div className="bg-card rounded-xl p-4 border border-white/5">
            <h2 className="text-sm font-bold text-sakura-pink mb-3">
              🤖 AIの読み
            </h2>
            <div className="bg-navy/50 rounded-lg p-4 border-l-4 border-sakura-pink">
              <p className="text-sm leading-relaxed text-gray-300">
                3歳牝馬重賞では<span className="text-white font-bold">前走上がり3Fが最重要指標</span>。
                瞬発力のある馬が好走しやすく、
                内枠有利の傾向は過去のデータでも顕著。
              </p>
              <p className="text-sm leading-relaxed text-gray-300 mt-2">
                Model B（オッズ除外）をメインに据えることで、
                <span className="text-white font-bold">市場が見落とす穴馬</span>を検出。
                デュアルモデルブレンド（A20+B80）で安定性とエッジを両立。
              </p>
            </div>
          </div>
        </motion.section>
      </main>

      <Navbar />
    </div>
  );
}
