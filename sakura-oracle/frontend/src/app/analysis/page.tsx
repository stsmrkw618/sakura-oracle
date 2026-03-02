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
  LineChart,
  Line,
  ReferenceLine,
  Tooltip,
  LabelList,
} from "recharts";
import Navbar from "@/components/Navbar";
import backtestAll from "@/data/backtest_all.json";

const fadeIn = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
};

// 特徴量重要度 — JSON動的データ優先、なければフォールバック
const featureImportanceFromJson = (backtestAll as Record<string, unknown>).feature_importance as
  | { name: string; key: string; value: number }[]
  | undefined;

// 特徴量の解説マップ（key → 説明）
const featureDescriptions: Record<string, string> = {
  pace_deviation: "レース前半のペースを距離帯別に偏差値化。50が平均、高いほどハイペース",
  speed_index: "タイム・距離・馬場を補正した独自指数。50が基準、高いほど速い",
  horse_number: "馬番（ゲート番号）。内枠/外枠の有利不利を反映",
  frame_number: "枠番（1〜8枠）。同枠の馬は同色の帽子",
  weight: "レース当日の馬体重（kg）。成長度合いやコンディションの指標",
  weight_diff: "前走からの馬体重増減（kg）。大幅増減は調子の変化を示唆",
  distance_m: "レースの距離（m）。1400〜2400mまで対応",
  grade_encoded: "レースのグレード。G1=5, G2=3, G3=3（格付け）",
  total_runs: "その馬の通算出走回数。キャリアの豊富さ",
  show_rate: "過去の複勝率（3着以内率）。安定感の指標",
  last1_finish: "前走の着順。直近の調子を反映",
  last1_last3f: "前走の上がり3F（ラスト600m）タイム。瞬発力の指標",
  last2_last3f: "2走前の上がり3Fタイム",
  last1_speed: "前走のスピード指数",
  avg_last3f: "過去走の上がり3F平均。持続的な瞬発力",
  best_last3f: "過去走の上がり3F最速値。ポテンシャルの上限",
  hanshin_runs: "阪神競馬場での出走回数。コース適性",
  jockey_win_rate: "騎手の累積勝率。腕前の指標",
  jockey_g1_wins: "騎手のG1累積勝利数。大舞台での実績",
  trainer_win_rate: "調教師の累積勝率。厩舎力",
  last1_start_pos: "前走の第1コーナー通過順位。脚質（逃げ/差し）を反映",
  last1_margin: "前走の着差（馬身）。勝ち方/負け方の程度",
  field_strength: "出走馬全体のオッズから算出した市場占有率。相手関係の強さ",
  odds: "単勝オッズ。市場の評価",
  popularity: "人気順（1〜18番人気）",
  last1_pace_deviation: "前走のペース偏差値。前走でどんなペースを経験したか",
  n_front_runners_est: "そのレースの推定逃げ・先行頭数。展開予想の指標",
  running_style_avg: "過去走の脚質コード平均（0=逃げ〜3=追込）",
  last1_running_style: "前走の脚質コード",
};

const featureImportance = featureImportanceFromJson
  ? featureImportanceFromJson.map((f) => ({ name: f.name, key: f.key, value: f.value, desc: featureDescriptions[f.key] || "" }))
  : [
      { name: "スピード指数", key: "speed_index", value: 0.12, desc: featureDescriptions["speed_index"] },
      { name: "馬体重", key: "weight", value: 0.12, desc: featureDescriptions["weight"] },
      { name: "場の強さ", key: "field_strength", value: 0.12, desc: featureDescriptions["field_strength"] },
      { name: "オッズ", key: "odds", value: 0.11, desc: featureDescriptions["odds"] },
      { name: "馬番", key: "horse_number", value: 0.09, desc: featureDescriptions["horse_number"] },
      { name: "枠番", key: "frame_number", value: 0.06, desc: featureDescriptions["frame_number"] },
      { name: "馬体重増減", key: "weight_diff", value: 0.06, desc: featureDescriptions["weight_diff"] },
      { name: "騎手勝率", key: "jockey_win_rate", value: 0.05, desc: featureDescriptions["jockey_win_rate"] },
      { name: "人気", key: "popularity", value: 0.04, desc: featureDescriptions["popularity"] },
      { name: "前走スタート位置", key: "last1_start_pos", value: 0.03, desc: featureDescriptions["last1_start_pos"] },
    ];

// 枠順別勝率 — JSON動的データ優先
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

// 人気別3着内率 — JSON動的データ優先
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

// 血統カテゴリ別勝率 — JSON動的データ優先
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

// 戦略比較データ（JSONから取得、フォールバック付き）
const strategyComparison = (backtestAll as Record<string, unknown>).strategy_comparison as
  | {
      n_races: number;
      budget_per_race: number;
      table: { label: string; box_agg: string; box_stb: string; nag_agg: string; nag_stb: string }[];
      pnl: { label: string; agg: number; stb: number }[];
      summary_text: string;
    }
  | undefined;

// バックテストデータ（JSONから取得）
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

// 組合せ馬券的中率（旧JSONには存在しない場合あり）
const comboHitRates = (backtestAll as Record<string, unknown>).combo_hit_rates as
  | {
      quinella_box3: number; wide_top2: number; trio_box3: number; trio_box5: number;
      quinella_box3_roi?: number; wide_top2_roi?: number; trio_box3_roi?: number; trio_box5_roi?: number;
      ev_trio_top5?: number; ev_trio_top5_roi?: number;
      ev_quinella_top3?: number; ev_quinella_top3_roi?: number;
    }
  | undefined;

// 信頼区間（旧JSONには存在しない場合あり）
const confidence = (summary as Record<string, unknown>).confidence as
  | {
      win_hit_rate_ci: [number, number];
      win_roi_ci: [number, number];
      show_hit_rate_ci: [number, number];
      show_roi_ci: [number, number];
      win_roi_pvalue: number;
    }
  | undefined;

// キャリブレーションデータ（旧JSONには存在しない場合あり）
const calibration = (backtestAll as Record<string, unknown>).calibration as
  | {
      win: { bin_center: number; predicted: number; observed: number; count: number }[];
      show: { bin_center: number; predicted: number; observed: number; count: number }[];
    }
  | undefined;

// ホールドアウト検証データ（旧JSONには存在しない場合あり）
const holdout = (backtestAll as Record<string, unknown>).holdout as
  | {
      cutoff_year: number;
      train: { n_races: number; win_hit_rate: number; win_roi: number; show_hit_rate: number; show_roi: number };
      test: { n_races: number; win_hit_rate: number; win_roi: number; show_hit_rate: number; show_roi: number };
      degradation: { win_roi_ratio: number };
    }
  | undefined;

// ジャックナイフ感度分析データ（旧JSONには存在しない場合あり）
const jackknife = (backtestAll as Record<string, unknown>).jackknife as
  | {
      n_races: number;
      base_win_roi: number;
      races: { label: string; win_roi_without: number; impact: number; win_return: number }[];
      roi_without_top1: number;
      roi_without_top3: number;
      roi_without_top5: number;
      min_roi: number;
      max_roi: number;
    }
  | undefined;

// 累積損益データ（定額ベット）
const cumulativePnl = (backtestAll as Record<string, unknown>).cumulative_pnl as
  | {
      per_race_investment: { win: number; quinella: number; wide: number; trio: number; total: number };
      history: { label: string; cum_win: number; cum_combo: number }[];
      total: { win: number; combo: number };
      n_races: number;
    }
  | undefined;

export default function AnalysisPage() {
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
        {/* AIの実力（信頼区間付き） */}
        <motion.section {...fadeIn} transition={{ delay: 0.1 }}>
          <div className="bg-card rounded-xl p-4 border border-white/5">
            <h2 className="text-sm font-bold mb-3">🏆 このAIの実力</h2>
            <p className="text-xs text-muted-foreground mb-3">
              過去{summary.n_races}レースのWalk-Forwardバックテスト
            </p>

            <div className="grid grid-cols-2 gap-3 mb-4">
              {/* ROIを主役に */}
              <div className="bg-navy/50 rounded-lg p-3 text-center border border-gold/20">
                <p className="text-[10px] text-gold mb-1">単勝回収率</p>
                <p className="font-mono text-lg font-bold text-gold">
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
              <div className="bg-navy/50 rounded-lg p-3 text-center border border-green-400/20">
                <p className="text-[10px] text-green-400 mb-1">複勝回収率</p>
                <p className="font-mono text-lg font-bold text-green-400">
                  {(summary.show_roi * 100).toFixed(0)}%
                </p>
                {confidence && (
                  <p className="text-[9px] text-muted-foreground font-mono">
                    95%CI: [{(confidence.show_roi_ci[0] * 100).toFixed(0)}–{(confidence.show_roi_ci[1] * 100).toFixed(0)}%]
                  </p>
                )}
              </div>
              {/* 的中率は参考指標 */}
              <div className="bg-navy/50 rounded-lg p-3 text-center">
                <p className="text-[10px] text-muted-foreground mb-1">1着的中率（参考）</p>
                <p className="font-mono text-lg font-bold text-muted-foreground">
                  {(summary.win_hit_rate * 100).toFixed(0)}%
                </p>
                {confidence && (
                  <p className="text-[9px] text-muted-foreground font-mono">
                    95%CI: [{(confidence.win_hit_rate_ci[0] * 100).toFixed(0)}–{(confidence.win_hit_rate_ci[1] * 100).toFixed(0)}%]
                  </p>
                )}
              </div>
              <div className="bg-navy/50 rounded-lg p-3 text-center">
                <p className="text-[10px] text-muted-foreground mb-1">複勝的中率（参考）</p>
                <p className="font-mono text-lg font-bold text-muted-foreground">
                  {(summary.show_hit_rate * 100).toFixed(0)}%
                </p>
                {confidence && (
                  <p className="text-[9px] text-muted-foreground font-mono">
                    95%CI: [{(confidence.show_hit_rate_ci[0] * 100).toFixed(0)}–{(confidence.show_hit_rate_ci[1] * 100).toFixed(0)}%]
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

        {/* ホールドアウト検証 */}
        {holdout && holdout.train.n_races > 0 && holdout.test.n_races > 0 && (
          <motion.section {...fadeIn} transition={{ delay: 0.11 }}>
            <div className="bg-card rounded-xl p-4 border border-white/5">
              <h2 className="text-sm font-bold mb-3">🔬 ホールドアウト検証</h2>
              <p className="text-xs text-muted-foreground mb-3">
                {holdout.cutoff_year}年以降を「未知データ」として分離検証
              </p>

              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-white/10">
                      <th className="text-left py-2 pr-2"></th>
                      <th className="text-center py-2 px-2">開発期間<br /><span className="text-[9px] text-muted-foreground">2021–{holdout.cutoff_year - 1}</span></th>
                      <th className="text-center py-2 px-2">検証期間<br /><span className="text-[9px] text-muted-foreground">{holdout.cutoff_year}–</span></th>
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      { label: "レース数", train: `${holdout.train.n_races}`, test: `${holdout.test.n_races}` },
                      { label: "単勝回収率", train: `${(holdout.train.win_roi * 100).toFixed(0)}%`, test: `${(holdout.test.win_roi * 100).toFixed(0)}%` },
                      { label: "複勝回収率", train: `${(holdout.train.show_roi * 100).toFixed(0)}%`, test: `${(holdout.test.show_roi * 100).toFixed(0)}%` },
                      { label: "1着的中率", train: `${(holdout.train.win_hit_rate * 100).toFixed(0)}%`, test: `${(holdout.test.win_hit_rate * 100).toFixed(0)}%` },
                      { label: "複勝的中率", train: `${(holdout.train.show_hit_rate * 100).toFixed(0)}%`, test: `${(holdout.test.show_hit_rate * 100).toFixed(0)}%` },
                    ].map((row) => (
                      <tr key={row.label} className="border-b border-white/5">
                        <td className="py-2 pr-2 text-muted-foreground">{row.label}</td>
                        <td className="py-2 px-2 text-center font-mono">{row.train}</td>
                        <td className="py-2 px-2 text-center font-mono">{row.test}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="mt-3 bg-navy/50 rounded-lg p-3">
                <p className="text-[10px] text-muted-foreground mb-1">劣化率（検証 / 開発）</p>
                <p className={`font-mono text-lg font-bold ${
                  holdout.degradation.win_roi_ratio >= 0.8 ? "text-green-400" :
                  holdout.degradation.win_roi_ratio >= 0.5 ? "text-yellow-400" :
                  "text-red-400"
                }`}>
                  {holdout.degradation.win_roi_ratio.toFixed(2)}
                  <span className="text-xs ml-2">
                    {holdout.degradation.win_roi_ratio >= 0.8 ? "頑健" :
                     holdout.degradation.win_roi_ratio >= 0.5 ? "注意" :
                     "過学習疑い"}
                  </span>
                </p>
                <p className="text-[9px] text-muted-foreground mt-1">
                  1.0に近いほど頑健（0.8以上=緑、0.5–0.8=黄、0.5未満=赤）
                </p>
              </div>
            </div>
          </motion.section>
        )}

        {/* ジャックナイフ感度分析 */}
        {jackknife && jackknife.races.length > 0 && (() => {
          // 横棒グラフ用データ: impactでソート済み（最も貢献=最も負のimpact → 先頭）
          // 表示は上位10件 + 下位5件
          const sorted = [...jackknife.races].sort((a, b) => a.impact - b.impact);
          const chartData = sorted.map((r) => ({
            label: r.label.replace(/\(.*\)/, "").trim(),
            fullLabel: r.label,
            impact: Math.round(r.impact * 100),  // %表示
            fill: r.impact < 0 ? "#EF4444" : "#22C55E",
          }));

          return (
            <motion.section {...fadeIn} transition={{ delay: 0.115 }}>
              <div className="bg-card rounded-xl p-4 border border-white/5">
                <h2 className="text-sm font-bold mb-3">🔍 感度分析（ジャックナイフ）</h2>
                <p className="text-xs text-muted-foreground mb-3">
                  各レースを1件ずつ除外した時のROI変動（赤=ROI貢献、緑=ROI低下要因）
                </p>

                {/* KPI指標 */}
                <div className="grid grid-cols-3 gap-2 mb-4">
                  <div className="bg-navy/50 rounded-lg p-2 text-center">
                    <p className="text-[9px] text-muted-foreground">Top1除外</p>
                    <p className={`font-mono text-sm font-bold ${jackknife.roi_without_top1 >= 1 ? "text-green-400" : "text-red-400"}`}>
                      {(jackknife.roi_without_top1 * 100).toFixed(0)}%
                    </p>
                  </div>
                  <div className="bg-navy/50 rounded-lg p-2 text-center">
                    <p className="text-[9px] text-muted-foreground">Top3除外</p>
                    <p className={`font-mono text-sm font-bold ${jackknife.roi_without_top3 >= 1 ? "text-green-400" : "text-red-400"}`}>
                      {(jackknife.roi_without_top3 * 100).toFixed(0)}%
                    </p>
                  </div>
                  <div className="bg-navy/50 rounded-lg p-2 text-center">
                    <p className="text-[9px] text-muted-foreground">Top5除外</p>
                    <p className={`font-mono text-sm font-bold ${jackknife.roi_without_top5 >= 1 ? "text-green-400" : "text-red-400"}`}>
                      {(jackknife.roi_without_top5 * 100).toFixed(0)}%
                    </p>
                  </div>
                </div>

                {/* 棒グラフ */}
                <ResponsiveContainer width="100%" height={Math.min(chartData.length * 22, 600)}>
                  <BarChart
                    data={chartData}
                    layout="vertical"
                    margin={{ left: 10, right: 30, top: 5, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#1A1A2E" />
                    <XAxis
                      type="number"
                      tick={{ fill: "#A0A0B0", fontSize: 9 }}
                      tickFormatter={(v: number) => `${v > 0 ? "+" : ""}${v}%`}
                    />
                    <YAxis
                      dataKey="label"
                      type="category"
                      width={100}
                      tick={{ fill: "#A0A0B0", fontSize: 8 }}
                    />
                    <Tooltip
                      content={({ payload }) => {
                        if (!payload || payload.length === 0) return null;
                        const d = payload[0].payload as { fullLabel: string; impact: number };
                        return (
                          <div className="bg-navy border border-white/10 rounded p-2 text-xs">
                            <p>{d.fullLabel}</p>
                            <p>ROI変動: {d.impact > 0 ? "+" : ""}{d.impact}%pt</p>
                          </div>
                        );
                      }}
                    />
                    <ReferenceLine x={0} stroke="#666" />
                    <Bar dataKey="impact" radius={[0, 4, 4, 0]} animationDuration={1200}>
                      {chartData.map((d, i) => (
                        <Cell key={i} fill={d.fill} />
                      ))}
                      <LabelList
                        dataKey="impact"
                        position="right"
                        formatter={((v: unknown) => {
                          const n = Number(v);
                          return isNaN(n) ? "" : `${n > 0 ? "+" : ""}${n}%`;
                        }) as (value: unknown) => string}
                        style={{ fill: "#A0A0B0", fontSize: 8 }}
                      />
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>

                {/* 結論 */}
                <div className="mt-3 bg-navy/50 rounded-lg p-3">
                  <p className="text-xs">
                    {jackknife.roi_without_top3 >= 1.0 ? (
                      <span className="text-green-400">
                        上位3レース除外でもROI {(jackknife.roi_without_top3 * 100).toFixed(0)}% → プラス収支を維持。少数レースへの依存リスクは限定的。
                      </span>
                    ) : (
                      <span className="text-yellow-400">
                        上位3レース除外でROI {(jackknife.roi_without_top3 * 100).toFixed(0)}% → 特定レースに依存している可能性あり。注意が必要。
                      </span>
                    )}
                  </p>
                </div>
              </div>
            </motion.section>
          );
        })()}

        {/* キャリブレーション曲線 */}
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

              {/* 解説 */}
              <div className="mt-4 space-y-3">
                <div className="bg-navy/50 rounded-lg p-3">
                  <p className="text-[11px] font-bold text-white mb-2">グラフの見方</p>
                  <div className="space-y-1.5 text-[10px] text-gray-300 leading-relaxed">
                    <p><span className="text-muted-foreground">横軸</span> = AIが出した予測確率、<span className="text-muted-foreground">縦軸</span> = 実際に当たった割合</p>
                    <p><span className="text-muted-foreground">点線（対角線）</span> = 「予測10%なら10%当たる」完璧なライン</p>
                    <p><span className="text-muted-foreground">点の大きさ</span> = そのビンのサンプル数（大きい = 信頼度高）</p>
                  </div>
                </div>

                <div className="bg-navy/50 rounded-lg p-3">
                  <p className="text-[11px] font-bold text-white mb-2">読み取り方</p>
                  <div className="space-y-1.5 text-[10px] text-gray-300 leading-relaxed">
                    <p>
                      <span className="text-green-400">対角線の上</span> = AIは控えめ（実際はもっと当たる）
                    </p>
                    <p>
                      <span className="text-red-400">対角線の下</span> = AIは過信（実際はそこまで当たらない）
                    </p>
                  </div>
                </div>

                {(() => {
                  const winData = calibration.win.filter(d => d.count >= 10);
                  const showData = calibration.show.filter(d => d.count >= 10);
                  const winOverconfident = winData.filter(d => d.predicted > d.observed).length;
                  const showOverconfident = showData.filter(d => d.predicted > d.observed).length;
                  const isWinOver = winOverconfident > winData.length / 2;
                  const isShowOver = showOverconfident > showData.length / 2;

                  return (
                    <div className="bg-navy/50 rounded-lg p-3 border-l-4 border-sakura-pink">
                      <p className="text-[11px] font-bold text-white mb-2">現在の傾向</p>
                      <div className="space-y-1.5 text-[10px] text-gray-300 leading-relaxed">
                        <p>
                          <span className="text-sakura-pink">単勝</span>:
                          {isWinOver
                            ? " 全体的に対角線の下 → やや過信傾向。キャリブレーター（Isotonic Regression）で補正済み"
                            : " 対角線付近 → 校正は良好"
                          }
                        </p>
                        <p>
                          <span className="text-gold">複勝</span>:
                          {isShowOver
                            ? " 低確率帯では過信だが、高確率帯では控えめ → 本命馬の複勝は信頼できる"
                            : " 対角線付近 → 校正は良好"
                          }
                        </p>
                        <p className="text-muted-foreground mt-1">
                          ※ このグラフは補正前の生予測。本番ではキャリブレーターが自動補正します
                        </p>
                      </div>
                    </div>
                  );
                })()}
              </div>
            </div>
          </motion.section>
        )}

        {/* 組合せ馬券的中率 */}
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
                  ...(comboHitRates.ev_trio_top5 != null ? [{ label: "三連複 EV Top5", rate: comboHitRates.ev_trio_top5, roi: comboHitRates.ev_trio_top5_roi, desc: "Harville EV比率上位5通り", cost: "5通り×100円" }] : []),
                  ...(comboHitRates.ev_quinella_top3 != null ? [{ label: "馬連 EV Top3", rate: comboHitRates.ev_quinella_top3, roi: comboHitRates.ev_quinella_top3_roi, desc: "Harville EV比率上位3通り", cost: "3通り×100円" }] : []),
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

        {/* 累積損益（定額ベット） */}
        {cumulativePnl && cumulativePnl.history.length > 0 && (
          <motion.section {...fadeIn} transition={{ delay: 0.18 }}>
            <div className="bg-card rounded-xl p-4 border border-white/5">
              <h2 className="text-sm font-bold mb-3">📈 累積損益（定額ベット）</h2>
              <p className="text-xs text-muted-foreground mb-3">
                毎レース¥{cumulativePnl.per_race_investment.total.toLocaleString()}定額投資の累積損益（{cumulativePnl.n_races}レース）
              </p>

              <ResponsiveContainer width="100%" height={250}>
                <LineChart
                  data={cumulativePnl.history}
                  margin={{ left: 10, right: 10, top: 5, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#1A1A2E" />
                  <XAxis
                    dataKey="label"
                    tick={false}
                    label={{ value: "レース（時系列）", position: "bottom", fill: "#A0A0B0", fontSize: 10, offset: -5 }}
                  />
                  <YAxis
                    tick={{ fill: "#A0A0B0", fontSize: 10 }}
                    tickFormatter={(v: number) => `¥${v.toLocaleString()}`}
                  />
                  <Tooltip
                    content={({ payload }) => {
                      if (!payload || payload.length === 0) return null;
                      const d = payload[0].payload as { label: string; cum_win: number; cum_combo: number };
                      return (
                        <div className="bg-navy border border-white/10 rounded p-2 text-xs">
                          <p className="font-bold mb-1">{d.label}</p>
                          <p><span className="text-[#FFD700]">全戦略:</span> ¥{d.cum_combo.toLocaleString()}</p>
                          <p><span className="text-[#A0A0B0]">単勝のみ:</span> ¥{d.cum_win.toLocaleString()}</p>
                        </div>
                      );
                    }}
                  />
                  {/* 損益分岐点 */}
                  <ReferenceLine y={0} stroke="#666" strokeDasharray="5 5" />
                  <Line
                    type="monotone"
                    dataKey="cum_win"
                    stroke="#A0A0B0"
                    strokeWidth={1.5}
                    dot={false}
                    name="単勝のみ"
                  />
                  <Line
                    type="monotone"
                    dataKey="cum_combo"
                    stroke="#FFD700"
                    strokeWidth={2}
                    dot={false}
                    name="全戦略"
                  />
                </LineChart>
              </ResponsiveContainer>

              <div className="flex justify-center gap-4 mt-1">
                <span className="text-[10px] text-gold">━ 全戦略（三連複+馬連+ワイド+単勝）</span>
                <span className="text-[10px] text-[#A0A0B0]">━ 単勝のみ</span>
              </div>

              {/* KPI指標 */}
              <div className="grid grid-cols-2 gap-2 mt-3">
                <div className="bg-navy/50 rounded-lg p-2 text-center">
                  <p className="text-[9px] text-muted-foreground">累積損益（全戦略）</p>
                  <p className={`font-mono text-sm font-bold ${cumulativePnl.total.combo >= 0 ? "text-green-400" : "text-red-400"}`}>
                    ¥{cumulativePnl.total.combo.toLocaleString()}
                  </p>
                  <p className="text-[9px] text-muted-foreground">
                    ROI {((cumulativePnl.total.combo + cumulativePnl.n_races * cumulativePnl.per_race_investment.total) / (cumulativePnl.n_races * cumulativePnl.per_race_investment.total) * 100).toFixed(0)}%
                  </p>
                </div>
                <div className="bg-navy/50 rounded-lg p-2 text-center">
                  <p className="text-[9px] text-muted-foreground">累積損益（単勝）</p>
                  <p className={`font-mono text-sm font-bold ${cumulativePnl.total.win >= 0 ? "text-green-400" : "text-red-400"}`}>
                    ¥{cumulativePnl.total.win.toLocaleString()}
                  </p>
                  <p className="text-[9px] text-muted-foreground">
                    ROI {((cumulativePnl.total.win + cumulativePnl.n_races * cumulativePnl.per_race_investment.win) / (cumulativePnl.n_races * cumulativePnl.per_race_investment.win) * 100).toFixed(0)}%
                  </p>
                </div>
              </div>
            </div>
          </motion.section>
        )}

        {/* 戦略比較: 強気 vs 安定 */}
        <motion.section {...fadeIn} transition={{ delay: 0.195 }}>
          <div className="bg-card rounded-xl p-4 border border-white/5">
            <h2 className="text-sm font-bold mb-3">🔥 戦略比較: 強気 vs 安定</h2>
            <p className="text-xs text-muted-foreground mb-3">
              {strategyComparison
                ? `${strategyComparison.n_races}レースBT。強気=Kelly/印順で穴馬軸、安定=勝率順で人気馬軸。¥${strategyComparison.budget_per_race.toLocaleString()}/R投資`
                : "50レースBT（v10）。強気=Kelly/印順で穴馬軸、安定=勝率順で人気馬軸。¥3,000/R投資"}
            </p>

            {/* 比較テーブル */}
            <div className="overflow-x-auto mb-4">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-white/10">
                    <th className="text-left py-1.5 text-muted-foreground font-normal">指標</th>
                    <th className="text-right py-1.5 text-gold font-bold">BOX強気</th>
                    <th className="text-right py-1.5 text-sakura-pink font-bold">BOX安定</th>
                    <th className="text-right py-1.5 text-gold font-bold">軸流し強気</th>
                    <th className="text-right py-1.5 text-sakura-pink font-bold">軸流し安定</th>
                  </tr>
                </thead>
                <tbody className="font-mono">
                  {(strategyComparison?.table ?? [
                    { label: "回収率", box_agg: "350%", box_stb: "415%", nag_agg: "178%", nag_stb: "384%" },
                    { label: "1回EV", box_agg: "3.50", box_stb: "4.15", nag_agg: "1.78", nag_stb: "3.84" },
                    { label: "当選率", box_agg: "18%", box_stb: "52%", nag_agg: "18%", nag_stb: "50%" },
                    { label: "最大DD", box_agg: "100%", box_stb: "51%", nag_agg: "100%", nag_stb: "26%" },
                    { label: "最終倍率", box_agg: "27.0x", box_stb: "28.4x", nag_agg: "9.7x", nag_stb: "21.5x" },
                  ]).map((row) => (
                    <tr key={row.label} className="border-b border-white/5">
                      <td className="py-1.5 text-muted-foreground font-sans">{row.label}</td>
                      <td className="text-right py-1.5">{row.box_agg}</td>
                      <td className="text-right py-1.5">{row.box_stb}</td>
                      <td className="text-right py-1.5">{row.nag_agg}</td>
                      <td className="text-right py-1.5">{row.nag_stb}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* 累積損益推移チャート（BOXモード） */}
            <h3 className="text-xs font-bold text-muted-foreground mb-2">
              累積損益推移（BOXモード）
            </h3>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart
                data={strategyComparison?.pnl ?? [
                  { label: "開始", agg: 0, stb: 0 },
                ]}
                margin={{ left: 10, right: 10, top: 5, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#1A1A2E" />
                <XAxis dataKey="label" tick={{ fill: "#A0A0B0", fontSize: 10 }} />
                <YAxis
                  tick={{ fill: "#A0A0B0", fontSize: 10 }}
                  tickFormatter={(v: number) => `¥${v.toLocaleString()}`}
                />
                <Tooltip
                  content={({ payload }) => {
                    if (!payload || payload.length === 0) return null;
                    const d = payload[0].payload as { label: string; agg: number; stb: number };
                    return (
                      <div className="bg-navy border border-white/10 rounded p-2 text-xs">
                        <p className="font-bold mb-1">{d.label}</p>
                        <p><span className="text-gold">強気:</span> ¥{d.agg.toLocaleString()}</p>
                        <p><span className="text-sakura-pink">安定:</span> ¥{d.stb.toLocaleString()}</p>
                      </div>
                    );
                  }}
                />
                {/* 損益分岐点 */}
                <ReferenceLine y={0} stroke="#666" strokeDasharray="5 5" />
                <Line type="monotone" dataKey="agg" stroke="#FFD700" strokeWidth={2} dot={false} name="強気" />
                <Line type="monotone" dataKey="stb" stroke="#E8879C" strokeWidth={2} dot={false} name="安定" />
              </LineChart>
            </ResponsiveContainer>

            <div className="flex justify-center gap-4 mt-1">
              <span className="text-[10px] text-gold">━ 強気（Kelly/印順）</span>
              <span className="text-[10px] text-sakura-pink">━ 安定（勝率順）</span>
            </div>

            {/* まとめ */}
            <div className="mt-3 p-3 bg-navy/50 rounded-lg border border-sakura-pink/20">
              <p className="text-xs text-muted-foreground leading-relaxed">
                {strategyComparison?.summary_text ?? (
                  <>
                    <span className="text-sakura-pink font-bold">安定モードが優秀</span>:
                    当選率3倍（18%→52%）、DD半減（100%→51%）でありながら回収率も+65pt上回る。
                    強気は途中2度破産するが一発逆転で巻き返す「ギャンブラー型」。
                    安定はコツコツ積み上げる「投資家型」。リスク許容度で選択を。
                  </>
                )}
              </p>
            </div>
          </div>
        </motion.section>

        {/* 特徴量重要度 */}
        <motion.section {...fadeIn} transition={{ delay: 0.2 }}>
          <div className="bg-card rounded-xl p-4 border border-white/5">
            <h2 className="text-sm font-bold mb-3">📊 特徴量重要度 Top10</h2>
            <p className="text-xs text-muted-foreground mb-3">
              LightGBMが予測時に各特徴量をどの程度使ったか（タップで解説）
            </p>
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
                <Tooltip
                  cursor={{ fill: "rgba(255,255,255,0.05)" }}
                  content={({ payload }) => {
                    if (!payload || payload.length === 0) return null;
                    const d = payload[0].payload as { name: string; key: string; value: number; desc?: string };
                    return (
                      <div className="bg-navy border border-white/10 rounded-lg p-3 text-xs max-w-[260px] shadow-lg">
                        <p className="font-bold text-white mb-1">{d.name}</p>
                        <p className="text-muted-foreground font-mono text-[10px] mb-1.5">{d.key}</p>
                        {d.desc && <p className="text-gray-300 leading-relaxed">{d.desc}</p>}
                        <p className="text-gold font-mono mt-1.5">寄与度: {(d.value * 100).toFixed(1)}%</p>
                      </div>
                    );
                  }}
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

        {/* 枠順別勝率 */}
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

        {/* 人気別3着内率 */}
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

        {/* 血統カテゴリ別勝率 */}
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

        {/* AIの読み */}
        <motion.section {...fadeIn} transition={{ delay: 0.6 }}>
          <div className="bg-card rounded-xl p-4 border border-white/5">
            <h2 className="text-sm font-bold text-sakura-pink mb-3">
              🤖 AIの読み
            </h2>
            <div className="bg-navy/50 rounded-lg p-4 border-l-4 border-sakura-pink">
              <p className="text-sm leading-relaxed text-gray-300">
                3歳牝馬重賞では<span className="text-white font-bold">レースペース（前半の流れ）</span>が最重要指標。
                ペース偏差値・馬体重・スピード指数の3つで予測力の約36%を占める。
                ハイペースで先行馬が崩れるか、スローで逃げ切るかの展開読みがカギ。
              </p>
              <p className="text-sm leading-relaxed text-gray-300 mt-2">
                Model B（オッズ除外）をメインに据えることで、
                <span className="text-white font-bold">市場が見落とす穴馬</span>を検出。
                デュアルモデルブレンド（A20+B80）により、
                組合せ馬券（馬連・三連複）で高回収率を実現。
              </p>
            </div>
          </div>
        </motion.section>
      </main>

      <Navbar />
    </div>
  );
}
