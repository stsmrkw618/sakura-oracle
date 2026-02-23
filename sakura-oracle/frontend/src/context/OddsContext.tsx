"use client";

import {
  createContext,
  useContext,
  useState,
  useMemo,
  useCallback,
  useEffect,
  useRef,
  type ReactNode,
} from "react";
import { useRace } from "@/context/RaceContext";
import {
  normalizeProbabilities,
  quinellaProb,
  wideProb,
  trioProb,
} from "@/lib/harville";

// --- Types ---

interface OddsEntry {
  win: number;
  show: number;
}

interface LiveHorse {
  horse_number: number;
  horse_name: string;
  win_prob: number;
  show_prob: number;
  ev_win: number;
  ev_show: number;
  kelly_win: number;
  kelly_show: number;
  mark: string;
  odds_win: number;
  odds_show: number;
  /** market implied probability (overround-normalized) */
  market_prob: number;
  /** original values from predictions.json */
  orig_ev_win: number;
  orig_ev_show: number;
  orig_mark: string;
  /** true if odds were changed from initial */
  oddsChanged: boolean;
}

interface Bet {
  type: string;
  targets: string;
  description: string;
  amount: number;
  ev: number;
  evReliable: boolean; // true=実オッズでEV計算済み
  odds: number | null;
  kelly: number;
  /** BT実績回収率（馬連=5.50, 三連複=4.74, ワイド=4.65, 単勝=2.65） */
  backtestRoi: number;
  /** 的中時リターン (単勝: amount × odds) */
  winReturn?: number;
  /** Harville的中確率 (組合せ馬券) */
  comboProb?: number;
  /** 組合せオッズ検索キー (例: "quinella-7-12") */
  comboKey?: string;
}

interface OddsContextValue {
  oddsMap: Record<number, OddsEntry>;
  updateOdds: (horseNumber: number, win: number, show: number) => void;
  resetOdds: () => void;
  liveHorses: LiveHorse[];
  liveBets: Bet[];
  comboOddsMap: Record<string, number>;
  updateComboOdds: (key: string, odds: number) => void;
  resetComboOdds: () => void;
  normProbs: Map<number, number>;
}

function storageKey(raceId: string | null) {
  const suffix = raceId ? `-${raceId}` : "";
  return {
    odds: `sakura-oracle-odds${suffix}`,
    combo: `sakura-oracle-combo-odds${suffix}`,
  };
}

// --- Initial odds from predictions ---

function buildInitialOdds(
  preds: { horse_number: number; odds: { win: number; show: number } }[],
): Record<number, OddsEntry> {
  const map: Record<number, OddsEntry> = {};
  for (const h of preds) {
    map[h.horse_number] = { win: h.odds.win, show: h.odds.show };
  }
  return map;
}

// --- Kelly calculation (mirrors predictor.py calc_kelly) ---

function calcKelly(prob: number, odds: number, fraction = 0.25): number {
  if (odds <= 1.0 || prob <= 0 || prob >= 1) return 0;
  const b = odds - 1; // net odds
  const f = (prob * b - (1 - prob)) / b;
  return Math.max(0, f * fraction);
}

// --- Mark calculation (mirrors predictor.py get_mark — Kelly-based) ---

function calcMark(
  horse: { kelly_win: number; show_prob: number },
  allHorses: { kelly_win: number }[]
): string {
  const kelly = horse.kelly_win;
  // rank by kelly descending (1 = highest)
  const kellyRank =
    allHorses.filter((h) => h.kelly_win > kelly).length + 1;

  if (kellyRank === 1 && kelly > 0.01) return "◎";
  if (kellyRank <= 3 && kelly > 0.005) return "○";
  if (kelly > 0.002) return "▲";
  if (horse.show_prob >= 0.2 && kelly > 0) return "△";
  return "×";
}

// --- Bet generation (Kelly-based — mirrors predictor.py) ---

function generateBets(
  horses: LiveHorse[],
  normProbs: Map<number, number>,
  comboOddsMap: Record<string, number>,
  budget = 3000,
): Bet[] {
  const MARK_ORDER: Record<string, number> = {
    "◎": 0, "○": 1, "▲": 2, "△": 3, "×": 4,
  };
  const sorted = [...horses].sort((a, b) => {
    const oa = MARK_ORDER[a.mark] ?? 99;
    const ob = MARK_ORDER[b.mark] ?? 99;
    if (oa !== ob) return oa - ob;
    return b.kelly_win - a.kelly_win;
  });

  const topHorses = sorted.filter((h) => ["◎", "○", "▲"].includes(h.mark));
  const bets: Bet[] = [];

  // nCr計算
  const comb = (n: number, r: number) => {
    if (r > n) return 0;
    let result = 1;
    for (let i = 0; i < r; i++) result = result * (n - i) / (i + 1);
    return result;
  };

  // BT実績ROI定数
  const BT_ROI: Record<string, number> = {
    "馬連": 5.50,
    "三連複": 4.74,
    "ワイド": 4.65,
    "単勝": 2.65,
  };

  // 単勝: ◎○▲の馬のみ（最大3頭）— Kelly > 0 かつ EV > 1.0
  const winTargets = topHorses.filter((h) => h.kelly_win > 0 && h.ev_win >= 1.0).slice(0, 3);
  for (const h of winTargets) {
    const amount = Math.max(100, Math.round((budget * h.kelly_win) / 100) * 100);
    bets.push({
      type: "単勝",
      targets: `${h.horse_number}番 ${h.horse_name}`,
      description: `${h.horse_name}が1着なら的中`,
      amount,
      ev: h.ev_win,
      evReliable: true,
      odds: h.odds_win,
      kelly: h.kelly_win,
      backtestRoi: BT_ROI["単勝"],
      winReturn: Math.round(amount * h.odds_win),
    });
  }

  // 馬連BOX（上位2-3頭）→ 個別組合せに展開
  if (topHorses.length >= 2) {
    const slice = topHorses.slice(0, 3);
    const n = slice.length;
    const numCombs = comb(n, 2);
    const avgKelly = slice.reduce((s, h) => s + h.kelly_win, 0) / slice.length;
    const totalAmount = Math.max(100 * numCombs, Math.round((budget * avgKelly) / 100) * 100);
    const perCombo = Math.max(100, Math.round(totalAmount / numCombs / 100) * 100);

    for (let i = 0; i < slice.length; i++) {
      for (let j = i + 1; j < slice.length; j++) {
        const a = slice[i], b = slice[j];
        const nums = [a.horse_number, b.horse_number].sort((x, y) => x - y);
        const comboKey = `quinella-${nums[0]}-${nums[1]}`;
        const prob = quinellaProb(normProbs, nums[0], nums[1]);
        const comboOdds = comboOddsMap[comboKey] ?? null;
        const ev = comboOdds ? Math.round(prob * comboOdds * 100) / 100 : 0;
        const kelly = comboOdds ? calcKelly(prob, comboOdds) : avgKelly;

        bets.push({
          type: "馬連",
          targets: `${nums[0]}-${nums[1]}`,
          description: `${a.horse_name}と${b.horse_name}が1着2着（順不問）`,
          amount: perCombo,
          ev,
          evReliable: comboOdds !== null,
          odds: comboOdds,
          kelly,
          backtestRoi: BT_ROI["馬連"],
          comboProb: prob,
          comboKey,
        });
      }
    }
  }

  // ワイド（◎-○）
  if (topHorses.length >= 2) {
    const [h1, h2] = topHorses;
    const nums = [h1.horse_number, h2.horse_number].sort((a, b) => a - b);
    const comboKey = `wide-${nums[0]}-${nums[1]}`;
    const prob = wideProb(normProbs, nums[0], nums[1]);
    const comboOdds = comboOddsMap[comboKey] ?? null;
    const ev = comboOdds ? Math.round(prob * comboOdds * 100) / 100 : 0;
    const avgKelly = (h1.kelly_win + h2.kelly_win) / 2;
    const kelly = comboOdds ? calcKelly(prob, comboOdds) : avgKelly;
    const amount = Math.max(100, Math.round((budget * (comboOdds ? kelly : avgKelly)) / 100) * 100);
    bets.push({
      type: "ワイド",
      targets: `${nums[0]}-${nums[1]}`,
      description: `${h1.horse_name}と${h2.horse_name}が両方3着以内なら的中`,
      amount,
      ev,
      evReliable: comboOdds !== null,
      odds: comboOdds,
      kelly,
      backtestRoi: BT_ROI["ワイド"],
      comboProb: prob,
      comboKey,
    });
  }

  // 三連複BOX(5): 常に上位5頭から全10通りを展開
  // バックテスト実績: 的中32%, 回収272% (BOX(3)の93%を大幅に上回る)
  {
    const top5 = sorted
      .filter((h) => h.win_prob > 0)
      .slice(0, 5);
    if (top5.length >= 3) {
      const n = top5.length;
      const numCombs = comb(n, 3);
      const avgKelly = top5.reduce((s, h) => s + h.kelly_win, 0) / top5.length;
      const totalAmount = Math.max(100 * numCombs, Math.round((budget * avgKelly * 0.5) / 100) * 100);
      const perCombo = Math.max(100, Math.round(totalAmount / numCombs / 100) * 100);

      for (let i = 0; i < top5.length; i++) {
        for (let j = i + 1; j < top5.length; j++) {
          for (let k = j + 1; k < top5.length; k++) {
            const a = top5[i], b = top5[j], c = top5[k];
            const nums = [a.horse_number, b.horse_number, c.horse_number].sort((x, y) => x - y);
            const comboKey = `trio-${nums[0]}-${nums[1]}-${nums[2]}`;
            const prob = trioProb(normProbs, nums[0], nums[1], nums[2]);
            const comboOdds = comboOddsMap[comboKey] ?? null;
            const ev = comboOdds ? Math.round(prob * comboOdds * 100) / 100 : 0;
            const kelly = comboOdds ? calcKelly(prob, comboOdds) : avgKelly;

            bets.push({
              type: "三連複",
              targets: `${nums[0]}-${nums[1]}-${nums[2]}`,
              description: `${a.horse_name}・${b.horse_name}・${c.horse_name}が1-2-3着（順不問）`,
              amount: perCombo,
              ev,
              evReliable: comboOdds !== null,
              odds: comboOdds,
              kelly,
              backtestRoi: BT_ROI["三連複"],
              comboProb: prob,
              comboKey,
            });
          }
        }
      }
    }
  }

  // ROI降順でソート（三連複→馬連→ワイド→単勝）
  bets.sort((a, b) => b.backtestRoi - a.backtestRoi);

  return bets;
}

// --- Context ---

const OddsContext = createContext<OddsContextValue | null>(null);

export function OddsProvider({ children }: { children: ReactNode }) {
  const { predictions, selectedRaceId } = useRace();
  const prevRaceIdRef = useRef(selectedRaceId);

  const initialOdds = useMemo(
    () => buildInitialOdds(predictions.predictions),
    [predictions],
  );

  const [oddsMap, setOddsMap] = useState<Record<number, OddsEntry>>(initialOdds);
  const [comboOddsMap, setComboOddsMap] = useState<Record<string, number>>({});

  // Reset odds when race changes
  useEffect(() => {
    if (prevRaceIdRef.current !== selectedRaceId) {
      prevRaceIdRef.current = selectedRaceId;

      // Load per-race localStorage
      const keys = storageKey(selectedRaceId);
      let restored = false;
      try {
        const stored = localStorage.getItem(keys.odds);
        if (stored) {
          setOddsMap(JSON.parse(stored));
          restored = true;
        }
      } catch { /* ignore */ }

      if (!restored) {
        setOddsMap(buildInitialOdds(predictions.predictions));
      }

      try {
        const stored = localStorage.getItem(keys.combo);
        if (stored) {
          setComboOddsMap(JSON.parse(stored));
        } else {
          setComboOddsMap({});
        }
      } catch {
        setComboOddsMap({});
      }
    } else {
      // Same race but predictions data refreshed (initial load) — sync initialOdds
      setOddsMap(initialOdds);
    }
  }, [predictions, selectedRaceId, initialOdds]);

  // Hydrate from localStorage on mount
  useEffect(() => {
    const keys = storageKey(selectedRaceId);
    try {
      const stored = localStorage.getItem(keys.odds);
      if (stored) {
        setOddsMap(JSON.parse(stored));
      }
    } catch { /* ignore */ }
    try {
      const stored = localStorage.getItem(keys.combo);
      if (stored) {
        setComboOddsMap(JSON.parse(stored));
      }
    } catch { /* ignore */ }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Persist to localStorage on change
  useEffect(() => {
    const keys = storageKey(selectedRaceId);
    try { localStorage.setItem(keys.odds, JSON.stringify(oddsMap)); } catch { /* ignore */ }
  }, [oddsMap, selectedRaceId]);

  useEffect(() => {
    const keys = storageKey(selectedRaceId);
    try { localStorage.setItem(keys.combo, JSON.stringify(comboOddsMap)); } catch { /* ignore */ }
  }, [comboOddsMap, selectedRaceId]);

  const updateOdds = useCallback(
    (horseNumber: number, win: number, show: number) => {
      setOddsMap((prev) => ({
        ...prev,
        [horseNumber]: { win, show },
      }));
    },
    []
  );

  const resetOdds = useCallback(() => {
    setOddsMap(initialOdds);
    const keys = storageKey(selectedRaceId);
    try { localStorage.removeItem(keys.odds); } catch { /* ignore */ }
  }, [initialOdds, selectedRaceId]);

  const updateComboOdds = useCallback((key: string, odds: number) => {
    setComboOddsMap((prev) => ({ ...prev, [key]: odds }));
  }, []);

  const resetComboOdds = useCallback(() => {
    setComboOddsMap({});
    const keys = storageKey(selectedRaceId);
    try { localStorage.removeItem(keys.combo); } catch { /* ignore */ }
  }, [selectedRaceId]);

  // Compute live horses with dynamic EV, Kelly, mark, market_prob
  const liveHorses = useMemo(() => {
    // Market implied probability: 1/odds normalized by overround
    const rawImplied = predictions.predictions.map((h) => {
      const odds = oddsMap[h.horse_number] || { win: h.odds.win, show: h.odds.show };
      return 1 / Math.max(odds.win, 1.01);
    });
    const totalImplied = rawImplied.reduce((s, v) => s + v, 0);

    const horses = predictions.predictions.map((h, idx) => {
      const odds = oddsMap[h.horse_number] || { win: h.odds.win, show: h.odds.show };
      const ev_win = Math.round(h.win_prob * odds.win * 100) / 100;
      const ev_show = Math.round(h.show_prob * odds.show * 100) / 100;
      const kelly_win = calcKelly(h.win_prob, odds.win);
      const kelly_show = calcKelly(h.show_prob, odds.show);
      const oddsChanged =
        odds.win !== h.odds.win || odds.show !== h.odds.show;
      const market_prob = totalImplied > 0 ? rawImplied[idx] / totalImplied : 0;
      return {
        horse_number: h.horse_number,
        horse_name: h.horse_name,
        win_prob: h.win_prob,
        show_prob: h.show_prob,
        ev_win,
        ev_show,
        kelly_win,
        kelly_show,
        mark: "", // will be computed below
        odds_win: odds.win,
        odds_show: odds.show,
        market_prob,
        orig_ev_win: h.ev_win,
        orig_ev_show: h.ev_show,
        orig_mark: h.mark,
        oddsChanged,
      };
    });

    // Recalculate marks based on Kelly
    for (const h of horses) {
      h.mark = calcMark(h, horses);
    }

    return horses;
  }, [oddsMap, predictions]);

  // Harville normalized probabilities
  const normProbs = useMemo(
    () => normalizeProbabilities(liveHorses),
    [liveHorses]
  );

  const liveBets = useMemo(
    () => generateBets(liveHorses, normProbs, comboOddsMap),
    [liveHorses, normProbs, comboOddsMap]
  );

  const value = useMemo<OddsContextValue>(
    () => ({
      oddsMap,
      updateOdds,
      resetOdds,
      liveHorses,
      liveBets,
      comboOddsMap,
      updateComboOdds,
      resetComboOdds,
      normProbs,
    }),
    [oddsMap, updateOdds, resetOdds, liveHorses, liveBets, comboOddsMap, updateComboOdds, resetComboOdds, normProbs]
  );

  return (
    <OddsContext.Provider value={value}>{children}</OddsContext.Provider>
  );
}

export function useOdds(): OddsContextValue {
  const ctx = useContext(OddsContext);
  if (!ctx) throw new Error("useOdds must be used within OddsProvider");
  return ctx;
}
