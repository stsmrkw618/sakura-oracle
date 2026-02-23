"use client";

import {
  createContext,
  useContext,
  useState,
  useMemo,
  useCallback,
  useEffect,
  type ReactNode,
} from "react";
import predictions from "@/data/predictions.json";

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
  amount: number;
  ev: number;
  odds: number | null;
}

interface OddsContextValue {
  oddsMap: Record<number, OddsEntry>;
  updateOdds: (horseNumber: number, win: number, show: number) => void;
  resetOdds: () => void;
  liveHorses: LiveHorse[];
  liveBets: Bet[];
}

const STORAGE_KEY = "sakura-oracle-odds";

// --- Initial odds from predictions.json ---

function buildInitialOdds(): Record<number, OddsEntry> {
  const map: Record<number, OddsEntry> = {};
  for (const h of predictions.predictions) {
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

function generateBets(horses: LiveHorse[], budget = 3000): Bet[] {
  const MARK_ORDER: Record<string, number> = {
    "◎": 0, "○": 1, "▲": 2, "△": 3, "×": 4,
  };
  const sorted = [...horses].sort((a, b) => {
    const oa = MARK_ORDER[a.mark] ?? 99;
    const ob = MARK_ORDER[b.mark] ?? 99;
    if (oa !== ob) return oa - ob;
    return b.kelly_win - a.kelly_win;
  });

  const kellyHorses = sorted.filter((h) => h.kelly_win > 0);
  const topHorses = sorted.filter((h) => ["◎", "○", "▲"].includes(h.mark));
  const bets: Bet[] = [];

  // 単勝: Kelly fraction > 0 の馬に比例配分
  for (const h of kellyHorses) {
    const amount = Math.max(100, Math.round((budget * h.kelly_win) / 100) * 100);
    bets.push({
      type: "単勝",
      targets: `${h.horse_number}番`,
      amount,
      ev: h.ev_win,
      odds: null,
    });
  }

  // 馬連BOX
  if (topHorses.length >= 2) {
    const slice = topHorses.slice(0, 3);
    const targets = slice.map((h) => h.horse_number).join("-");
    const avgEv = slice.reduce((s, h) => s + h.ev_win, 0) / slice.length;
    const avgKelly = slice.reduce((s, h) => s + h.kelly_win, 0) / slice.length;
    const amount = Math.max(100, Math.round((budget * avgKelly) / 100) * 100);
    bets.push({
      type: "馬連BOX",
      targets,
      amount,
      ev: Math.round(avgEv * 100) / 100,
      odds: null,
    });
  }

  // 三連複BOX
  if (topHorses.length >= 3) {
    const top5 = sorted
      .filter((h) => ["◎", "○", "▲", "△"].includes(h.mark))
      .slice(0, 5);
    if (top5.length >= 3) {
      const targets = top5.map((h) => h.horse_number).join("-");
      const avgEv = top5.reduce((s, h) => s + h.ev_win, 0) / top5.length;
      const avgKelly = top5.reduce((s, h) => s + h.kelly_win, 0) / top5.length;
      const amount = Math.max(100, Math.round((budget * avgKelly * 0.5) / 100) * 100);
      bets.push({
        type: "三連複BOX",
        targets,
        amount,
        ev: Math.round(avgEv * 100) / 100,
        odds: null,
      });
    }
  }

  // ワイド
  if (topHorses.length >= 2) {
    const [h1, h2] = topHorses;
    const avgKelly = (h1.kelly_win + h2.kelly_win) / 2;
    const amount = Math.max(100, Math.round((budget * avgKelly) / 100) * 100);
    bets.push({
      type: "ワイド",
      targets: `${h1.horse_number}-${h2.horse_number}`,
      amount,
      ev: Math.round(((h1.ev_win + h2.ev_win) / 2) * 100) / 100,
      odds: null,
    });
  }

  return bets;
}

// --- Context ---

const OddsContext = createContext<OddsContextValue | null>(null);

export function OddsProvider({ children }: { children: ReactNode }) {
  const initialOdds = useMemo(() => buildInitialOdds(), []);

  const [oddsMap, setOddsMap] = useState<Record<number, OddsEntry>>(() => {
    // Try loading from localStorage (deferred to useEffect for SSR safety)
    return initialOdds;
  });

  // Hydrate from localStorage on mount
  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        const parsed = JSON.parse(stored) as Record<number, OddsEntry>;
        setOddsMap(parsed);
      }
    } catch {
      // ignore parse errors
    }
  }, []);

  // Persist to localStorage on change
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(oddsMap));
    } catch {
      // ignore quota errors
    }
  }, [oddsMap]);

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
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch {
      // ignore
    }
  }, [initialOdds]);

  // Compute live horses with dynamic EV, Kelly, mark
  const liveHorses = useMemo(() => {
    const horses = predictions.predictions.map((h) => {
      const odds = oddsMap[h.horse_number] || { win: h.odds.win, show: h.odds.show };
      const ev_win = Math.round(h.win_prob * odds.win * 100) / 100;
      const ev_show = Math.round(h.show_prob * odds.show * 100) / 100;
      const kelly_win = calcKelly(h.win_prob, odds.win);
      const kelly_show = calcKelly(h.show_prob, odds.show);
      const oddsChanged =
        odds.win !== h.odds.win || odds.show !== h.odds.show;
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
  }, [oddsMap]);

  const liveBets = useMemo(() => generateBets(liveHorses), [liveHorses]);

  const value = useMemo<OddsContextValue>(
    () => ({ oddsMap, updateOdds, resetOdds, liveHorses, liveBets }),
    [oddsMap, updateOdds, resetOdds, liveHorses, liveBets]
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
