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

// --- Mark calculation (mirrors predictor.py get_mark) ---

function calcMark(
  horse: { ev_win: number; win_prob: number; show_prob: number },
  allHorses: { ev_win: number; win_prob: number; show_prob: number }[]
): string {
  // rank by win_prob descending (1 = highest)
  const winRank =
    allHorses.filter((h) => h.win_prob > horse.win_prob).length + 1;
  // rank by ev_win descending
  const evRank =
    allHorses.filter((h) => h.ev_win > horse.ev_win).length + 1;

  if (winRank === 1 && horse.ev_win >= 1.3) return "◎";
  if (winRank <= 2 || evRank <= 2) return "○";
  if (horse.ev_win >= 1.5) return "▲";
  if (horse.show_prob >= 0.3) return "△";
  return "×";
}

// --- Bet generation (mirrors predictor.py L421-462) ---

function generateBets(horses: LiveHorse[]): Bet[] {
  const MARK_ORDER: Record<string, number> = {
    "◎": 0, "○": 1, "▲": 2, "△": 3, "×": 4,
  };
  const sorted = [...horses].sort((a, b) => {
    const oa = MARK_ORDER[a.mark] ?? 99;
    const ob = MARK_ORDER[b.mark] ?? 99;
    if (oa !== ob) return oa - ob;
    return b.win_prob - a.win_prob;
  });

  const topHorses = sorted.filter((h) => ["◎", "○", "▲"].includes(h.mark));
  const honmei = sorted[0];
  const bets: Bet[] = [];

  if (honmei && honmei.ev_win >= 1.0) {
    bets.push({
      type: "単勝",
      targets: `${honmei.horse_number}番`,
      amount: 500,
      ev: honmei.ev_win,
      odds: null,
    });
  }
  if (topHorses.length >= 2) {
    const targets = topHorses
      .slice(0, 3)
      .map((h) => h.horse_number)
      .join("-");
    const avgEv =
      topHorses.slice(0, 3).reduce((s, h) => s + h.ev_win, 0) /
      Math.min(topHorses.length, 3);
    bets.push({
      type: "馬連BOX",
      targets,
      amount: 600,
      ev: Math.round(avgEv * 100) / 100,
      odds: null,
    });
  }
  if (topHorses.length >= 3) {
    const top5 = sorted.slice(0, 5);
    const targets = top5.map((h) => h.horse_number).join("-");
    const avgEv = top5.reduce((s, h) => s + h.ev_win, 0) / top5.length;
    bets.push({
      type: "三連複BOX",
      targets,
      amount: 1200,
      ev: Math.round(avgEv * 100) / 100,
      odds: null,
    });
  }
  if (topHorses.length >= 2) {
    bets.push({
      type: "ワイド",
      targets: `${topHorses[0].horse_number}-${topHorses[1].horse_number}`,
      amount: 700,
      ev:
        Math.round(
          ((topHorses[0].ev_win + topHorses[1].ev_win) / 2) * 100
        ) / 100,
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

  // Compute live horses with dynamic EV, mark
  const liveHorses = useMemo(() => {
    const horses = predictions.predictions.map((h) => {
      const odds = oddsMap[h.horse_number] || { win: h.odds.win, show: h.odds.show };
      const ev_win = Math.round(h.win_prob * odds.win * 100) / 100;
      const ev_show = Math.round(h.show_prob * odds.show * 100) / 100;
      const oddsChanged =
        odds.win !== h.odds.win || odds.show !== h.odds.show;
      return {
        horse_number: h.horse_number,
        horse_name: h.horse_name,
        win_prob: h.win_prob,
        show_prob: h.show_prob,
        ev_win,
        ev_show,
        mark: "", // will be computed below
        orig_ev_win: h.ev_win,
        orig_ev_show: h.ev_show,
        orig_mark: h.mark,
        oddsChanged,
      };
    });

    // Recalculate marks based on new EVs
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
