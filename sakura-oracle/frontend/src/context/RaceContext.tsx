"use client";

import {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  useMemo,
  type ReactNode,
} from "react";
import defaultPredictions from "@/data/predictions.json";

// --- Types ---

export interface RaceIndexEntry {
  id: string;
  label: string;
  date: string;
  course: string;
}

export type PredictionsData = typeof defaultPredictions;

interface RaceContextValue {
  races: RaceIndexEntry[];
  selectedRaceId: string | null;
  selectRace: (id: string) => void;
  predictions: PredictionsData;
  isLoading: boolean;
}

const STORAGE_KEY = "sakura-oracle-selected-race";
const DEFAULT_RACE_ID = "__default__";

const RaceContext = createContext<RaceContextValue | null>(null);

export function RaceProvider({ children }: { children: ReactNode }) {
  const [races, setRaces] = useState<RaceIndexEntry[]>([]);
  const [selectedRaceId, setSelectedRaceId] = useState<string | null>(null);
  const [predictions, setPredictions] = useState<PredictionsData>(defaultPredictions);
  const [isLoading, setIsLoading] = useState(false);

  // Fetch race index on mount
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch("/races/index.json");
        if (!res.ok) return;
        const data: RaceIndexEntry[] = await res.json();
        if (cancelled || !Array.isArray(data) || data.length === 0) return;
        setRaces(data);

        // Restore from localStorage or default to first
        const stored = localStorage.getItem(STORAGE_KEY);
        const validId = data.find((r) => r.id === stored)?.id ?? data[0].id;
        setSelectedRaceId(validId);
      } catch {
        // No index.json — use default static data (backward compat)
        setSelectedRaceId(DEFAULT_RACE_ID);
      }
    })();
    return () => { cancelled = true; };
  }, []);

  // Fetch race data when selection changes
  useEffect(() => {
    if (!selectedRaceId || selectedRaceId === DEFAULT_RACE_ID) {
      setPredictions(defaultPredictions);
      return;
    }

    let cancelled = false;
    (async () => {
      setIsLoading(true);
      try {
        const res = await fetch(`/races/${selectedRaceId}.json`);
        if (!res.ok) throw new Error("fetch failed");
        const data: PredictionsData = await res.json();
        if (!cancelled) setPredictions(data);
      } catch {
        if (!cancelled) setPredictions(defaultPredictions);
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [selectedRaceId]);

  // Persist selection
  useEffect(() => {
    if (selectedRaceId && selectedRaceId !== DEFAULT_RACE_ID) {
      try { localStorage.setItem(STORAGE_KEY, selectedRaceId); } catch { /* ignore */ }
    }
  }, [selectedRaceId]);

  const selectRace = useCallback((id: string) => {
    setSelectedRaceId(id);
  }, []);

  const value = useMemo<RaceContextValue>(
    () => ({ races, selectedRaceId, selectRace, predictions, isLoading }),
    [races, selectedRaceId, selectRace, predictions, isLoading],
  );

  return (
    <RaceContext.Provider value={value}>{children}</RaceContext.Provider>
  );
}

export function useRace(): RaceContextValue {
  const ctx = useContext(RaceContext);
  if (!ctx) throw new Error("useRace must be used within RaceProvider");
  return ctx;
}
