"use client";

import { RaceProvider } from "@/context/RaceContext";
import { OddsProvider } from "@/context/OddsContext";

export default function Providers({ children }: { children: React.ReactNode }) {
  return (
    <RaceProvider>
      <OddsProvider>{children}</OddsProvider>
    </RaceProvider>
  );
}
