"use client";

import { OddsProvider } from "@/context/OddsContext";

export default function Providers({ children }: { children: React.ReactNode }) {
  return <OddsProvider>{children}</OddsProvider>;
}
