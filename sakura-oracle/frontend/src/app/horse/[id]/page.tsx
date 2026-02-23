import predictions from "@/data/predictions.json";
import HorseDetailClient from "./HorseDetailClient";

export function generateStaticParams() {
  return predictions.predictions.map((h) => ({
    id: String(h.horse_number),
  }));
}

export default function HorseDetailPage() {
  return <HorseDetailClient />;
}
