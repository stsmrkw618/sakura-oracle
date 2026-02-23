import HorseDetailClient from "./HorseDetailClient";

export function generateStaticParams() {
  return Array.from({ length: 18 }, (_, i) => ({
    id: String(i + 1),
  }));
}

export default function HorseDetailPage() {
  return <HorseDetailClient />;
}
