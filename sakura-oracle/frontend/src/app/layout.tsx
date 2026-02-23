import type { Metadata } from "next";
import { Noto_Sans_JP, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import Providers from "@/components/Providers";
import predictions from "@/data/predictions.json";

const notoSansJP = Noto_Sans_JP({
  subsets: ["latin"],
  variable: "--font-noto",
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains",
  display: "swap",
});

const raceName = predictions.race_info.name;

export const metadata: Metadata = {
  title: `SAKURA ORACLE | ${raceName} AI予測`,
  description: `AIが導き出した${raceName}の最強予測。LightGBMによる勝率・期待値分析で最適な買い目を提案。`,
  openGraph: {
    title: `SAKURA ORACLE | ${raceName} AI予測`,
    description: `AIが導き出した${raceName}の最強予測`,
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ja">
      <body
        className={`${notoSansJP.variable} ${jetbrainsMono.variable} font-sans antialiased bg-navy-dark text-white`}
      >
        <Providers>
          <div className="mx-auto max-w-[430px] min-h-screen relative">
            {children}
          </div>
        </Providers>
      </body>
    </html>
  );
}
