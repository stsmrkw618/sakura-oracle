"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import Navbar from "@/components/Navbar";
import MarkBadge from "@/components/MarkBadge";

/* フェードインアニメーション */
const fadeIn = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
};

/* アコーディオン項目の型 */
interface AccordionItem {
  key: string;
  title: string;
  content: React.ReactNode;
}

/* ─── セクション3: 買い目ガイドのアコーディオン項目 ─── */
const betGuideItems: AccordionItem[] = [
  {
    key: "box-vs-axis",
    title: "BOX vs ◎軸流し — どちらを選ぶ？",
    content: (
      <div className="space-y-2">
        <table className="w-full text-xs border-collapse">
          <thead>
            <tr className="border-b border-white/10">
              <th className="text-left py-1.5 text-muted-foreground">方式</th>
              <th className="text-left py-1.5 text-muted-foreground">特徴</th>
              <th className="text-left py-1.5 text-muted-foreground">向いてる人</th>
            </tr>
          </thead>
          <tbody className="text-muted-foreground">
            <tr className="border-b border-white/10">
              <td className="py-1.5 font-medium text-white">BOX</td>
              <td className="py-1.5">選んだ馬の全組合せを均等に買う</td>
              <td className="py-1.5">初心者・手堅く</td>
            </tr>
            <tr>
              <td className="py-1.5 font-medium text-white">◎軸流し</td>
              <td className="py-1.5">本命◎を軸に相手へ流す</td>
              <td className="py-1.5">◎に自信あり</td>
            </tr>
          </tbody>
        </table>
        <p className="text-xs text-muted-foreground leading-relaxed">
          迷ったら <span className="text-white font-medium">BOX</span> がおすすめ。
          AIの上位馬をまんべんなくカバーできます。
        </p>
      </div>
    ),
  },
  {
    key: "ticket-types",
    title: "馬券の種類 — 三連複・馬連・ワイド・単勝",
    content: (
      <table className="w-full text-xs border-collapse">
        <thead>
          <tr className="border-b border-white/10">
            <th className="text-left py-1.5 text-muted-foreground">種類</th>
            <th className="text-left py-1.5 text-muted-foreground">条件</th>
            <th className="text-left py-1.5 text-muted-foreground">難易度</th>
          </tr>
        </thead>
        <tbody className="text-muted-foreground">
          <tr className="border-b border-white/10">
            <td className="py-1.5 font-medium text-white">単勝</td>
            <td className="py-1.5">1着を当てる</td>
            <td className="py-1.5">★☆☆</td>
          </tr>
          <tr className="border-b border-white/10">
            <td className="py-1.5 font-medium text-white">ワイド</td>
            <td className="py-1.5">3着以内の2頭を当てる</td>
            <td className="py-1.5">★☆☆</td>
          </tr>
          <tr className="border-b border-white/10">
            <td className="py-1.5 font-medium text-white">馬連</td>
            <td className="py-1.5">1-2着の2頭を当てる（順不同）</td>
            <td className="py-1.5">★★☆</td>
          </tr>
          <tr>
            <td className="py-1.5 font-medium text-white">三連複</td>
            <td className="py-1.5">1-2-3着の3頭を当てる（順不同）</td>
            <td className="py-1.5">★★★</td>
          </tr>
        </tbody>
      </table>
    ),
  },
  {
    key: "card-reading",
    title: "各カードの見方 — オッズ・EV・Kelly・金額",
    content: (
      <div className="space-y-2 text-xs text-muted-foreground leading-relaxed">
        <div className="grid grid-cols-2 gap-2">
          <div className="bg-navy/50 rounded-lg p-2">
            <span className="text-white font-medium">オッズ</span>
            <p className="mt-0.5">的中時の払戻倍率。100円賭けてオッズ10倍なら1,000円。</p>
          </div>
          <div className="bg-navy/50 rounded-lg p-2">
            <span className="text-white font-medium">EV（期待値）</span>
            <p className="mt-0.5">1.0以上ならプラス期待。高いほどお得。</p>
          </div>
          <div className="bg-navy/50 rounded-lg p-2">
            <span className="text-white font-medium">Kelly%</span>
            <p className="mt-0.5">予算に対する最適な賭け割合。数値が大きい＝自信あり。</p>
          </div>
          <div className="bg-navy/50 rounded-lg p-2">
            <span className="text-white font-medium">金額</span>
            <p className="mt-0.5">予算スライダーに応じた推奨購入額。100円単位に丸め済み。</p>
          </div>
        </div>
      </div>
    ),
  },
  {
    key: "budget-slider",
    title: "予算スライダーの使い方",
    content: (
      <div className="space-y-2 text-xs text-muted-foreground leading-relaxed">
        <p>
          買い目ページ上部のスライダーで <span className="text-white font-medium">投資予算（1,000〜50,000円）</span> を設定します。
        </p>
        <p>
          スライダーを動かすと、各買い目カードの「金額」がリアルタイムで変わります。
          Kelly基準に基づいて期待値の高い買い目に多く、低い買い目に少なく配分されます。
        </p>
        <p>
          初心者は <span className="text-white font-medium">3,000〜5,000円</span> がおすすめ。
        </p>
      </div>
    ),
  },
  {
    key: "estimated-vs-confirmed",
    title: "「推定」と「確定」の違い",
    content: (
      <div className="space-y-2 text-xs text-muted-foreground leading-relaxed">
        <table className="w-full border-collapse">
          <thead>
            <tr className="border-b border-white/10">
              <th className="text-left py-1.5 text-muted-foreground">状態</th>
              <th className="text-left py-1.5 text-muted-foreground">意味</th>
            </tr>
          </thead>
          <tbody>
            <tr className="border-b border-white/10">
              <td className="py-1.5 text-yellow-400 font-medium">推定オッズ</td>
              <td className="py-1.5">過去データからAIが予測した参考値</td>
            </tr>
            <tr>
              <td className="py-1.5 text-green-400 font-medium">確定オッズ</td>
              <td className="py-1.5">当日入力した実際のオッズ（こちらが正確）</td>
            </tr>
          </tbody>
        </table>
        <p>
          レース当日に予測ページの「オッズ更新」トグルをONにして、
          netkeibaの実際のオッズを入力すると、EVが正確になり買い目の精度が上がります。
        </p>
      </div>
    ),
  },
];

/* ─── セクション6: FAQ項目 ─── */
const faqItems: AccordionItem[] = [
  {
    key: "ev-below-1",
    title: "EVが1.0未満の買い目は買わない方がいい？",
    content: (
      <p className="text-xs text-muted-foreground leading-relaxed">
        EV 1.0未満は「期待値マイナス」なので、基本的には見送り推奨です。
        ただし推定オッズの段階では目安程度。
        当日の確定オッズ入力後にEVが変わることもあるので、確定後に最終判断しましょう。
      </p>
    ),
  },
  {
    key: "box-axis-switch",
    title: "BOXと軸流しはいつでも切り替えられる？",
    content: (
      <p className="text-xs text-muted-foreground leading-relaxed">
        はい。買い目ページ上部の「BOX / ◎軸流し」ボタンでいつでも切替可能です。
        切り替えると買い目リストと金額配分がリアルタイムで変わります。
      </p>
    ),
  },
  {
    key: "how-much-profit",
    title: "いくら儲かるの？",
    content: (
      <div className="text-xs text-muted-foreground leading-relaxed space-y-1">
        <p>
          過去50レースのバックテストでは、三連複BOXで回収率 <span className="text-gold font-bold">474%</span>、
          馬連BOXで <span className="text-gold font-bold">550%</span> でした（分析ページで詳細が見られます）。
        </p>
        <p>
          ただしこれは過去の成績であり、将来を保証するものではありません。
          あくまでエンタメとして楽しんでください！
        </p>
      </div>
    ),
  },
];

export default function HowToPage() {
  /* アコーディオン状態管理（セクション3用） */
  const [betOpen, setBetOpen] = useState<string | null>(null);
  /* アコーディオン状態管理（FAQ用） */
  const [faqOpen, setFaqOpen] = useState<string | null>(null);

  const toggleBet = (key: string) =>
    setBetOpen(betOpen === key ? null : key);
  const toggleFaq = (key: string) =>
    setFaqOpen(faqOpen === key ? null : key);

  return (
    <main className="min-h-screen bg-background pb-24">
      {/* ─── ヘッダー（sticky） ─── */}
      <motion.header
        {...fadeIn}
        className="sticky top-0 z-40 bg-navy-dark/95 backdrop-blur-md border-b border-white/5 px-4 py-3"
      >
        <h1 className="text-lg font-bold">📖 使い方ガイド</h1>
        <p className="text-xs text-muted-foreground">
          競馬初心者でも迷わない！
        </p>
      </motion.header>

      <div className="max-w-[430px] mx-auto px-4 py-4 space-y-4">
        {/* ─── セクション1: このアプリって何？ ─── */}
        <motion.section
          {...fadeIn}
          transition={{ delay: 0.05 }}
        >
          <h2 className="text-sm font-bold mb-2">🌸 このアプリって何？</h2>
          <div className="bg-card rounded-xl p-4 border border-white/5">
            <p className="text-xs text-muted-foreground leading-relaxed mb-3">
              桜花賞をAIで予測するアプリです。4つのページで構成されています。
            </p>
            <table className="w-full text-xs border-collapse">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="text-left py-1.5 text-muted-foreground">ページ</th>
                  <th className="text-left py-1.5 text-muted-foreground">何ができる？</th>
                </tr>
              </thead>
              <tbody className="text-muted-foreground">
                <tr className="border-b border-white/10">
                  <td className="py-1.5 font-medium text-white">🏠 TOP</td>
                  <td className="py-1.5">レース選択・トップ画面</td>
                </tr>
                <tr className="border-b border-white/10">
                  <td className="py-1.5 font-medium text-white">📊 予測</td>
                  <td className="py-1.5">各馬の勝率・印・EVを確認</td>
                </tr>
                <tr className="border-b border-white/10">
                  <td className="py-1.5 font-medium text-white">💰 買い目</td>
                  <td className="py-1.5">おすすめ馬券と金額配分</td>
                </tr>
                <tr>
                  <td className="py-1.5 font-medium text-white">📈 分析</td>
                  <td className="py-1.5">過去50レースの成績データ</td>
                </tr>
              </tbody>
            </table>
          </div>
        </motion.section>

        {/* ─── セクション2: 予測ページの見方 ─── */}
        <motion.section
          {...fadeIn}
          transition={{ delay: 0.1 }}
        >
          <h2 className="text-sm font-bold mb-2">📊 予測ページの見方</h2>
          <div className="bg-card rounded-xl p-4 border border-white/5 space-y-3">
            {/* 3つの数字 */}
            <div>
              <h3 className="text-xs font-bold mb-1.5">3つの数字の意味</h3>
              <div className="grid grid-cols-3 gap-2">
                <div className="bg-navy/50 rounded-lg p-2 text-center">
                  <p className="text-gold font-bold text-sm">勝率</p>
                  <p className="text-[10px] text-muted-foreground mt-0.5">
                    1着になる確率
                  </p>
                </div>
                <div className="bg-navy/50 rounded-lg p-2 text-center">
                  <p className="text-sakura-pink font-bold text-sm">複勝率</p>
                  <p className="text-[10px] text-muted-foreground mt-0.5">
                    3着以内に入る確率
                  </p>
                </div>
                <div className="bg-navy/50 rounded-lg p-2 text-center">
                  <p className="text-green-400 font-bold text-sm">EV</p>
                  <p className="text-[10px] text-muted-foreground mt-0.5">
                    100円あたりの期待リターン
                  </p>
                </div>
              </div>
            </div>

            {/* 印の見方 */}
            <div>
              <h3 className="text-xs font-bold mb-1.5">印の見方</h3>
              <table className="w-full text-xs border-collapse">
                <thead>
                  <tr className="border-b border-white/10">
                    <th className="text-left py-1.5 text-muted-foreground w-16">印</th>
                    <th className="text-left py-1.5 text-muted-foreground">意味</th>
                    <th className="text-left py-1.5 text-muted-foreground">目安</th>
                  </tr>
                </thead>
                <tbody className="text-muted-foreground">
                  <tr className="border-b border-white/10">
                    <td className="py-1.5"><MarkBadge mark="◎" size="sm" /></td>
                    <td className="py-1.5">本命</td>
                    <td className="py-1.5">最も勝つ可能性が高い</td>
                  </tr>
                  <tr className="border-b border-white/10">
                    <td className="py-1.5"><MarkBadge mark="○" size="sm" /></td>
                    <td className="py-1.5">対抗</td>
                    <td className="py-1.5">本命に次ぐ有力馬</td>
                  </tr>
                  <tr className="border-b border-white/10">
                    <td className="py-1.5"><MarkBadge mark="▲" size="sm" /></td>
                    <td className="py-1.5">単穴</td>
                    <td className="py-1.5">穴をあける可能性あり</td>
                  </tr>
                  <tr className="border-b border-white/10">
                    <td className="py-1.5"><MarkBadge mark="△" size="sm" /></td>
                    <td className="py-1.5">連下</td>
                    <td className="py-1.5">2-3着候補</td>
                  </tr>
                  <tr>
                    <td className="py-1.5"><MarkBadge mark="×" size="sm" /></td>
                    <td className="py-1.5">消し</td>
                    <td className="py-1.5">今回は厳しい</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </motion.section>

        {/* ─── セクション3: 買い目ガイドの使い方（メイン） ─── */}
        <motion.section
          {...fadeIn}
          transition={{ delay: 0.15 }}
        >
          <h2 className="text-sm font-bold mb-2">💰 買い目ガイドの使い方</h2>
          <div className="bg-card rounded-xl border border-white/5 divide-y divide-white/5">
            {betGuideItems.map((item) => (
              <div key={item.key} className="px-4">
                <button
                  onClick={() => toggleBet(item.key)}
                  className="w-full text-left text-sm py-3 flex items-center justify-between"
                >
                  <span className="font-medium text-xs">{item.title}</span>
                  <span className="text-muted-foreground text-xs ml-2 shrink-0">
                    {betOpen === item.key ? "▲" : "▼"}
                  </span>
                </button>
                {betOpen === item.key && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    className="pb-3"
                  >
                    {item.content}
                  </motion.div>
                )}
              </div>
            ))}
          </div>
        </motion.section>

        {/* ─── セクション4: 当日のオッズ入力 ─── */}
        <motion.section
          {...fadeIn}
          transition={{ delay: 0.2 }}
        >
          <h2 className="text-sm font-bold mb-2">🎯 当日のオッズ入力</h2>
          <div className="bg-card rounded-xl p-4 border border-white/5">
            <p className="text-xs text-muted-foreground leading-relaxed mb-3">
              レース当日にオッズを入力すると、期待値（EV）と買い目がより正確になります。
            </p>
            <ol className="space-y-2">
              {[
                {
                  step: "1",
                  title: "予測ページで「オッズ更新」をON",
                  desc: "各馬カードにオッズ入力欄が表示されます",
                },
                {
                  step: "2",
                  title: "netkeibaで単勝オッズを確認",
                  desc: "レース発走10〜15分前がベスト",
                },
                {
                  step: "3",
                  title: "各馬のオッズを入力",
                  desc: "入力するとEVと買い目がリアルタイムで更新されます",
                },
              ].map((item) => (
                <li key={item.step} className="flex gap-3">
                  <span className="bg-sakura-pink/20 text-sakura-pink rounded-full w-6 h-6 flex items-center justify-center text-xs font-bold shrink-0">
                    {item.step}
                  </span>
                  <div>
                    <p className="text-xs font-medium text-white">{item.title}</p>
                    <p className="text-[10px] text-muted-foreground">{item.desc}</p>
                  </div>
                </li>
              ))}
            </ol>
          </div>
        </motion.section>

        {/* ─── セクション5: 実際に馬券を買うとき ─── */}
        <motion.section
          {...fadeIn}
          transition={{ delay: 0.25 }}
        >
          <h2 className="text-sm font-bold mb-2">🏇 実際に馬券を買うとき</h2>
          <div className="bg-card rounded-xl p-4 border border-white/5">
            <ol className="space-y-2">
              {[
                {
                  step: "1",
                  title: "予算を決める",
                  desc: "買い目ページのスライダーで設定（初心者は3,000〜5,000円）",
                },
                {
                  step: "2",
                  title: "買い目を確認",
                  desc: "AI推奨TOP10の馬券種・組合せ・金額をチェック",
                },
                {
                  step: "3",
                  title: "馬券を購入",
                  desc: "JRAのネット投票（即PAT等）またはマークカードで購入",
                },
                {
                  step: "4",
                  title: "レースを楽しむ！",
                  desc: "的中時は自動で口座に払戻されます",
                },
              ].map((item) => (
                <li key={item.step} className="flex gap-3">
                  <span className="bg-gold/20 text-gold rounded-full w-6 h-6 flex items-center justify-center text-xs font-bold shrink-0">
                    {item.step}
                  </span>
                  <div>
                    <p className="text-xs font-medium text-white">{item.title}</p>
                    <p className="text-[10px] text-muted-foreground">{item.desc}</p>
                  </div>
                </li>
              ))}
            </ol>
          </div>
        </motion.section>

        {/* ─── セクション6: よくある疑問（FAQ） ─── */}
        <motion.section
          {...fadeIn}
          transition={{ delay: 0.3 }}
        >
          <h2 className="text-sm font-bold mb-2">❓ よくある疑問</h2>
          <div className="bg-card rounded-xl border border-white/5 divide-y divide-white/5">
            {faqItems.map((item) => (
              <div key={item.key} className="px-4">
                <button
                  onClick={() => toggleFaq(item.key)}
                  className="w-full text-left text-sm py-3 flex items-center justify-between"
                >
                  <span className="font-medium text-xs">{item.title}</span>
                  <span className="text-muted-foreground text-xs ml-2 shrink-0">
                    {faqOpen === item.key ? "▲" : "▼"}
                  </span>
                </button>
                {faqOpen === item.key && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    className="pb-3"
                  >
                    {item.content}
                  </motion.div>
                )}
              </div>
            ))}
          </div>
        </motion.section>
      </div>

      <Navbar />
    </main>
  );
}
