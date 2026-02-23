# SAKURA ORACLE — 桜花賞AI予測システム

## プロジェクト概要
- **目的**: 2026年4月12日の桜花賞（阪神芝1600m）をAIで予測し、スマホで閲覧できるWebアプリを作る
- **利用シーン**: 懇親会でスマホから閲覧、当日のオッズ入力で期待値リアルタイム更新

## 技術スタック
- **フロントエンド**: Next.js 14 (App Router) + TypeScript + Tailwind CSS + shadcn/ui + Recharts + framer-motion
- **ML/データ**: Python 3.11 + LightGBM + pandas + BeautifulSoup
- **デプロイ**: Vercel（静的JSON配置、APIサーバー不要）

## ディレクトリ構造
```
sakura-oracle/
├── frontend/           # Next.js アプリ（スマホファースト）
│   ├── src/app/        # App Router ページ
│   ├── src/components/ # UIコンポーネント
│   └── src/data/       # predictions.json配置
├── ml/                 # Python 予測パイプライン
│   ├── scraper/        # netkeibaスクレイパー
│   ├── model/          # LightGBMモデル
│   └── output/         # JSON出力先
├── data/               # 生データCSV
└── docs/               # 設計書
```

## コーディングルール

### TypeScript / React
- TypeScriptはstrictモード
- コンポーネントは全て`"use client"`（クライアントコンポーネント）
- スマホファースト: 375px基準、max-width: 430pxで中央寄せ
- フォント: Noto Sans JP（本文）、JetBrains Mono（数値）
- アニメーション: framer-motion使用

### デザインカラー
- Background: `#0F0F1A`（ダークモード基調）
- Primary（桜ピンク）: `#E8879C`
- Accent（ゴールド）: `#FFD700`
- Navy: `#1A1A2E`
- Text: `#FFFFFF` / `#A0A0B0`

### 印バッジカラー
- ◎ 本命: ゴールド `#FFD700`
- ○ 対抗: 桜ピンク `#E8879C`
- ▲ 単穴: オレンジ `#FF8C00`
- △ 連下: グレー `#6B7280`
- × 消し: ダークグレー `#374151`

### 枠色（競馬正式）
1白, 2黒, 3赤, 4青, 5黄, 6緑, 7橙, 8桃

### Python
- type hints必須、docstring必須
- スクレイピング: 3〜7秒のランダム間隔（固定間隔NG）
- User-Agentヘッダー必須
- DB版URL（db.netkeiba.com）優先
- キャッシュ機能（pickle）で途中再開対応
