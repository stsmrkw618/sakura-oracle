/**
 * Harville確率エンジン
 *
 * LightGBMの個別勝率から、Harvilleモデルを使って
 * 組合せ馬券（馬連・ワイド・三連複）の的中確率を導出する。
 *
 * Pure関数モジュール（React依存なし）
 */

/** 確率を正規化（LightGBMのwin_probは合計≠1.0なため） */
export function normalizeProbabilities(
  horses: { horse_number: number; win_prob: number }[]
): Map<number, number> {
  const total = horses.reduce((s, h) => s + h.win_prob, 0);
  const map = new Map<number, number>();
  for (const h of horses) {
    map.set(h.horse_number, total > 0 ? h.win_prob / total : 0);
  }
  return map;
}

/**
 * Harville: P(A 1着, B 2着) = p_A × p_B / (1 - p_A)
 */
function exactaProb(
  normProbs: Map<number, number>,
  first: number,
  second: number
): number {
  const pA = normProbs.get(first) ?? 0;
  const pB = normProbs.get(second) ?? 0;
  if (pA >= 1) return 0;
  return pA * (pB / (1 - pA));
}

/**
 * 馬連: P(AとBが1-2着, 順不問)
 * = P(A1st,B2nd) + P(B1st,A2nd)
 */
export function quinellaProb(
  normProbs: Map<number, number>,
  a: number,
  b: number
): number {
  return exactaProb(normProbs, a, b) + exactaProb(normProbs, b, a);
}

/**
 * Harville: P(A 1着, B 2着, C 3着)
 * = p_A × [p_B / (1 - p_A)] × [p_C / (1 - p_A - p_B)]
 */
function trifectaProb(
  normProbs: Map<number, number>,
  first: number,
  second: number,
  third: number
): number {
  const pA = normProbs.get(first) ?? 0;
  const pB = normProbs.get(second) ?? 0;
  const pC = normProbs.get(third) ?? 0;
  const denom1 = 1 - pA;
  const denom2 = 1 - pA - pB;
  if (denom1 <= 0 || denom2 <= 0) return 0;
  return pA * (pB / denom1) * (pC / denom2);
}

/**
 * 三連複: P(A,B,Cが全員1-2-3着, 順不問)
 * = 6通りの順列の合計
 */
export function trioProb(
  normProbs: Map<number, number>,
  a: number,
  b: number,
  c: number
): number {
  return (
    trifectaProb(normProbs, a, b, c) +
    trifectaProb(normProbs, a, c, b) +
    trifectaProb(normProbs, b, a, c) +
    trifectaProb(normProbs, b, c, a) +
    trifectaProb(normProbs, c, a, b) +
    trifectaProb(normProbs, c, b, a)
  );
}

/**
 * ワイド: P(AとBが両方3着以内)
 * = Σ(全他馬k) trio(A, B, k)
 */
export function wideProb(
  normProbs: Map<number, number>,
  a: number,
  b: number
): number {
  let sum = 0;
  const keys = Array.from(normProbs.keys());
  for (const k of keys) {
    if (k === a || k === b) continue;
    sum += trioProb(normProbs, a, b, k);
  }
  return sum;
}
