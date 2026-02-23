"use client";

import { useState, useEffect } from "react";

interface OddsInputProps {
  value: number;
  onChange: (v: number) => void;
  className?: string;
  placeholder?: string;
}

/**
 * 数値入力コンポーネント。
 * ローカルに文字列状態を持ち、入力途中（空文字・無効値）でも
 * UIが壊れないようにする。有効な値のみ親に伝搬。
 */
export default function OddsInput({ value, onChange, className, placeholder }: OddsInputProps) {
  const [localValue, setLocalValue] = useState(String(value));

  // 外部からの値変更（リセット等）を同期
  useEffect(() => {
    setLocalValue(String(value));
  }, [value]);

  return (
    <input
      type="number"
      step="0.1"
      min="1"
      value={localValue}
      placeholder={placeholder}
      onChange={(e) => {
        setLocalValue(e.target.value);
        const v = parseFloat(e.target.value);
        if (!isNaN(v) && v > 0) {
          onChange(v);
        }
      }}
      className={className}
    />
  );
}
