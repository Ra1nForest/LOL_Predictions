import { useLayoutEffect, useRef } from "react";
import { reducedMotion, tween } from "./chartKit";

interface Props {
  value: number;
  /** 显示格式; 补间途中拿到的是小数, 取整在这里做 */
  fmt?: (n: number) => string;
  ms?: number;
  /** 计数类 (人头、塔、龙、等级…): 不滚动, 直接跳到新值, 增加时弹一下 */
  pop?: boolean;
  className?: string;
}

/**
 * 会动的数字。值变了就从当前显示的数平滑滚过去 (中途又变就从半路接着滚)。
 *
 * **直接改 DOM 的 textContent, 不走 React 状态**: 回放每秒推一帧, 记分板上三四十个数同时在滚,
 * 要是每个都 setState, 整块记分板 (连同二十张装备图) 会以 60 帧重渲染。这样只有这个 span 在变。
 * 为此 span 不交给 React 渲染子节点 —— 两边同时管一个文本节点, React 下次更新会写到已经
 * 被替换掉的旧节点上。
 */
export function Num({ value, fmt = (n) => String(Math.round(n)), ms = 600, pop = false, className }: Props) {
  const ref = useRef<HTMLSpanElement>(null);
  const shown = useRef<number | null>(null);
  const fmtRef = useRef(fmt);
  fmtRef.current = fmt;

  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const from = shown.current;
    const still = from === null || from === value || !Number.isFinite(from) || !Number.isFinite(value);
    if (still || reducedMotion() || pop) {
      shown.current = value;
      el.textContent = fmtRef.current(value);
      if (pop && !still && value > from! && !reducedMotion()) {
        el.animate?.(
          [{ transform: "scale(1)" }, { transform: "scale(1.35)", offset: 0.3 }, { transform: "scale(1)" }],
          { duration: 450, easing: "ease-out" },
        );
      }
      return;
    }
    return tween(ms, (k) => {
      const x = k === 1 ? value : from! + (value - from!) * k;
      shown.current = x;
      el.textContent = fmtRef.current(x);
    });
  }, [value, ms, pop]);

  // 格式变了 (比如切了语言) 而值没变: 按当前显示的数重写一遍
  useLayoutEffect(() => {
    if (ref.current && shown.current !== null) ref.current.textContent = fmtRef.current(shown.current);
  });

  return <span ref={ref} className={className ? `num ${className}` : "num"} />;
}
