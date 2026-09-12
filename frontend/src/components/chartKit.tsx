import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import type { TimelinePoint } from "../api/types";

/**
 * 图表共用的几件小事: 按容器真实宽度画、数值补间、走势线"长出新的一段"、跟随鼠标的数值气泡。
 * 动画一律尊重系统的"减少动态效果"设置。
 */

export const reducedMotion = () =>
  typeof window !== "undefined" && !!window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

const ease = (t: number) => 1 - Math.pow(1 - t, 3);

/**
 * 容器宽度 (CSS 像素)。图表按它画, 1 个 SVG 单位 = 1 像素。
 *
 * 原来是固定 900 宽的 viewBox 整体缩放 —— 手机上宽度只剩三分之一, 字号 11 缩成四五个像素,
 * 线也细得看不清。按真实宽度画之后, 字就是 11 像素, 高度另外按宽度给, 不需要横向滑动。
 */
export function useWidth<E extends HTMLElement>(fallback = 900) {
  const ref = useRef<E>(null);
  const [w, setW] = useState(fallback);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const set = () => setW(Math.max(260, Math.round(el.getBoundingClientRect().width)));
    set();
    const ro = new ResizeObserver(set);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, w] as const;
}

/**
 * 0→1 的进度动画, 每一步回调 step(k), 最后一次保证是 k = 1。返回取消函数。
 *
 * 另挂一个定时器兜底: 页面不绘制时 (后台标签、被收起的内嵌窗口) requestAnimationFrame 会
 * **整个停掉**, 实测每秒 0 次 —— 只靠它的话, 数字和走势线会一直停在旧值上。定时器在后台
 * 也还会跑 (最慢每秒一次), 到点没走完就直接落到终点。
 */
export function tween(ms: number, step: (k: number) => void, curve: (t: number) => number = ease): () => void {
  const t0 = performance.now();
  let raf = 0;
  let done = false;
  const tick = (now: number) => {
    if (done) return;
    const t = Math.min(1, (now - t0) / ms);
    if (t >= 1) done = true;
    step(t >= 1 ? 1 : curve(t));
    if (!done) raf = requestAnimationFrame(tick);
  };
  // 起点先同步画一次: 走势线每秒追加一个点, 下一个点到的时候上一段动画会被取消 ——
  // rAF 停着的时候兜底定时器永远等不到, 不先画一次的话线就一直不长
  step(0);
  raf = requestAnimationFrame(tick);
  const timer = setTimeout(() => {
    if (done) return;
    done = true;
    cancelAnimationFrame(raf);
    step(1);
  }, ms + 150);
  return () => {
    done = true;
    cancelAnimationFrame(raf);
    clearTimeout(timer);
  };
}

/** 数值补间: 目标变了就从当前显示的值平滑过渡过去, 中途又变了就从半路接着走 */
export function useTween(target: number, ms = 650): number {
  const [v, setV] = useState(target);
  const cur = useRef(target);
  useEffect(() => {
    const from = cur.current;
    if (reducedMotion() || !Number.isFinite(target) || !Number.isFinite(from)) {
      cur.current = target;
      setV(target);
      return;
    }
    if (from === target) return;
    return tween(ms, (k) => {
      const x = k === 1 ? target : from + (target - from) * k;
      cur.current = x;
      setV(x);
    });
  }, [target, ms]);
  return v;
}

/** 两个走势点之间插值 —— 新点从上一个末端"长"出来用 */
export function lerpPoint(a: TimelinePoint, b: TimelinePoint, k: number): TimelinePoint {
  const L = (x: number, y: number) => x + (y - x) * k;
  return {
    ...b,
    minute: L(a.minute, b.minute),
    golddiff: L(a.golddiff, b.golddiff),
    probability_blue:
      a.probability_blue != null && b.probability_blue != null
        ? L(a.probability_blue, b.probability_blue)
        : b.probability_blue,
  };
}

const samePoint = (a: TimelinePoint, b: TimelinePoint) =>
  a.minute === b.minute && a.golddiff === b.golddiff && a.probability_blue === b.probability_blue;

/**
 * 走势线新增的一段: 新点从原来的末端长出来, 横轴跟着伸长。
 * 只处理"在末尾追加"这一种情况; 换局、点被重新抽样 (超过 60 个点时会重排) 就直接跳过去,
 * 硬做过渡反而会让整条线扭一下。
 *
 * 直播回放时每秒追加一个点 (web/player.ts), 要的是**一直在往前走**的感觉:
 * - 新的一段用**匀速**、时长取"上一个点到这一个点隔了多久", 于是这一段刚长完下一个点
 *   正好到, 线和横轴连续地滑, 不会每秒"冲一下、停一下" (缓出曲线就是那样)。
 *   轮询那种隔几秒才来一个点的情况仍用缓出。
 * - 回放每秒发三五帧, 大多和上一帧内容相同 (上游约 1Hz 才变): 内容没变就什么都不做,
 *   否则正在长的那一段每次都会被打断、直接跳到终点。
 */
export function useGrowingSeries(series: TimelinePoint[], ms = 750): TimelinePoint[] {
  const [out, setOut] = useState(series);
  const prev = useRef(series);
  const stop = useRef<() => void>(() => {});
  const lastAt = useRef(0);
  useEffect(() => () => stop.current(), []);
  useEffect(() => {
    const old = prev.current;
    if (old.length === series.length && old.every((p, i) => samePoint(p, series[i]!))) return;
    prev.current = series;
    stop.current();
    const appended =
      old.length > 0 &&
      series.length > old.length &&
      old.every((p, i) => Math.abs(series[i]!.minute - p.minute) < 1e-9);
    const now0 = performance.now();
    const gap = now0 - lastAt.current;
    lastAt.current = now0;
    if (!appended || reducedMotion()) {
      setOut(series);
      return;
    }
    const base = series.slice(0, old.length);
    const from = series[old.length - 1]!;
    const added = series.slice(old.length);
    // 逐秒流: 上一次追加就在一两秒前 → 匀速, 时长跟着节奏走
    const stream = added.length === 1 && gap < 2500;
    const dur = stream ? Math.min(1500, Math.max(300, gap)) : ms;
    stop.current = tween(
      dur,
      (k) => setOut(k === 1 ? series : [...base, ...added.map((p) => lerpPoint(from, p, k))]),
      stream ? (t) => t : ease,
    );
  }, [series, ms]);
  return out;
}

/** 横轴刻度间隔 (分钟): 两个刻度之间至少留 minGap 像素 */
export function pickTicks(maxM: number, plotW: number, minGap = 60): number {
  for (const s of [1, 2, 5, 10, 15, 20, 30]) if ((plotW * s) / Math.max(maxM, 1) >= minGap) return s;
  return 30;
}

/** 游戏内时间 12.4 → "12:24" */
export const clock = (m: number) => {
  const s = Math.max(0, Math.round(m * 60));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
};

/** 离指针最近的那个点 (按横坐标) */
export function nearestIndex(xs: number[], px: number): number {
  let best = 0;
  for (let i = 1; i < xs.length; i++) if (Math.abs(xs[i]! - px) < Math.abs(xs[best]! - px)) best = i;
  return best;
}

/** 首次出现时曲线从左往右画出来 —— 包在 clipPath 里用; 之后每次轮询不重播 */
export function RevealClip({ id }: { id: string }) {
  return (
    <clipPath id={id}>
      <rect x="0" y="0" width="100%" height="100%">
        {!reducedMotion() && <animate attributeName="width" from="0%" to="100%" dur="0.9s" fill="freeze" />}
      </rect>
    </clipPath>
  );
}

/**
 * 跟随指针的数值气泡。放在点的右边, 快出右边界时翻到左边; 纵向跟着点, 但不越出图表上下沿。
 * 纯展示, 不接收指针 —— 否则指针一压上去, 图表就收到 leave, 气泡闪一下消失。
 */
export function ChartTip({ x, y, W, H, children }: { x: number; y: number; W: number; H: number; children: ReactNode }) {
  const flip = x > W - 180;
  // 按百分比定位, 不按像素: W 来自 ResizeObserver, 容器变宽变窄的那一刻它可能还是旧值 ——
  // 按像素放的话气泡会飞出图表 (实测窄屏上跑到 821px, 把整页撑宽)。百分比永远跟着容器走。
  const top = Math.min(H - 40, Math.max(40, y));
  return (
    <div
      className={`chart-tip${flip ? " flip" : ""}`}
      style={{ left: `${(x / W) * 100}%`, top: `${(top / H) * 100}%` }}
    >
      {children}
    </div>
  );
}
