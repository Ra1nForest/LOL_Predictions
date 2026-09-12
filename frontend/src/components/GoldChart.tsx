import { useState } from "react";
import type { TimelinePoint } from "../api/types";
// 这个文件里 T 是图表的上边距, 翻译函数换个名字引进来
import { T as tr } from "../i18n";
import { ChartTip, clock, nearestIndex, pickTicks, RevealClip, useGrowingSeries, useWidth } from "./chartKit";

interface Props {
  series: TimelinePoint[];
  blueName: string;
  redName: string;
}

const L = 52;
const R = 16;
const T = 12;
const B = 24;

const kfmt = (g: number) => `${g > 0 ? "+" : g < 0 ? "−" : ""}${(Math.abs(g) / 1000).toFixed(1)}k`;

/**
 * 经济差走势。零线两侧分别染色, 谁在上面谁领先。
 * 两侧的峰值各画一条虚线并标上大小 —— "最多领先过多少"是看经济图时最常问的一句,
 * 原来只能对着纵轴刻度估。尺寸按容器真实宽度画, 见 chartKit.useWidth。
 */
export function GoldChart({ series, blueName, redName }: Props) {
  const [box, W] = useWidth<HTMLDivElement>();
  const H = Math.round(Math.min(240, Math.max(180, W * 0.22)));
  const pts = useGrowingSeries(series);
  const [hover, setHover] = useState<number | null>(null);

  if (!pts.length) {
    return (
      <div ref={box} className="chart-wrap">
        <svg viewBox={`0 0 ${W} ${H}`} className="chart">
          <text x={W / 2} y={H / 2} textAnchor="middle" fontSize="14" fill="var(--ink-3)">
            {tr("还没有经济数据")}
          </text>
        </svg>
      </div>
    );
  }

  const maxM = Math.max(...pts.map((p) => p.minute), 1);
  // 上下对称, 这样零线永远在正中间 —— 否则领先方一换边, 图形会整个跳一下
  const span = Math.max(1000, ...pts.map((p) => Math.abs(p.golddiff)));
  const x = (m: number) => L + ((W - L - R) * m) / maxM;
  const y = (g: number) => T + ((H - T - B) * (1 - g / span)) / 2;
  const zero = y(0);

  const path = pts
    .map((p, i) => `${i ? "L" : "M"}${x(p.minute)} ${y(p.golddiff)}`)
    .join(" ");
  const last = pts[pts.length - 1]!;
  const area = `${path} L${x(last.minute)} ${zero} L${x(pts[0]!.minute)} ${zero} Z`;

  const step = span > 8000 ? 5000 : span > 3000 ? 2000 : 1000;
  const gridVals: number[] = [];
  for (let g = -span; g <= span; g += step) if (Math.abs(g) > 1) gridVals.push(g);

  const ticks = pickTicks(maxM, W - L - R);
  const xLabels: number[] = [];
  for (let m = 0; m <= maxM; m += ticks) xLabels.push(m);

  // 两侧峰值: 蓝方最多领先多少、红方最多领先多少 (没领先过的那一侧不画)
  const peaks: { g: number; side: "blue" | "red"; m: number }[] = [];
  const hiP = pts.reduce((a, p) => (p.golddiff > a.golddiff ? p : a), pts[0]!);
  const loP = pts.reduce((a, p) => (p.golddiff < a.golddiff ? p : a), pts[0]!);
  if (hiP.golddiff > 0) peaks.push({ g: hiP.golddiff, side: "blue", m: hiP.minute });
  if (loP.golddiff < 0) peaks.push({ g: loP.golddiff, side: "red", m: loP.minute });

  const hp = hover != null && hover < pts.length ? pts[hover]! : null;
  const onMove = (e: React.PointerEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - r.left) * W) / r.width;
    setHover(nearestIndex(pts.map((p) => x(p.minute)), px));
  };

  return (
    <div ref={box} className="chart-wrap">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="chart"
        onPointerMove={onMove}
        onPointerDown={onMove}
        onPointerLeave={() => setHover(null)}
      >
        <defs>
          <clipPath id="g-above">
            <rect x={L} y={T} width={W - L - R} height={zero - T} />
          </clipPath>
          <clipPath id="g-below">
            <rect x={L} y={zero} width={W - L - R} height={H - B - zero} />
          </clipPath>
          <RevealClip id="gc-reveal" />
        </defs>

        {gridVals.map((g) => (
          <g key={g}>
            <line x1={L} x2={W - R} y1={y(g)} y2={y(g)} stroke="var(--hair-2)" />
            <text x={L - 8} y={y(g) + 3.5} textAnchor="end" fontSize="10.5" fill="var(--ink-3)">
              {g > 0 ? "+" : ""}
              {Math.round(g / 1000)}k
            </text>
          </g>
        ))}
        <line x1={L} x2={W - R} y1={zero} y2={zero} stroke="var(--ink-3)" strokeDasharray="4 3" />

        {xLabels.map((m) => (
          <text key={m} x={x(m)} y={H - 6} textAnchor="middle" fontSize="10.5" fill="var(--ink-3)">
            {m}
          </text>
        ))}

        {/* 峰值虚线 + 标注。标注放在左侧 —— 右上/右下角是"X 领先"那两行字, 而绝对值
            更大的那个峰正好顶在图的上沿或下沿 (纵轴按它定的), 放右边一定撞上。 */}
        {peaks.map((pk) => (
          <g key={pk.side} pointerEvents="none">
            <line
              x1={L}
              x2={W - R}
              y1={y(pk.g)}
              y2={y(pk.g)}
              stroke={`var(--${pk.side})`}
              strokeWidth="1"
              strokeDasharray="5 4"
              opacity="0.7"
            />
            <text
              x={L + 6}
              y={pk.side === "blue" ? y(pk.g) + 13 : y(pk.g) - 5}
              fontSize="10.5"
              fontWeight="600"
              fill={`var(--${pk.side})`}
            >
              {tr("峰值 {v}", { v: kfmt(pk.g) })} · {clock(pk.m)}
            </text>
          </g>
        ))}

        <g clipPath="url(#gc-reveal)">
          <path d={area} fill="var(--blue)" opacity="0.2" clipPath="url(#g-above)" />
          <path d={area} fill="var(--red)" opacity="0.2" clipPath="url(#g-below)" />
          <path d={path} fill="none" stroke="var(--ink-2)" strokeWidth="1.8" strokeLinejoin="round" />
        </g>

        <text x={W - R} y={T + 11} textAnchor="end" fontSize="10.5" fill="var(--blue)">
          {tr("{team} 领先", { team: blueName })}
        </text>
        <text x={W - R} y={H - B - 2} textAnchor="end" fontSize="10.5" fill="var(--red)">
          {tr("{team} 领先", { team: redName })}
        </text>

        {hp && (
          <g pointerEvents="none">
            <line
              x1={x(hp.minute)}
              x2={x(hp.minute)}
              y1={T}
              y2={H - B}
              stroke="var(--ink-3)"
              strokeWidth="1"
              strokeDasharray="3 3"
            />
            <circle
              cx={x(hp.minute)}
              cy={y(hp.golddiff)}
              r="4.5"
              fill={hp.golddiff >= 0 ? "var(--blue)" : "var(--red)"}
              stroke="var(--panel-strong)"
              strokeWidth="2"
            />
          </g>
        )}
      </svg>
      {hp && (
        <ChartTip x={x(hp.minute)} y={y(hp.golddiff)} W={W} H={H}>
          <div className="tip-time">{clock(hp.minute)}</div>
          <div className={hp.golddiff > 0 ? "blue" : hp.golddiff < 0 ? "red" : ""}>
            {hp.golddiff === 0 ? (
              tr("经济持平")
            ) : (
              <>
                <b>{hp.golddiff > 0 ? blueName : redName}</b> +{Math.round(Math.abs(hp.golddiff)).toLocaleString()}
              </>
            )}
          </div>
          <div>
            {tr("击杀")} <span className="blue">{hp.blue_kills}</span> : <span className="red">{hp.red_kills}</span>
          </div>
        </ChartTip>
      )}
    </div>
  );
}
