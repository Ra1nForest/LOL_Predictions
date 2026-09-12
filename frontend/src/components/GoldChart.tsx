import type { TimelinePoint } from "../api/types";

interface Props {
  series: TimelinePoint[];
  blueName: string;
  redName: string;
}

const W = 900;
const H = 170;
const L = 52;
const R = 16;
const T = 12;
const B = 24;

/** 经济差走势。零线两侧分别染色, 谁在上面谁领先。 */
export function GoldChart({ series, blueName, redName }: Props) {
  if (!series.length) {
    return (
      <svg viewBox={`0 0 ${W} ${H}`} className="chart">
        <text x={W / 2} y={H / 2} textAnchor="middle" fontSize="14" fill="var(--ink-3)">
          还没有经济数据
        </text>
      </svg>
    );
  }

  const maxM = Math.max(...series.map((p) => p.minute), 1);
  // 上下对称, 这样零线永远在正中间 —— 否则领先方一换边, 图形会整个跳一下
  const span = Math.max(1000, ...series.map((p) => Math.abs(p.golddiff)));
  const x = (m: number) => L + ((W - L - R) * m) / maxM;
  const y = (g: number) => T + ((H - T - B) * (1 - g / span)) / 2;
  const zero = y(0);

  const path = series
    .map((p, i) => `${i ? "L" : "M"}${x(p.minute)} ${y(p.golddiff)}`)
    .join(" ");
  const last = series[series.length - 1]!;
  const area = `${path} L${x(last.minute)} ${zero} L${x(series[0]!.minute)} ${zero} Z`;

  const step = span > 8000 ? 5000 : span > 3000 ? 2000 : 1000;
  const gridVals: number[] = [];
  for (let g = -span; g <= span; g += step) if (Math.abs(g) > 1) gridVals.push(g);

  const ticks = maxM > 30 ? 10 : 5;
  const xLabels: number[] = [];
  for (let m = 0; m <= maxM; m += ticks) xLabels.push(m);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart">
      <defs>
        <clipPath id="g-above">
          <rect x={L} y={T} width={W - L - R} height={zero - T} />
        </clipPath>
        <clipPath id="g-below">
          <rect x={L} y={zero} width={W - L - R} height={H - B - zero} />
        </clipPath>
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

      <path d={area} fill="var(--blue)" opacity="0.2" clipPath="url(#g-above)" />
      <path d={area} fill="var(--red)" opacity="0.2" clipPath="url(#g-below)" />
      <path d={path} fill="none" stroke="var(--ink-2)" strokeWidth="1.8" strokeLinejoin="round" />

      <text x={W - R} y={T + 11} textAnchor="end" fontSize="10.5" fill="var(--blue)">
        {blueName} 领先
      </text>
      <text x={W - R} y={H - B - 2} textAnchor="end" fontSize="10.5" fill="var(--red)">
        {redName} 领先
      </text>
    </svg>
  );
}
