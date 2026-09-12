import { useMemo, useState } from "react";
import type { TimelinePoint } from "../api/types";
// 这个文件里 T 是图表的上边距, 翻译函数换个名字引进来
import { T as tr } from "../i18n";
import { ChartTip, clock, nearestIndex, pickTicks, RevealClip, useGrowingSeries, useWidth } from "./chartKit";

interface Props {
  series: TimelinePoint[];
  pMin: number | null;
  pMax: number | null;
  blueName: string;
  redName: string;
  /** 赛前概率 —— 画成参考线, 让"局势移动了多少"看得见 */
  pregame: number | null;
  /** BP 后概率, 有就用它当曲线起点 */
  postdraft: number | null;
  /** 末端标记画成什么。三态而不是布尔 ——
   *  "live"     还在推进  -> 呼吸的光点
   *  "paused"   暂停或帧流中断 -> 定住的暂停符号 (**不是**奖杯: 这局还没分
   *             胜负, 画奖杯等于宣布结果, 而暂停随时可能继续)
   *  "finished" 正常打完  -> 奖杯, 用领先方的颜色 */
  endState: "live" | "paused" | "finished";
}

const L = 46;
const R = 16;
const T = 14;
const B = 26;

/**
 * 胜率走势。
 *
 * 两个设计是有意为之, 迁移时原样保留:
 *
 * 1. **纵轴对称**, 中线 50/50, 上下都标到 100 —— 上面那个 100 是蓝方的,
 *    下面那个是红方的。标签始终是**领先方**的胜率。
 * 2. **死区画成斜线**。校准把概率夹在 [p_min, p_max] 里, 够不到的那两段
 *    是模型**不能**表达的区间。把它画出来, 比让人以为"顶到 99% 就是极限"
 *    要诚实 —— 这是 README 里专门讲的那件事。
 *
 * 尺寸按容器的真实宽度画 (见 chartKit.useWidth), 手机上也是真的 11 像素字, 不需要横向滑动。
 */
export function WinChart({
  series,
  pMin,
  pMax,
  blueName,
  redName,
  pregame,
  postdraft,
  endState,
}: Props) {
  const [box, W] = useWidth<HTMLDivElement>();
  // 桌面上宽高比和原来差不多; 手机上宽度小, 高度保底 220, 纵向不至于被压扁
  const H = Math.round(Math.min(300, Math.max(220, W * 0.3)));
  const grown = useGrowingSeries(series);
  const [hover, setHover] = useState<number | null>(null);

  const pts = useMemo(() => {
    // 曲线从第 0 分钟起, 起点是 BP 后的概率。局内模型最早只在第 10 分钟
    // 训练过, 前几分钟本来就没有局内读数 —— 但"这局开打时怎么看"是有答案
    // 的。所以给早期的点**补概率**, 不是补点。
    const seed = postdraft ?? pregame;
    let s = grown;
    if (seed != null) {
      s = s.map((p) =>
        p.minute < 3 && p.probability_blue == null
          ? { ...p, probability_blue: seed }
          : p,
      );
      if (!s.some((p) => p.minute <= 0.02)) {
        s = [
          {
            minute: 0,
            golddiff: 0,
            blue_kills: 0,
            red_kills: 0,
            probability_blue: seed,
          },
          ...s,
        ];
      }
    }
    return s;
  }, [grown, pregame, postdraft]);

  const drawable = pts.filter(
    (p): p is TimelinePoint & { probability_blue: number } =>
      p.probability_blue != null,
  );

  if (!drawable.length) {
    return (
      <div ref={box} className="chart-wrap">
        <svg viewBox={`0 0 ${W} ${H}`} className="chart">
          <text x={W / 2} y={H / 2} textAnchor="middle" fontSize="15" fill="var(--ink-3)">
            {tr("还没有胜率数据")}
          </text>
        </svg>
      </div>
    );
  }

  const maxM = Math.max(...pts.map((p) => p.minute), 1);
  const x = (m: number) => L + ((W - L - R) * m) / maxM;
  const y = (p: number) => T + (H - T - B) * (1 - p);
  const mid = y(0.5);

  // 当前领先方的颜色 —— 末端圆点和线条尾段要一致, 否则最新那一刻看着是
  // 两种颜色。0.5 归给蓝方只是为了有个确定归属, 视觉上此时点正落在中线。
  const lastP = drawable[drawable.length - 1]!.probability_blue;
  const leadColor = lastP >= 0.5 ? "var(--blue)" : "var(--red)";

  const line = drawable.map((p) => [x(p.minute), y(p.probability_blue)] as const);
  const path = line.map(([px, py], i) => `${i ? "L" : "M"}${px} ${py}`).join(" ");
  const areaTop = `${path} L${line[line.length - 1]![0]} ${mid} L${line[0]![0]} ${mid} Z`;

  const endX = line[line.length - 1]![0];
  const endY = line[line.length - 1]![1];

  // 两条参考线 (赛前 Stage 1 / BP 后 Stage 2) 的标签位置。
  //
  // **两条线可能几乎重合** —— BP 只在赛前基础上微调, 实测 73% 和 74% 那次
  // 两个标签直接叠成了 "B赛前 Karmine Corp 74%" 这种乱码。
  // 所以靠得太近时把一个顶到线上方、另一个压到线下方; 分得开时都放在线上方。
  const LABEL_GAP = 13;
  const refLines = (() => {
    const rows = [
      pregame != null ? { p: pregame, label: tr("赛前"), dash: "2 4" } : null,
      postdraft != null ? { p: postdraft, label: tr("BP 后"), dash: "6 3" } : null,
    ].filter((r): r is { p: number; label: string; dash: string } => r != null);
    if (rows.length < 2 || Math.abs(y(rows[0]!.p) - y(rows[1]!.p)) >= LABEL_GAP) {
      return rows.map((r) => ({ ...r, ty: y(r.p) - 5 }));
    }
    // 上面那条 (y 更小) 的标签往上, 下面那条往下 —— 顺序和线本身一致,
    // 读者不会把标签认到另一条线上。
    const [hi, lo] = y(rows[0]!.p) <= y(rows[1]!.p) ? [0, 1] : [1, 0];
    const out = [...rows] as ({ p: number; label: string; dash: string; ty: number })[];
    out[hi] = { ...rows[hi]!, ty: y(rows[hi]!.p) - 6 };
    out[lo] = { ...rows[lo]!, ty: y(rows[lo]!.p) + 12 };
    return out;
  })();

  const ticks = pickTicks(maxM, W - L - R);
  const xLabels: number[] = [];
  for (let m = 0; m <= maxM; m += ticks) xLabels.push(m);

  const hp = hover != null && hover < drawable.length ? drawable[hover]! : null;
  const onMove = (e: React.PointerEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - r.left) * W) / r.width;
    setHover(nearestIndex(line.map(([lx]) => lx), px));
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
          <pattern
            id="hatch"
            width="7"
            height="7"
            patternUnits="userSpaceOnUse"
            patternTransform="rotate(45)"
          >
            <rect width="7" height="7" fill="transparent" />
            <line x1="0" y1="0" x2="0" y2="7" stroke="var(--dead)" strokeWidth="2.4" />
          </pattern>
          <clipPath id="clip-above">
            <rect x={L} y={T} width={W - L - R} height={mid - T} />
          </clipPath>
          <clipPath id="clip-below">
            <rect x={L} y={mid} width={W - L - R} height={H - B - mid} />
          </clipPath>
          <RevealClip id="wc-reveal" />
        </defs>

        {/* 死区: 模型够不到的区间。标签按"领先方"口径写, 免得读者自己换算 */}
        {pMax != null && pMax < 0.995 && (
          <>
            <rect
              x={L}
              y={y(1)}
              width={W - L - R}
              height={Math.max(0, y(pMax) - y(1))}
              fill="url(#hatch)"
              opacity="0.8"
            />
            <text x={L + 6} y={y(1) + 12} fontSize="11" fill="var(--ink-3)">
              {tr("{team} 胜率超过 {n}% 时只显示到此", { team: blueName, n: Math.round(pMax * 100) })}
            </text>
          </>
        )}
        {pMin != null && pMin > 0.005 && (
          <>
            <rect
              x={L}
              y={y(pMin)}
              width={W - L - R}
              height={Math.max(0, y(0) - y(pMin))}
              fill="url(#hatch)"
              opacity="0.8"
            />
            <text x={L + 6} y={y(0) - 5} fontSize="11" fill="var(--ink-3)">
              {tr("{team} 胜率超过 {n}% 时只显示到此", { team: redName, n: Math.round((1 - pMin) * 100) })}
            </text>
          </>
        )}

        {/* 50% 中线画成加粗实线, 不再写"势均力敌" —— 那几个字和赛前/BP 后参考线的
            标签挤在同一个高度, 实测叠成一团 (2026-09-13 截图)。中线本身就是"持平"。 */}
        {[0, 0.25, 0.5, 0.75, 1].map((v) => (
          <g key={v}>
            <line
              x1={L}
              x2={W - R}
              y1={y(v)}
              y2={y(v)}
              stroke={v === 0.5 ? "var(--ink-3)" : "var(--hair-2)"}
              strokeWidth={v === 0.5 ? 1.8 : 1}
            />
            <text
              x={L - 8}
              y={y(v) + 3.5}
              textAnchor="end"
              fontSize="11"
              fill={
                v > 0.5 ? "var(--blue)" : v < 0.5 ? "var(--red)" : "var(--ink-3)"
              }
            >
              {Math.round(Math.max(v, 1 - v) * 100)}
            </text>
          </g>
        ))}

        {xLabels.map((m) => (
          <text
            key={m}
            x={x(m)}
            y={H - 8}
            textAnchor="middle"
            fontSize="11"
            fill="var(--ink-3)"
          >
            {m}
          </text>
        ))}

        {/* 两条参考线: 赛前 (Stage 1) 和 BP 后 (Stage 2)。
            **写领先方的名字和它的胜率**, 不是蓝方的原始概率 —— 纵轴是对称的,
            标签始终按"领先方"口径, 所以 39% 那种数在轴上根本读不出来:
            那条线所在的高度, 轴上写的是红方的 61。和自己坐标轴对不上的数字,
            比不标还糟。
            虚线样式分开: 赛前是短点 (2 4), BP 后是长划 (6 3), 两条挨得近时
            仍分得清谁是谁。 */}
        {refLines.map((r) => (
          <g key={r.label}>
            <line
              x1={L}
              x2={W - R}
              y1={y(r.p)}
              y2={y(r.p)}
              stroke="var(--ink-3)"
              strokeWidth="1"
              strokeDasharray={r.dash}
            />
            <text x={W - R} y={r.ty} textAnchor="end" fontSize="10.5" fill="var(--ink-3)">
              {r.label} {r.p >= 0.5 ? blueName : redName}{" "}
              {Math.round(Math.max(r.p, 1 - r.p) * 100)}%
            </text>
          </g>
        ))}

        {/* 曲线、面积和末端标记包在 reveal 里: 第一次出现时从左往右画出来 */}
        <g clipPath="url(#wc-reveal)">
          {/* 高于中线的部分染蓝, 低于染红 —— 用 clip 而不是分段路径 */}
          <path d={areaTop} fill="var(--blue)" opacity="0.2" clipPath="url(#clip-above)" />
          <path d={areaTop} fill="var(--red)" opacity="0.2" clipPath="url(#clip-below)" />

          {/* 线条也按领先方染色: 同一条路径画两遍, 分别用上/下半区的 clip 裁掉。
              和上面面积填充用的是同一套 clip —— 好处是**穿越 50% 的那一点自动
              就是分界**, 不需要去算交点、切分路径, 也就不会在交点附近出现颜色
              和面积对不上的缝。描边宽度被中线一分为二, 视觉上正好是"谁领先谁
              的颜色"。 */}
          <path
            d={path}
            fill="none"
            stroke="var(--blue)"
            strokeWidth="2.2"
            strokeLinejoin="round"
            clipPath="url(#clip-above)"
          />
          <path
            d={path}
            fill="none"
            stroke="var(--red)"
            strokeWidth="2.2"
            strokeLinejoin="round"
            clipPath="url(#clip-below)"
          />
          {/* 末端标记。进行中画成呼吸的光点 —— 让"这条线还在长"和"这局已经
              定格"一眼可分; 之前两者长得一模一样, 只能靠别处的文字去猜。
              结束了换成奖杯, 并且**放在领先方那一侧的颜色上** —— 那一侧就是
              模型认为赢面更大的一方。
              动画走 SVG 原生 <animate>: 不依赖 CSS 类, 导出成图片也带着。 */}
          {endState === "live" ? (
            <g>
              <circle
                cx={endX}
                cy={endY}
                r="4"
                fill="none"
                stroke={leadColor}
                strokeWidth="1.6"
                opacity="0.5"
              >
                <animate attributeName="r" values="4;11;4" dur="1.8s" repeatCount="indefinite" />
                <animate
                  attributeName="opacity"
                  values="0.5;0;0.5"
                  dur="1.8s"
                  repeatCount="indefinite"
                />
              </circle>
              <circle cx={endX} cy={endY} r="4" fill={leadColor}>
                <animate attributeName="r" values="4;5.2;4" dur="1.8s" repeatCount="indefinite" />
              </circle>
            </g>
          ) : endState === "paused" ? (
            <g>
              <title>{tr("暂停 / 数据中断 —— 胜负未定")}</title>
              {/* 定住的暂停符号。**刻意不画奖杯**: 暂停时这一局还没分胜负,
                  画奖杯就是在宣布一个还没发生的结果, 而这正是本项目最要避免的
                  那种"看着很确定的错话"。 */}
              <circle cx={endX} cy={endY} r="6.5" fill={leadColor} opacity="0.18" />
              <circle cx={endX} cy={endY} r="6.5" fill="none" stroke={leadColor} strokeWidth="1.4" />
              <rect x={endX - 2.4} y={endY - 3} width="1.6" height="6" rx="0.6" fill={leadColor} />
              <rect x={endX + 0.8} y={endY - 3} width="1.6" height="6" rx="0.6" fill={leadColor} />
            </g>
          ) : (
            <g transform={`translate(${endX - 7} ${endY - 8})`} fill={leadColor}>
              <title>{tr("这一局已结束")}</title>
              {/* 奖杯: 杯身 + 两只耳 + 杯脚 + 底座 */}
              <path d="M4.2 0h5.6v4.1a2.8 2.8 0 0 1-5.6 0Z" />
              <path
                d="M4.2 0.7H2.1a1.5 1.5 0 0 0 0 3h.9M9.8 0.7h2.1a1.5 1.5 0 0 1 0 3h-.9"
                fill="none"
                stroke={leadColor}
                strokeWidth="1.1"
              />
              <path d="M6.4 6.6h1.2v2.4H6.4z" />
              <path d="M3.9 9h6.2c.3 0 .5.2.5.5v1c0 .3-.2.5-.5.5H3.9a.5.5 0 0 1-.5-.5v-1c0-.3.2-.5.5-.5Z" />
            </g>
          )}
        </g>

        {/* 指针所在的那一刻: 竖线 + 曲线上的点 */}
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
              cy={y(hp.probability_blue)}
              r="4.5"
              fill={hp.probability_blue >= 0.5 ? "var(--blue)" : "var(--red)"}
              stroke="var(--panel-strong)"
              strokeWidth="2"
            />
          </g>
        )}
      </svg>
      {hp && (
        <ChartTip x={x(hp.minute)} y={y(hp.probability_blue)} W={W} H={H}>
          <div className="tip-time">{clock(hp.minute)}</div>
          <div className={hp.probability_blue >= 0.5 ? "blue" : "red"}>
            <b>{hp.probability_blue >= 0.5 ? blueName : redName}</b>{" "}
            {Math.round(Math.max(hp.probability_blue, 1 - hp.probability_blue) * 100)}%
          </div>
          <div>
            {tr("经济差")}{" "}
            <span className={hp.golddiff > 0 ? "blue" : hp.golddiff < 0 ? "red" : ""}>
              {hp.golddiff > 0 ? "+" : hp.golddiff < 0 ? "−" : ""}
              {Math.round(Math.abs(hp.golddiff)).toLocaleString()}
            </span>
          </div>
          <div>
            {tr("击杀")} <span className="blue">{hp.blue_kills}</span> : <span className="red">{hp.red_kills}</span>
          </div>
        </ChartTip>
      )}
    </div>
  );
}
