import { useState } from "react";
import type { Reason } from "../api/types";
import { WhyBubbles } from "./WhyBubbles";

interface Props {
  blueName: string;
  redName: string;
  probabilityBlue: number;
  pMin: number | null;
  pMax: number | null;
  /** Stage 1 —— 只看队伍历史 */
  pregame: number | null;
  /** Stage 2 —— 已计入双方阵容 */
  postdraft?: number | null;
  /** 有依据就把中间那个"胜率"变成可点的"胜率解析"。
   *  弹窗**挂在这个组件里面**, 于是赛前/BP 后/局内三段自动都有 ——
   *  放在外面的话每处都要各自接一遍线, 迟早漏掉一处。 */
  reasons?: Reason[];
}

/**
 * 概率 —— 页面的英雄元素。
 *
 * 两件事是设计上刻意的, 不要"简化"掉:
 *   · 数字虽然大, 但真正的读数是**刻度上的位置**, 不是那个百分数
 *   · 校准够不到的两段画成斜线死区。模型**不能**表达的区间要看得见,
 *     否则"顶到 99%"会被当成"它认为就是 99%"。README 专门讲过这件事。
 */
export function ProbabilityScale({
  blueName,
  redName,
  probabilityBlue,
  pMin,
  pMax,
  pregame,
  postdraft,
  reasons,
}: Props) {
  const [why, setWhy] = useState(false);
  const canExplain = !!reasons && reasons.length > 0;
  const lo = pMin ?? 0;
  const hi = pMax ?? 1;
  const p = probabilityBlue;
  const split = (v: number) => {
    const s = (v * 100).toFixed(1).split(".");
    return { int: s[0]!, dec: s[1]! };
  };
  const b = split(p);
  const r = split(1 - p);

  return (
    <div>
      <div className="prob">
        <div className="prob-side">
          <div className="prob-label">{blueName}</div>
          <div className="prob-num prob-blue">
            {b.int}
            <small>.{b.dec}%</small>
          </div>
        </div>
        {/* 中间这块从一个静态的"胜率"变成入口: 点开看模型凭什么给这个数。
            "主要依据"原来常驻在页面下方, 但它是**偶尔才看**的东西, 摊在那里
            既占地方又打断阅读。
            气泡**从这四个字原地长出来** (wrap 是定位锚点), 视线不用挪。 */}
        <div className="prob-mid-wrap">
          {canExplain ? (
            <button
              className={`prob-mid link${why ? " on" : ""}`}
              onClick={() => setWhy((v) => !v)}
              aria-expanded={why}
            >
              胜率解析
            </button>
          ) : (
            <div className="prob-mid">胜率</div>
          )}
          {why && canExplain && (
            <WhyBubbles reasons={reasons!} onClose={() => setWhy(false)} />
          )}
        </div>
        <div className="prob-side right">
          <div className="prob-label">{redName}</div>
          <div className="prob-num prob-red">
            {r.int}
            <small>.{r.dec}%</small>
          </div>
        </div>
      </div>

      <svg viewBox="0 0 1000 26" className="chart" preserveAspectRatio="none" height="26">
        <defs>
          <pattern
            id="hatch-scale"
            width="6"
            height="6"
            patternUnits="userSpaceOnUse"
            patternTransform="rotate(45)"
          >
            <line x1="0" y1="0" x2="0" y2="6" stroke="var(--dead)" strokeWidth="2" />
          </pattern>
          <linearGradient id="fill-blue" x1="0" x2="1">
            <stop offset="0" stopColor="var(--blue)" stopOpacity="0.55" />
            <stop offset="1" stopColor="var(--blue)" />
          </linearGradient>
          <linearGradient id="fill-red" x1="1" x2="0">
            <stop offset="0" stopColor="var(--red)" stopOpacity="0.55" />
            <stop offset="1" stopColor="var(--red)" />
          </linearGradient>
        </defs>

        {/* 红方那一段 = 剩下的部分 */}
        <rect x="0" y="9" width="1000" height="8" rx="4" fill="url(#fill-red)" opacity="0.9" />
        <rect x="0" y="9" width={1000 * p} height="8" rx="4" fill="url(#fill-blue)" />

        {/* 校准够不到的两端 */}
        {lo > 0.005 && <rect x="0" y="9" width={1000 * lo} height="8" fill="url(#hatch-scale)" />}
        {hi < 0.995 && (
          <rect x={1000 * hi} y="9" width={1000 * (1 - hi)} height="8" fill="url(#hatch-scale)" />
        )}

        {/* 两个参照刻度: 赛前 (Stage 1) 短点, BP 后 (Stage 2) 长划。
            虚线样式分开, 两条挨得近时仍分得清谁是谁。 */}
        {pregame != null && (
          <line
            x1={1000 * pregame}
            x2={1000 * pregame}
            y1="4"
            y2="22"
            stroke="var(--ink-3)"
            strokeWidth="1.5"
            strokeDasharray="2 3"
            vectorEffect="non-scaling-stroke"
          />
        )}
        {postdraft != null && (
          <line
            x1={1000 * postdraft}
            x2={1000 * postdraft}
            y1="2"
            y2="24"
            stroke="var(--ink-2)"
            strokeWidth="1.5"
            strokeDasharray="6 3"
            vectorEffect="non-scaling-stroke"
          />
        )}
        <line
          x1={1000 * p}
          x2={1000 * p}
          y1="1"
          y2="25"
          stroke="var(--ink)"
          strokeWidth="2"
          vectorEffect="non-scaling-stroke"
        />
      </svg>

      <div className="scale-foot">
        <span>{(pMin != null || pMax != null) && "斜线区间超出模型输出范围"}</span>
        {/* 和刻度本身一样, 按**领先方**口径写。这一条刻度左右两端各是一方的
            胜率, 单写一个 39% 读者无从知道那是谁的 —— 而它旁边那两个大数字
            写的正是两边各自的胜率, 39 哪个都对不上。 */}
        <span>
          {([
            { lab: "赛前", v: pregame },
            { lab: "BP 后", v: postdraft ?? null },
          ] as { lab: string; v: number | null }[])
            .filter((x) => x.v != null)
            .map(
              ({ lab, v }) =>
                `${lab} ${v! >= 0.5 ? blueName : redName} ${Math.round(
                  Math.max(v!, 1 - v!) * 100,
                )}%`,
            )
            .join(" · ")}
        </span>
      </div>
    </div>
  );
}
