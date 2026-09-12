/**
 * 特征翻译层 —— explain.py 的浏览器版。
 *
 * **文案表不在这里抄写**: STAT / LIVE / META / INTERACTION / TONE 这些由
 * tools/export_web_model.py 从 explain.py 原样导出成 explain.json, 这里只移植逻辑。
 * 这样 explain.py 改一句话, 重新导出就跟上了, 不会出现两份文案慢慢漂开。
 * 写在函数体里的少数句子只能手工移植 —— 由 scripts/golden.mjs 逐字核对。
 *
 * 数字一律走 pyfmt: Python 的 .1f / .0f / 千分位都是四舍六入五成双。
 */
import { pyComma, pyFixed, pyRound } from "./pyfmt.ts";

export interface ExplainFile {
  format: string;
  /** (中文名, 单位, 小数位, 极性, diff 句式, 单方句式, 合计句式) */
  stat: Record<string, [string, string, number, number, string, string, string | null]>;
  alias: Record<string, string>;
  /** (中文名, 单位, 小数位) */
  live: Record<string, [string, string, number]>;
  meta: Record<string, string>;
  interaction: Record<string, string>;
  /** 有序 —— _player_champ 按这个顺序试 */
  role_cn: [string, string][];
  skip_in_reasons: string[];
  tone: [number, string][];
}

export interface EvidenceItem {
  feature: string;
  value: number;
  importance: number;
}

const P = "%";
const own = (o: object, k: string) => Object.prototype.hasOwnProperty.call(o, k);
const fill = (tpl: string, vars: Record<string, string>) =>
  Object.entries(vars).reduce((s, [k, v]) => s.replaceAll(`{${k}}`, v), tpl);

/** explain._fmt: 数值 + 单位, 大数走 k, 百分比走 % */
function fmt(v: number, unit: string, dec: number): string {
  const a = Math.abs(v);
  if (unit === P) return `${pyFixed(a * 100, dec)}%`;
  if (unit === "k") return `${pyFixed(a / 1000, dec)}k`;
  if (a >= 10000) return `${pyFixed(a / 1000, 1)}k${unit}`;
  return `${pyComma(a, dec)}${unit}`;
}

/** explain._split: diff_avg_towers → ["diff", "towers"] */
function split(ex: ExplainFile, name: string): [string | null, string] {
  let pre: string | null = null;
  for (const p of ["diff_", "sum_", "b_", "r_"]) {
    if (name.startsWith(p)) {
      pre = p.slice(0, -1);
      name = name.slice(p.length);
      break;
    }
  }
  for (const p of ["avg_", "pre_"]) {
    if (name.startsWith(p)) name = name.slice(p.length);
  }
  return [pre, own(ex.alias, name) ? ex.alias[name]! : name];
}

function scaling(name: string, v: number, blue: string, red: string): string | null {
  if (name === "scaling_sum") return null;
  if (name === "b_scaling" || name === "r_scaling") {
    const team = name[0] === "b" ? blue : red;
    const tone = v > 0.5 ? "偏后期" : v < -0.5 ? "偏前期" : "前后期均衡";
    return `${team} 的阵容${tone}`;
  }
  if (Math.abs(v) < 0.3) return "双方阵容的强势期相当";
  const [late, early] = v > 0 ? [blue, red] : [red, blue];
  return `${late} 的阵容更偏后期, ${early} 需要在前期建立优势`;
}

function streak(pre: string | null, v: number, blue: string, red: string): string | null {
  const n = Math.trunc(v);
  if (pre === "diff") return null;
  if (n === 0) return null;
  const team = pre === "b" ? blue : red; // 和 Python 一样: 不是 b 就算红方 (sum_/裸 streak 也是)
  return `${team} 当前${n > 0 ? "连胜" : "连败"} ${Math.abs(n)} 场`;
}

function playerChamp(ex: ExplainFile, name: string, v: number, blue: string, red: string): string | null {
  for (const [side, team] of [
    ["b_", blue],
    ["r_", red],
  ] as const) {
    if (!name.startsWith(side)) continue;
    const rest = name.slice(2);
    for (const [role, cn] of ex.role_cn) {
      if (rest === `${role}_champ_wr`) return `${team} ${cn} 用这个英雄历史胜率 ${pyFixed(v * 100, 0)}%`;
      if (rest === `${role}_champ_games`) {
        const n = Math.trunc(v);
        if (n === 0) return `${team} ${cn} 没用过这个英雄`;
        return `${team} ${cn} 用过这个英雄 ${n} 场`;
      }
      if (rest === `${role}_comfort`) return null;
    }
  }
  return null;
}

/** explain.explain: 把一个 (字段名, 值) 翻译成一句人话。null = 不值得展示。 */
export function explain(
  ex: ExplainFile,
  name: string,
  value: number | null | undefined,
  blue = "蓝方",
  red = "红方",
): string | null {
  if (value === null || value === undefined || Number.isNaN(value)) return null;
  const v = value;

  if (own(ex.meta, name)) {
    if (name.startsWith("is_")) return null;
    return v ? "季后赛" : null;
  }
  if (own(ex.interaction, name)) {
    if (Math.abs(v) < 1e-9) return null;
    return `${ex.interaction[name]} —— 当前利于 ${v > 0 ? blue : red}`;
  }
  if (name === "scaling_diff" || name === "b_scaling" || name === "r_scaling" || name === "scaling_sum") {
    return scaling(name, v, blue, red);
  }
  if (name === "T") return `按第 ${Math.trunc(v)} 分钟的模型切片评估`;

  if (name === "killsum") return `双方已合计击杀 ${Math.trunc(v)} 个人头`;
  if (name === "golddiff_norm") {
    if (Math.abs(v) < 1e-9) return "双方经济持平";
    return `${v > 0 ? blue : red} 的经济领先占场上总经济的 ${pyFixed(Math.abs(v) * 100, 1)}%`;
  }
  if (own(ex.live, name)) {
    const [cn, unit, dec] = ex.live[name]!;
    if (Math.abs(v) < 1e-9) return `双方${cn}持平`;
    return `${v > 0 ? blue : red} ${cn}领先 ${fmt(v, unit, dec)}`;
  }

  if (name.startsWith("diff_") && name.endsWith("_comfort")) {
    const role = name.slice(5, -8);
    const hit = ex.role_cn.find(([r]) => r === role);
    if (hit) {
      if (Math.abs(v) < 1) return null;
      return `${v > 0 ? blue : red} ${hit[1]} 对这个英雄比对位多 ${Math.abs(Math.trunc(v))} 场经验`;
    }
  }

  const pc = playerChamp(ex, name, v, blue, red);
  if (pc) return pc;

  const [pre, base] = split(ex, name);
  if (base === "streak") return streak(pre, v, blue, red);
  if (!own(ex.stat, base) || pre === null) return null;
  const [cn, unit, dec, pol, tDiff, tSide, tSum] = ex.stat[base]!;

  if (pre === "diff") {
    const shown = fmt(v, unit, dec);
    // 格式化之后是 0 的, 说「持平」比说「多 0.00 条」诚实
    if (Math.abs(v) < 1e-9 || !/[1-9]/.test(shown)) return `双方${cn}持平`;
    const who = pol === -1 ? (v > 0 ? red : blue) : v > 0 ? blue : red;
    return fill(tDiff, { who, v: fmt(v, unit, dec) });
  }
  if (pre === "sum") {
    if (!tSum) return null;
    return fill(tSum, { v: fmt(v, unit, dec) });
  }
  return fill(tSide, { team: pre === "b" ? blue : red, v: fmt(v, unit, dec) });
}

/** explain.explain_evidence: 整体翻译, 丢掉翻译不出来的, 同一个基础指标只留权重最高的那条 */
export function explainEvidence(
  ex: ExplainFile,
  items: EvidenceItem[] | null | undefined,
  blue = "蓝方",
  red = "红方",
  k = 6,
): { text: string; weight: number }[] {
  const out: { text: string; weight: number }[] = [];
  const seen = new Set<string>();
  const bases = new Set<string>();
  const skip = new Set(ex.skip_in_reasons);
  for (const it of items ?? []) {
    if (skip.has(it.feature)) continue;
    const txt = explain(ex, it.feature, it.value, blue, red);
    if (!txt || seen.has(txt)) continue;
    const base = split(ex, it.feature)[1];
    if (bases.has(base)) continue;
    bases.add(base);
    seen.add(txt);
    out.push({ text: txt, weight: pyRound(it.importance || 0, 4) });
    if (out.length >= k) break;
  }
  return out;
}

const WHEN: Record<string, string> = { "1_pre_draft": "赛前", "2_post_draft": "BP 结束", "4_ingame": "局内" };

/** explain.summarize: 一句话总结 */
export function summarize(
  ex: ExplainFile,
  prob: number,
  blue: string,
  red: string,
  stage: string | null = null,
  minute: number | null = null,
): string {
  const lead = prob >= 0.5 ? blue : red;
  const p = Math.max(prob, 1 - prob);
  const tone = fill(ex.tone.find(([th]) => p < th)![1], { lead });
  let when = stage !== null && own(WHEN, stage) ? WHEN[stage]! : "";
  if (minute !== null) when = `第 ${Math.trunc(minute)} 分钟`;
  return when ? `${when} · ${tone}` : tone;
}

/** explain.confidence_note: 把模型指标翻译成「这个数字有多可信」 */
export function confidenceNote(m: Record<string, unknown>): string {
  const num = (k: string) => (typeof m[k] === "number" ? (m[k] as number) : null);
  const acc = num("accuracy_at_this_slice") || num("accuracy");
  const base = num("baseline");
  // m.get("ece_cal", m.get("ece")): 键在就用它 (哪怕是 None), 不在才退到 ece
  const ece = own(m, "ece_cal") ? num("ece_cal") : num("ece");
  const parts: string[] = [];
  if (acc) {
    let s = `留出测试集准确率 ${pyFixed(acc * 100, 0)}%`;
    if (base) s += ` (基线 ${pyFixed(base * 100, 0)}%)`;
    parts.push(s);
  }
  if (ece !== null) {
    const q = ece < 0.03 ? "良好" : ece < 0.06 ? "可用" : "偏差偏大";
    parts.push(`概率校准${q} (ECE ${pyFixed(ece, 3)})`);
  }
  return parts.join(" · ");
}

/** explain.range_note: 模型够不到的区间 —— 前端的死区就是这个 */
export function rangeNote(pMin: number | null | undefined, pMax: number | null | undefined): string {
  if (pMin === null || pMin === undefined || pMax === null || pMax === undefined) return "";
  return (
    `该模型的概率输出被校准限制在 ${pyFixed(pMin * 100, 0)}%–${pyFixed(pMax * 100, 0)}%。` +
    `超出这个范围的一边倒对局, 它只会顶到边界, 不会给出更极端的数字。`
  );
}
