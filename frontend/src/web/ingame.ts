/**
 * Stage 4 (局内胜率) 的浏览器版
 * ================================
 * `ingame_service.IngameModel` 的逐行移植 (make_row → 特征向量 → 500 棵树 → 保序回归),
 * 外加 api._ingame_core 组响应的那一段。存在的理由只有一个: GitHub Pages 只能放
 * 静态文件, 跑不了 Python。
 *
 * **这是第二份实现, 所以必须证明它和第一份是同一份。**
 * 这个项目几乎每个 bug 都是"一个看起来合理的错数字", 从不报错。训练在 Python、
 * 推理在 JS, 正是 CLAUDE.md 第一条 (训练和推理必须读同一份逻辑) 要防的形状。
 * 证明靠 scripts/golden.mjs: 用真实直播帧当输入, 26 个特征逐位相等、警告逐字相等、
 * 概率误差 < 1e-6, 整个响应逐字段相等, 才算过。改这个文件之前先跑它, 改完再跑。
 *
 * 数值上几处必须照抄的细节 —— 每一处抄错都不会报错, 只会偏:
 *   1. 缺失值先填 -999, 比较和累加在 float32 上 (见 xgb.ts)
 *   2. 切片 min(slices, key=...) 平局取**第一个**: 12.5 分钟算 10, 不算 15
 *   3. np.mean 的求和顺序: 这里 (≤5 个数) 就是从左到右依次加再除以 n。**这条是
 *      实测出来的, 不是读源码推的** —— 照 numpy 源码推出的 a[0] + (a[1]+a[2]+...)
 *      在 494 组真实阵容上只对了 317 组, 依次相加 494/494 (numpy 2.3.5)。
 *      两种顺序差一个 ulp, 阵容强势期那一族特征 (b/r_scaling 及其交互项) 全会偏
 *   4. 警告里的 {minute:.0f} 是 Python 的"四舍六入五成双": 12.5 写 "12";
 *      JS 的 toFixed(0) 会写 "13" (见 pyfmt.ts)
 *   5. "胜率解析"用的特征值先 round(v, 3) 再翻译 —— evidence() 里就是这么做的,
 *      golddiff_norm 这类小数会因此影响文案的最后一位
 */
import type { ExplainFile, EvidenceItem } from "./explain.ts";
import { confidenceNote, explainEvidence, rangeNote, summarize } from "./explain.ts";
import { pyFixed, pyPct, pyRound } from "./pyfmt.ts";
import type { PreCtx, Stage1 } from "./stage1.ts";
import { preContext } from "./stage1.ts";
import type { TeamsFile } from "./teams.ts";
import type { Forest, TreeModelFile } from "./xgb.ts";
import { loadForest, predictRaw, toVector } from "./xgb.ts";

// ══════════════════════════════════════════════════════════
//  文件格式 (由 tools/export_web_model.py 生成)
// ══════════════════════════════════════════════════════════

export interface IngameModelFile extends TreeModelFile {
  variant: string;
  calibrator: string;
  iso_x: number[];
  iso_y: number[];
  clip: number;
  platt: { a: number; b: number };
  p_min: number | null;
  p_max: number | null;
  slices: number[];
  gold_total_median: Record<string, number>;
  scaling_index: Record<string, number>;
  has_xpdiff: boolean;
  metrics: {
    accuracy?: number | null;
    ece?: number | null;
    per_T?: Record<string, { accuracy?: number; n?: number; high_conf_share?: number }> | null;
  };
  trained_through: string;
}

export interface IngameModel {
  forest: Forest;
  calibrator: string;
  isoX: number[];
  isoY: number[];
  clip: number;
  plattA: number;
  plattB: number;
  pMin: number | null;
  pMax: number | null;
  slices: number[];
  goldTotalMedian: Map<string, number>;
  /** 用 Map 而不是普通对象: 普通对象上 "constructor" 这种键会查到原型链上 */
  scaling: Map<string, number>;
  hasXpdiff: boolean;
  metrics: IngameModelFile["metrics"];
  trainedThrough: string;
}

export function loadIngame(file: IngameModelFile): IngameModel {
  return {
    forest: loadForest(file),
    calibrator: file.calibrator,
    isoX: file.iso_x,
    isoY: file.iso_y,
    clip: file.clip,
    plattA: file.platt.a,
    plattB: file.platt.b,
    pMin: file.p_min,
    pMax: file.p_max,
    slices: file.slices,
    goldTotalMedian: new Map(Object.entries(file.gold_total_median)),
    scaling: new Map(Object.entries(file.scaling_index)),
    hasXpdiff: file.has_xpdiff,
    metrics: file.metrics,
    trainedThrough: file.trained_through,
  };
}

// ══════════════════════════════════════════════════════════
//  组一行特征 —— IngameModel.make_row
// ══════════════════════════════════════════════════════════

/** 一个时点的局面 */
export interface IngameInput {
  minute: number;
  golddiff: number;
  csdiff: number;
  blueKills: number;
  redKills: number;
  /** 蓝方总经济。null = 拿不到, 按切片中位数估 (会带警告) */
  goldTotal: number | null;
  blueChamps: string[] | null;
  redChamps: string[] | null;
  league: string;
}

/** min(slices, key=lambda t: abs(t - minute)) —— 平局取第一个 */
export function nearestSlice(slices: number[], minute: number): number {
  let best = slices[0]!;
  let bestD = Math.abs(best - minute);
  for (let i = 1; i < slices.length; i++) {
    const s = slices[i]!;
    const d = Math.abs(s - minute);
    if (d < bestD) {
      best = s;
      bestD = d;
    }
  }
  return best;
}

/**
 * np.mean(vals), vals 最多 5 个 (一方的阵容)。依次相加再除以 n —— 见头注释第 3 条,
 * 这是用 494 组真实阵容对出来的。numpy 在 n >= 8 时换成分块累加, 这里用不到。
 */
function npMean(vals: number[]): number {
  let s = 0;
  for (const v of vals) s += v;
  return s / vals.length;
}

/** IngameModel.comp_scaling: 认得的英雄不到 3 个就算不出来 */
function compScaling(scaling: Map<string, number>, champs: string[]): number {
  const vals: number[] = [];
  for (const c of champs) {
    const v = scaling.get(c);
    if (v !== undefined) vals.push(v);
  }
  return vals.length < 3 ? NaN : npMean(vals);
}

export interface BuiltRow {
  row: Record<string, number>;
  T: number;
  warnings: string[];
}

/** IngameModel.make_row。preRow 由调用方给 (api._pre_context 的 pre_row), 不认识的队给 null。 */
export function buildRow(model: IngameModel, preRow: Record<string, number> | null, inp: IngameInput): BuiltRow {
  const warn: string[] = [];
  const minute = inp.minute;
  const T = nearestSlice(model.slices, minute);
  if (Math.abs(T - minute) > 2.5) {
    warn.push(
      `你填的是第 ${pyFixed(minute, 0)} 分钟, 但模型只在 10/15/20/25 分钟` +
        `这几个时间点训练过。已按最接近的 ${T} 分钟评估, ` +
        `相差 ${pyFixed(Math.abs(T - minute), 0)} 分钟, 精度会下降。`,
    );
  }
  if (inp.league === "LPL") {
    warn.push(
      "LPL 在局内训练集里样本偏薄 (约 9%, 且全部来自 2026 年) " +
        "—— Oracle's Elixir 在 2022-2025 不提供 LPL 的分钟级快照。" +
        "结论在 LPL 上的可靠性弱于其他三大赛区。",
    );
  }

  let bSc = NaN;
  let rSc = NaN;
  // Python 的 `if blue_champs and red_champs:` —— None 和空列表都算没有
  if (inp.blueChamps && inp.blueChamps.length > 0 && inp.redChamps && inp.redChamps.length > 0) {
    bSc = compScaling(model.scaling, inp.blueChamps);
    rSc = compScaling(model.scaling, inp.redChamps);
    if (Number.isNaN(bSc) || Number.isNaN(rSc)) {
      warn.push("这几个英雄的历史样本不足, 算不出阵容强势期 —— " + "「领先但阵容吃亏」这类隐患本次无法识别。");
    }
  } else {
    warn.push("没填双方阵容, 无法判断谁的阵容更耐拖 —— " + "同样的经济差在不同阵容下含义相反。");
  }
  const scDiff = Number.isNaN(bSc) || Number.isNaN(rSc) ? NaN : bSc - rSc;

  let goldTotal = inp.goldTotal;
  if (goldTotal === null) {
    goldTotal = model.goldTotalMedian.get(String(T)) ?? 1750 * T;
    warn.push(`没填蓝方总经济, 按第 ${T} 分钟的实测中位数 ${goldTotal} 估算, ` + `相对经济差可能有偏。`);
  }

  // 实时数据永远没有经验差 (xpdiff=None)。live 变体训练时就不看它。
  if (model.hasXpdiff) {
    warn.push("没有经验差数据, 而当前模型把它当作输入 —— " + "结果不可靠。实时数据应当走 live 变体模型。");
  } else {
    warn.push("本次没有经验差数据(实时数据源不提供), 已改用" + "训练时就不含经验差的模型, 而不是用 0 顶替。");
  }

  const g = inp.golddiff;
  // Python 的 `golddiff / gold_total if gold_total else nan`: 0 算没有
  const goldNorm = goldTotal ? g / goldTotal : NaN;
  const row: Record<string, number> = {
    T,
    golddiff: g,
    csdiff: inp.csdiff,
    killdiff: inp.blueKills - inp.redKills,
    killsum: inp.blueKills + inp.redKills,
    golddiff_norm: goldNorm,
    b_scaling: bSc,
    r_scaling: rSc,
    scaling_diff: scDiff,
    scaling_sum: Number.isNaN(scDiff) ? NaN : bSc + rSc,
    is_lpl: inp.league === "LPL" ? 1 : 0,
    is_lck: inp.league === "LCK" ? 1 : 0,
    is_lec: inp.league === "LEC" ? 1 : 0,
    is_lcs: inp.league === "LCS" ? 1 : 0,
  };

  if (!Number.isNaN(scDiff)) {
    row.x_gold_scaling = g * scDiff;
    row.x_goldnorm_scaling = goldNorm * scDiff;
    row.lead_by_early_comp = g * -scDiff;
  }

  // Python 的 `if pre_row:` —— 空字典也算没有
  if (preRow && Object.keys(preRow).length > 0) {
    for (const [k, v] of Object.entries(preRow)) {
      if (model.forest.featureSet.has(k)) row[k] = v;
    }
  } else {
    warn.push("没有可用的赛前队伍统计, 本次只看局内数据。");
  }
  return { row, T, warnings: warn };
}

// ══════════════════════════════════════════════════════════
//  预测 —— IngameModel.predict / calibrate
// ══════════════════════════════════════════════════════════

/** np.interp(x, xp, fp), 标量版, 连同它的求值顺序 */
function npInterp(x: number, xp: number[], fp: number[]): number {
  const last = xp.length - 1;
  if (Number.isNaN(x)) return NaN;
  if (x > xp[last]!) return fp[last]!;
  if (x < xp[0]!) return fp[0]!;
  if (x === xp[last]!) return fp[last]!;
  let lo = 0;
  let hi = last; // 不变量: xp[lo] <= x < xp[hi]
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xp[mid]! <= x) lo = mid;
    else hi = mid;
  }
  const j = lo;
  if (xp[j] === x) return fp[j]!;
  const slope = (fp[j + 1]! - fp[j]!) / (xp[j + 1]! - xp[j]!);
  let r = slope * (x - xp[j]!) + fp[j]!;
  if (Number.isNaN(r)) {
    r = slope * (x - xp[j + 1]!) + fp[j + 1]!;
    if (Number.isNaN(r) && fp[j] === fp[j + 1]) r = fp[j]!;
  }
  return r;
}

export function calibrate(model: IngameModel, raw: number): number {
  if (model.calibrator === "isotonic") {
    const v = npInterp(raw, model.isoX, model.isoY);
    return Math.min(Math.max(v, model.clip), 1.0 - model.clip);
  }
  return 1.0 / (1.0 + Math.exp(-(model.plattA * raw + model.plattB)));
}

export interface Prediction {
  raw: number;
  cal: number;
  x: Float64Array;
}

export function predictRow(model: IngameModel, row: Record<string, number>): Prediction {
  const x = toVector(model.forest.features, row);
  const raw = predictRaw(model.forest, x);
  return { raw, cal: calibrate(model, raw), x };
}

/**
 * 概率有没有真的被校准的上下限夹住 —— IngameModel.clip_warning。
 * 必须看**实际输出**, 不能按经济差猜 (那条陈旧断言在 Python 那边踩过一次)。
 */
export function clipWarning(model: IngameModel, cal: number): string | null {
  const eps = 1e-6;
  if (model.pMax !== null && cal >= model.pMax - eps) {
    return (
      `概率已经顶到模型的表达上限 ${pyPct(model.pMax, 0)} —— 这种局面的实际` +
      `胜率可能更高。这是校准的固有上限, 不是判断失误。`
    );
  }
  if (model.pMin !== null && cal <= model.pMin + eps) {
    return (
      `概率已经触到模型的表达下限 ${pyPct(model.pMin, 0)} —— 这种局面的实际` +
      `胜率可能更低。这是校准的固有上限, 不是判断失误。`
    );
  }
  return null;
}

/** IngameModel.evidence: 值先 round 到 3 位、重要性 round 到 4 位, 再按 (round 过的) 重要性排序 */
function evidence(model: IngameModel, row: Record<string, number>, k = 6): EvidenceItem[] {
  const out: EvidenceItem[] = [];
  for (const f of model.forest.features) {
    const v = row[f];
    if (v === undefined || Number.isNaN(v)) continue;
    out.push({ feature: f, value: pyRound(v, 3), importance: pyRound(model.forest.importance.get(f) ?? 0, 4) });
  }
  out.sort((a, b) => b.importance - a.importance);
  return out.slice(0, k);
}

/** api._ingame_disclaimer —— 数字和校准方法全部从 metrics 读, 不写死 */
function disclaimer(model: IngameModel): string {
  const m = model.metrics;
  const acc = m.accuracy ?? null;
  const ece = m.ece ?? null;
  const t25 = m.per_T?.["25"]?.accuracy ?? null;
  const how = model.calibrator === "isotonic" ? "保序回归" : "Platt";
  const bits: string[] = [];
  if (acc !== null) {
    let s = `局内模型在留出测试集上准确率约 ${pyPct(acc, 0)}`;
    if (t25 !== null) s += `, 25 分钟时约 ${pyPct(t25, 0)}`;
    bits.push(s + "。");
  }
  if (ece !== null) bits.push(`概率经${how}校准 (ECE ≈ ${pyFixed(ece, 3)})。`);
  bits.push("非投注建议。");
  return bits.join("");
}

/** 喂给局内模型的那一帧 (LiveState.to_state 的形状, xpdiff 永远是 None) */
export interface IngameState {
  minute: number;
  golddiff: number;
  csdiff: number;
  blueKills: number;
  redKills: number;
  goldTotal: number | null;
}

export interface IngameArgs {
  blue: string;
  red: string;
  league: string;
  state: IngameState;
  blueChamps: string[] | null;
  redChamps: string[] | null;
  /** 调用方已经算好的赛前上下文 (曲线每个点共用一份)。不给就在这里算。 */
  preCtx?: PreCtx | null;
}

/**
 * api._ingame_core 的浏览器版 (include_pregame=True, raw=False) —— 看板头条和
 * 走势图每个点用的都是它。返回的对象和服务器的 prediction 逐字段相同。
 */
export function ingameResponse(
  model: IngameModel,
  s1: Stage1,
  teams: TeamsFile,
  ex: ExplainFile,
  a: IngameArgs,
  today: string,
): Record<string, unknown> {
  let preRow: Record<string, number> | null = null;
  let preP: number | null = null;
  const warns: string[] = [];
  if (a.preCtx) {
    preRow = a.preCtx.preRow;
    preP = a.preCtx.preP;
    warns.push(...a.preCtx.warnings);
  } else {
    try {
      const c = preContext(s1, teams, a.blue, a.red, a.league, today);
      preRow = c.preRow;
      preP = c.preP;
      warns.push(...c.warnings);
    } catch (e) {
      warns.push(`赛前特征不可用: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  const st = a.state;
  const b = buildRow(model, preRow, {
    minute: st.minute,
    golddiff: st.golddiff,
    csdiff: st.csdiff,
    blueKills: st.blueKills,
    redKills: st.redKills,
    goldTotal: st.goldTotal,
    blueChamps: a.blueChamps,
    redChamps: a.redChamps,
    league: a.league,
  });
  warns.push(...b.warnings);
  const p = predictRow(model, b.row);
  const cw = clipWarning(model, p.cal);
  if (cw) warns.push(cw);

  const perT = model.metrics.per_T?.[String(b.T)] ?? {};
  const card: Record<string, unknown> = {
    accuracy_at_this_slice: perT.accuracy ?? null,
    high_conf_share_at_this_slice: perT.high_conf_share ?? null,
    overall_accuracy: model.metrics.accuracy ?? null,
    ece: model.metrics.ece ?? null,
    p_min: model.pMin,
    p_max: model.pMax,
    trained_through: model.trainedThrough,
  };
  const out: Record<string, unknown> = {
    blue_team: a.blue,
    red_team: a.red,
    league: a.league,
    minute: st.minute,
    slice_used: b.T,
    probability_blue: pyRound(p.cal, 4),
    probability_raw: pyRound(p.raw, 4),
    summary: summarize(ex, p.cal, a.blue, a.red, null, st.minute),
    model: card,
    warnings: warns,
    disclaimer: disclaimer(model),
  };
  if (preP !== null) {
    out.pregame_probability_blue = pyRound(preP, 4);
    out.shift_from_pregame = pyRound(p.cal - preP, 4);
    const d = p.cal - preP;
    if (Math.abs(d) >= 0.05) {
      out.shift_note = `相比赛前, 局势已向 ${d > 0 ? a.blue : a.red} 移动 ${pyFixed(Math.abs(d) * 100, 0)} 个百分点`;
    }
  }
  card.note = confidenceNote(card);
  card.range_note = rangeNote(model.pMin, model.pMax);
  out.reasons = explainEvidence(ex, evidence(model, b.row), a.blue, a.red);
  return out;
}
