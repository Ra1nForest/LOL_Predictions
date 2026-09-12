/**
 * Stage 1 (赛前) 的浏览器版 —— api.Stage("pre_draft") + FeatureStore.make_row (不带 BP)
 * + /predict 路由里组响应的那一段。
 *
 * 79 个特征全部来自两队的滚动统计 (diff_/b_/r_/sum_) 加赛区和季后赛标记, 所以一张
 * 每队一行的表 (teams.json) 就够, 不需要把五年 CSV 搬进浏览器。
 *
 * 和局内模型一样是第二份实现, 由 scripts/golden.mjs 对着 /predict 的真实响应逐字段核对。
 */
import { pyRound, pyStr } from "./pyfmt.ts";
import type { ExplainFile, EvidenceItem } from "./explain.ts";
import { confidenceNote, explainEvidence, rangeNote, summarize } from "./explain.ts";
import type { TeamsFile } from "./teams.ts";
import { daysBetween, snapshot, staleness } from "./teams.ts";
import type { Forest, TreeModelFile } from "./xgb.ts";
import { loadForest, predictRaw, toVector } from "./xgb.ts";

export interface Stage1File extends TreeModelFile {
  calibrator: string;
  platt: { a: number; b: number };
  /** api.MODEL_KEYS 那几项 —— 模型卡只用到这些 */
  metrics: Record<string, number>;
}

export interface Stage1 {
  forest: Forest;
  a: number;
  b: number;
  metrics: Record<string, number>;
}

export function loadStage1(file: Stage1File): Stage1 {
  if (file.calibrator !== "platt") throw new Error(`赛前模型校准器是 ${file.calibrator}, 这里只实现了 Platt`);
  return { forest: loadForest(file), a: file.platt.a, b: file.platt.b, metrics: file.metrics };
}

/**
 * FeatureStore.make_row(blue, red, league) —— 不带 BP 的那条路。
 * 队伍不认识就抛错, 信息和服务器一字不差 (看板和 /predict 都会把它原样展示)。
 */
export function makeRow(
  teams: TeamsFile,
  blue: string | null,
  red: string | null,
  league: string,
  playoffs: number,
  today: string,
): { row: Record<string, number>; warnings: string[] } {
  const warn: string[] = [];
  const st = staleness(teams, league, today);
  if (st.message) warn.push(st.message);
  const bs = snapshot(teams, blue);
  const rs = snapshot(teams, red);
  if (!bs) throw new Error(`unknown or insufficient history: ${pyStr(blue)}`);
  if (!rs) throw new Error(`unknown or insufficient history: ${pyStr(red)}`);
  for (const [nm, s] of [
    [blue!, bs],
    [red!, rs],
  ] as const) {
    if (s.nGames < 5) warn.push(`${nm} 仅有 ${s.nGames} 场历史, 统计不稳定`);
    const gap = daysBetween(today, s.lastDate);
    if (gap >= 14) warn.push(`${nm} 最后一场是 ${s.lastDate} (${gap} 天前), 近期状态未知`);
  }

  const m: Record<string, number> = {
    is_lpl: league === "LPL" ? 1 : 0,
    is_lck: league === "LCK" ? 1 : 0,
    is_lec: league === "LEC" ? 1 : 0,
    is_lcs: league === "LCS" ? 1 : 0,
    playoffs,
  };
  const sums = new Set(teams.sum_feats);
  for (const c of teams.roll_cols) {
    const bv = bs.values.get(c) ?? NaN;
    const rv = rs.values.get(c) ?? NaN;
    m[`diff_${c}`] = bv - rv;
    m[`b_${c}`] = bv;
    m[`r_${c}`] = rv;
    if (sums.has(c)) m[`sum_${c}`] = bv + rv;
  }
  return { row: m, warnings: warn };
}

/** Stage.predict: 原始概率 + Platt 校准 */
export function predictStage1(s1: Stage1, row: Record<string, number>): { raw: number; cal: number } {
  const raw = predictRaw(s1.forest, toVector(s1.forest.features, row));
  return { raw, cal: 1.0 / (1.0 + Math.exp(-(s1.a * raw + s1.b))) };
}

/** Stage.top_evidence: 按全局特征重要性取前 k 个有值的特征 (排序稳定, 同分保持特征顺序) */
export function topEvidence(s1: Stage1, row: Record<string, number>, k = 8): EvidenceItem[] {
  const items: EvidenceItem[] = [];
  for (const f of s1.forest.features) {
    const v = row[f];
    if (v === undefined || Number.isNaN(v)) continue;
    items.push({ feature: f, value: v, importance: s1.forest.importance.get(f) ?? 0 });
  }
  items.sort((x, y) => y.importance - x.importance);
  return items.slice(0, k);
}

export interface PreCtx {
  preRow: Record<string, number>;
  preP: number;
  warnings: string[];
}

/** api._pre_context: 局内模型要的赛前特征 + 赛前概率。队伍不认识就抛错。 */
export function preContext(
  s1: Stage1,
  teams: TeamsFile,
  blue: string | null,
  red: string | null,
  league: string,
  today: string,
): PreCtx {
  const { row, warnings } = makeRow(teams, blue, red, league, 0, today);
  const preRow: Record<string, number> = {};
  for (const [k, v] of Object.entries(row)) {
    if (k.startsWith("diff_avg_")) preRow[`diff_pre_${k.slice("diff_avg_".length)}`] = v;
  }
  if (Object.prototype.hasOwnProperty.call(row, "diff_rolling_wr")) preRow.diff_pre_wr = row.diff_rolling_wr!;
  return { preRow, preP: predictStage1(s1, row).cal, warnings };
}

const MODEL_KEYS = ["accuracy", "baseline", "lift", "ece_cal", "p_min", "p_max"];

/** api._model_card */
function modelCard(s1: Stage1): Record<string, unknown> {
  const d: Record<string, unknown> = {};
  for (const k of MODEL_KEYS) if (Object.prototype.hasOwnProperty.call(s1.metrics, k)) d[k] = s1.metrics[k];
  d.note = confidenceNote(d);
  d.range_note = rangeNote(d.p_min as number | undefined, d.p_max as number | undefined);
  return d;
}

export interface PredictBody {
  blue_team: string;
  red_team: string;
  league: string;
  playoffs?: boolean;
}

/**
 * POST /predict 的浏览器版 (不带 BP、不带 agent_review)。
 * 队伍不认识时服务器回 400, 这里抛出同样文字的 Error。
 */
export function predictResponse(
  s1: Stage1,
  teams: TeamsFile,
  ex: ExplainFile,
  body: PredictBody,
  today: string,
): Record<string, unknown> {
  const blue = body.blue_team;
  const red = body.red_team;
  const league = body.league;
  const playoffs = !!body.playoffs;
  const { row, warnings } = makeRow(teams, blue, red, league, playoffs ? 1 : 0, today);

  const st = staleness(teams, league, today);
  const { raw, cal } = predictStage1(s1, row);
  const stage = {
    probability_blue: pyRound(cal, 4),
    probability_raw: pyRound(raw, 4),
    summary: summarize(ex, cal, blue, red, "1_pre_draft"),
    model: modelCard(s1),
    reasons: explainEvidence(ex, topEvidence(s1, row), blue, red),
  };

  const disc = [
    "概率来自留出测试集校准的统计模型, 非投注建议。",
    "模型概率输出范围有限, 对一边倒的对局判别力不足。",
    "walk-forward 估计的期望 lift 约 +0.13 ± 0.10, 单场预测不确定性大。",
  ];
  if (st.level === "very_stale") disc.unshift(`⚠ 本次预测基于 ${st.days} 天前的数据, 结果不可靠。`);
  else if (st.level === "stale") disc.unshift(`⚠ 数据滞后 ${st.days} 天, 未反映最近比赛。`);

  return {
    blue_team: blue,
    red_team: red,
    league,
    playoffs,
    data_quality: {
      league_data_through: st.last_date ?? null,
      days_old: st.days,
      level: st.level,
      reliable: st.level === "fresh",
    },
    warnings,
    stages: { "1_pre_draft": stage },
    final: {
      probability_source: "1_pre_draft",
      summary: summarize(ex, cal, blue, red, "1_pre_draft"),
      note: "Stage 3 为 advisory-only, 不影响此概率",
      probability_blue: pyRound(cal, 4),
      probability_red: pyRound(1 - cal, 4),
      implied_fair_odds_blue: pyRound(1 / cal, 3),
      implied_fair_odds_red: pyRound(1 / (1 - cal), 3),
    },
    disclaimer: disc.join(" "),
  };
}
