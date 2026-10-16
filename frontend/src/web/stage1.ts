/**
 * Stage 1 (赛前) 的浏览器版 —— api.Stage("pre_draft") + FeatureStore.make_row (不带 BP)
 * + /predict 路由里组响应的那一段。
 *
 * 79 个特征全部来自两队的滚动统计 (diff_/b_/r_/sum_) 加赛区和季后赛标记, 所以一张
 * 每队一行的表 (teams.json) 就够, 不需要把五年 CSV 搬进浏览器。
 *
 * 校准层 (api.Stage, train.py 的文档): 文件里有 logit_elo 段时是
 * σ((logit(clip(p_raw))·c₁ + (elo_gap·scale)·c₂) + b), elo_gap 是两队的赛区内 Elo 差 (teams.json 的 elo,
 * 见 eloRating) —— 它不进树, 只进这一层; 没有 (旧导出) 时是 Platt σ(a·p_raw + b)。exp / log 用 fdlibm.ts 的
 * 移植, 不用 Math.exp / Math.log: 看板上的 BP 后 / 赛前概率不取整就输出, 要和 Python 逐位相同。
 *
 * 和局内模型一样是第二份实现, 由 scripts/golden.mjs 对着 /predict 的真实响应逐字段核对。
 */
import { exp, log } from "./fdlibm.ts";
import { pyRound, pyStr } from "./pyfmt.ts";
import type { ExplainFile, EvidenceItem } from "./explain.ts";
import { confidenceNote, explainEvidence, rangeNote, summarize } from "./explain.ts";
import type { EloFile, TeamsFile } from "./teams.ts";
import { daysBetween, snapshot, staleness } from "./teams.ts";
import type { Forest, TreeModelFile } from "./xgb.ts";
import { loadForest, predictRaw, toVector } from "./xgb.ts";

/** train.py 写进 calib 的 logit_elo 段 (export_web_model._calib 原样导出) */
export interface LogitElo {
  coef: [number, number];
  intercept: number;
  logit_clip: number;
  elo_scale: number;
  /** Elo 那一条依据的满份量 (它在校准层里的份量, train.elo_weight) */
  elo_weight: number;
  /** 校准段上 elo_gap 的标准差 (Elo 分): Elo 那条依据按 |差| / elo_sd 缩放 (见 topEvidence)。早期导出没有 */
  elo_sd?: number;
  elo: { column: string; k: number; init: number; regress: number };
}

export interface Stage1File extends TreeModelFile {
  /**
   * 导出时永远写 "platt" —— 给发布那一刻还开着的旧 bundle 读的 (旧版 loadStage1 见到别的值就抛错, 连带局内
   * 看板一起挂掉; 见 export_web_model._calib)。这里不看它, 校准方式由 logit_elo 段在不在决定
   */
  calibrator: string;
  /** 兼容 Platt (train.py 同一校准段上拟的); 没有 logit_elo 段 (旧导出) 时它就是校准层 */
  platt: { a: number; b: number };
  logit_elo?: LogitElo;
  /** api.MODEL_KEYS 那几项 —— 模型卡只用到这些 */
  metrics: Record<string, number>;
}

export interface Stage1 {
  forest: Forest;
  a: number;
  b: number;
  /** null = Platt (旧导出) */
  logitElo: LogitElo | null;
  metrics: Record<string, number>;
}

/** Elo 差在行里的列名 (feature_store.ELO_COL) */
export const ELO_COL = "elo_gap";

export function loadStage1(file: Stage1File): Stage1 {
  // 有 logit_elo 段就是 logit_elo (api.Stage 的 calibrator == "logit_elo"), 没有就是 Platt。calibrator 字段只留给
  // 旧 bundle (见 Stage1File); 早期的导出也可能写着 "logit_elo", 同样由 logit_elo 段决定
  const logitElo: LogitElo | null = file.logit_elo ?? null;
  if (!logitElo && file.calibrator !== "platt") throw new Error(`赛前模型校准器是 ${file.calibrator}, 但文件里没有 logit_elo 段`);
  return { forest: loadForest(file), a: file.platt.a, b: file.platt.b, logitElo, metrics: file.metrics };
}

type EloPool = EloFile["pools"][string];

/**
 * feature_store.elo_rating: 某队在某池里的评分; 请求年份晚于池子最后一个比赛日的年份就向池均值回归
 * 0.5^(差几年) —— 回归是整池一起的, 不按这支队最后一场的年份算。不在池里的队是初始分, 不回归。
 */
export function eloRating(elo: EloFile, pool: EloPool | undefined, team: string, year: number): number {
  if (!pool || !Object.prototype.hasOwnProperty.call(pool.teams, team)) return elo.init;
  let r = pool.teams[team]![0];
  if (year > pool.year) {
    const f = (1 - elo.regress) ** (year - pool.year);
    r = pool.mean + f * (r - pool.mean);
  }
  return r;
}

/**
 * FeatureStore.elo_gap: 蓝减红的赛区内 Elo, 用 league 这个池 (国际赛同母赛区时调用方传母赛区)。
 * year = 请求的年份 (本地的 today, 和服务器的 pd.Timestamp.now() 同一个口径)。
 * 旧导出没有 elo 表就返回 NaN —— logit_elo 校准层见到 NaN 会抛错, 不悄悄当 0。
 */
export function eloGap(teams: TeamsFile, blue: string, red: string, league: string, year: number): number {
  const elo = teams.elo;
  if (!elo) return NaN;
  const pool = Object.prototype.hasOwnProperty.call(elo.pools, league) ? elo.pools[league] : undefined;
  return eloRating(elo, pool, blue, year) - eloRating(elo, pool, red, year);
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
    // 不进树, 只进校准层; 请求年份 = today 的年份 (本地日期, 和 staleness 同一个 today)
    [ELO_COL]: eloGap(teams, blue!, red!, league, Number(today.slice(0, 4))),
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

/** Stage.predict: 原始概率 + 校准层 (logit_elo, 旧导出是 Platt) */
export function predictStage1(s1: Stage1, row: Record<string, number>): { raw: number; cal: number } {
  const raw = predictRaw(s1.forest, toVector(s1.forest.features, row));
  const le = s1.logitElo;
  if (!le) return { raw, cal: 1.0 / (1.0 + exp(-(s1.a * raw + s1.b))) };
  const g = row[ELO_COL];
  // 和 api.Stage 一样不退回 Platt: 兼容 Platt 是给旧版服务的, 悄悄换过去就是另一个模型的数
  if (g === undefined || Number.isNaN(g)) throw new Error(`这一行没有 ${ELO_COL}, 算不了 logit_elo 校准`);
  // feature_store.calib_logit + logit_elo_prob, 运算顺序照抄; exp / log 是 fdlibm 的移植 (两边逐位相同)
  const q = Math.min(Math.max(raw, le.logit_clip), 1 - le.logit_clip);
  const z = log(q / (1 - q)) * le.coef[0] + g * le.elo_scale * le.coef[1] + le.intercept;
  return { raw, cal: 1.0 / (1.0 + exp(-z)) };
}

/**
 * Stage.top_evidence: 按权重取前 k 条 (排序稳定, 同分保持原顺序)。树特征用全局重要性; logit_elo 时
 * Elo 差放在最前也算一条 —— 它不在树里, 不这样就永远进不了理由。权重 = elo_weight × min(1, |差| / elo_sd):
 * 差距到典型大小才拿满份量, 两队只差几分不会排第一 (理由见 api.Stage.top_evidence); 没有 elo_sd 就不缩放。
 */
export function topEvidence(s1: Stage1, row: Record<string, number>, k = 8): EvidenceItem[] {
  const items: EvidenceItem[] = [];
  const le = s1.logitElo;
  if (le) {
    const g = row[ELO_COL];
    if (g !== undefined && !Number.isNaN(g)) {
      let w = le.elo_weight;
      if (le.elo_sd) w = w * Math.min(1, Math.abs(g) / le.elo_sd);
      items.push({ feature: ELO_COL, value: g, importance: w });
    }
  }
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

/**
 * api._model_card。logit_elo 没有硬上下限: 卡上不给 p_min / p_max (前端拿不到就不画"够不到"的斜线死区),
 * 范围只在 range_note 里按"出现过的范围, 不是上下限"说; Platt (旧导出) 照旧
 */
function modelCard(s1: Stage1): Record<string, unknown> {
  const d: Record<string, unknown> = {};
  for (const k of MODEL_KEYS) if (Object.prototype.hasOwnProperty.call(s1.metrics, k)) d[k] = s1.metrics[k];
  d.note = confidenceNote(d);
  if (s1.logitElo) {
    const lo = d.p_min as number | undefined;
    const hi = d.p_max as number | undefined;
    delete d.p_min;
    delete d.p_max;
    d.range_note = rangeNote(lo, hi, false);
  } else {
    d.range_note = rangeNote(d.p_min as number | undefined, d.p_max as number | undefined);
  }
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
    // logit_elo 没有硬上下限, "输出范围有限"不再成立 (同 api.predict)
    ...(s1.logitElo ? [] : ["模型概率输出范围有限, 对一边倒的对局判别力不足。"]),
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
