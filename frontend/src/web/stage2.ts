/**
 * Stage 2 (BP 后) 的浏览器版 —— api._predict_core: FeatureStore.make_row 带 draft 的那条路
 * + Stage("post_draft").predict。只给看板那条"BP 后"参考线用, 所以只要一个概率, 不组响应。
 *
 * 队伍那部分和赛前完全一样 (stage1.makeRow); 多出来的 BP 特征靠两张查表:
 *   champ_league  赛区 → 英雄 → [胜场, 场数]     至少 5 场才有胜率
 *   player_champ  选手 → 英雄 → [胜场, 场数]     至少 3 场才有胜率, 场数本身也是特征
 * 存整数, 在这里做同一次除法 —— np.mean 对 0/1 列表就是精确求和再除以 n, 逐位相同。
 * 英雄按 championKey 存和查, 选手名在进来之前已经去掉战队前缀 (board.draftFromMeta)。
 *
 * 由 scripts/golden.mjs 第 4 组对着 api._board_draft + api._predict_core 核对。
 */
import { npMean } from "./ingame.ts";
import { championKey } from "./names.ts";
import type { Stage1, Stage1File } from "./stage1.ts";
import { loadStage1, makeRow, predictStage1 } from "./stage1.ts";
import type { TeamsFile } from "./teams.ts";
import { toVector } from "./xgb.ts";

/** [胜场, 场数] */
type Tally = [number, number];

export interface Stage2File extends Stage1File {
  champion_alias: Record<string, string>;
  champ_league: Record<string, Record<string, Tally>>;
  player_champ: Record<string, Record<string, Tally>>;
}

export interface Stage2 extends Stage1 {
  alias: Map<string, string>;
  /** 用 Map 而不是普通对象: 选手名 "constructor" 这种键会查到原型链上 */
  champLeague: Map<string, Map<string, Tally>>;
  playerChamp: Map<string, Map<string, Tally>>;
}

/** 看板组出来的 BP: {side: {role: {player, champion}}}, role 是 top/jng/mid/bot/sup */
export type Draft = Record<"blue" | "red", Record<string, { player: string; champion: string }>>;

const nested = (o: Record<string, Record<string, Tally>>) =>
  new Map(Object.entries(o).map(([k, v]) => [k, new Map(Object.entries(v))]));

export function loadStage2(file: Stage2File): Stage2 {
  return {
    ...loadStage1(file),
    alias: new Map(Object.entries(file.champion_alias)),
    champLeague: nested(file.champ_league),
    playerChamp: nested(file.player_champ),
  };
}

const ROLES = ["top", "jng", "mid", "bot", "sup"];

/** FeatureStore.champ_winrate (before=None, min_g=5) */
function champWinrate(s2: Stage2, champion: string, league: string): number {
  const t = s2.champLeague.get(league)?.get(championKey(champion, s2.alias));
  return t && t[1] >= 5 ? t[0] / t[1] : NaN;
}

/** FeatureStore.player_champ_stats: (胜率 —— 不到 3 场是 NaN, 场数) */
function playerChampStats(s2: Stage2, player: string, champion: string): [number, number] {
  const t = s2.playerChamp.get(player)?.get(championKey(champion, s2.alias));
  const n = t ? t[1] : 0;
  return [t && n >= 3 ? t[0] / n : NaN, n];
}

/** FeatureStore.make_row(..., draft=...) —— 队伍不认识就抛错, 文字和服务器一致 */
export function makeDraftRow(
  s2: Stage2,
  teams: TeamsFile,
  blue: string | null,
  red: string | null,
  league: string,
  draft: Draft,
  today: string,
): Record<string, number> {
  const { row: m } = makeRow(teams, blue, red, league, 0, today);
  const bw: number[] = [];
  const rw: number[] = [];
  const bc: number[] = [];
  const rc: number[] = [];
  for (const [side, key, accw, accc] of [
    ["b", "blue", bw, bc],
    ["r", "red", rw, rc],
  ] as const) {
    const picks = draft[key] ?? {};
    for (const role of ROLES) {
      const ent = picks[role];
      if (!ent) continue; // make_row 在这里只记一条警告; 看板只要概率
      const cw = champWinrate(s2, ent.champion, league);
      if (!Number.isNaN(cw)) accw.push(cw);
      const [pw, ng] = playerChampStats(s2, ent.player, ent.champion);
      accc.push(ng);
      m[`${side}_${role}_champ_wr`] = pw;
      m[`${side}_${role}_champ_games`] = ng;
    }
  }
  m.b_avg_champ_wr = bw.length ? npMean(bw) : NaN;
  m.r_avg_champ_wr = rw.length ? npMean(rw) : NaN;
  if (bw.length && rw.length) {
    m.diff_avg_champ_wr = m.b_avg_champ_wr - m.r_avg_champ_wr;
    m.sum_avg_champ_wr = m.b_avg_champ_wr + m.r_avg_champ_wr;
  }
  const sum = (a: number[]) => a.reduce((x, y) => x + y, 0);
  m.b_comfort = sum(bc);
  m.r_comfort = sum(rc);
  m.diff_comfort = sum(bc) - sum(rc);
  m.sum_comfort = sum(bc) + sum(rc);
  for (const role of ROLES) {
    m[`diff_${role}_comfort`] = (m[`b_${role}_champ_games`] ?? 0) - (m[`r_${role}_champ_games`] ?? 0);
  }
  return m;
}

/** Stage.predict 组的那个向量 (NaN → -999), 给黄金测试逐位核对用 */
export function draftVector(s2: Stage2, row: Record<string, number>) {
  return toVector(s2.forest.features, row);
}

/** api._predict_core: 带 BP 的一行特征 → Stage 2 的校准概率 */
export function predictCore(
  s2: Stage2,
  teams: TeamsFile,
  blue: string | null,
  red: string | null,
  league: string,
  draft: Draft,
  today: string,
): number {
  return predictStage1(s2, makeDraftRow(s2, teams, blue, red, league, draft, today)).cal;
}
