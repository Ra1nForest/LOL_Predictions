/**
 * 看板和比赛列表 —— api.py 里 /esports/live、/esports/upcoming、/esports/board 的浏览器版。
 *
 * 输出和服务器逐字段同形, 前端组件一行不用改, 包括 BP 后 (Stage 2) 那条参考线
 * (postdraft_probability_blue, 见 stage2.ts 和 draftFromMeta)。
 *
 * 正确性由 scripts/diff-board.mjs 保证: 对已打完的比赛和一个全新的 Python 进程逐字段比对。
 *
 * 国际赛 (LEAGUE_IDS 里四大赛区之后那几个) 的规矩和 api.py 一样, 全在 boardPlan 一处定:
 *   · 两队同一个母赛区: 赛前 / BP 后按母赛区算 (pregameLeague), 局内模型仍拿赛事本身当赛区
 *   · 跨赛区, 或有队名映射不出来: 没有赛前和 BP 后, 局内模型不带赛前特征、不渐变, 第 3 分钟起
 *     就是它自己的数; 映射不出来的队用 lolesports 原名当标签, 绝不拿去查队伍表
 *   · 其中两队都能映射进跨赛区模型 C 的状态 (xregion.json, 含四大之外的队)、本赛事赛程取得到、两队母赛区
 *     不同时: 赛前数来自 C (xregionBoard, = api._xregion_board), 开局 3-15 分钟以它为锚渐变, 局内模型仍不带
 *     赛前特征; 没有 BP 后 (加英雄胜率的候选 D 没过闸)。C 算不出来就一字不改地保持上一条
 * 四大赛区两队都认识时这些分支一个都不走, 国内看板和原来逐字段相同 (也不多发任何上游请求)。
 */
import type { ExplainFile } from "./explain.ts";
import type { IngameModel } from "./ingame.ts";
import { NO_PRE_STATS, ingameResponse } from "./ingame.ts";
import type { PreCtx, Stage1 } from "./stage1.ts";
import { preContext } from "./stage1.ts";
import type { Draft, Stage2 } from "./stage2.ts";
import { predictCore } from "./stage2.ts";
import type { TeamsFile } from "./teams.ts";
import { homeLeague, knownTeams, mapTeam, norm } from "./teams.ts";
import type { LiveState, Match, MetaEntry, TimelineRow } from "./feed.ts";
import { DDRAGON, Feed, LEAGUE_IDS, PERSISTED, isMajor, parseTs19, predictable, toState } from "./feed.ts";
import { oePlayerName } from "./names.ts";
import { pyRound } from "./pyfmt.ts";
import type { Series, XregionState } from "./xregion.ts";
import { boardPredict, coefficientsFor, replayFramesNeeded, seriesWins, xregionKnown } from "./xregion.ts";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Json = any;

export interface Models {
  ingame: IngameModel;
  s1: Stage1;
  s2: Stage2;
  ex: ExplainFile;
  /**
   * 跨赛区模型 C 的状态 (public/web/xregion.json)。null / 缺席 = 没加载 (下载失败等):
   * 跨赛区看板保持原来的行为, 和服务器读不到 artifacts/xregion.json 时一样
   */
  xr?: XregionState | null;
}

/** 看板/列表失败时带状态码的错误, 对应服务器的 HTTPException */
export class HttpError extends Error {
  // 显式字段而不是构造函数参数属性: Node 直接跑 .ts 时只剥类型, 不支持参数属性
  readonly status: number;
  constructor(status: number, msg: string) {
    super(msg);
    this.name = "HttpError";
    this.status = status;
  }
}

/** api._warn: 失败会改变输出数字的地方, 把说明带进响应 (浏览器里只能 console) */
function warn(where: string, e: unknown): string {
  const msg = `${where}: ${e instanceof Error ? `${e.name}: ${e.message}` : String(e)}`;
  console.warn(`⚠ ${msg}`);
  return msg;
}

/** api._clinched: 按赛制算需要几胜 —— BO3 打到 1-0 不算结束 */
function clinched(m: Match): boolean {
  const need = Math.floor((m.best_of || 1) / 2) + 1;
  return m.teams.some((t) => (t.game_wins || 0) >= need);
}

/** api._refresh_wins: 比分换成 20 秒缓存的那一份。取不到就原样不动 */
async function refreshWins(m: Match, feed: Feed): Promise<boolean> {
  let wins: Map<string, number>;
  try {
    wins = await feed.matchWins(m.match_id, norm);
  } catch {
    return false;
  }
  if (!wins.size) return false;
  let changed = false;
  for (const t of m.teams) {
    let w = wins.get(norm(t.api_name));
    if (w === undefined) w = wins.get(norm(t.code || ""));
    if (w !== undefined && w !== t.game_wins) {
      t.game_wins = w;
      changed = true;
    }
  }
  return changed;
}

// ══════════════════════════════════════════════════════════
//  国际赛: 映射队名用哪份表、赛前/BP 后按哪个赛区算、为什么不给
// ══════════════════════════════════════════════════════════

/**
 * api._known_for: 映射这个赛区的队名时用哪份已知队伍表 (喂给 mapTeam)。
 * 四大赛区用本赛区的表; 国际赛用四大赛区的并集 —— 参赛队来自各个赛区, 而按赛事名去查恒为空表,
 * 空表 (修好之后) 会让 mapTeam 一律返回 null。所有映射队名的地方只走这一处。
 */
export function knownFor(teams: TeamsFile, league: string | null | undefined): string[] {
  return isMajor(league) ? knownTeams(teams, String(league).toUpperCase()) : teams.known_all;
}

/**
 * api.pregame_league: 赛前 (Stage 1) 和 BP 后 (Stage 2) 按哪个赛区的口径算; null = 不给这两段。
 *   · 四大赛区: 原样返回比赛的赛区 —— 国内比赛的行为一点都不变
 *   · 国际赛, 两队都认识且母赛区相同: 返回那个母赛区 (实测这类比赛上 Stage 1/2 和国内表现一样)
 *   · 其余 (跨赛区, 或有一队不认识): null。Stage 1/2 的特征全是"赛区内的相对量", 看不见赛区之间
 *     的差距; 历史跨赛区国际赛上准确率约 50%, 却和国内一样自信 (research/gate_international.py)
 * Stage 4 不受影响, 仍拿比赛本身的赛区。由 test:golden 的 intl_cases 和 Python 对账。
 */
export function pregameLeague(
  teams: TeamsFile,
  blue: string | null | undefined,
  red: string | null | undefined,
  matchLeague: string | null | undefined,
): string | null {
  if (isMajor(matchLeague)) return matchLeague as string;
  if (!blue || !red) return null;
  const hb = homeLeague(teams, blue);
  return hb !== null && hb === homeLeague(teams, red) ? hb : null;
}

/**
 * api._pregame_of: 按这场比赛**此刻**的模型队名定下 m.pregame_league 并返回。
 * matchJson 每次序列化都重算 —— 看板会在中途重新映射队名, 存一份旧值就可能和队名对不上。
 */
function pregameOf(teams: TeamsFile, m: Match): string | null {
  const two = m.teams.length === 2;
  m.pregame_league = pregameLeague(
    teams,
    two ? m.teams[0]!.model_name : null,
    two ? m.teams[1]!.model_name : null,
    m.league,
  );
  return m.pregame_league;
}

/** 看板上"为什么没有赛前/BP 后"的说法, 和 api._CROSS_REGION_NOTE 一字不差 (改了要同步 i18n.ts) */
const CROSS_REGION_NOTE =
  "跨赛区对阵: 赛前和 BP 后模型只在赛区内战上验证过, 在历史跨赛区国际赛上没有预测力, " +
  "这里不显示; 开局 3 分钟起直接用局内模型";

/**
 * 跨赛区模型 C 出数时, 上面那句换成这句 (whyKind = "xregion")。和 api._XREGION_NOTE 一字不差, 数字来自
 * research/gate_cross_region.py 的 HOLDOUT; i18n.ts 有对应的英文模板。
 */
const XREGION_NOTE =
  "跨赛区对阵: 赛前数字来自跨赛区模型 (只看两队和所在赛区的历史战绩, 不看英雄), " +
  "历史上准确率约 65%, 而且偏自信 —— 它说 85% 的局实际约赢 76%";

/** api._xregion_note: C 的那句说明; 母赛区在状态里没有跨赛区国际赛记录 (偏移全靠先验) 的, 点名补一句 */
export function xregionNote(noHistory: string[]): string {
  let s = XREGION_NOTE;
  if (noHistory.length) s += `; ${noHistory.join(" 和 ")} 此前没有跨赛区国际赛记录, 这个数主要靠先验`;
  return s;
}

/**
 * api._xregion_pregame: 这一局还没有帧 (没开打) 时看板的 prediction —— 只有 C 的赛前数。原来这时看板不给
 * prediction, 界面自己拿 Stage 1 去问 /predict; 跨赛区的比赛 pregame_league 为 null, 什么都不显示。C 的数要
 * 重放本赛事的赛程才算得出来, 只能由看板带回去。没有 BP 后那条线 (候选 D 没过闸)。
 */
export function xregionPregame(p: number, why: string): Record<string, unknown> {
  return {
    too_early: false,
    minute: null,
    pregame_probability_blue: p,
    postdraft_probability_blue: null,
    probability_blue: p,
    source: "xregion",
    warnings: [why],
  };
}

export type PregameKind = "unmapped" | "cross_region" | "intl_home" | "xregion";

/**
 * api._pregame_reason: [kind, 一句话] 或 [null, null]。kind:
 *   unmapped      有队名映射不出来 —— 没有赛前和 BP 后, 只有局内模型
 *   cross_region  国际赛跨赛区 (两队都认识) —— 同上, 但原因是模型没在跨赛区上验证过
 *   intl_home     国际赛同母赛区 —— 照常给赛前和 BP 后, 但要说明口径是两队母赛区的内战
 * 四大赛区两队都认识时返回 [null, null], 国内看板一个字都不多。
 * 只吃最小的形状 (lolesports 原名 + 模型队名), 看板 (feed 的 Match) 和回放器 (看板响应里的
 * match) 共用。
 */
export function pregameReason(
  ts: { api: string | null | undefined; model: string | null | undefined }[],
  league: string | null | undefined,
  pgLeague: string | null | undefined,
): [PregameKind, string] | [null, null] {
  if (ts.length !== 2) return [null, null];
  const miss = ts.filter((t) => t.model == null).map((t) => t.api || "");
  if (miss.length) {
    return [
      "unmapped",
      `${miss.join(" 和 ")} 不在四大赛区的数据里, 没有赛前和 BP 后预测; ` + "开局 3 分钟起直接用局内模型",
    ];
  }
  if (pgLeague == null) return ["cross_region", CROSS_REGION_NOTE];
  if (!isMajor(league)) {
    return ["intl_home", `国际赛: 赛前和 BP 后按两队所在的 ${pgLeague} 内战口径计算, 国际赛的场次不计入近况`];
  }
  return [null, null];
}

/**
 * api._board_warnings: 看板局内阶段的提醒, "为什么没有赛前/BP 后"放第一条。跨赛区 / 队名不认识时
 * 是**故意**不带赛前特征, 局内模型那句"没有可用的赛前队伍统计"说错了原因又和第一条重复, 按原文去掉。
 * 只在看板这一层做 (头条和 player.ts 的直播回放共用), ingameResponse 本身不动。
 * 跨赛区模型 C 出数时 (xregion) 同样不带赛前特征 (C 的数只当开局渐变的锚), 也去掉那一句。
 */
export function boardWarnings(warns: string[], whyKind: PregameKind | null, why: string | null): string[] {
  let out = [...warns];
  if (whyKind === "unmapped" || whyKind === "cross_region" || whyKind === "xregion") {
    out = out.filter((w) => w !== NO_PRE_STATS);
  }
  if (why) out.unshift(why);
  return out;
}

/**
 * api._early_prediction: 开局头 3 分钟 (局内模型还用不上) 那一段的 prediction, 纯函数。
 * test:diff 只取终局帧, 碰不到这一段; 由 test:golden 的 intl_cases 逐条对账。
 * whyKind = "xregion" (C 出数): p1 是 C 的赛前数, source 写 "xregion", 那句说明放进 warnings (同 intl_home)。
 */
export function earlyPrediction(
  minute: number | null,
  p1: number | null,
  p2: number | null,
  whyKind: PregameKind | null,
  why: string | null,
): Record<string, unknown> {
  const head =
    (minute !== null ? `开局第 ${minute} 分钟, ` : "刚开局 (局内时间还没同步出来), ") +
    "局内模型最早在第 10 分钟的切片上训练过。";
  let tail: string;
  if (p1 !== null || p2 !== null) {
    tail = "这里显示的是" + (p2 !== null ? "BP 后" : "赛前") + "的概率 —— 它不看局内数据, 但一直有效。";
  } else if (whyKind === "unmapped" || whyKind === "cross_region") {
    tail = why + "。";
  } else {
    // 两段都算不出来 (队伍历史不足、出错)。原来这里仍然写"这里显示的是赛前的概率",
    // 而 probability_blue 明明是 null —— 说了一个页面上并不存在的数
    tail = "这一局算不出赛前和 BP 后的概率, 开局 3 分钟起才有局内胜率。";
  }
  const pred: Record<string, unknown> = {
    too_early: true,
    minute,
    pregame_probability_blue: p1,
    postdraft_probability_blue: p2,
    probability_blue: p2 ?? p1,
    source: p2 !== null ? "post_draft" : p1 !== null ? (whyKind === "xregion" ? "xregion" : "pre_draft") : null,
    note: head + tail,
  };
  if (whyKind === "intl_home" || whyKind === "xregion") pred.warnings = [why];
  return pred;
}

/**
 * 看板上一局怎么算 —— api.esports_board 里 two / use_pre / bl / rl / why 那几行。头条、曲线、
 * 逐秒走势 (fineTimeline) 和直播回放 (player.ts) 全按它来, 四处说法才一致。
 *   blue / red  标签: 模型队名, 映射不出来就是 lolesports 原名 —— 只用于摘要和理由里的队名
 *   league      比赛本身的赛区, 给局内模型 (国际赛 is_* 全 0)
 *   preLeague   use_pre 成立时 = pregame_league: Stage 1/2 按它算、局内模型带赛前特征、开局渐变;
 *               null = 都不做 (跨赛区 / 有队名不认识)。四大赛区两队都认识时就是比赛赛区,
 *               国内看板不变
 *   xr          跨赛区模型 C 的赛前数 (preLeague 为 null 时才可能有): 不带赛前特征, 但开局 3-15 分钟以它为
 *               锚渐变; whyKind / why 换成 C 的说明。null = 没有 C, 和原来一样
 * 不是两队 (null) 就什么都不算。
 */
export interface BoardPlan {
  blue: string;
  red: string;
  league: string;
  preLeague: string | null;
  xr: number | null;
  whyKind: PregameKind | null;
  why: string | null;
}

/** 看板响应里 C 的那一块 (api._xregion_board 的 public): 回放器和逐秒走势从这里拿锚和说明 */
export interface XregionPublic {
  probability_blue: number;
  game_number: number;
  blue: string;
  red: string;
  home_blue: string | null;
  home_red: string | null;
  no_history: string[];
  /** 这一局还没有帧 (不知道谁蓝方): probability_blue 是两种选边的平均, 不能当按帧选边的局内渐变锚 */
  side_neutral: boolean;
  replayed: number;
  updated: number;
  deduped: number;
  oe_asof: string;
}

/** boardPlan 的 xr 参数: C 的数 (未舍入) 和没有跨赛区记录的母赛区 */
export interface PlanXr {
  p: number;
  noHistory: string[];
}

/**
 * 看板响应 → boardPlan 的 xr 参数 (player.ts / static-api.fine 用; 看板里没有 C 就是 null)。
 * 两种选边的平均 (side_neutral: 看板取到时这一局还没有帧) 不当锚: 回放器手里已经有帧、知道谁蓝方, 拿一个
 * "不知道谁蓝方"的数去渐变会差出半个蓝方优势。看板一有帧就按帧的选边重算, 下一轮轮询就换上。
 */
export function planXrOf(board: {
  xregion?: { probability_blue: number; no_history: string[]; side_neutral?: boolean } | null;
}): PlanXr | null {
  const x = board.xregion;
  return x && typeof x.probability_blue === "number" && !x.side_neutral
    ? { p: x.probability_blue, noHistory: x.no_history ?? [] }
    : null;
}

export function boardPlan(
  ts: { api: string | null | undefined; model: string | null | undefined }[],
  league: string,
  pgLeague: string | null | undefined,
  xr: PlanXr | null = null,
): BoardPlan | null {
  if (ts.length !== 2) return null;
  const usePre = ts.every((t) => t.model != null) && pgLeague != null;
  let [whyKind, why] = pregameReason(ts, league, pgLeague) as [PregameKind | null, string | null];
  // C 出数时原来那句 ("跨赛区不显示" / "不在四大赛区的数据里") 换成 C 的说明; 没有 C 就一字不改
  const useXr = !usePre && xr !== null;
  if (useXr) [whyKind, why] = ["xregion", xregionNote(xr.noHistory)];
  return {
    blue: ts[0]!.model || ts[0]!.api || "",
    red: ts[1]!.model || ts[1]!.api || "",
    league,
    preLeague: usePre ? pgLeague! : null,
    xr: useXr ? xr.p : null,
    whyKind,
    why,
  };
}

/** api._match_json */
function matchJson(teams: TeamsFile, m: Match): Record<string, unknown> {
  return {
    match_id: m.match_id,
    league: m.league,
    start_time: m.start_time,
    state: m.state,
    best_of: m.best_of,
    predictable: predictable(m),
    // 赛前 / BP 后按哪个赛区算, null = 不给 (跨赛区或有队不认识) —— 见 pregameLeague
    pregame_league: pregameOf(teams, m),
    teams: m.teams.map((t) => ({
      name: t.api_name,
      code: t.code,
      model_name: t.model_name,
      known: t.model_name !== null,
      game_wins: t.game_wins,
      image: t.image,
    })),
    unknown_teams: m.teams.filter((t) => t.model_name === null).map((t) => t.api_name),
  };
}

// ══════════════════════════════════════════════════════════
//  列表
// ══════════════════════════════════════════════════════════

/** GET /esports/live —— getLive 会漏, 再拿最近开赛的比赛逐个查帧补上 */
export async function liveList(feed: Feed, teams: TeamsFile): Promise<{ count: number; matches: Json[] }> {
  const found: Match[] = [];
  const seen = new Set<string>();
  try {
    for (const m of await feed.live()) {
      found.push(m);
      seen.add(m.match_id);
    }
  } catch (e) {
    console.warn(`getLive 失败: ${e}`);
  }
  try {
    const cands: Match[] = [];
    for (const [lg] of LEAGUE_IDS) {
      // 按赛区各自 try (同 api.esports_live / upcomingList): 一个国际赛的赛程取不到, 不能把四大的
      // 候选连同"局间休息"那段判断一起丢掉
      let ms: Match[];
      try {
        ms = await feed.schedule(lg, knownFor(teams, lg));
      } catch (e) {
        console.warn(`${lg} 赛程取不到, 这一轮跳过: ${e}`);
        continue;
      }
      for (const m of ms) {
        if (seen.has(m.match_id) || clinched(m)) continue;
        cands.push(m);
      }
    }
    for (const m of await feed.liveByFrames(cands)) {
      if (!seen.has(m.match_id)) {
        found.push(m);
        seen.add(m.match_id);
      }
    }
    // 局间休息: 赢过局、没分出胜负、此刻没有一局有帧 —— 比赛仍在进行, 不该从列表里消失
    const now = Date.now();
    for (const m of cands) {
      if (seen.has(m.match_id)) continue;
      if (!m.teams.some((t) => (t.game_wins || 0) > 0)) continue;
      const ts = parseTs19(m.start_time);
      if (Number.isNaN(ts)) continue;
      if (now - 6 * 3600_000 <= ts && ts <= now) {
        m.between_games = true;
        found.push(m);
        seen.add(m.match_id);
      }
    }
  } catch (e) {
    console.warn(`live_by_frames 失败: ${e}`);
  }

  const out: Json[] = [];
  for (const m of found) {
    const known = knownFor(teams, m.league);
    for (const t of m.teams) t.model_name = mapTeam(teams, t.api_name, known);
    await refreshWins(m, feed);
    // 比分刷新后重判: 已经打完的系列赛不能赖在直播列表里
    if (m.between_games && clinched(m)) continue;
    out.push({ ...matchJson(teams, m), between_games: !!m.between_games });
  }
  return { count: out.length, matches: out };
}

/** GET /esports/upcoming —— 不按 state 过滤 (它会把没开赛的比赛标成 completed) */
export async function upcomingList(
  feed: Feed,
  teams: TeamsFile,
  limit = 24,
  pastHours = 5,
): Promise<{ count: number; matches: Json[]; errors: string[] }> {
  const now = Date.now();
  const out: Json[] = [];
  const errs: string[] = [];
  for (const [lg] of LEAGUE_IDS) {
    try {
      for (const m of await feed.schedule(lg, knownFor(teams, lg))) {
        if (!m.start_time) continue;
        const ts = parseTs19(m.start_time);
        if (Number.isNaN(ts)) continue;
        if (ts < now - pastHours * 3600_000) continue;
        if (m.teams.some((t) => (t.game_wins || 0) > 0)) continue;
        // 两边都是 TBD 的占位赛不列 (api.esports_upcoming 同): 只有时间没有对阵, 会挤掉真比赛
        if (m.teams.every((t) => ["", "TBD"].includes((t.api_name ?? "").toUpperCase()))) continue;
        out.push(matchJson(teams, m));
      }
    } catch (e) {
      errs.push(`${lg}: ${e instanceof Error ? e.message : String(e)}`);
    }
  }
  const key = (x: Json) => x.start_time || "";
  out.sort((a, b) => (key(a) < key(b) ? -1 : key(a) > key(b) ? 1 : 0));
  return { count: out.length, matches: out.slice(0, limit), errors: errs };
}

// ══════════════════════════════════════════════════════════
//  看板
// ══════════════════════════════════════════════════════════

function sidesOf(g: Json): { blue?: string; red?: string } {
  const out: Record<string, string> = {};
  for (const t of (g && g.teams) || []) if (t.side === "blue" || t.side === "red") out[t.side] = t.id ?? null;
  return out;
}

/** api._playable_games: 有帧即算数, 和 games[].state 取并集 */
function playableGames(games: Json[], framed: Map<string, LiveState>): Json[] {
  return games.filter((g) => g.state === "inProgress" || g.state === "completed" || framed.has(g.id));
}

async function playersJson(feed: Feed, gameId: string, ver: string): Promise<Json[]> {
  const icon = (i: number) => ({ id: i, icon: `${DDRAGON}/cdn/${ver}/img/item/${i}.png` });
  return (await feed.players(gameId)).map((p) => ({
    participant_id: p.participant_id,
    side: p.side,
    role: p.role,
    summoner_name: p.summoner_name,
    champion: p.champion,
    champion_icon: p.champion ? `${DDRAGON}/cdn/${ver}/img/champion/${p.champion}.png` : null,
    level: p.level,
    kills: p.kills,
    deaths: p.deaths,
    assists: p.assists,
    cs: p.cs,
    gold: p.gold,
    kill_participation: p.kill_participation,
    damage_share: p.damage_share,
    wards_placed: p.wards_placed,
    wards_destroyed: p.wards_destroyed,
    current_health: p.current_health,
    max_health: p.max_health,
    items: p.items.map(icon),
    trinket: p.trinket ? icon(p.trinket) : null,
    stats: p.stats,
    perks: p.perks,
    abilities: p.abilities,
  }));
}

/** 五路对位的经济差 —— 看板和回放器 (player.ts) 共用 */
export function lanesOf(ps: Json[]): Json[] {
  const alias: Record<string, string> = { jng: "jungle", bot: "bottom", sup: "support" };
  const lanes: Json[] = [];
  for (const lane of ["top", "jungle", "mid", "bottom", "support"]) {
    const pick = (side: string) => ps.find((p) => p.side === side && (alias[p.role] ?? p.role) === lane);
    const b = pick("blue");
    const r = pick("red");
    if (b && r) lanes.push({ lane, gold_diff: b.gold - r.gold, blue_gold: b.gold, red_gold: r.gold });
  }
  return lanes;
}

/**
 * 走势曲线逐点算胜率要的上下文: 赛前特征只和队伍有关、阵容整局不变, 整条曲线取一次。
 * blue / red 是标签: 队名映射不出来时是 lolesports 原名 —— 那时 preLeague 必为 null,
 * 原名不会进任何查表 (includePregame=false 的那条路根本不碰队名)。
 */
interface CurveCtx {
  blue: string;
  red: string;
  /** 比赛本身的赛区 —— 局内模型的 is_*; 国际赛全 0, 和训练时一样 */
  league: string;
  /**
   * 赛前口径 (看板的 use_pre 成立时 = pregame_league)。null = 不带赛前特征、不渐变
   * (跨赛区 / 队名不认识), 同头条
   */
  preLeague: string | null;
  preCtx: PreCtx | null;
  ln: [string[], string[]] | null;
  /** BP 后概率 (没有就 null, 渐变退回赛前) —— 曲线和头条同一个渐变 */
  prior: number | null;
  /** 跨赛区模型 C 的赛前数 (boardPlan.xr): preLeague 为 null 时以它为锚渐变, 同头条 */
  xr: number | null;
}

async function curveCtx(
  feed: Feed,
  teams: TeamsFile,
  s1: Stage1,
  blue: string,
  red: string,
  league: string,
  preLeague: string | null,
  today: string,
  gameId: string,
  prior: number | null,
  xr: number | null = null,
): Promise<CurveCtx> {
  // 赛前特征只和队伍有关, 整条曲线算一次就够。没有赛前口径时不算 (同头条)
  let preCtx: PreCtx | null = null;
  if (preLeague !== null) {
    try {
      preCtx = preContext(s1, teams, blue, red, preLeague, today);
    } catch (e) {
      warn("曲线取赛前特征失败, 曲线会偏离头条", e);
    }
  }
  let ln: [string[], string[]] | null = null;
  try {
    ln = await lineups(feed, gameId);
  } catch (e) {
    warn("曲线取阵容失败, 曲线会偏离头条", e);
  }
  return { blue, red, league, preLeague, preCtx, ln, prior, xr };
}

/** 曲线上一行的胜率, 和看板头条同一个函数。失败给 null, 调用方沿用上一个点 */
function curveProb(m: Models, teams: TeamsFile, today: string, c: CurveCtx, row: TimelineRow): { p: unknown } | null {
  try {
    const base = {
      blue: c.blue,
      red: c.red,
      league: c.league,
      state: {
        minute: Math.min(60, Math.max(3, row.minute)),
        golddiff: row.golddiff,
        csdiff: row.csdiff ?? 0,
        blueKills: row.blue_kills,
        redKills: row.red_kills,
        goldTotal: row.blue_gold,
      },
      blueChamps: c.ln ? c.ln[0] : null,
      redChamps: c.ln ? c.ln[1] : null,
    };
    // 有赛前口径: 带赛前特征 (和头条一致) + 从 BP 后渐变; 没有但有 C: 不带赛前特征, 从 C 渐变;
    // 都没有: 局内模型自己的数。三种都同头条
    const res = ingameResponse(
      m.ingame,
      m.s1,
      teams,
      m.ex,
      c.preLeague !== null
        ? { ...base, preCtx: c.preCtx, blend: true, blendWith: c.prior, preLeague: c.preLeague }
        : c.xr !== null
          ? { ...base, includePregame: false, blend: true, blendWith: c.xr }
          : { ...base, includePregame: false },
      today,
    );
    return { p: res.probability_blue };
  } catch (e) {
    console.warn(`曲线点 ${row.minute}min 失败: ${e}`);
    return null;
  }
}

/**
 * 逐秒走势 (静态站的走势图用, 见 feed.fineRows): 每一行配上胜率, 和看板曲线同一套 (curveCtx / curveProb)。
 * 看板自己的 timeline 不动 —— 它是 test:diff 拿去和 Python 逐字段比的东西。
 * 每补完一批交一次。胜率按 (帧, 分钟数) 记在 memo 里: 传同一个 memo 反复重算, 只有新的点真算 ——
 * 键里带分钟数, 是因为暂停区间事后才认出来时, 同一帧的局内时间会被修正, 那就得重算。
 * blue / red / league / preLeague 的取法和看板曲线一样, 见 boardPlan。
 */
export async function fineTimeline(
  feed: Feed,
  teams: TeamsFile,
  models: Models,
  today: string,
  a: {
    gameId: string;
    blue: string;
    red: string;
    league: string;
    preLeague: string | null;
    upto: number;
    prior: number | null;
    /** 跨赛区模型 C 的赛前数 (看板响应的 xregion.probability_blue, 经 boardPlan); 没有就 null */
    xr?: number | null;
  },
  onBatch: (pts: Json[]) => void,
  memo: Map<string, { p: unknown } | null> = new Map(),
): Promise<void> {
  const c = await curveCtx(
    feed,
    teams,
    models.s1,
    a.blue,
    a.red,
    a.league,
    a.preLeague,
    today,
    a.gameId,
    a.prior,
    a.xr ?? null,
  );
  await feed.fineRows(a.gameId, a.upto, (rows) => {
    let lastP: unknown = null;
    onBatch(
      rows.map((row) => {
        if (row.minute >= 3) {
          const key = `${row.t}|${row.minute}`;
          if (!memo.has(key)) memo.set(key, curveProb(models, teams, today, c, row));
          const got = memo.get(key);
          if (got) lastP = got.p;
        }
        return {
          minute: row.minute,
          golddiff: row.golddiff,
          blue_kills: row.blue_kills,
          red_kills: row.red_kills,
          probability_blue: lastP,
        };
      }),
    );
  });
}

async function lineups(feed: Feed, gameId: string): Promise<[string[], string[]] | null> {
  const meta = await feed.gameMetadata(gameId);
  const bl: string[] = [];
  const rd: string[] = [];
  for (const v of meta.values()) {
    if (v.side === "blue" && v.champion) bl.push(v.champion);
    if (v.side === "red" && v.champion) rd.push(v.champion);
  }
  return bl.length === 5 && rd.length === 5 ? [bl, rd] : null;
}

const ROLE_OE = new Map([
  ["jungle", "jng"],
  ["bottom", "bot"],
  ["support", "sup"],
]);

/**
 * api._board_draft 的纯函数部分: 元数据里的十个人 → {side: {role: {player, champion}}}。
 * 选手名去掉战队简称前缀 (oePlayerName); 英雄名原样, 查表时再按 championKey 对齐。
 * 凑不齐十个人返回 null。黄金测试直接调它, 所以和取数分开。
 */
export function draftFromMeta(players: Iterable<MetaEntry>, codes: (string | null | undefined)[]): Draft | null {
  const draft: Draft = { blue: {}, red: {} };
  for (const mm of players) {
    const role = mm.role == null ? mm.role : (ROLE_OE.get(mm.role) ?? mm.role);
    const nm = oePlayerName(mm.summoner_name, codes);
    const ch = mm.champion;
    const side = mm.side;
    if (role && ch && nm && (side === "blue" || side === "red")) draft[side][role] = { player: nm, champion: ch };
  }
  return Object.keys(draft.blue).length + Object.keys(draft.red).length === 10 ? draft : null;
}

/** api._board_draft */
async function boardDraft(feed: Feed, gameId: string, minfo: Match): Promise<Draft | null> {
  return draftFromMeta(
    (await feed.gameMetadata(gameId)).values(),
    minfo.teams.map((t) => t.code),
  );
}

// ══════════════════════════════════════════════════════════
//  跨赛区模型 C (xregion.ts) 在看板上的那一段 —— api._xregion_series / _xregion_frames / _xregion_board
//
//  只在"非四大赛区、pregame_league 为 null (跨赛区, 或有队不在四大)"的看板上用; 国内看板和同母赛区的国际赛
//  看板根本不进来 (连上游请求都不多发)。闸门只认赛事内在线更新的版本, 所以必须重放本赛事 OE 还没收录的已完成局
//  —— 赛程取不到就不给 C, 不退回"只用 OE"的那个没过闸的版本。重放规则全在 xregion.ts (纯函数), 这里只取数。
// ══════════════════════════════════════════════════════════

/**
 * api._xregion_series: 本赛事 (同一个 lolesports tournament) 已完赛的系列赛 + 当前系列赛 → boardPredict 的输入。
 * 赛程取不到返回 null。别的系列赛的队名 / 比分取 getCompletedEvents; 当前系列赛的队序和比分取 getEventDetails
 * (20 秒缓存, 和页面顶上的比分同一份), 队名取赛程那一份 (minfo) —— getEventDetails 有时把队名写成全大写,
 * 别名表是按赛程的写法收的。
 */
async function xregionSeries(feed: Feed, teams: TeamsFile, minfo: Match, xknown: string[]): Promise<Series[] | null> {
  const tid = (await feed.eventTournament(minfo.match_id)) || (await feed.leagueTournament(minfo.league, minfo.start_time));
  if (!tid) return null;
  const evs = await feed.completedEvents(tid);
  const det = await feed.matchDetail(minfo.match_id);
  if (evs === null || det === null || det.teams.length !== 2 || !minfo.start_time) return null;
  const series: Series[] = [];
  for (const e of evs) {
    const m = e && typeof e === "object" && e.match && typeof e.match === "object" ? e.match : {};
    const ts: Json[] = Array.isArray(m.teams) ? m.teams : [];
    if (!m.id || String(m.id) === String(minfo.match_id) || ts.length !== 2 || !e.startTime) continue;
    series.push({
      match_id: String(m.id),
      start: e.startTime,
      best_of: (m.strategy && typeof m.strategy === "object" ? m.strategy.count : null) ?? null,
      teams: ts.map((t: Json) => ({
        name: t.name ?? null,
        code: t.code ?? null,
        oe: mapTeam(teams, t.name, xknown),
        wins: Math.trunc(Number((t.result && typeof t.result === "object" ? t.result.gameWins : 0) || 0)),
        id: null,
      })),
    });
  }
  const names = new Map<string, string>();
  for (const t of minfo.teams) for (const k of [t.code, t.api_name]) if (k) names.set(norm(k), t.api_name);
  const cur = det.teams.map((t) => {
    const nm = names.get(norm(t.name)) || names.get(norm(t.code)) || t.name;
    return { name: nm, code: t.code, oe: mapTeam(teams, nm, xknown), wins: Math.trunc(Number(t.wins || 0)), id: t.id };
  });
  series.push({ match_id: String(minfo.match_id), start: minfo.start_time, best_of: minfo.best_of, teams: cur });
  return series;
}

/**
 * api._xregion_frames: 把 replayFramesNeeded 列出的终局帧取好, 填进 series[*].frames["<局号>"]。
 * 别的系列赛先取一次局号 → gameId、队伍 id (feed.matchIds: 照常的 20 秒缓存, id 齐了另记一份, 不把共用的
 * getEventDetails 按一小时缓存); 当前系列赛用 matchDetail。任何一步取不到就记 null —— 那个系列赛在
 * decodeSeries 里整体退回交替序, 不让一个请求失败拖掉整块看板的 C。
 */
async function xregionFrames(feed: Feed, series: Series[], need: [string, number][], currentId: string): Promise<void> {
  const by = new Map(series.map((s) => [s.match_id, s]));
  const gids = new Map<string, Map<number, string>>();
  for (const [mid, i] of need) {
    const s = by.get(mid)!;
    if (!gids.has(mid)) {
      gids.set(mid, new Map());
      let det: { teams: { id: string | null; name: string | null; code: string | null }[]; games: { number: number | null; id: string | null }[] } | null = null;
      try {
        det = mid === currentId ? await feed.matchDetail(mid) : await feed.matchIds(mid);
      } catch (e) {
        warn(`跨赛区模型取 ${mid} 的对局详情失败, 这个系列赛按交替序`, e);
        det = null;
      }
      if (det) {
        for (const t of s.teams) {
          if (t.id) continue;
          const hit = det.teams.find(
            (d) => norm(d.name) === norm(t.name) || (!!d.code && norm(d.code) === norm(t.code)),
          );
          t.id = hit ? hit.id : null;
        }
        const g = new Map<number, string>();
        for (const x of det.games) if (x.number && x.id) g.set(Math.trunc(Number(x.number)), x.id);
        gids.set(mid, g);
      }
    }
    let fr = null;
    const gid = gids.get(mid)!.get(i);
    if (gid) {
      try {
        fr = await feed.finalFrame(gid);
      } catch (e) {
        warn(`跨赛区模型取 ${gid} 的终局帧失败, 这个系列赛按交替序`, e);
      }
    }
    (s.frames ??= {})[String(i)] = fr;
  }
}

/**
 * api._xregion_board: 这块看板上 C 的赛前数, 或 null (= 保持原来的行为和原来的每一句话)。
 * 调用方已判断: 非四大赛区、pregame_league 为 null、两队。这里再要求: 状态已加载、这个赛区键有系数、两队都能
 * 映射成 C 状态里的队名、本赛事赛程取得到、两队母赛区都查得到且不同、当前系列赛的比分跟得上局号。局号: 选中的
 * 那一局; 还没有选中的局 (没开打) 就是比分之和 + 1。蓝红按 minfo.teams 的顺序 —— 有帧时已经按本局选边排过
 * (sidesKnown); 没帧时那只是赛程顺序, 给两种选边的平均 (boardPredict)。
 */
async function xregionBoard(
  feed: Feed,
  teams: TeamsFile,
  state: XregionState | null | undefined,
  minfo: Match,
  chosen: Json,
  sidesKnown: boolean,
): Promise<{ p: number; why: string; public: XregionPublic } | null> {
  if (!state) return null;
  const key = String(minfo.league).toUpperCase();
  if (coefficientsFor(state, key) === null) return null;
  const xknown = xregionKnown(state);
  const blue = mapTeam(teams, minfo.teams[0]!.api_name, xknown);
  const red = mapTeam(teams, minfo.teams[1]!.api_name, xknown);
  if (!blue || !red) return null;
  let res;
  let number: number;
  try {
    const series = await xregionSeries(feed, teams, minfo, xknown);
    if (series === null) {
      console.warn(`跨赛区模型: 取不到 ${minfo.match_id} 所在赛事的赛程, 这场不给 C`);
      return null;
    }
    const cur = series[series.length - 1]!;
    number =
      chosen && chosen.number ? Math.trunc(Number(chosen.number)) : seriesWins(cur).reduce((a, b) => a + b, 0) + 1;
    const al = teams.aliases;
    await xregionFrames(feed, series, replayFramesNeeded(state, series, minfo.match_id, number, al), minfo.match_id);
    res = boardPredict(state, key, blue, red, series, minfo.match_id, number, al, sidesKnown);
  } catch (e) {
    warn("跨赛区模型重放失败, 这场不给 C", e);
    return null;
  }
  if (res.withheld === "score_lag") {
    console.warn(`跨赛区模型: ${minfo.match_id} 第 ${number} 局开打了, 比分还没算上前一局, 这次不给 C`);
  }
  const p = res.probability_blue;
  if (p === null) return null;
  return {
    p,
    why: xregionNote(res.no_history),
    public: {
      probability_blue: p,
      game_number: number,
      blue,
      red,
      home_blue: res.home_blue,
      home_red: res.home_red,
      no_history: res.no_history,
      side_neutral: res.side_neutral,
      replayed: res.replayed,
      updated: res.updated,
      deduped: res.deduped,
      oe_asof: state.oe_asof,
    },
  };
}

/** GET /esports/board/{match_id} */
export async function buildBoard(
  feed: Feed,
  teams: TeamsFile,
  models: Models,
  today: string,
  matchId: string,
  gameId: string | null = null,
  curves = true,
  points = 60,
): Promise<Record<string, unknown>> {
  const { ingame, s1, ex } = models;
  let games: Json[];
  try {
    games = await feed.games(matchId);
  } catch (e) {
    throw new HttpError(502, `对局详情拉取失败: ${e instanceof Error ? e.message : String(e)}`);
  }

  // 这场比赛的基本信息 —— 从直播列表或赛程里找
  // 每一处上游请求各自 try (同 api.py): 一个赛区的赛程取不到只跳过它自己, 不让排在前面的一个
  // 国际赛把后面赛区的比赛一起"找不到"
  let minfo: Match | null = null;
  try {
    minfo = (await feed.live()).find((m) => m.match_id === matchId) ?? null;
  } catch (e) {
    warn("查找比赛信息失败", e);
  }
  if (!minfo) {
    for (const [lg] of LEAGUE_IDS) {
      let ms: Match[];
      try {
        ms = await feed.schedule(lg, knownFor(teams, lg));
      } catch (e) {
        warn("查找比赛信息失败", e);
        continue;
      }
      minfo = ms.find((m) => m.match_id === matchId) ?? null;
      if (minfo) break;
    }
  }
  if (minfo) {
    const known = knownFor(teams, minfo.league);
    for (const t of minfo.teams) t.model_name = mapTeam(teams, t.api_name, known);
    await refreshWins(minfo, feed);
  }

  const framed = new Map<string, LiveState>();
  for (const [g, st] of await feed.probeGames(matchId)) if (st) framed.set(g.id, st);
  const playable = playableGames(games, framed);
  let chosen: Json = null;
  let st: LiveState | null = null;
  if (gameId) {
    chosen = games.find((g) => g.id === gameId) ?? null;
    if (chosen) st = await feed.window(chosen.id);
  } else {
    const cur = await feed.currentGame(matchId);
    if (cur) [chosen, st] = cur;
    else if (playable.length) {
      chosen = playable[playable.length - 1];
      st = framed.get(chosen.id) ?? (await feed.window(chosen.id));
    }
  }
  // 选定之后再取一次, 这回要扣掉暂停的精确分钟数
  if (chosen && st !== null) st = (await feed.window(chosen.id, { preciseMinute: true })) ?? st;

  // 按本局选边给队伍排序: [0] 永远是蓝方 —— 以帧为准, 见 api.esports_board
  let sideNote: string | null = null;
  let sideWarn: string | null = null;
  if (minfo && chosen && st !== null) {
    const byId = new Map<string, string>();
    try {
      const ev = await feed.get(`${PERSISTED}/getEventDetails?hl=en-US&id=${matchId}`, feed.liveTtl);
      for (const t of ((ev?.data?.event || {}).match || {}).teams || []) if (t.id) byId.set(String(t.id), t.name ?? null);
    } catch (e) {
      sideWarn = warn("取队名失败, 无法确认本局选边", e);
    }
    let fs: { blue?: string; red?: string } = {};
    try {
      fs = await feed.frameSides(st.game_id);
    } catch (e) {
      sideWarn = sideWarn || warn("取帧选边失败", e);
    }
    const bname = byId.get(fs.blue ?? "") ?? null;
    if (bname && minfo.teams.length === 2) {
      // 归一化比对: getEventDetails 全大写, schedule 不是
      const want = norm(bname);
      const cur = minfo.teams.map((t) => norm(t.api_name));
      if (cur[0] !== want && cur[1] === want) {
        minfo.teams.reverse();
        sideNote = "按本局选边调整了左右";
      } else if (cur[0] !== want && cur[1] !== want) {
        sideWarn = sideWarn || `本局蓝方是 ${bname}, 但和赛程里的队名都对不上 —— 左右未调整, 可能是反的`;
      }
      const sd = sidesOf(chosen);
      if (sd.blue && fs.blue && String(sd.blue) !== fs.blue) {
        const pname = byId.get(String(sd.blue)) || sd.blue;
        sideWarn =
          sideWarn ||
          `两个官方源对本局选边说法不一: 实时帧说蓝方是 ${bname}, ` +
            `赛程接口说是 ${pname}。页面上的数字全部按实时帧分组, ` +
            `因此以帧为准; 若转播画面显示的是另一边, 以转播为准。`;
      }
    } else if (fs.blue && !bname) {
      sideWarn = sideWarn || "拿不到本局的队伍对照, 左右可能不准";
    } else if (!fs.blue) {
      sideWarn = sideWarn || "这一局的帧里没有选边信息, 左右可能不准";
    }
  } else if (minfo && chosen) {
    sideWarn = "本局还没有实时帧, 无法确认选边 —— 左右按赛程顺序排, 不代表蓝红";
  }

  const ver = await feed.ddragonVersion();
  const playableSet = new Set(playable);
  const out: Record<string, unknown> = {
    match_id: matchId,
    match: minfo ? matchJson(teams, minfo) : null,
    side_note: sideNote,
    side_warning: sideWarn,
    games: games.map((g) => ({
      number: g.number ?? null,
      game_id: g.id ?? null,
      persisted_state: g.state ?? null,
      playable: playableSet.has(g),
      sides: sidesOf(g),
    })),
    selected_game_id: chosen ? (chosen.id ?? null) : null,
    ddragon_version: ver,
    live: null,
    prediction: null,
    timeline: [],
    note: "以帧里的 gameState 为准, persisted_state 会滞后。",
  };
  // 跨赛区模型 C (见 xregionBoard): 只看非四大赛区、pregame_league 为 null 的两队对阵。算得出来才多一个
  // "xregion" 键; 国内和同母赛区的看板根本不进这个分支 (也不多发任何上游请求)。
  let xr: { p: number; why: string; public: XregionPublic } | null = null;
  if (minfo && minfo.teams.length === 2 && !isMajor(minfo.league) && pregameOf(teams, minfo) === null) {
    // 有帧 = 左右已按本局的帧排成蓝 / 红; 没帧时只是赛程顺序 (C 改给两种选边的平均)
    xr = await xregionBoard(feed, teams, models.xr, minfo, chosen, !!(chosen && st));
    if (xr) out.xregion = xr.public;
  }
  if (!chosen || !st) {
    // 还没有帧: 原来这里不给 prediction (界面自己拿 Stage 1 问赛前); C 的数只有看板算得出来
    if (xr) out.prediction = xregionPregame(xr.p, xr.why);
    return out;
  }

  const live: Record<string, unknown> = {
    game_id: st.game_id,
    game_number: chosen.number ?? null,
    game_state: st.game_state,
    minute: st.minute,
    frame_time: st.frame_time,
    state: toState(st),
    in_progress: st.live,
    teams: st.teams,
    stale_seconds: st.stale_seconds,
    paused_seconds: pyRound(st.paused_seconds, 0),
    stalled: st.stalled,
  };
  if (st.stalled) {
    live.note =
      `这一局的数据流已经 ${Math.floor(st.stale_seconds / 60)} 分钟没有更新了 —— ` +
      `帧里还写着 ${st.game_state}, 但它停在 ${st.frame_time}, 不是实时战况。` +
      (chosen.state === "completed" ? "赛程也已经把这一局标为结束。" : "");
  }
  out.live = live;
  try {
    const ps = await playersJson(feed, st.game_id, ver);
    live.players = ps;
    live.lanes = lanesOf(ps);
  } catch (e) {
    live.players = [];
    live.lanes = [];
    console.warn(`players() 失败: ${e}`);
  }

  let boardP2: number | null = null; // BP 后概率; 曲线开局那段的渐变也要用 (ingame.blendPrior)
  const MIN_MINUTE = 3;
  // 赛前 / BP 后能不能给、按哪个赛区算, 局内模型拿什么当队名 —— 见 boardPlan / pregameLeague。
  // plan 为 null = 不是两队, 什么都不算。plan.preLeague 为 null = 不算 Stage 1/2、不渐变、局内模型
  // 不带赛前特征; 队名映射不出来时 blue/red 是 lolesports 原名, 只当标签, 不进任何查表。
  // xr: 跨赛区模型 C 的赛前数 (上面算过), 只可能出现在 preLeague 为 null 的看板上 —— 有它时赛前 = C、开局
  // 3-15 分钟以 C 为锚渐变, 局内模型照旧不带赛前特征; 没有 BP 后 (候选 D 没过闸)
  const plan = minfo
    ? boardPlan(
        minfo.teams.map((t) => ({ api: t.api_name, model: t.model_name })),
        minfo.league,
        pregameOf(teams, minfo),
        xr ? { p: xr.p, noHistory: xr.public.no_history } : null,
      )
    : null;
  if (st.minute === null || st.minute < MIN_MINUTE) {
    // 局内模型给不了, 赛前和 BP 后这两段一直有效 —— 照给 (有赛前口径时; 跨赛区有 C 时赛前就是 C 的数)
    let p1: number | null = plan ? plan.xr : null;
    let p2: number | null = null;
    if (minfo && plan && plan.preLeague !== null) {
      try {
        p1 = preContext(s1, teams, plan.blue, plan.red, plan.preLeague, today).preP;
      } catch {
        /* 队伍不认识就没有赛前概率 */
      }
      try {
        const draft = await boardDraft(feed, st.game_id, minfo);
        if (draft) p2 = predictCore(models.s2, teams, plan.blue, plan.red, plan.preLeague, draft, today);
      } catch {
        /* 和服务器一样: BP 后算不出来就只给赛前 */
      }
    }
    boardP2 = p2;
    out.prediction = earlyPrediction(st.minute, p1, p2, plan ? plan.whyKind : null, plan ? plan.why : null);
  } else if (minfo && plan) {
    try {
      if (st.minute > 60) throw new Error(`局内分钟数 ${st.minute} 超出局内模型的输入范围 (3-60)`);
      let ln: [string[], string[]] | null = null;
      try {
        ln = await lineups(feed, st.game_id);
      } catch (e) {
        warn("头条取阵容失败, 本次预测不含阵容强势期", e);
      }
      // BP 后 (第二段) 的概率, 同 api.py —— 先算: 头条开局那段要从它渐变过渡到局内模型。
      // 凑不齐十个人就不给这条线, 不猜 (渐变就退回赛前概率)。没有赛前口径就不算
      let p2: number | null = null;
      if (plan.preLeague !== null) {
        try {
          const draft = await boardDraft(feed, st.game_id, minfo);
          if (draft) p2 = predictCore(models.s2, teams, plan.blue, plan.red, plan.preLeague, draft, today);
        } catch (e) {
          console.warn(`Stage2 参考线跳过: ${e instanceof Error ? e.message : String(e)}`);
        }
      }
      boardP2 = p2;
      const base = {
        blue: plan.blue,
        red: plan.red,
        league: minfo.league,
        state: {
          minute: st.minute,
          golddiff: st.golddiff,
          csdiff: st.csdiff,
          blueKills: st.blue_kills,
          redKills: st.red_kills,
          goldTotal: st.gold_total,
        },
        blueChamps: ln ? ln[0] : null,
        redChamps: ln ? ln[1] : null,
      };
      // 没有赛前口径 (跨赛区 / 队名不认识): 不带赛前特征, 第 3 分钟起就是局内模型自己的数 —— 有 C 时开局
      // 3-15 分钟从 C 的数渐变过渡 (采纳理由同国内的 BP 后锚: 不显著变差 + 连续; 跨赛区历史局上对硬切
      // 系列赛 t +2.27, 方向是变好, 第 3 分钟的跳变 18.6 → 0.11 个百分点)。
      // includePregame=false 是显式的 —— 不能让 ingameResponse 退回按比赛赛区去算 preContext
      const pred = ingameResponse(
        ingame,
        s1,
        teams,
        ex,
        plan.preLeague !== null
          ? { ...base, blend: true, blendWith: p2, preLeague: plan.preLeague }
          : plan.xr !== null
            ? { ...base, includePregame: false, blend: true, blendWith: plan.xr }
            : { ...base, includePregame: false },
        today,
      ) as Record<string, unknown>;
      if (plan.preLeague === null && plan.xr !== null) {
        pred.pregame_probability_blue = pyRound(plan.xr, 4);
        pred.pregame_source = "xregion";
      }
      // 原因放第一条; 故意不带赛前特征时去掉局内模型那句"没有可用的赛前队伍统计"
      pred.warnings = boardWarnings(pred.warnings as string[], plan.whyKind, plan.why);
      if (p2 !== null) pred.postdraft_probability_blue = p2;
      out.prediction = pred;
    } catch (e) {
      out.prediction = { error: e instanceof Error ? e.message : String(e) };
    }
  }

  // 曲线。条件和头条的局内段一样 (两队齐了): 队名映射不出来也画局内模型的线, 赛前口径同头条
  if (curves && plan) {
    try {
      const upto = st.frame_time ? parseTs19(st.frame_time) : NaN;
      const rows = Feed.downsample(await feed.goldTimeline(st.game_id, Number.isNaN(upto) ? undefined : upto), points);
      const c = await curveCtx(
        feed,
        teams,
        s1,
        plan.blue,
        plan.red,
        plan.league,
        plan.preLeague,
        today,
        st.game_id,
        boardP2,
        plan.xr,
      );
      const series: Json[] = [];
      let lastP: unknown = null;
      for (const row of rows) {
        const pt: Record<string, unknown> = {
          minute: row.minute,
          golddiff: row.golddiff,
          blue_kills: row.blue_kills,
          red_kills: row.red_kills,
        };
        if (row.minute >= 3) {
          const got = curveProb(models, teams, today, c, row);
          if (got) lastP = got.p;
        }
        pt.probability_blue = lastP;
        series.push(pt);
      }
      out.timeline = series;
    } catch (e) {
      console.warn(`timeline 失败: ${e}`);
    }
  }
  return out;
}
