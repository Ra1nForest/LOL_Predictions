/**
 * 看板和比赛列表 —— api.py 里 /esports/live、/esports/upcoming、/esports/board 的浏览器版。
 *
 * 输出和服务器逐字段同形, 前端组件一行不用改。唯一有意为之的差别:
 * **不算 BP 后 (Stage 2) 那条参考线** —— 它要带上一万多行的选手-英雄表, 而静态站只做
 * 赛程 + 赛前预测 + 局内看板。postdraft_probability_blue 因此恒为 null。
 *
 * 正确性由 scripts/diff-board.mjs 保证: 对已打完的比赛和本机 API 逐字段比对。
 */
import type { ExplainFile } from "./explain.ts";
import type { IngameModel } from "./ingame.ts";
import { ingameResponse } from "./ingame.ts";
import type { PreCtx, Stage1 } from "./stage1.ts";
import { preContext } from "./stage1.ts";
import type { TeamsFile } from "./teams.ts";
import { knownTeams, mapTeam, norm } from "./teams.ts";
import type { LiveState, Match } from "./feed.ts";
import { DDRAGON, Feed, LEAGUE_IDS, PERSISTED, parseTs19, predictable, toState } from "./feed.ts";
import { pyRound } from "./pyfmt.ts";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Json = any;

export interface Models {
  ingame: IngameModel;
  s1: Stage1;
  ex: ExplainFile;
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

/** api._match_json */
function matchJson(m: Match): Record<string, unknown> {
  return {
    match_id: m.match_id,
    league: m.league,
    start_time: m.start_time,
    state: m.state,
    best_of: m.best_of,
    predictable: predictable(m),
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
      for (const m of await feed.schedule(lg, knownTeams(teams, lg))) {
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
    const known = knownTeams(teams, m.league);
    for (const t of m.teams) t.model_name = mapTeam(teams, t.api_name, known);
    await refreshWins(m, feed);
    // 比分刷新后重判: 已经打完的系列赛不能赖在直播列表里
    if (m.between_games && clinched(m)) continue;
    out.push({ ...matchJson(m), between_games: !!m.between_games });
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
      for (const m of await feed.schedule(lg, knownTeams(teams, lg))) {
        if (!m.start_time) continue;
        const ts = parseTs19(m.start_time);
        if (Number.isNaN(ts)) continue;
        if (ts < now - pastHours * 3600_000) continue;
        if (m.teams.some((t) => (t.game_wins || 0) > 0)) continue;
        out.push(matchJson(m));
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
  let minfo: Match | null = null;
  try {
    minfo = (await feed.live()).find((m) => m.match_id === matchId) ?? null;
    if (!minfo) {
      for (const [lg] of LEAGUE_IDS) {
        minfo = (await feed.schedule(lg, knownTeams(teams, lg))).find((m) => m.match_id === matchId) ?? null;
        if (minfo) break;
      }
    }
  } catch (e) {
    warn("查找比赛信息失败", e);
  }
  if (minfo) {
    const known = knownTeams(teams, minfo.league);
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
    match: minfo ? matchJson(minfo) : null,
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
  if (!chosen || !st) return out;

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
    const alias: Record<string, string> = { jng: "jungle", bot: "bottom", sup: "support" };
    const lanes: Json[] = [];
    for (const lane of ["top", "jungle", "mid", "bottom", "support"]) {
      const pick = (side: string) => ps.find((p) => p.side === side && (alias[p.role] ?? p.role) === lane);
      const b = pick("blue");
      const r = pick("red");
      if (b && r) lanes.push({ lane, gold_diff: b.gold - r.gold, blue_gold: b.gold, red_gold: r.gold });
    }
    live.lanes = lanes;
  } catch (e) {
    live.players = [];
    live.lanes = [];
    console.warn(`players() 失败: ${e}`);
  }

  const MIN_MINUTE = 3;
  if (st.minute === null || st.minute < MIN_MINUTE) {
    // 局内模型给不了, 赛前那一段一直有效 —— 照给
    let p1: number | null = null;
    const p2: number | null = null; // BP 后 (Stage 2) 浏览器版不做, 见文件头
    if (minfo && predictable(minfo)) {
      try {
        p1 = preContext(s1, teams, minfo.teams[0]!.model_name, minfo.teams[1]!.model_name, minfo.league, today).preP;
      } catch {
        /* 队伍不认识就没有赛前概率 */
      }
    }
    out.prediction = {
      too_early: true,
      minute: st.minute,
      pregame_probability_blue: p1,
      postdraft_probability_blue: p2,
      probability_blue: p2 ?? p1,
      source: p2 !== null ? "post_draft" : "pre_draft",
      note:
        (st.minute !== null ? `开局第 ${st.minute} 分钟, ` : "刚开局 (局内时间还没同步出来), ") +
        "局内模型最早在第 10 分钟的切片上训练过。这里显示的是" +
        (p2 !== null ? "BP 后" : "赛前") +
        "的概率 —— 它不看局内数据, 但一直有效。",
    };
  } else if (minfo && predictable(minfo)) {
    try {
      if (st.minute > 60) throw new Error(`局内分钟数 ${st.minute} 超出局内模型的输入范围 (3-60)`);
      let ln: [string[], string[]] | null = null;
      try {
        ln = await lineups(feed, st.game_id);
      } catch (e) {
        warn("头条取阵容失败, 本次预测不含阵容强势期", e);
      }
      out.prediction = ingameResponse(
        ingame,
        s1,
        teams,
        ex,
        {
          blue: minfo.teams[0]!.model_name!,
          red: minfo.teams[1]!.model_name!,
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
        },
        today,
      );
    } catch (e) {
      out.prediction = { error: e instanceof Error ? e.message : String(e) };
    }
  }

  // 曲线
  if (curves && minfo && predictable(minfo)) {
    try {
      const upto = st.frame_time ? parseTs19(st.frame_time) : NaN;
      const rows = Feed.downsample(await feed.goldTimeline(st.game_id, Number.isNaN(upto) ? undefined : upto), points);
      const b = minfo.teams[0]!.model_name!;
      const r = minfo.teams[1]!.model_name!;
      // 赛前特征只和队伍有关, 整条曲线算一次
      let preCtx: PreCtx | null = null;
      try {
        preCtx = preContext(s1, teams, b, r, minfo.league, today);
      } catch (e) {
        warn("曲线取赛前特征失败, 曲线会偏离头条", e);
      }
      let ln: [string[], string[]] | null = null;
      try {
        ln = await lineups(feed, st.game_id);
      } catch (e) {
        warn("曲线取阵容失败, 曲线会偏离头条", e);
      }
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
          try {
            const res = ingameResponse(
              ingame,
              s1,
              teams,
              ex,
              {
                blue: b,
                red: r,
                league: minfo.league,
                state: {
                  minute: Math.min(60, Math.max(3, row.minute)),
                  golddiff: row.golddiff,
                  csdiff: row.csdiff ?? 0,
                  blueKills: row.blue_kills,
                  redKills: row.red_kills,
                  goldTotal: row.blue_gold,
                },
                blueChamps: ln ? ln[0] : null,
                redChamps: ln ? ln[1] : null,
                preCtx,
              },
              today,
            );
            lastP = res.probability_blue;
          } catch (e) {
            console.warn(`曲线点 ${row.minute}min 失败: ${e}`);
          }
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
