/**
 * lolesports 数据接入 —— esports_feed.py 的浏览器版。
 * ==================================================
 * 逐函数移植, 连同那些踩过的坑 (完整记录见 esports_feed.py 模块顶部, 这里不再重复):
 *   · startingTime 必须对齐 10 秒再减 lag, 否则 204
 *   · 帧里的 gameState 才是权威, persisted 的 state 两个方向都会错
 *   · 不带 startingTime 拿到的是开局帧, 只能用来定开局时刻
 *   · 赛程的 startTime 是转播开始, 不是开局
 *   · 结束后的比赛, 任何更晚的窗口都返回同一批冻结帧
 *
 * 和服务器版的区别只有运行环境: 浏览器直连上游 (两个主机都回 Access-Control-Allow-Origin: *,
 * 2026-09-12 实测), 流量摊在每个访客自己的 IP 上; 并发用 Promise 池代替线程池;
 * 缓存按过期时间定期清理 —— 浏览器标签页一挂几个小时, 不清会越攒越多。
 *
 * 正确性由 scripts/diff-board.mjs 保证: 对已经打完的比赛, 本机 API 和这里拉同一批
 * 上游数据, 整个看板响应逐字段比对。
 */
import { pyRound } from "./pyfmt.ts";
import type { TeamsFile } from "./teams.ts";
import { mapTeam } from "./teams.ts";

export const PERSISTED = "https://esports-api.lolesports.com/persisted/gw";
export const FEED = "https://feed.lolesports.com/livestats/v1";
/** 公开的共享 key, lolesports.com 前端自己就用这个。不是机密。 */
export const API_KEY = "0TvQnueqKa5mxJntVWt0w4LpLfEkrV1Ta8rQBb9Z";
/** 有序 —— 和 Python 的 dict 顺序一致, 列表里比赛的先后取决于它 */
export const LEAGUE_IDS: [string, string][] = [
  ["LCK", "98767991310872058"],
  ["LPL", "98767991314006698"],
  ["LEC", "98767991302996019"],
  ["LCS", "98767991299243165"],
];
export const DDRAGON = "https://ddragon.leagueoflegends.com";
const DDRAGON_FALLBACK_VER = "16.16.1";

const LAG_STEPS = [20, 30, 40, 50, 60, 90, 120, 150, 210, 300];
const FALLBACK_LAGS = [60, 90, 120, 150, 210, 300];
const START_GOLD = 2500;
const FIRST_INCOME_SEC = 33;
// 见 esports_feed.CALIB_SPAN_SEC / START_RETRY_SEC: 那批窗口定型前, 校准失败只说明"还没发布"
const CALIB_SPAN_SEC = 180;
const START_RETRY_SEC = 30;
const FREEZE_SLACK = 5;
const PLAY_GOLD_BASE = 1000;
const PLAY_GOLD_PER_MIN = 800;
export const LIVE_STALE_SEC = 600;
const OBS_SETTLE_SEC = 180;
const POOL = 8;

/** 小符文碎片 —— 不在 runesReforged.json 里, 见 esports_feed.STAT_SHARDS */
const STAT_SHARDS: Record<number, string> = {
  5001: "成长生命值", 5002: "护甲", 5003: "魔法抗性",
  5005: "攻击速度", 5007: "技能急速", 5008: "适应之力",
  5010: "移动速度", 5011: "生命值", 5012: "抗性成长",
  5013: "韧性和减速抗性",
};

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Json = any;

// ── 数据结构 ────────────────────────────────────────────────

export interface TeamRef {
  api_name: string;
  code: string;
  model_name: string | null;
  game_wins: number | null;
  image: string | null;
}

export interface Match {
  match_id: string;
  league: string;
  start_time: string;
  state: string;
  best_of: number | null;
  teams: TeamRef[];
  between_games: boolean;
}

export const predictable = (m: Match) => m.teams.length === 2 && m.teams.every((t) => t.model_name !== null);

export interface TeamObjectives {
  gold: number;
  kills: number;
  towers: number;
  inhibitors: number;
  barons: number;
  dragons: number;
  dragon_types: string[];
  cs: number;
}

export interface LiveState {
  game_id: string;
  game_state: string;
  minute: number | null;
  golddiff: number;
  csdiff: number;
  blue_kills: number;
  red_kills: number;
  gold_total: number;
  frame_time: string;
  live: boolean;
  teams: { blue: TeamObjectives; red: TeamObjectives };
  stale_seconds: number;
  paused_seconds: number;
  stalled: boolean;
}

/** LiveState.to_state —— xpdiff 永远是 None: 帧里没有 XP 字段, 0 是编造的 */
export const toState = (s: LiveState) => ({
  minute: s.minute,
  golddiff: s.golddiff,
  xpdiff: null,
  csdiff: s.csdiff,
  blue_kills: s.blue_kills,
  red_kills: s.red_kills,
  gold_total: s.gold_total,
});

export interface Player {
  participant_id: number;
  side: string;
  role: string | null;
  summoner_name: string | null;
  champion: string | null;
  level: number;
  kills: number;
  deaths: number;
  assists: number;
  cs: number;
  gold: number;
  kill_participation: number | null;
  damage_share: number | null;
  wards_placed: number | null;
  wards_destroyed: number | null;
  current_health: number | null;
  max_health: number | null;
  items: number[];
  trinket: number | null;
  stats: Record<string, number | null>;
  perks: Record<string, unknown>;
  abilities: string[];
}

export interface MetaEntry {
  summoner_name: string | null;
  champion: string | null;
  role: string | null;
  side: string;
}

export interface TimelineRow {
  t: string;
  minute: number;
  golddiff: number;
  blue_gold: number;
  red_gold: number;
  blue_kills: number;
  red_kills: number;
  csdiff: number;
  [k: string]: unknown;
}

type Obs = [tt: number, first: number, last: number, frozen: boolean, gold: number];

export class FeedError extends Error {
  constructor(msg: string) {
    super(msg);
    this.name = "FeedError";
  }
}

// ── 小工具 ──────────────────────────────────────────────────

/** Python 的 dict.get(k, default): 键在就给值 (哪怕是 null), 不在才给默认值 */
function pget(o: Json, k: string, d: Json): Json {
  return o && typeof o === "object" && Object.prototype.hasOwnProperty.call(o, k) ? o[k] : d;
}
/** `x or {}` */
const orObj = (x: Json): Json => (x && typeof x === "object" ? x : {});
/** `x or []` */
const orArr = (x: Json): Json[] => (Array.isArray(x) && x.length ? x : []);

/** _https: 队标 URL 升到 https, 否则 HTTPS 页面上会被当混合内容静默拦掉 */
function https(url: string | null | undefined): string | null {
  if (url && url.startsWith("http://")) return "https://" + url.slice("http://".length);
  return url ?? null;
}

/** 帧时间戳 → 毫秒。和 Python 一样只取前 19 位 (到秒), 丢掉毫秒 */
export function frameTs(frame: Json): number {
  return Date.parse(String(frame.rfc460Timestamp).slice(0, 19) + "Z");
}

/** "2026-09-12T07:28:39..." 这种字符串的前 19 位 → 毫秒; 解析不了是 NaN */
export function parseTs19(s: string): number {
  return Date.parse(s.slice(0, 19) + "Z");
}

async function pool<T, R>(items: T[], n: number, fn: (t: T) => Promise<R>): Promise<R[]> {
  const out = new Array<R>(items.length);
  let next = 0;
  const workers = Array.from({ length: Math.min(n, items.length) }, async () => {
    while (next < items.length) {
      const k = next++;
      out[k] = await fn(items[k]!);
    }
  });
  await Promise.all(workers);
  return out;
}

// ── 客户端 ──────────────────────────────────────────────────

export interface FeedOptions {
  teams: TeamsFile;
  fetchImpl?: typeof fetch;
  sleep?: (ms: number) => Promise<void>;
  /**
   * 符文名用 Data Dragon 的哪种语言。默认 zh_CN, 和 esports_feed.py 一样 ——
   * 差分测试拿看板整个响应和 Python 逐字段比, 这里不能随界面语言变; 只有网页在
   * 英文界面下才传 en_US (static-api.ts)。
   */
  runeLocale?: string;
}

export class Feed {
  scheduleTtl = 300;
  liveTtl = 20;
  windowTtl = 3;
  noFramesTtl = 30;
  minLag = 20;
  lagProbeSec = 20;
  retries = 3;
  timeoutMs = 15_000;

  private teams: TeamsFile;
  private fetchImpl: typeof fetch;
  private sleep: (ms: number) => Promise<void>;
  private runeLocale: string;
  private cache = new Map<string, { exp: number; val: Json }>();
  private inflight = new Map<string, Promise<Json>>();
  private writes = 0;
  private starts = new Map<string, number>();
  private startRetry = new Map<string, number>();
  private obs = new Map<string, Map<number, Obs>>();
  private meta = new Map<string, Map<number, MetaEntry>>();
  private noFrames = new Map<string, number>();
  private lag = new Map<string, number>();
  private lagProbe = new Map<string, number>();

  constructor(opts: FeedOptions) {
    this.teams = opts.teams;
    this.fetchImpl = opts.fetchImpl ?? ((...a) => fetch(...a));
    this.sleep = opts.sleep ?? ((ms) => new Promise((r) => setTimeout(r, ms)));
    this.runeLocale = opts.runeLocale ?? "zh_CN";
  }

  // -- 底层 --

  /**
   * _get: 带缓存的 GET。204 / 大多数 4xx = "这一刻没数据", 不是错误。
   * emptyTtl: 空结果用更短的缓存 —— 一次瞬时 404 不能焊死成一小时的缺口。
   * 同一个 URL 正在飞的请求直接复用, 不重复发。
   */
  async get(url: string, ttl: number, keyHdr = true, emptyTtl?: number): Promise<Json> {
    const now = Date.now();
    const hit = this.cache.get(url);
    if (hit && hit.exp > now) return hit.val;
    const flying = this.inflight.get(url);
    if (flying) return flying;
    const p = this.fetchRetry(url, keyHdr)
      .then((payload) => {
        const keep = payload !== null ? ttl : (emptyTtl ?? ttl);
        this.cache.set(url, { exp: now + keep * 1000, val: payload });
        if (++this.writes % 200 === 0) this.prune();
        return payload;
      })
      .finally(() => this.inflight.delete(url));
    this.inflight.set(url, p);
    return p;
  }

  private prune() {
    const now = Date.now();
    for (const [k, v] of this.cache) if (v.exp <= now) this.cache.delete(k);
  }

  private async fetchRetry(url: string, keyHdr: boolean): Promise<Json> {
    const headers: Record<string, string> = { Accept: "application/json" };
    if (keyHdr) headers["x-api-key"] = API_KEY;
    let last = "";
    for (let attempt = 0; attempt < this.retries; attempt++) {
      const ctl = new AbortController();
      const timer = setTimeout(() => ctl.abort(), this.timeoutMs);
      try {
        const r = await this.fetchImpl(url, { headers, signal: ctl.signal });
        if (r.status === 204) return null;
        if (r.status === 200) return await r.json();
        if (r.status >= 400 && r.status < 500 && r.status !== 403 && r.status !== 429) return null;
        last = `HTTP ${r.status}`; // 观测到过瞬时 403, 退避后重试
      } catch (e) {
        last = e instanceof Error ? e.message : String(e);
      } finally {
        clearTimeout(timer);
      }
      await this.sleep(1500 * (attempt + 1));
    }
    throw new FeedError(`${url} 取数失败: ${last}`);
  }

  /** _lagged: 对齐 10 秒后再减 lag */
  static lagged(lagSec: number, at?: number): string {
    const t = Math.floor((at ?? Date.now()) / 10_000) * 10_000;
    return new Date(t - lagSec * 1000).toISOString().slice(0, 19) + "Z";
  }

  // -- 赛程 / 直播 --

  private matches(payload: Json, known: string[] | null, league: string): Match[] {
    const out: Match[] = [];
    for (const e of orArr(orObj(orObj(orObj(payload).data).schedule).events)) {
      const m = orObj(e.match);
      if (!m.id) continue;
      const lg = orObj(e.league).name || league;
      const teams: TeamRef[] = [];
      for (const t of pget(m, "teams", [])) {
        const nm = t.name ?? null;
        teams.push({
          api_name: nm,
          code: pget(t, "code", ""),
          model_name: mapTeam(this.teams, nm, known),
          game_wins: orObj(t.result).gameWins ?? null,
          image: https(t.image),
        });
      }
      out.push({
        match_id: m.id,
        league: lg,
        start_time: pget(e, "startTime", ""),
        state: pget(e, "state", ""),
        best_of: orObj(m.strategy).count ?? null,
        teams,
        between_games: false,
      });
    }
    return out;
  }

  async schedule(league: string, known: string[] | null): Promise<Match[]> {
    const lid = LEAGUE_IDS.find(([k]) => k === league.toUpperCase())?.[1];
    if (!lid) throw new FeedError(`未知赛区 ${league}`);
    const p = await this.get(`${PERSISTED}/getSchedule?hl=en-US&leagueId=${lid}`, this.scheduleTtl);
    return this.matches(p, known, league.toUpperCase());
  }

  async live(known: string[] | null = null): Promise<Match[]> {
    const p = await this.get(`${PERSISTED}/getLive?hl=en-US`, this.liveTtl);
    return this.matches(p, known, "").filter((m) => LEAGUE_IDS.some(([k]) => k === m.league.toUpperCase()));
  }

  private eventUrl(matchId: string) {
    return `${PERSISTED}/getEventDetails?hl=en-US&id=${matchId}`;
  }

  async event(matchId: string): Promise<Json> {
    return orObj(orObj(orObj(await this.get(this.eventUrl(matchId), this.liveTtl)).data).event);
  }

  async games(matchId: string): Promise<Json[]> {
    return pget(orObj((await this.event(matchId)).match), "games", [] as Json[]);
  }

  /** 系列赛比分 {队名归一化 → 赢下的局数} —— 20 秒缓存的那一份, 见 esports_feed.match_wins */
  async matchWins(matchId: string, norm: (s: string) => string): Promise<Map<string, number>> {
    const out = new Map<string, number>();
    for (const t of orArr(orObj((await this.event(matchId)).match).teams)) {
      const w = orObj(t.result).gameWins;
      if (w === null || w === undefined) continue;
      for (const key of [t.name, t.code]) if (key) out.set(norm(key), Math.trunc(w));
    }
    return out;
  }

  // -- 实时帧 --

  async rawWindow(gameId: string, starting: string | null, ttl: number, emptyTtl?: number): Promise<Json> {
    let url = `${FEED}/window/${gameId}`;
    if (starting) url += `?startingTime=${starting}`;
    return this.get(url, ttl, false, emptyTtl);
  }

  private static objectives(t: Json) {
    const d = t.dragons;
    return {
      towers: t.towers || 0,
      inhibitors: t.inhibitors || 0,
      barons: t.barons || 0,
      dragons: Array.isArray(d) ? d.length : d || 0,
      dragon_types: Array.isArray(d) ? d.filter((x: unknown) => typeof x === "string") : [],
    };
  }

  /**
   * _game_start: 开局时刻。赛程的 startTime 是转播时间, 不能用。
   * 校准失败而那批窗口还没定型时, 先用 frames[0] 顶着但不缓存, 隔 START_RETRY_SEC
   * 再试 —— 失败的退回值进长缓存会让整局分钟数偏 8~20 秒 (见 esports_feed._game_start)。
   */
  async gameStart(gameId: string): Promise<number | null> {
    const cached = this.starts.get(gameId);
    if (cached !== undefined) return cached;
    const p = await this.rawWindow(gameId, null, 3600);
    const frames = orArr(orObj(p).frames);
    if (!frames.length) return null;
    const raw = frameTs(frames[0]);
    const now = Date.now();
    if (now < (this.startRetry.get(gameId) ?? 0)) return raw;
    let start = await this.calibrateStart(gameId, raw);
    if (start === null) {
      if ((now - raw) / 1000 <= CALIB_SPAN_SEC + OBS_SETTLE_SEC) {
        this.startRetry.set(gameId, now + START_RETRY_SEC * 1000);
        return raw;
      }
      start = raw; // 窗口都定型了还算不出来: 这一局确实校不了
    }
    this.starts.set(gameId, start);
    // 此前一直按 frames[0] 算, 暂停观测是按旧零点分的桶, 作废重扫
    if (this.startRetry.delete(gameId)) this.obs.delete(gameId);
    return start;
  }

  /** _calibrate_start: 零点校到游戏内 0:00 —— 靠"队伍金币第一次超过 2500"的那一帧 */
  private async calibrateStart(gameId: string, raw: number): Promise<number | null> {
    const times = Array.from({ length: CALIB_SPAN_SEC / 10 }, (_, i) => raw + i * 10_000);
    const now = Date.now();
    let batches: Json[][];
    try {
      batches = await pool(times, POOL, async (t) => {
        // 还没定型的窗口只短缓存, 否则重试拿回的都是同一批空结果
        const settled = (now - t) / 1000 > OBS_SETTLE_SEC;
        const p = await this.rawWindow(gameId, Feed.lagged(0, t), settled ? 3600 : this.windowTtl, this.windowTtl);
        return orArr(orObj(p).frames);
      });
    } catch {
      return null;
    }
    const all = batches.flat();
    const key = (f: Json) => f.rfc460Timestamp || "";
    all.sort((a, b) => (key(a) < key(b) ? -1 : key(a) > key(b) ? 1 : 0));
    const seen = new Set<string>();
    let first: number | null = null;
    for (const fr of all) {
      const ts = fr.rfc460Timestamp;
      if (!ts || seen.has(ts)) continue;
      seen.add(ts);
      const bg = pget(orObj(fr.blueTeam), "totalGold", 0);
      const rg = pget(orObj(fr.redTeam), "totalGold", 0);
      if (!bg && !rg) continue;
      if (bg > START_GOLD || rg > START_GOLD) {
        first = frameTs(fr);
        break;
      }
    }
    if (first === null) return null;
    const t0 = first - FIRST_INCOME_SEC * 1000;
    if (!(raw - 120_000 <= t0 && t0 <= raw + 10_000)) return null;
    return t0;
  }

  // ── 暂停 (见 esports_feed.pause_spans 的长注释) ──

  async pauseSeconds(gameId: string, upto: number, stepSec = 60): Promise<number> {
    let s = 0;
    for (const [b, e] of await this.pauseSpans(gameId, upto, stepSec)) {
      if (b < upto) s += Math.max(0, (Math.min(e, upto) - b) / 1000);
    }
    return s;
  }

  async pauseSpans(gameId: string, upto: number, stepSec = 60): Promise<[number, number][]> {
    const start = await this.gameStart(gameId);
    if (start === null || upto <= start) return [];
    const times: number[] = [];
    for (let t = start; t <= upto; t += stepSec * 1000) times.push(t);
    const bucket = (tt: number) => Math.floor(Math.trunc((tt - start) / 1000) / stepSec);

    const now = Date.now();
    const cached = new Map(this.obs.get(gameId) ?? []);
    // 只补扫没有观测、或者观测还可能变的窗口
    const need = times.filter((tt) => !cached.has(bucket(tt)) || (now - tt) / 1000 <= OBS_SETTLE_SEC);
    if (need.length) {
      const framesets = await pool(need, POOL, async (tt) => {
        const settled = (now - tt) / 1000 > 600;
        const p = await this.rawWindow(gameId, Feed.lagged(0, tt), settled ? 3600 : this.windowTtl, this.windowTtl);
        return orArr(orObj(p).frames);
      });
      const fresh = new Map<number, Obs>();
      need.forEach((tt, i) => {
        const fr = framesets[i]!.filter((f: Json) => f.rfc460Timestamp);
        // 空窗是"不知道", 不能算暂停, 也不写进缓存
        if (!fr.length) return;
        const first = frameTs(fr[0]);
        const last = frameTs(fr[fr.length - 1]);
        const lf = fr[fr.length - 1];
        const gold = pget(orObj(lf.blueTeam), "totalGold", 0) + pget(orObj(lf.redTeam), "totalGold", 0);
        fresh.set(bucket(tt), [tt, first, last, last < tt - FREEZE_SLACK * 1000, gold]);
      });
      for (const [k, v] of fresh) cached.set(k, v);
      const store = this.obs.get(gameId) ?? new Map<number, Obs>();
      for (const [k, v] of fresh) store.set(k, v);
      this.obs.set(gameId, store);
    }

    const seq = times.filter((tt) => cached.has(bucket(tt))).map((tt) => cached.get(bucket(tt))!);
    const spans: [number, number][] = [];
    let i = 0;
    while (i < seq.length) {
      if (!seq[i]![3]) {
        i++;
        continue;
      }
      let j = i;
      while (j < seq.length && seq[j]![3]) j++;
      if (j < seq.length && seq[j]![1] > seq[i]![2]) {
        const begin = seq[i]![2];
        const end = seq[j]![1];
        // 断流不是暂停: 跨过这一段经济涨了一大截, 说明比赛照打
        const gapMin = (end - begin) / 1000 / 60;
        const grew = (seq[j]![4] || 0) - (seq[i]![4] || 0);
        if (grew > PLAY_GOLD_BASE + PLAY_GOLD_PER_MIN * gapMin) {
          i = j;
          continue;
        }
        spans.push([begin, end]);
      }
      i = j;
    }
    return spans;
  }

  /**
   * window(): 当前帧。取不到就返回 null —— 绝不回退到默认调用 (那给的是开局帧)。
   * 自适应 lag、兜底阶梯、"整条链都没帧就记一会儿"都和服务器一样, 见 esports_feed.window。
   */
  async window(
    gameId: string,
    opts: { at?: number; fallbackLags?: number[]; preciseMinute?: boolean } = {},
  ): Promise<LiveState | null> {
    const at = opts.at;
    const fallback = opts.fallbackLags ?? FALLBACK_LAGS;
    const now = Date.now();
    if (at === undefined) {
      const until = this.noFrames.get(gameId);
      if (until && until > now) return null;
    }
    const fetchLag = async (lag: number): Promise<Json[]> =>
      orArr(orObj(await this.rawWindow(gameId, Feed.lagged(lag, at), this.windowTtl)).frames);

    let base = fallback[0]!;
    let probe: number | null = null;
    if (at === undefined) {
      base = this.lag.get(gameId) ?? fallback[0]!;
      if (base > this.minLag && now >= (this.lagProbe.get(gameId) ?? 0)) {
        probe = base - 10;
        this.lagProbe.set(gameId, now + this.lagProbeSec * 1000);
      }
    }

    let frames: Json[];
    let usedLag: number;
    if (probe !== null) {
      const [gp, gb] = await Promise.all([fetchLag(probe), fetchLag(base)]);
      if (gp.length) {
        frames = gp;
        usedLag = probe;
        this.lag.set(gameId, probe);
      } else {
        frames = gb;
        usedLag = base;
      }
    } else {
      frames = await fetchLag(base);
      usedLag = base;
    }

    const ladder = [...new Set([...LAG_STEPS, ...fallback])].sort((a, b) => a - b);
    const rest = frames.length ? [] : ladder.filter((l) => l > usedLag);
    if (rest.length) {
      const got = await Promise.all(rest.map(fetchLag));
      for (let k = 0; k < rest.length; k++) {
        if (got[k]!.length) {
          frames = got[k]!;
          usedLag = rest[k]!;
          break;
        }
      }
      if (frames.length && at === undefined) this.lagProbe.set(gameId, now + this.lagProbeSec * 2 * 1000);
    }
    if (!frames.length) {
      if (at === undefined) {
        this.noFrames.set(gameId, now + this.noFramesTtl * 1000);
        this.lag.delete(gameId);
      }
      return null;
    }
    if (at === undefined) this.lag.set(gameId, usedLag);
    this.noFrames.delete(gameId);

    const f = frames[frames.length - 1];
    const b = f.blueTeam;
    const r = f.redTeam;
    const csOf = (t: Json) => pget(t, "participants", []).reduce((s: number, x: Json) => s + pget(x, "creepScore", 0), 0);

    let minute: number | null = null;
    let paused = 0;
    const start = await this.gameStart(gameId);
    if (start !== null) {
      const elapsed = (frameTs(f) - start) / 1000;
      if (opts.preciseMinute) paused = await this.pauseSeconds(gameId, frameTs(f));
      minute = Math.max(0, Math.floor((elapsed - paused) / 60));
    }
    const stale = Math.max(0, Math.trunc((Date.now() - frameTs(f)) / 1000));
    const gs = pget(f, "gameState", "");
    const stalled = gs === "in_game" && stale > LIVE_STALE_SEC;
    const side = (t: Json): TeamObjectives => {
      const d = pget(t, "dragons", undefined);
      return {
        gold: pget(t, "totalGold", 0),
        kills: pget(t, "totalKills", 0),
        towers: pget(t, "towers", 0),
        inhibitors: pget(t, "inhibitors", 0),
        barons: pget(t, "barons", 0),
        dragons: Array.isArray(d) ? (d.length ? d.length : 0) : d || 0,
        dragon_types: Array.isArray(d) && d.length ? d : [],
        cs: csOf(t),
      };
    };
    return {
      game_id: gameId,
      game_state: gs,
      minute,
      golddiff: pget(b, "totalGold", 0) - pget(r, "totalGold", 0),
      csdiff: csOf(b) - csOf(r),
      blue_kills: pget(b, "totalKills", 0),
      red_kills: pget(r, "totalKills", 0),
      gold_total: pget(b, "totalGold", 0),
      frame_time: pget(f, "rfc460Timestamp", ""),
      live: gs === "in_game" && !stalled,
      teams: { blue: side(b), red: side(r) },
      stale_seconds: stale,
      paused_seconds: paused,
      stalled,
    };
  }

  // -- 选手面板 --

  async ddragonVersion(): Promise<string> {
    const key = "__ddragon_ver__";
    const hit = this.cache.get(key);
    if (hit && hit.exp > Date.now()) return hit.val;
    let v: string;
    try {
      const r = await this.fetchImpl(`${DDRAGON}/api/versions.json`);
      v = (await r.json())[0];
      if (typeof v !== "string") throw new Error("版本号格式不对");
    } catch {
      v = DDRAGON_FALLBACK_VER;
    }
    this.cache.set(key, { exp: Date.now() + 6 * 3600_000, val: v });
    return v;
  }

  /** 符文 id → {name, icon, style}。拿不到就是空表, 调用方退回显示原始 id */
  async runeIndex(): Promise<Map<number, { name: string | null; icon: string | null; style: string | null }>> {
    const key = "__runes__";
    const hit = this.cache.get(key);
    if (hit && hit.exp > Date.now()) return hit.val;
    const idx = new Map<number, { name: string | null; icon: string | null; style: string | null }>();
    try {
      const ver = await this.ddragonVersion();
      const r = await this.fetchImpl(`${DDRAGON}/cdn/${ver}/data/${this.runeLocale}/runesReforged.json`);
      for (const style of await r.json()) {
        const sname = style.name ?? null;
        idx.set(style.id, { name: sname, icon: style.icon ?? null, style: sname });
        for (const slot of orArr(style.slots)) {
          for (const p of orArr(slot.runes)) idx.set(p.id, { name: p.name ?? null, icon: p.icon ?? null, style: sname });
        }
      }
    } catch {
      idx.clear();
    }
    this.cache.set(key, { exp: Date.now() + 6 * 3600_000, val: idx });
    return idx;
  }

  /** 饰品 id 集合, 按 item.json 的 "Trinket" 标签 —— 跟着版本走, 不写死 */
  async trinketIds(): Promise<Set<number>> {
    const key = "__trinkets__";
    const hit = this.cache.get(key);
    if (hit && hit.exp > Date.now()) return hit.val;
    const out = new Set<number>();
    try {
      const ver = await this.ddragonVersion();
      const r = await this.fetchImpl(`${DDRAGON}/cdn/${ver}/data/en_US/item.json`);
      for (const [iid, d] of Object.entries(orObj((await r.json()).data)) as [string, Json][]) {
        if (orArr(d.tags).includes("Trinket")) out.add(Number(iid));
      }
    } catch {
      out.clear();
    }
    this.cache.set(key, { exp: Date.now() + 6 * 3600_000, val: out });
    return out;
  }

  private async perks(meta: Json): Promise<Record<string, unknown>> {
    if (!meta || (typeof meta === "object" && !Object.keys(meta).length)) return {};
    const idx = await this.runeIndex();
    const ver = await this.ddragonVersion();
    const one = (pid: Json) => {
      const info = pid === null || pid === undefined ? undefined : idx.get(pid);
      if (!info) {
        const shard = typeof pid === "number" && Object.prototype.hasOwnProperty.call(STAT_SHARDS, pid);
        return { id: pid ?? null, name: shard ? STAT_SHARDS[pid] : null, icon: null, shard };
      }
      return { id: pid, name: info.name, icon: info.icon ? `${DDRAGON}/cdn/img/${info.icon}` : null };
    };
    return {
      style: one(meta.styleId),
      sub_style: one(meta.subStyleId),
      perks: orArr(meta.perks).map(one),
      ddragon_version: ver,
    };
  }

  private async metaPayload(gameId: string): Promise<Json> {
    for (const at of [null, Feed.lagged(60), Feed.lagged(150)]) {
      const p = await this.rawWindow(gameId, at, at === null ? 3600 : this.windowTtl);
      const md = orObj(orObj(p).gameMetadata);
      if (Object.keys(md).length) return md;
    }
    return {};
  }

  /** {participantId: {summoner_name, champion, role, side}} —— 只缓存完整的十个人 */
  async gameMetadata(gameId: string): Promise<Map<number, MetaEntry>> {
    const cached = this.meta.get(gameId);
    if (cached) return cached;
    const md = await this.metaPayload(gameId);
    const out = new Map<number, MetaEntry>();
    for (const [side, key] of [
      ["blue", "blueTeamMetadata"],
      ["red", "redTeamMetadata"],
    ] as const) {
      for (const m of pget(orObj(md[key]), "participantMetadata", [])) {
        out.set(m.participantId ?? null, {
          summoner_name: m.summonerName ?? null,
          champion: m.championId ?? null,
          role: m.role ?? null,
          side,
        });
      }
    }
    if (out.size === 10) this.meta.set(gameId, out);
    return out;
  }

  /** 本局蓝红方的 esportsTeamId —— 以帧为准, 见 esports_feed.frame_sides */
  async frameSides(gameId: string): Promise<{ blue?: string; red?: string }> {
    const md = await this.metaPayload(gameId);
    const out: { blue?: string; red?: string } = {};
    for (const [side, key] of [
      ["blue", "blueTeamMetadata"],
      ["red", "redTeamMetadata"],
    ] as const) {
      const tid = orObj(md[key]).esportsTeamId;
      if (tid) out[side] = String(tid);
    }
    return out;
  }

  /** esports_feed.players —— lag 的取法见那边的注释 (和头条同一个自适应 lag, 取不到退回 60) */
  async players(gameId: string, at?: number): Promise<Player[]> {
    const lags = at === undefined ? [...new Set([this.lag.get(gameId) ?? 60, 60])] : [60];
    let frames: Json[] = [];
    let lag = 60;
    for (const l of lags) {
      const p = await this.get(`${FEED}/details/${gameId}?startingTime=${Feed.lagged(l, at)}`, this.windowTtl, false);
      frames = orArr(orObj(p).frames);
      if (frames.length) {
        lag = l;
        break;
      }
    }
    if (!frames.length) return [];
    const meta = await this.gameMetadata(gameId);

    // 血量只有 window 里有, 按 participantId 并进来; 拿不到就是没血量
    const hp = new Map<number, Json>();
    try {
      const w = await this.rawWindow(gameId, Feed.lagged(lag, at), this.windowTtl);
      const wf = orArr(orObj(w).frames);
      if (wf.length) {
        for (const k of ["blueTeam", "redTeam"]) {
          for (const x of pget(orObj(wf[wf.length - 1][k]), "participants", [])) hp.set(x.participantId ?? null, x);
        }
      }
    } catch {
      /* 没血量不影响其余字段 */
    }

    const trinkets = await this.trinketIds();
    const out: Player[] = [];
    for (const x of pget(frames[frames.length - 1], "participants", [])) {
      const pid = x.participantId ?? null;
      const m = meta.get(pid) ?? ({} as Partial<MetaEntry>);
      const h = hp.get(pid) ?? {};
      const rawItems: number[] = orArr(x.items).filter((i: Json) => i);
      const trinket = rawItems.find((i) => trinkets.has(i)) ?? null;
      const plain = rawItems.filter((i) => i !== trinket).slice(0, 6);
      out.push({
        participant_id: pid,
        side: m.side || ((pid || 0) <= 5 ? "blue" : "red"),
        role: m.role ?? null,
        summoner_name: m.summoner_name ?? null,
        champion: m.champion ?? null,
        level: pget(x, "level", 0),
        kills: pget(x, "kills", 0),
        deaths: pget(x, "deaths", 0),
        assists: pget(x, "assists", 0),
        cs: pget(x, "creepScore", 0),
        gold: pget(x, "totalGoldEarned", 0),
        kill_participation: x.killParticipation ?? null,
        damage_share: x.championDamageShare ?? null,
        wards_placed: x.wardsPlaced ?? null,
        wards_destroyed: x.wardsDestroyed ?? null,
        current_health: h.currentHealth ?? null,
        max_health: h.maxHealth ?? null,
        items: plain,
        trinket,
        stats: {
          attack_damage: x.attackDamage ?? null,
          ability_power: x.abilityPower ?? null,
          armor: x.armor ?? null,
          magic_resist: x.magicResistance ?? null,
          attack_speed: x.attackSpeed ?? null,
          crit: x.criticalChance ?? null,
          life_steal: x.lifeSteal ?? null,
          tenacity: x.tenacity ?? null,
        },
        perks: await this.perks(x.perkMetadata),
        abilities: orArr(x.abilities).filter((a: Json) => a),
      });
    }
    return out;
  }

  // -- 时间线 --

  /** gold_timeline: 每分钟一个点, 横轴是扣掉暂停的局内时间。见 esports_feed.gold_timeline */
  async goldTimeline(gameId: string, upto?: number, stepSec = 60): Promise<TimelineRow[]> {
    const start = await this.gameStart(gameId);
    if (start === null) return [];
    const now = Date.now();
    let end = Math.min(upto ?? now, now);
    if (end <= start) return [];
    end = Math.min(end, start + 120 * 60_000);

    const times: number[] = [];
    for (let t = start; t <= end + stepSec * 1000; t += stepSec * 1000) times.push(t);
    const framesets = await pool(times, POOL, async (tt) => {
      const settled = (now - tt) / 1000 > 600;
      const p = await this.rawWindow(gameId, Feed.lagged(0, tt), settled ? 3600 : this.windowTtl, this.windowTtl);
      return orArr(orObj(p).frames);
    });

    const spans = await this.pauseSpans(gameId, end);
    const ingameSecs = (ts: number) => {
      let paused = 0;
      for (const [b, e] of spans) if (b < ts) paused += Math.max(0, (Math.min(e, ts) - b) / 1000);
      return (ts - start) / 1000 - paused;
    };

    const rows = new Map<string, TimelineRow>();
    for (const frames of framesets) {
      for (let k = frames.length - 1; k >= 0; k--) {
        const f = frames[k];
        const b = orObj(f.blueTeam);
        const r = orObj(f.redTeam);
        const bg = pget(b, "totalGold", 0);
        const rg = pget(r, "totalGold", 0);
        if (!bg && !rg) continue;
        const ts = f.rfc460Timestamp;
        if (!ts || rows.has(ts)) break;
        const secs = ingameSecs(frameTs(f));
        const bo = Feed.objectives(b);
        const ro = Feed.objectives(r);
        const cs = (t: Json) => pget(t, "participants", []).reduce((s: number, x: Json) => s + pget(x, "creepScore", 0), 0);
        rows.set(ts, {
          t: ts,
          minute: pyRound(secs / 60, 2), // Python 的 round(secs / 60, 2)
          blue_towers: bo.towers,
          red_towers: ro.towers,
          blue_inhibitors: bo.inhibitors,
          red_inhibitors: ro.inhibitors,
          blue_barons: bo.barons,
          red_barons: ro.barons,
          blue_dragons: bo.dragons,
          red_dragons: ro.dragons,
          blue_dragon_types: bo.dragon_types,
          red_dragon_types: ro.dragon_types,
          golddiff: bg - rg,
          blue_gold: bg,
          red_gold: rg,
          blue_kills: pget(b, "totalKills", 0),
          red_kills: pget(r, "totalKills", 0),
          csdiff: cs(b) - cs(r),
          state: pget(f, "gameState", ""),
        });
        break;
      }
    }
    return [...rows.values()].sort((a, b) => (a.t < b.t ? -1 : a.t > b.t ? 1 : 0));
  }

  static downsample<T>(rows: T[], n = 60): T[] {
    if (rows.length <= n) return rows;
    const step = rows.length / n;
    const picked = Array.from({ length: n }, (_, i) => rows[Math.min(rows.length - 1, Math.trunc(i * step))]!);
    if (picked[picked.length - 1] !== rows[rows.length - 1]) picked.push(rows[rows.length - 1]!);
    return picked;
  }

  // -- 当前局 --

  async probeGames(matchId: string): Promise<[Json, LiveState | null][]> {
    const gs = (await this.games(matchId)).filter((g: Json) => g.id);
    if (!gs.length) return [];
    const states = await Promise.all(
      gs.map((g: Json) =>
        this.window(g.id).catch((e) => {
          if (e instanceof FeedError) return null;
          throw e;
        }),
      ),
    );
    return gs.map((g: Json, i: number) => [g, states[i]!]);
  }

  /** current_game: 真正在打的那一局。完全不看 games[].state, 卡住的 in_game 局也交出去 */
  async currentGame(matchId: string): Promise<[Json, LiveState] | null> {
    const pairs = await this.probeGames(matchId);
    for (let i = pairs.length - 1; i >= 0; i--) {
      const [g, st] = pairs[i]!;
      if (st && st.live) return [g, st];
    }
    for (let i = pairs.length - 1; i >= 0; i--) {
      const [g, st] = pairs[i]!;
      if (st && st.game_state === "in_game") return [g, st];
    }
    return null;
  }

  /** live_by_frames: 最近开赛的比赛里, 谁的帧显示真的在打 */
  async liveByFrames(matches: Match[], withinHours = 5): Promise<Match[]> {
    const now = Date.now();
    const cands = matches.filter((m) => {
      if (!m.start_time) return false;
      const ts = parseTs19(m.start_time);
      return !Number.isNaN(ts) && now - withinHours * 3600_000 <= ts && ts <= now + 10 * 60_000;
    });
    const hit = await Promise.all(cands.map((m) => this.currentGame(m.match_id).catch(() => null)));
    return cands.filter((_, i) => hit[i]);
  }
}
