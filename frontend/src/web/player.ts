/**
 * 直播回放器 —— 让看板一直在动, 而不是每几秒跳一下。
 *
 * 数据源的实时帧其实很密: 一个 10 秒窗口里 38~50 帧, 帧间隔中位约 200ms (2026-09-13 实测)。
 * 看板原来每次轮询只取窗口的最后一帧, 其余全扔了。这里像直播推流一样: 每个窗口一发布就
 * 整段取回放进缓冲, 画面时钟固定在"现在 − D", 按真实速度一帧一帧往前播 —— 经济一直在跳、
 * 血条实时掉血回血、补刀一刀一刀加, 胜率每帧用浏览器里的局内模型重算。
 *
 * D 取多少是整件事的关键。窗口 [S, S+10) 要到 S+P 才发布 (P = 上游发布延迟, 实测约 50~60 秒)。
 * 播放头走到 S+10 时下一个窗口必须已经到手, 即 D ≥ P。所以 D = 实测 P + 0.3 秒, 就是"不断档"
 * 的最低延迟 —— 比原来"只看最新一帧"平均只多约 5 秒 (原来显示的是 P−10 到 P 之间, 一跳一跳的),
 * 换来每秒 5 帧的连续画面。P 从每个窗口实际到手的时刻现量: 上游变慢就自动拉长, 变快就略微
 * 加速追上去 (最多 1.25 倍速), 永不倒退。
 *
 * 取窗口不走固定周期: 在预计发布的时刻前 1.5 秒开始, 每 0.5 秒试一次, 一出来就拿到。
 * 暂停/断流: 缓冲播完就停在最后一帧等, 不编造。
 * 只在静态版里用 —— 胜率要在浏览器里重算, 模型只在静态版里有。
 */
import type { BoardResponse, Player as PlayerJson, Prediction, TimelinePoint } from "../api/types.ts";
import type { Models } from "./board.ts";
import { lanesOf } from "./board.ts";
import type { Feed } from "./feed.ts";
import { DDRAGON, teamObjectives } from "./feed.ts";
import { ingameResponse } from "./ingame.ts";
import type { TeamsFile } from "./teams.ts";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Json = any;

/** 同一时刻的 window 帧 (队伍总计、血量) 和 details 帧 (装备、属性) */
export interface PlayFrame {
  ts: number;
  w: Json;
  d: Json | null;
}

const MARGIN_MS = 300;
const POLL_MS = 500;
const TICK_MS = 100;
const KEEP_MS = 180_000;
const iso = (ms: number) => new Date(ms).toISOString().slice(0, 19) + "Z";
const floor10 = (ms: number) => Math.floor(ms / 10_000) * 10_000;
/** 帧时间戳要毫秒精度 —— frameTs() 只取到秒, 而帧间隔才 200ms */
const tsOf = (f: Json) => Date.parse(String(f.rfc460Timestamp));

export class LivePlayer {
  private frames: PlayFrame[] = [];
  private lastDetails: Json | null = null;
  private next = 0;
  /** 实测发布延迟: 窗口起点 → 我们第一次拿到它, 毫秒 */
  private delays: number[] = [];
  private initialDelay = 60_000;
  private playhead = 0;
  private lastWall = 0;
  private tickT: ReturnType<typeof setInterval> | undefined;
  private pollT: ReturnType<typeof setTimeout> | undefined;
  private stopped = false;
  private finished = false;
  private cur: PlayFrame | null = null;

  private feed: Feed;
  readonly gameId: string;
  private onFrame: (f: PlayFrame) => void;

  constructor(feed: Feed, gameId: string, onFrame: (f: PlayFrame) => void) {
    this.feed = feed;
    this.gameId = gameId;
    this.onFrame = onFrame;
  }

  /** 此刻画面比真实比赛晚多少毫秒 */
  delayMs(): number {
    return this.cur ? Date.now() - this.cur.ts : NaN;
  }

  async start(): Promise<void> {
    const lag = this.feed.currentLag(this.gameId) ?? 60;
    const s0 = floor10(Date.now()) - lag * 1000;
    const t0 = Date.now();
    // 起步多取一个窗口: 缓冲里先有 20 秒, 播放头从更早处起步也不会断
    const [a, b] = await Promise.all([this.fetchWindow(s0 - 10_000), this.fetchWindow(s0)]);
    if (this.stopped) return;
    this.next = b === "none" ? s0 : s0 + 10_000;
    // 窗口 s0 此刻已经拿到了, 所以 P ≤ 现在 − s0。先按这个上界放, 保证不断档;
    // 之后每个窗口到手的时刻会把它校准下来
    this.initialDelay = t0 - (b === "none" ? s0 - 10_000 : s0) + MARGIN_MS;
    void a;
    const first = this.frames[0];
    this.playhead = first ? Math.max(first.ts, Date.now() - this.targetDelay()) : 0;
    this.lastWall = Date.now();
    this.tickT = setInterval(() => this.step(), TICK_MS);
    this.schedulePoll();
  }

  stop(): void {
    this.stopped = true;
    if (this.tickT) clearInterval(this.tickT);
    if (this.pollT) clearTimeout(this.pollT);
  }

  private targetDelay(): number {
    const d = this.delays.slice(-3);
    return (d.length ? Math.max(...d) : this.initialDelay) + MARGIN_MS;
  }

  /** 取一个窗口放进缓冲: new = 有新帧; stale = 有帧但都是旧的 (暂停时的冻结帧); none = 还没发布 */
  private async fetchWindow(s: number): Promise<"new" | "stale" | "none"> {
    const st = iso(s);
    // 空结果不缓存 (emptyTtl 0): 还没发布的窗口过 0.5 秒要重新问
    const w = await this.feed.rawWindow(this.gameId, st, 3600, 0).catch(() => null);
    const wf: Json[] = w?.frames ?? [];
    if (!wf.length) return "none";
    // window 先到就先播; details 同一时刻的帧随后并进来, 取不到就沿用上一帧的装备和属性
    const d = await this.feed.rawDetails(this.gameId, st, 3600, 0).catch(() => null);
    const dmap = new Map<number, Json>((d?.frames ?? []).map((f: Json) => [tsOf(f), f]));
    const have = new Set(this.frames.map((f) => f.ts));
    let fresh = 0;
    for (const f of wf) {
      const ts = tsOf(f);
      if (!Number.isFinite(ts) || have.has(ts)) continue;
      const df = dmap.get(ts) ?? null;
      if (df) this.lastDetails = df;
      this.frames.push({ ts, w: f, d: df ?? this.lastDetails });
      have.add(ts);
      if (ts >= s) fresh++;
    }
    this.frames.sort((x, y) => x.ts - y.ts);
    return fresh ? "new" : "stale";
  }

  private schedulePoll(): void {
    if (this.stopped || this.finished) return;
    const p = this.delays.length ? Math.min(...this.delays.slice(-3)) : this.initialDelay - MARGIN_MS;
    const wait = Math.max(POLL_MS, this.next + p - 1500 - Date.now());
    this.pollT = setTimeout(() => void this.poll(), wait);
  }

  private async poll(): Promise<void> {
    if (this.stopped) return;
    const p = this.delays.length ? Math.min(...this.delays.slice(-3)) : this.initialDelay - MARGIN_MS;
    // 标签页在后台被冻过、或长时间断流: 直接跳到最新的窗口, 不去补中间那些
    if (Date.now() - (this.next + p) > 30_000) this.next = floor10(Date.now() - p);
    const r = await this.fetchWindow(this.next);
    if (this.stopped) return;
    if (r === "new") {
      this.delays.push(Date.now() - this.next);
      if (this.delays.length > 6) this.delays.shift();
      this.next += 10_000;
      this.schedulePoll();
    } else if (r === "stale") {
      this.next += 10_000; // 暂停: 窗口里是冻结的旧帧, 往下一个窗口走
      this.schedulePoll();
    } else {
      this.pollT = setTimeout(() => void this.poll(), POLL_MS);
    }
  }

  private step(): void {
    const now = Date.now();
    const dt = now - this.lastWall;
    this.lastWall = now;
    const latest = this.frames[this.frames.length - 1]?.ts;
    if (latest === undefined) return;
    const target = now - this.targetDelay();
    if (target - this.playhead > 30_000) this.playhead = target; // 后台冻过: 直接跳, 不快进半天
    // 延迟比目标大就略微加速追, 比目标小 (上游刚变慢) 就略微放慢; 缓冲播完就停在最后一帧
    const rate = Math.min(1.25, Math.max(0.9, 1 + (target - this.playhead) / 8000));
    this.playhead = Math.min(this.playhead + dt * rate, latest);

    let i = this.frames.length - 1;
    while (i > 0 && this.frames[i]!.ts > this.playhead) i--;
    const f = this.frames[i]!;
    if (f !== this.cur) {
      this.cur = f;
      if (f.w?.gameState === "finished") this.finished = true;
      this.onFrame(f);
    }
    const cut = this.playhead - KEEP_MS;
    if (this.frames.length > 50 && this.frames[0]!.ts < cut) this.frames = this.frames.filter((x) => x.ts >= cut);
  }
}

export interface ViewCtx {
  models: Models;
  teams: TeamsFile;
  today: string;
  /** 这一局的开局时刻 (Feed.gameStart) */
  start: number | null;
  trinkets: Set<number>;
  /** 回放期间逐秒积累的走势点 (liveView 往里追加), 一次回放一份 */
  trail: TimelinePoint[];
}

/**
 * 用回放到的这一帧改写看板: 记分板、队伍总计、胜率、走势图末端都换成这一刻的。
 * 选手的名字/英雄/符文这些不随帧变的沿用看板; 胜率和看板头条同一个函数 (ingameResponse),
 * 分钟数同一个取法 (开局到这一帧、扣掉暂停、向下取整)。
 * 走势图只画到播放头 —— 看板自己的曲线比回放新几秒, 不截的话图会跑在记分板前面。
 */
export function liveView(board: BoardResponse, pf: PlayFrame, ctx: ViewCtx): BoardResponse {
  const lv = board.live!;
  const w = pf.w;
  const d = pf.d;
  const ver = board.ddragon_version ?? "";
  const icon = (i: number) => ({ id: i, icon: `${DDRAGON}/cdn/${ver}/img/item/${i}.png` });
  const hp = new Map<number, Json>();
  for (const k of ["blueTeam", "redTeam"]) for (const x of w?.[k]?.participants ?? []) hp.set(x.participantId, x);
  const det = new Map<number, Json>((d?.participants ?? []).map((x: Json) => [x.participantId, x]));

  const players: PlayerJson[] = lv.players.map((p) => {
    const x = det.get(p.participant_id);
    const h = hp.get(p.participant_id);
    if (!x && !h) return p;
    const raw: number[] = (x?.items ?? []).filter((i: number) => i);
    const trinket = raw.find((i) => ctx.trinkets.has(i)) ?? null;
    return {
      ...p,
      level: x?.level ?? h?.level ?? p.level,
      kills: x?.kills ?? h?.kills ?? p.kills,
      deaths: x?.deaths ?? h?.deaths ?? p.deaths,
      assists: x?.assists ?? h?.assists ?? p.assists,
      cs: x?.creepScore ?? h?.creepScore ?? p.cs,
      gold: x?.totalGoldEarned ?? p.gold,
      kill_participation: x?.killParticipation ?? p.kill_participation,
      damage_share: x?.championDamageShare ?? p.damage_share,
      wards_placed: x?.wardsPlaced ?? p.wards_placed,
      wards_destroyed: x?.wardsDestroyed ?? p.wards_destroyed,
      current_health: h?.currentHealth ?? p.current_health,
      max_health: h?.maxHealth ?? p.max_health,
      items: x ? raw.filter((i) => i !== trinket).slice(0, 6).map(icon) : p.items,
      trinket: x ? (trinket ? icon(trinket) : null) : p.trinket,
      stats: x
        ? {
            attack_damage: x.attackDamage ?? null,
            ability_power: x.abilityPower ?? null,
            armor: x.armor ?? null,
            magic_resist: x.magicResistance ?? null,
            attack_speed: x.attackSpeed ?? null,
            crit: x.criticalChance ?? null,
            life_steal: x.lifeSteal ?? null,
            tenacity: x.tenacity ?? null,
          }
        : p.stats,
      abilities: x ? (x.abilities ?? []).filter(Boolean) : p.abilities,
    };
  });

  const teams = { blue: teamObjectives(w.blueTeam), red: teamObjectives(w.redTeam) };
  const golddiff = teams.blue.gold - teams.red.gold;
  const live = {
    ...lv,
    game_state: w.gameState ?? lv.game_state,
    frame_time: String(w.rfc460Timestamp),
    in_progress: w.gameState === "in_game",
    stalled: false,
    stale_seconds: Math.max(0, Math.trunc((Date.now() - pf.ts) / 1000)),
    teams,
    players,
    lanes: lanesOf(players),
  };

  const minuteF =
    ctx.start !== null ? Math.max(0, (pf.ts - ctx.start) / 60_000 - (lv.paused_seconds ?? 0) / 60) : null;
  const minute = minuteF === null ? null : Math.floor(minuteF);
  live.minute = minute;
  const pr = board.prediction;
  if (minute === null || minuteF === null || minute < 3 || minute > 60 || !board.match || !pr) {
    return { ...board, live };
  }

  const champs = (side: string) => players.filter((p) => p.side === side && p.champion).map((p) => p.champion!);
  const bch = champs("blue");
  const rch = champs("red");
  const res = ingameResponse(
    ctx.models.ingame,
    ctx.models.s1,
    ctx.teams,
    ctx.models.ex,
    {
      blue: board.match.teams[0]!.model_name!,
      red: board.match.teams[1]!.model_name!,
      league: board.match.league,
      state: {
        minute,
        golddiff,
        csdiff: teams.blue.cs - teams.red.cs,
        blueKills: teams.blue.kills,
        redKills: teams.red.kills,
        goldTotal: teams.blue.gold,
      },
      blueChamps: bch.length === 5 ? bch : null,
      redChamps: rch.length === 5 ? rch : null,
      // 同看板头条: 开局那段从 BP 后渐变过渡到局内模型
      blend: true,
      blendWith: pr.postdraft_probability_blue ?? null,
    },
    ctx.today,
  ) as unknown as Prediction;
  const prediction: Prediction = { ...res, postdraft_probability_blue: pr.postdraft_probability_blue ?? null };

  // 走势图按秒记: 回放每走过 1 秒游戏时间, 往轨迹末尾追加一个点。打开页面之前的部分
  // 仍是看板的每分钟一个点 (补全整局的逐秒数据要把几百个窗口全取一遍); 两段在轨迹起点接上。
  // 游戏时间倒退 (暂停时长刚被轮询更新, 分钟数往回收) 时, 把跑过头的点撤掉
  const tip: TimelinePoint = {
    minute: minuteF,
    golddiff,
    blue_kills: teams.blue.kills,
    red_kills: teams.red.kills,
    probability_blue: prediction.probability_blue,
  };
  const trail = ctx.trail;
  while (trail.length && trail[trail.length - 1]!.minute > minuteF) trail.pop();
  if (!trail.length || minuteF - trail[trail.length - 1]!.minute >= 1 / 60) trail.push(tip);
  const timeline: TimelinePoint[] = [...board.timeline.filter((p) => p.minute < trail[0]!.minute), ...trail];
  return { ...board, live, prediction, timeline };
}
