/**
 * 跨赛区模型 C (分层 Elo) —— xregion.py 里看板用到的那一段的浏览器版: 引擎 (Engine 的重放 / 预测子集)、
 * 赛事内重放的纯函数 (终局帧判胜负、比分约束解码、OE 去重、排序) 和 boardPredict。
 *
 * 为什么有它、为什么必须重放, 见 xregion.py 的模块文档; 这里只说移植时要守的规矩:
 *   · **运算顺序一个都不许改。** 偏移 o 的衰减是分段乘的 (先衰减到 t1 再到 t2, 和一步到 t2 差最后一位),
 *     所以 off() 必须在和 Python 同样的时刻、同样的次数被调用: 每一局跨赛区局开打时两个赛区各一次, 预测时
 *     再各一次。同一时间戳的几局先全部取赛前值、再统一更新 (同时开打的局互相看不到)。
 *   · 新赛区的先验 = 迁入队伍原赛区偏移的均值, 用 pairwiseMean —— numpy 那种 8 路分块的成对求和, 顺序累加
 *     会差最后一两位。
 *   · **exp 不用 Math.exp**, 用下面的 exp (fdlibm e_exp.c 的逐行移植, Python 那边是 xregion.portable_exp):
 *     各平台的 exp 末位互不相同 (本机 Python 的 math.exp 是 MSVC 运行库的, 和 V8 有 7% 的输入差一位, Safari
 *     用系统 libm), 只有只用 IEEE 加减乘除的同一段算法两边才逐位相同。
 *   · 状态 (public/web/xregion.json) 是 export_web_model.py 从 artifacts/xregion.json 原样复制的同一份;
 *     本模块只读它, 不改它 (fromState 复制一份再动)。
 * 全部由 test:golden 的 xregion_cases (引擎、predict、整条重放) 和 Python 逐位对账, test:diff 对整块看板。
 * 取数 (赛程、终局帧) 不在这里, 在 board.ts 的 xregionBoard (= api._xregion_board)。
 */

// ── 状态文件的形状 (xregion.build_state, schema 1) ──

export interface XregionTeam {
  pool: string;
  home?: string;
  r: number;
  ng: number;
  last_regular?: string;
  last_regular_t: number;
}

export interface XregionLeague {
  o: number | null;
  prior: number | null;
  last_t: number | null;
  last?: string | null;
  has_crossregion_history: boolean;
  crossregion_games?: number;
  major?: boolean;
}

export interface XregionCoef {
  a: number;
  b_o: number;
  b_r: number;
  c?: number;
  event?: string | null;
  cutoff?: string | null;
  n_train?: number;
}

/** [蓝, 红, 局号, 开打时间 ISO (UTC), 蓝方胜, 跨赛区状态] —— xregion.GAME_FIELDS */
export type OeGame = [string, string, number | null, string, number, string];

export interface XregionState {
  schema: number;
  built_at: string;
  oe_asof: string;
  oe_asof_t: number;
  params: {
    h: number;
    init: number;
    eta: number;
    kint: number;
    hl: number;
    dnm: number;
    S: number;
    majors: string[];
    [k: string]: unknown;
  };
  teams: Record<string, XregionTeam>;
  leagues: Record<string, XregionLeague>;
  migr: Record<string, string[]>;
  coefficients: Record<string, XregionCoef>;
  events: Record<string, { games: OeGame[]; [k: string]: unknown }>;
  [k: string]: unknown;
}

const own = (o: object | null | undefined, k: string) => !!o && Object.prototype.hasOwnProperty.call(o, k);

// ── 可移植的 exp (xregion.portable_exp 的逐行移植) ──

const F64 = new Float64Array(1);
const U32 = new Uint32Array(F64.buffer); // 小端: [低位字, 高位字]
const FD_HALF = [0.5, -0.5];
const FD_LN2HI = [6.9314718036912381649e-1, -6.9314718036912381649e-1]; // 0x3fe62e42 fee00000
const FD_LN2LO = [1.90821492927058770002e-10, -1.90821492927058770002e-10]; // 0x3dea39ef 35793c76
const FD_INVLN2 = 1.442695040888963387;
const FD_P1 = 1.66666666666666019037e-1;
const FD_P2 = -2.77777777770155933842e-3;
const FD_P3 = 6.61375632143793436117e-5;
const FD_P4 = -1.6533902205465251539e-6;
const FD_P5 = 4.13813679705723846039e-8;
const FD_HUGE = 1.0e300;
const FD_TWOM1000 = 9.3326361850321887899e-302; // 2**-1000
const FD_O_THRESHOLD = 7.09782712893383973096e2;
const FD_U_THRESHOLD = -7.4513321910194110842e2;

/**
 * fdlibm 的 __ieee754_exp: 只用 IEEE 加减乘除和 2 的整数次幂缩放, 在任何平台上都是同一个结果 ——
 * 和 Python 的 xregion.portable_exp 逐位相同 (test:golden 的 exp 用例), 也和 V8 的 Math.exp 相同
 * (V8 本来就是 fdlibm, 只对 x = 1 特判返回 Math.E —— 这里不跟), 但不依赖浏览器: Safari 的 Math.exp 是系统 libm。
 */
export function exp(x: number): number {
  F64[0] = x;
  let hx = U32[1]! >>> 0;
  const lx = U32[0]! >>> 0;
  const xsb = (hx >>> 31) & 1;
  hx = (hx & 0x7fffffff) >>> 0;
  if (hx >= 0x40862e42) {
    // |x| >= 709.78...
    if (hx >= 0x7ff00000) {
      if (((hx & 0xfffff) | lx) !== 0) return x + x; // NaN
      return xsb === 0 ? x : 0.0; // exp(±inf)
    }
    if (x > FD_O_THRESHOLD) return Infinity;
    if (x < FD_U_THRESHOLD) return 0.0;
  }
  let k = 0;
  let hi = 0;
  let lo = 0;
  if (hx > 0x3fd62e42) {
    // |x| > 0.5 ln2: 约简到 [-0.5 ln2, 0.5 ln2]
    if (hx < 0x3ff0a2b2) {
      hi = x - FD_LN2HI[xsb]!;
      lo = FD_LN2LO[xsb]!;
      k = 1 - xsb - xsb;
    } else {
      k = Math.trunc(FD_INVLN2 * x + FD_HALF[xsb]!); // 向零取整, 同 C 的 (int)
      const t = k;
      hi = x - t * FD_LN2HI[0]!;
      lo = t * FD_LN2LO[0]!;
    }
    x = hi - lo;
  } else if (hx < 0x3e300000) {
    // |x| < 2**-28
    if (FD_HUGE + x > 1) return 1 + x;
  }
  const t = x * x;
  const c = x - t * (FD_P1 + t * (FD_P2 + t * (FD_P3 + t * (FD_P4 + t * FD_P5))));
  if (k === 0) return 1 - ((x * c) / (c - 2.0) - x);
  const y = 1 - (lo - (x * c) / (2.0 - c) - hi);
  // fdlibm 直接往 y 的指数位上加 k —— 乘 2 的整数次幂是精确的, 结果一样; k = 1024 拆两步, 同 Python
  if (k >= -1021) return k < 1024 ? y * 2 ** k : y * 2 * 2 ** 1023;
  return y * 2 ** (k + 1000) * FD_TWOM1000;
}

// ── 常数 (xregion.py 同名) ──

/** Elo 刻度: 差 S 分 = logit 差 1 (状态里的 params.S 是同一个数, golden.mjs 核对) */
export const S = 400 / Math.log(10);
const LN2 = Math.log(2);
export const STALE_HOME_DAYS = 365;
/** 同一系列赛相邻两局在引擎里差 1 秒 */
export const REPLAY_GAP_S = 1;
/** 去重窗口: OE 那局的开打时间最早比赛程 startTime 早 6 小时, 最晚晚 18 小时 (跨 UTC 零点也在窗里) */
export const DEDUP_BEFORE_S = 6 * 3600;
export const DEDUP_AFTER_S = 18 * 3600;

/**
 * xregion.pairwise_mean: float(np.mean(xs)) 的逐位等价写法 —— 8 个以上的数用 8 路分块的成对求和,
 * 超过 128 个再对半递归。顺序累加在 8 个以上时差最后一两位。
 */
export function pairwiseMean(xs: number[]): number {
  const a = xs.map(Number);
  if (!a.length) throw new Error("空列表没有均值");
  const psum = (lo: number, n: number): number => {
    if (n < 8) {
      let s = 0;
      for (let i = lo; i < lo + n; i++) s += a[i]!;
      return s;
    }
    if (n <= 128) {
      const r = a.slice(lo, lo + 8);
      let i = 8;
      while (i < n - (n % 8)) {
        for (let j = 0; j < 8; j++) r[j]! += a[lo + i + j]!;
        i += 8;
      }
      // ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7)), 和 numpy / xregion.py 同一个结合顺序
      let s = ((r[0]! + r[1]!) + (r[2]! + r[3]!)) + ((r[4]! + r[5]!) + (r[6]! + r[7]!));
      while (i < n) {
        s += a[lo + i]!;
        i++;
      }
      return s;
    }
    let n2 = Math.floor(n / 2);
    n2 -= n2 % 8;
    return psum(lo, n2) + psum(lo + n2, n - n2);
  };
  return psum(0, a.length) / a.length;
}

// ── 时间 ──

/**
 * xregion.epoch_s: ISO 时间 → 1970 起的整秒 (向下取整)。**必须带时区**: 不带时区的串在 JS 里按本地时间解析,
 * Python 按 UTC —— 两边会差几个小时, 所以和 Python 一样直接拒绝 (抛错, 看板因此不给 C)。
 */
export function epochS(iso: string): number {
  const s = String(iso);
  if (!/(?:[zZ]|[+-]\d{2}:?\d{2})$/.test(s)) throw new Error(`时间必须带时区: '${s}'`);
  const ms = Date.parse(s);
  if (Number.isNaN(ms)) throw new Error(`解析不了的时间: '${s}'`);
  return Math.floor(ms / 1000);
}

/** xregion.game_t: 系列赛第 n 局的引擎时间 = (startTime 整秒 + (n − 1) × REPLAY_GAP_S) / 86400 (天) */
export function gameT(start: string, n: number): number {
  return (epochS(start) + (Math.trunc(Number(n)) - 1) * REPLAY_GAP_S) / 86400;
}

// ── 引擎 ──

/** Engine.play 的一局: OE 队名, t = 引擎时间 (天), t 不得递减 */
export interface ReplayGame {
  t: number;
  blue: string;
  red: string;
  blue_win: number;
  [k: string]: unknown;
}

/**
 * xregion.Engine 里看板要的子集: fromState → play (重放 OE 截止之后的跨赛区局) → predict。
 * 常规联赛 / 杯赛那几条更新路径 (pregame 的 ec = 0 / 1) 只在 Python 走整表时用, 这里没有 ——
 * 重放只喂国际赛, 而国际赛里只有跨赛区局更新 (同母赛区的局 intl_same = false, 母赛区查不到的局不出数也不更新)。
 */
export class Engine {
  readonly r = new Map<string, number>();
  readonly pool = new Map<string, string>();
  readonly ng = new Map<string, number>();
  readonly last = new Map<string, number>();
  readonly o = new Map<string, number>();
  readonly oprior = new Map<string, number>();
  readonly olast = new Map<string, number>();
  readonly migr = new Map<string, string[]>();
  private readonly h: number;
  private readonly eta: number;
  private readonly kint: number;
  private readonly hl: number;
  private readonly dnm: number;
  private readonly majors: Set<string>;

  private constructor(p: XregionState["params"]) {
    this.h = Number(p.h);
    this.eta = Number(p.eta);
    this.kint = Number(p.kint);
    this.hl = Number(p.hl);
    this.dnm = Number(p.dnm);
    this.majors = new Set(p.majors);
  }

  /** Engine.from_state: 只读 state, 不改它 */
  static fromState(state: Pick<XregionState, "params" | "teams" | "leagues" | "migr">): Engine {
    const e = new Engine(state.params);
    for (const [team, v] of Object.entries(state.teams)) {
      e.pool.set(team, v.pool);
      e.r.set(team, Number(v.r));
      e.ng.set(team, Math.trunc(Number(v.ng)));
      e.last.set(team, Number(v.last_regular_t));
    }
    for (const [L, v] of Object.entries(state.leagues)) {
      if (v.has_crossregion_history) {
        e.o.set(L, Number(v.o));
        e.oprior.set(L, Number(v.prior));
        e.olast.set(L, Number(v.last_t));
      }
    }
    for (const [L, src] of Object.entries(state.migr ?? {})) e.migr.set(L, [...src]);
    return e;
  }

  /**
   * Engine.off: 赛区偏移 —— 懒初始化 (四大 0, 其余 −dnm, 迁入队伍原赛区偏移的均值优先) + 向先验的指数衰减,
   * 并把 olast 推到 t (会改状态)。t 早于 olast 时不动。
   */
  off(L: string, t: number): number {
    const o = this.o;
    if (!o.has(L)) {
      const src = (this.migr.get(L) ?? []).filter((x) => o.has(x)).map((x) => o.get(x)!);
      const pri = src.length ? pairwiseMean(src) : this.majors.has(L) ? 0 : -this.dnm;
      o.set(L, pri);
      this.oprior.set(L, pri);
      this.olast.set(L, t);
    } else if (this.hl !== Infinity) {
      const dt = t - this.olast.get(L)!;
      if (dt > 0) {
        const pr = this.oprior.get(L)!;
        o.set(L, pr + (o.get(L)! - pr) * exp((-LN2 * dt) / this.hl));
        this.olast.set(L, t);
      }
    }
    return o.get(L)!;
  }

  /** Engine.home: 母赛区 = Elo 池; 最后一场常规联赛早于 t 超过 365 天就是 null (服务时不按阵容认队) */
  home(team: string, t: number): string | null {
    const L = this.pool.get(team);
    const last = this.last.get(team);
    if (L === undefined || last === undefined || t - last > STALE_HOME_DAYS) return null;
    return L;
  }

  /**
   * Engine.play: 按顺序重放已打完的国际赛局, 返回实际更新了评分的局数。t 递减直接抛错 (不悄悄重排)。
   * t 相同的连续几局是一组: 先全部取赛前值, 再按给定顺序统一更新。
   */
  play(games: ReplayGame[]): number {
    const ts = games.map((g) => Number(g.t));
    for (let k = 1; k < ts.length; k++) {
      if (ts[k]! < ts[k - 1]!) throw new Error("重放的局必须按时间排好 (t 不递减)");
    }
    let nUpd = 0;
    let i = 0;
    while (i < games.length) {
      let j = i;
      while (j < games.length && ts[j] === ts[i]) j++;
      const upd: { k: number; b: string; rd: string; pe: number; hb: string; hr: string }[] = [];
      for (let k = i; k < j; k++) {
        const g = games[k]!;
        const b = g.blue;
        const rd = g.red;
        const t = ts[k]!;
        const hb = this.home(b, t);
        const hr = this.home(rd, t);
        if (hb === null || hr === null || hb === hr) continue; // 同母赛区 / 查不到: 不出数也不更新
        const ob = this.off(hb, t);
        const orr = this.off(hr, t);
        const vb = this.r.get(b)!;
        const vr = this.r.get(rd)!;
        const pe = 1 / (1 + exp(-(ob + vb + this.h - orr - vr) / S));
        upd.push({ k, b, rd, pe, hb, hr });
      }
      for (const u of upd) {
        const d = Number(games[u.k]!.blue_win) - u.pe;
        this.o.set(u.hb, this.o.get(u.hb)! + this.eta * d);
        this.o.set(u.hr, this.o.get(u.hr)! - this.eta * d);
        if (this.kint) {
          this.r.set(u.b, this.r.get(u.b)! + this.kint * d);
          this.r.set(u.rd, this.r.get(u.rd)! - this.kint * d);
        }
        nUpd++;
      }
      i = j;
    }
    return nUpd;
  }

  /** Engine.features: t 时刻的 [o_蓝, o_红, r_蓝, r_红]; 两队母赛区都查得到且不同才有。会把 o 衰减到 t */
  features(blue: string, red: string, t: number): [number, number, number, number] | null {
    const hb = this.home(blue, t);
    const hr = this.home(red, t);
    if (hb === null || hr === null || hb === hr) return null;
    const ob = this.off(hb, t);
    const orr = this.off(hr, t);
    return [ob, orr, this.r.get(blue)!, this.r.get(red)!];
  }

  predict(coef: XregionCoef, blue: string, red: string, t: number): number | null {
    const f = this.features(blue, red, t);
    return f === null ? null : logistic(coef, f[0], f[1], f[2], f[3]);
  }
}

/** xregion.logistic: p = σ(a + b_o·Δo/S + b_r·Δr/S), 运算顺序同 Python */
export function logistic(coef: XregionCoef, ob: number, orr: number, rb: number, rr: number): number {
  const xo = (ob - orr) / S;
  const xr = (rb - rr) / S;
  const z = coef.a + coef.b_o * xo + coef.b_r * xr;
  return 1 / (1 + exp(-z));
}

/** 这个 lolesports 赛区键的系数 (按 xregion.py 的口径转大写); 没有就是 null */
export function coefficientsFor(state: Pick<XregionState, "coefficients">, leagueKey: unknown): XregionCoef | null {
  const k = String(leagueKey).toUpperCase();
  return own(state.coefficients, k) ? state.coefficients[k]! : null;
}

/** xregion.predict: 导出的状态 + 重放 (Engine.play 的格式) → t 时刻蓝方胜率; 没有系数 / 母赛区不明 / 同母赛区 → null */
export function predict(
  state: XregionState,
  leagueKey: unknown,
  blue: string,
  red: string,
  t: number,
  games: ReplayGame[] = [],
): number | null {
  const coef = coefficientsFor(state, leagueKey);
  if (coef === null) return null;
  const eng = Engine.fromState(state);
  eng.play(games);
  return eng.predict(coef, blue, red, t);
}

// ══════════════════════════════════════════════════════════
//  看板: 赛事内重放 (规则见 xregion.py 文件末尾那一段的注释)
// ══════════════════════════════════════════════════════════

/** 终局帧 (Feed.finalFrame / esports_feed.final_frame) */
export interface FinalFrame {
  blue_id: string | null;
  red_id: string | null;
  towers: number[];
  inhibitors: number[];
  gold: number[];
}

export interface SeriesTeam {
  name: string | null;
  code?: string | null;
  /** OE 队名 (map_team 到 C 的已知表); 映射不出来是 null */
  oe: string | null;
  wins: number | null;
  /** esportsTeamId (和终局帧的蓝红对照用); 不知道是 null */
  id: string | null;
}

/** 一场 lolesports 比赛 (系列赛) —— board_predict 的输入 */
export interface Series {
  match_id: string;
  /** 赛程 startTime, 必须带时区 */
  start: string;
  best_of: number | null;
  teams: SeriesTeam[];
  /** "<局号>" → 终局帧; 只有 replayFramesNeeded 列出的局才需要 */
  frames?: Record<string, FinalFrame | null>;
}

export function seriesWins(s: Series): number[] {
  return s.teams.map((t) => Math.trunc(Number(t.wins || 0)));
}

/** 按赛制算是否已分出胜负; best_of 不知道就当没分出 */
export function seriesClinched(s: Series): boolean {
  const bo = s.best_of;
  return !!bo && Math.max(...seriesWins(s)) >= Math.floor(Math.trunc(Number(bo)) / 2) + 1;
}

/** OE 里已有的局: "<局号>" → [胜方下标, 蓝方下标] (xregion.oe_lookup 的结果; Python 那边键是整数) */
export type Known = Record<string, readonly [number, number]>;

const knownNumbers = (known: Known | null | undefined): number[] =>
  Object.keys(known ?? {})
    .map((k) => Math.trunc(Number(k)))
    .sort((a, b) => a - b);

/**
 * 解码要终局帧的局号: 有一方 0 胜 (BO1、横扫) 不要; 分出胜负的系列赛最后一局归胜者, 也不要;
 * OE 里已有的局 (known 的键) 按 OE 的胜负, 也不要
 */
export function framesNeeded(s: Series, known?: Known | null): number[] {
  const wins = seriesWins(s);
  const n = wins.reduce((a, b) => a + b, 0);
  if (Math.min(...wins) === 0) return [];
  const last = seriesClinched(s) ? n : null;
  const kn = new Set(knownNumbers(known));
  const out: number[] = [];
  for (let i = 1; i <= n; i++) if (i !== last && !kn.has(i)) out.push(i);
  return out;
}

/**
 * xregion.frame_verdict: 终局帧 → [胜方下标, 蓝方下标, [|塔差|, |水晶差|, |经济差|]] 或 null (下标指 teams 的顺序)。
 * 塔多的赢; 塔平看水晶; 再平看总经济; 三样都平 → null。蓝方是哪队只看帧的 esportsTeamId, 必须正好是这两队。
 */
export function frameVerdict(
  frame: FinalFrame | null | undefined | Record<string, unknown>,
  ids: (string | null | undefined)[],
): [number, number, [number, number, number]] | null {
  if (!frame || typeof frame !== "object" || !Object.keys(frame).length || !ids[0] || !ids[1]) return null;
  const f = frame as Record<string, unknown>;
  const bid = f.blue_id;
  const rid = f.red_id;
  let blue: number;
  if (bid === ids[0] && rid === ids[1]) blue = 0;
  else if (bid === ids[1] && rid === ids[0]) blue = 1;
  else return null;
  const keys = ["towers", "inhibitors", "gold"] as const;
  if (keys.some((k) => !Array.isArray(f[k]) || (f[k] as unknown[]).length !== 2)) return null;
  const d = keys.map((k) => {
    const v = f[k] as unknown[];
    return Math.trunc(Number(v[0])) - Math.trunc(Number(v[1]));
  });
  const lead = d.find((x) => x !== 0) ?? 0;
  if (lead === 0) return null;
  return [lead > 0 ? blue : 1 - blue, blue, [Math.abs(d[0]!), Math.abs(d[1]!), Math.abs(d[2]!)]];
}

export interface Decoded {
  number: number;
  /** teams 的下标 */
  winner: number;
  blue: number;
  how: "oe" | "score" | "frames" | "alternate";
}

/**
 * xregion.decode_series: 系列赛已完成的 n 局 (n = 比分之和) 各自的胜方, 局号 1..n。
 * OE 里已有的局 (known) 按 OE, 先从比分里扣掉 → 比分定的 (BO1 / 横扫 / 决胜局归胜者) → 终局帧 + 比分约束
 * (把握最大的先定, 配额用完的改判) → 有一局没有终局帧就把剩下的局整体退回交替序 (领先方先赢, 平分时 teams[0] 先)。
 * 每队胜局数一定等于比分; OE 的胜负和比分对不上就抛错 (看板因此不给 C)。
 */
export function decodeSeries(s: Series, known?: Known | null): Decoded[] {
  const wins = seriesWins(s);
  const n = wins.reduce((a, b) => a + b, 0);
  if (n === 0) return [];
  const out = new Map<number, [number, number, Decoded["how"]]>();
  const quota = [...wins];
  for (const i of knownNumbers(known)) {
    if (i < 1 || i > n) continue;
    const [w, bl] = known![String(i)]!;
    out.set(i, [Math.trunc(Number(w)), Math.trunc(Number(bl)), "oe"]);
    quota[Math.trunc(Number(w))]!--;
  }
  if (Math.min(...quota) < 0) throw new Error(`系列赛 ${s.match_id}: OE 里已有的局和比分 [${wins.join(", ")}] 对不上`);
  const result = () =>
    Array.from({ length: n }, (_, k) => {
      const [winner, blue, how] = out.get(k + 1)!;
      return { number: k + 1, winner, blue, how };
    });
  if (Math.min(...wins) === 0) {
    const w = wins[0] ? 0 : 1;
    for (let i = 1; i <= n; i++) if (!out.has(i)) out.set(i, [w, 0, "score"]);
    return result();
  }
  if (seriesClinched(s) && !out.has(n)) {
    const w = wins[0]! > wins[1]! ? 0 : 1;
    if (quota[w] === 0) throw new Error(`系列赛 ${s.match_id}: OE 里胜者在决胜局之前就赢满了 (比分 [${wins.join(", ")}])`);
    out.set(n, [w, 0, "score"]);
    quota[w]!--;
  }
  const need = framesNeeded(s, known);
  const frames = s.frames || {};
  const ids = s.teams.map((t) => t.id);
  const ver = new Map(need.map((i) => [i, frameVerdict(frames[String(i)], ids)] as const));
  if ([...ver.values()].every((v) => v !== null)) {
    const k = (i: number) => ver.get(i)![2];
    const order = [...need].sort((a, b) => {
      const ka = k(a);
      const kb = k(b);
      return kb[0] - ka[0] || kb[1] - ka[1] || kb[2] - ka[2] || a - b;
    });
    for (const i of order) {
      const [w0, blue] = ver.get(i)!;
      let w = w0;
      if (quota[w] === 0) w = 1 - w;
      quota[w]!--;
      out.set(i, [w, blue, "frames"]);
    }
  } else {
    let pref = wins[0]! >= wins[1]! ? 0 : 1;
    for (const i of need) {
      const w = quota[pref]! > 0 ? pref : 1 - pref;
      quota[w]!--;
      const v = ver.get(i);
      out.set(i, [w, v ? v[1] : 0, "alternate"]);
      pref = 1 - pref;
    }
  }
  return result();
}

export interface OeIndex {
  /** "队名\u0000局号" → [[对手队名, 开打时间整秒, 这一方赢了, 这一方是蓝方], …] */
  by: Map<string, [string, number, boolean, boolean][]>;
  /** 状态里的现役队名 */
  teams: Set<string>;
  /** 别名表 {旧名: 现名} (teams.json 的 aliases = esports_feed.TEAM_ALIASES): 只用来认 OE 用过期旧名记的局 */
  aliases: Record<string, string>;
}

/** xregion.oe_index: OE 国际赛局的去重索引 (不分赛事; 每局两个队名各记一条) */
export function oeIndex(
  state: Pick<XregionState, "events"> & { teams?: Record<string, unknown> },
  aliases?: Record<string, string> | null,
): OeIndex {
  const by = new Map<string, [string, number, boolean, boolean][]>();
  const add = (k: string, v: [string, number, boolean, boolean]) => {
    const arr = by.get(k);
    if (arr) arr.push(v);
    else by.set(k, [v]);
  };
  for (const ev of Object.values(state.events)) {
    for (const [b, r, g, date, y] of ev.games) {
      if (g === null || g === undefined) continue;
      const x = epochS(date);
      const n = Math.trunc(Number(g));
      const bw = Math.trunc(Number(y)) === 1;
      add(`${b}\u0000${n}`, [r, x, bw, true]);
      add(`${r}\u0000${n}`, [b, x, !bw, false]);
    }
  }
  return { by, teams: new Set(Object.keys(state.teams ?? {})), aliases: { ...(aliases ?? {}) } };
}

/**
 * xregion.oe_lookup: 这一局 OE 里有没有 (= 已经计入状态); 有就返回 OE 的 [胜方, 蓝方] (下标 0 = a, 1 = b), 没有是 null。
 * 同一局号; OE 的开打时间落在 [startTime − 6 小时, startTime + 18 小时]; 两队对得上 (不论蓝红) —— 至少一方队名
 * 相同, 另一方要么也相同, 要么 OE 那边是过期旧名 (不是现役队) 且别名表把它指向另一方 (2026 EWC 上 OE 用旧名
 * "Team Secret" 记了 Team Secret Whales)。窗口里命中几条时取离 startTime 最近的 (一样近取先出现的)。
 */
export function oeLookup(idx: OeIndex, a: string, b: string, number: number, start: string): [number, number] | null {
  const s = epochS(start);
  const lo = s - DEDUP_BEFORE_S;
  const hi = s + DEDUP_AFTER_S;
  const n = Math.trunc(Number(number));
  let best: [number, number, number] | null = null;
  const pairs = [
    [a, b],
    [b, a],
  ] as const;
  pairs.forEach(([me, other], k) => {
    for (const [opp, x, won, blue] of idx.by.get(`${me}\u0000${n}`) ?? []) {
      if (
        lo <= x &&
        x <= hi &&
        (opp === other || (!idx.teams.has(opp) && own(idx.aliases, opp) && idx.aliases[opp] === other))
      ) {
        const d = Math.abs(x - s);
        if (best === null || d < best[0]) best = [d, won ? k : 1 - k, blue ? k : 1 - k];
      }
    }
  });
  const hit = best as [number, number, number] | null;
  return hit === null ? null : [hit[1], hit[2]];
}

/** xregion.in_oe: oeLookup 命中与否 */
export function inOe(idx: OeIndex, a: string, b: string, number: number, start: string): boolean {
  return oeLookup(idx, a, b, number, start) !== null;
}

/** xregion._replay_selection: [[系列赛下标, 要重放的局号…, OE 已有局的胜负]] 和计数 —— 还没取终局帧、还没解码 */
function replaySelection(
  state: XregionState,
  series: Series[],
  currentId: string,
  number: number,
  aliases?: Record<string, string> | null,
): { sel: [number, number[], Known][]; deduped: number; unmapped: number; score_lag: boolean } {
  const cur = series.find((s) => s.match_id === currentId);
  if (!cur) throw new Error(`系列赛列表里没有当前比赛 ${currentId}`);
  const cs = epochS(cur.start);
  const idx = oeIndex(state, aliases);
  const sel: [number, number[], Known][] = [];
  let deduped = 0;
  let unmapped = 0;
  let lag = false;
  series.forEach((s, k) => {
    if (s.teams.length !== 2) return;
    const n = seriesWins(s).reduce((x, y) => x + y, 0);
    let hi: number;
    if (s.match_id === currentId) {
      lag = lag || Math.trunc(Number(number)) - 1 > n;
      hi = Math.min(n, Math.trunc(Number(number)) - 1);
    } else if (epochS(s.start) < cs) hi = n;
    else return;
    if (hi <= 0) return;
    const a = s.teams[0]!.oe;
    const b = s.teams[1]!.oe;
    if (!a || !b) {
      unmapped += hi; // 映射不出 OE 队名: 引擎里也没有母赛区, 本来就不会更新
      return;
    }
    const known: Record<string, [number, number]> = {};
    for (let i = 1; i <= n; i++) {
      const hit = oeLookup(idx, a, b, i, s.start);
      if (hit !== null) known[String(i)] = hit;
    }
    const keep: number[] = [];
    for (let i = 1; i <= hi; i++) if (!own(known, String(i))) keep.push(i);
    deduped += hi - keep.length;
    if (keep.length) sel.push([k, keep, known]);
  });
  return { sel, deduped, unmapped, score_lag: lag };
}

/**
 * xregion.replay_frames_needed: 要去取终局帧的局 [[match_id, 局号]]。只看有局要重放的系列赛 (整个系列赛都在
 * OE 里就一个请求都不发); 一个系列赛只要有一局要重放, 就要它全部的非定局 (比分约束是对整个系列赛的), OE 里
 * 已有的局除外。当前系列赛比分落后 (score_lag, 反正不给 C) 时一个都不取。
 */
export function replayFramesNeeded(
  state: XregionState,
  series: Series[],
  currentId: string,
  number: number,
  aliases?: Record<string, string> | null,
): [string, number][] {
  const { sel, score_lag } = replaySelection(state, series, currentId, number, aliases);
  if (score_lag) return [];
  return sel.flatMap(([k, , known]) =>
    framesNeeded(series[k]!, known).map((i) => [series[k]!.match_id, i] as [string, number]),
  );
}

export interface ReplayedGame extends ReplayGame {
  match_id: string;
  number: number;
  how: Decoded["how"];
}

/** xregion.board_replay: 要重放的局 (已解码、已排序, Engine.play 的输入) 和计数 */
export function boardReplay(
  state: XregionState,
  series: Series[],
  currentId: string,
  number: number,
  aliases?: Record<string, string> | null,
): { games: ReplayedGame[]; deduped: number; unmapped: number; score_lag: boolean } {
  const { sel, deduped, unmapped, score_lag } = replaySelection(state, series, currentId, number, aliases);
  const rows: { g: ReplayedGame; start: number; mid: string }[] = [];
  for (const [k, keep, known] of sel) {
    const s = series[k]!;
    const dec = new Map(decodeSeries(s, known).map((d) => [d.number, d]));
    const names = [s.teams[0]!.oe!, s.teams[1]!.oe!];
    const start = epochS(s.start);
    for (const i of keep) {
      const d = dec.get(i)!;
      rows.push({
        g: {
          t: gameT(s.start, i),
          blue: names[d.blue]!,
          red: names[1 - d.blue]!,
          blue_win: d.winner === d.blue ? 1 : 0,
          match_id: s.match_id,
          number: i,
          how: d.how,
        },
        start,
        mid: String(s.match_id),
      });
    }
  }
  // 排序键: t, 开赛时间, match_id, 局号 —— 不用输入列表里的位置: 同一时刻开打的两个系列赛, 组内的更新顺序
  // (浮点累加的顺序) 不能随上游返回的顺序变
  rows.sort(
    (x, y) =>
      x.g.t - y.g.t ||
      x.start - y.start ||
      (x.mid < y.mid ? -1 : x.mid > y.mid ? 1 : 0) ||
      x.g.number - y.g.number,
  );
  return { games: rows.map((r) => r.g), deduped, unmapped, score_lag };
}

export interface BoardPredictResult {
  probability_blue: number | null;
  t: number;
  home_blue: string | null;
  home_red: string | null;
  no_history: string[];
  /** "score_lag" = 当前系列赛的比分还没跟上局号, 不给数; 否则 null */
  withheld: "score_lag" | null;
  /** 这一局还没有帧 (sidesKnown = false): 给的是两种选边的平均 */
  side_neutral: boolean;
  replayed: number;
  updated: number;
  deduped: number;
  unmapped: number;
  games: ReplayedGame[];
}

/**
 * xregion.board_predict: 导出的状态 + 本赛事重放 → 第 number 局开打时的蓝方胜率。blue / red 是 OE 队名。
 * probability_blue 为 null = 不给 (没有这个赛区键的系数、母赛区查不到或过期、两队同母赛区、withheld)。
 * sidesKnown = false (这一局还没有帧, 队序只是赛程顺序): (p(A 蓝) + 1 − p(B 蓝)) / 2 —— 截距 a 是蓝方优势,
 * 按任意的队序当蓝方会让数挪 5-7 个百分点。no_history = 两队母赛区里在状态中没有跨赛区国际赛记录的。
 */
export function boardPredict(
  state: XregionState,
  leagueKey: unknown,
  blue: string,
  red: string,
  series: Series[],
  currentId: string,
  number: number,
  aliases?: Record<string, string> | null,
  sidesKnown = true,
): BoardPredictResult {
  const rep = boardReplay(state, series, currentId, number, aliases);
  const cur = series.find((s) => s.match_id === currentId)!;
  const t = gameT(cur.start, number);
  const eng = Engine.fromState(state);
  const nUpd = eng.play(rep.games);
  const hb = eng.home(blue, t);
  const hr = eng.home(red, t);
  const coef = coefficientsFor(state, leagueKey);
  const withheld = rep.score_lag ? "score_lag" : null;
  let p: number | null = null;
  const f = coef !== null && withheld === null ? eng.features(blue, red, t) : null;
  if (f !== null) {
    const [ob, orr, rb, rr] = f;
    p = logistic(coef!, ob, orr, rb, rr);
    if (!sidesKnown) p = (p + (1 - logistic(coef!, orr, ob, rr, rb))) / 2;
  }
  const noHist = [hb, hr].filter(
    (L): L is string => L !== null && !(own(state.leagues, L) && state.leagues[L]!.has_crossregion_history),
  );
  return {
    probability_blue: p,
    t,
    home_blue: hb,
    home_red: hr,
    no_history: noHist,
    withheld,
    side_neutral: !sidesKnown,
    replayed: rep.games.length,
    updated: nUpd,
    deduped: rep.deduped,
    unmapped: rep.unmapped,
    games: rep.games,
  };
}

/** C 的已知队伍表 (map_team 的 known): 状态里的全部 OE 队名, 文件顺序 (已按 Python 的 sorted 排好, 不要重排) */
const knownCache = new WeakMap<object, string[]>();
export function xregionKnown(state: XregionState): string[] {
  let k = knownCache.get(state);
  if (!k) knownCache.set(state, (k = Object.keys(state.teams)));
  return k;
}
