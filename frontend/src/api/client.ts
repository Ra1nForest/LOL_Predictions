import type {
  BoardResponse,
  HealthResponse,
  MatchListResponse,
} from "./types";

export class ApiError extends Error {
  readonly status?: number;
  constructor(message: string, status?: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

/**
 * 静态版 (GitHub Pages) 开关。`vite build --mode pages` 会把它编译成常量 true, 这时
 * 所有数据改由浏览器自己算 (src/web/), 不请求任何服务器。
 *
 * 两种模式给出的数据同形, 组件不需要知道自己跑在哪种模式下 —— 只有静态版确实没有的
 * 功能 (AI 辩论) 要用它藏起来。
 */
export const IS_STATIC = import.meta.env.VITE_STATIC === "1";

/** 静态版的实现按需加载: 服务器版的包里完全不带它 */
const loadStatic = () => import("../web/static-api").then((m) => m.staticApi);
type StaticApi = Awaited<ReturnType<typeof loadStatic>>;

/** 静态版里抛出的错误统一包成 ApiError, 和服务器版的报错走同一条显示路径 */
async function viaStatic<T>(run: (s: StaticApi) => Promise<T>): Promise<T> {
  try {
    return await run(await loadStatic());
  } catch (e) {
    if (e instanceof ApiError) throw e;
    const status =
      e && typeof e === "object" && "status" in e && typeof (e as { status: unknown }).status === "number"
        ? (e as { status: number }).status
        : undefined;
    throw new ApiError(e instanceof Error ? e.message : String(e), status);
  }
}

/**
 * 带超时的 JSON 取数。
 *
 * /esports/board 会向上游 lolesports 扇出好几个请求, 冷路径实测能到十几秒,
 * 所以默认超时给得比一般前端宽。超时必须**主动中止**而不是干等 ——
 * 轮询场景下堆积的请求会把后端拖垮。
 */
async function fetchJSON<T>(url: string, timeoutMs = 90_000): Promise<T> {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), timeoutMs);
  try {
    const res = await fetch(url, { signal: ctl.signal });
    if (!res.ok) {
      let detail = `HTTP ${res.status}`;
      try {
        const body = (await res.json()) as { detail?: unknown };
        if (typeof body.detail === "string") detail = body.detail;
      } catch {
        /* 响应体不是 JSON, 用状态码就够了 */
      }
      throw new ApiError(detail, res.status);
    }
    return (await res.json()) as T;
  } catch (e) {
    if (e instanceof ApiError) throw e;
    if (e instanceof DOMException && e.name === "AbortError") {
      throw new ApiError("请求超时");
    }
    throw new ApiError(e instanceof Error ? e.message : String(e));
  } finally {
    clearTimeout(timer);
  }
}

export const api = {
  health: () => fetchJSON<HealthResponse>("/health", 20_000),

  live: () =>
    IS_STATIC ? viaStatic((s) => s.live()) : fetchJSON<MatchListResponse>("/esports/live", 60_000),

  upcoming: (limit = 24, pastHours = 5) =>
    IS_STATIC
      ? viaStatic((s) => s.upcoming(limit, pastHours))
      : fetchJSON<MatchListResponse>(
          `/esports/upcoming?limit=${limit}&past_hours=${pastHours}`,
          60_000,
        ),

  /**
   * 观赛板。
   *
   * `curves=false` 是**首屏用的**: 曲线要把整局按分钟逐个窗口拉一遍
   * (37 分钟的比赛 ≈ 32 个上游请求), 冷缓存下实测 5–14 秒; 不要曲线只要
   * 473ms。两者请求的是同一批 window URL 且共用服务端缓存, 所以先要头条、
   * 再要曲线**不会多付网络** —— 只是把等待挪到了用户已经有东西看之后。
   * 热缓存下整个板子 50ms 上下, 所以只有第一次点进某场比赛才有区别。
   */
  board: (
    matchId: string,
    gameId?: string | null,
    opts: { points?: number; curves?: boolean } = {},
  ) => {
    const { points = 60, curves = true } = opts;
    if (IS_STATIC) return viaStatic((s) => s.board(matchId, gameId ?? null, points, curves));
    const q = new URLSearchParams({ points: String(points) });
    if (gameId) q.set("game_id", gameId);
    if (!curves) q.set("curves", "false");
    return fetchJSON<BoardResponse>(`/esports/board/${matchId}?${q}`);
  },
};

/** 已经预取过的比赛, 避免重复打同一场。只增不减 —— 一次会话里预取一遍就够,
 *  服务端缓存本身有 TTL, 真过期了正常轮询会自己补上。 */
const warmed = new Set<string>();

/**
 * 预热某场比赛的缓存。
 *
 * 为什么需要: 冷路径的成本**不在曲线上, 也不在模型上**, 而在
 * `window(precise_minute=True)` 的暂停扫描 —— 它要把整局按分钟逐个窗口拉一遍
 * (~33 个上游请求), 而单次往返实测 548–681ms, 8 线程下就是 5–11 秒。
 * 那是到 feed.lolesports.com 的固有延迟, 挤不出来 (`_game_start` 已有 6 小时
 * 缓存, 扫描本身已经并行)。
 *
 * 所以不是让首次请求变快, 是让它在用户点击**之前**就发生: 用户读列表的那几秒
 * 正好够。命中之后板子只要 ~50ms。静态版同理, 只是缓存在浏览器自己的内存里。
 *
 * 刻意不返回 Promise、不抛错 —— 调用方不该为预取写 try/catch, 失败了正常
 * 打开时照常请求即可。
 */
export function prefetchBoard(matchId: string): void {
  if (warmed.has(matchId)) return;
  warmed.add(matchId);
  const run = IS_STATIC
    ? viaStatic((s) => s.board(matchId, null, 60, true))
    : fetchJSON(`/esports/board/${matchId}?points=60`);
  void run.catch(() => {
    // 预取失败没有后果, 但要允许再试一次
    warmed.delete(matchId);
  });
}

/** POST 一个 JSON, 拿回 JSON。 */
async function postJSON<T>(url: string, body: unknown, timeoutMs: number): Promise<T> {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), timeoutMs);
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: ctl.signal,
    });
    if (!res.ok) {
      let detail = `HTTP ${res.status}`;
      try {
        const b = (await res.json()) as { detail?: unknown };
        if (typeof b.detail === "string") detail = b.detail;
        else if (Array.isArray(b.detail)) detail = "请求格式不对: " + JSON.stringify(b.detail[0]);
      } catch {
        /* 不是 JSON 就用状态码 */
      }
      throw new ApiError(detail, res.status);
    }
    return (await res.json()) as T;
  } catch (e) {
    if (e instanceof ApiError) throw e;
    if (e instanceof DOMException && e.name === "AbortError") throw new ApiError("请求超时");
    throw new ApiError(e instanceof Error ? e.message : String(e));
  } finally {
    clearTimeout(timer);
  }
}

const NOT_IN_STATIC = (what: string) => Promise.reject(new ApiError(`静态版不提供${what}`));

export const mutations = {
  teams: (league?: string) =>
    IS_STATIC
      ? NOT_IN_STATIC("队伍列表")
      : fetchJSON<import("./types").TeamsResponse>(
          league ? `/teams?league=${encodeURIComponent(league)}` : "/teams",
          30_000,
        ),

  predict: (body: import("./types").PredictRequestBody) =>
    IS_STATIC
      ? viaStatic((s) => s.predict(body))
      : postJSON<import("./types").PredictResponse>("/predict", body, 120_000),

  /**
   * Stage 3。静态版没有 —— 它要调 NVIDIA 的 API, key 不能放进网页。
   *
   * 超时必须**大于后端自己的预算**, 否则前端先放弃, 那次 LLM 调用的钱白花
   * 而且用户看到的是"请求超时"而不是结论。debate.py 的总预算是 300 秒
   * (还带 VERDICT_RESERVE 给裁定留量), 实测带检索的一次跑了约 240 秒 ——
   * 之前这里恰好也写 240, 是擦着边过的。留足到 380。
   */
  debate: (body: import("./types").DebateRequestBody) =>
    IS_STATIC
      ? NOT_IN_STATIC("AI 辩论")
      : postJSON<import("./types").DebateResponse>("/debate", body, 380_000),
};
