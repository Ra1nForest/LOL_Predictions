/**
 * 静态站的"后端" —— 在浏览器里代替 api.py。
 *
 * 只有 `vite build --mode pages` 构建出来的版本 (GitHub Pages) 会加载这个模块;
 * 服务器版照旧请求 api.py, 包里也不带它 (client.ts 里是动态 import)。
 * 输出和服务器同形, 组件感知不到差别。
 *
 * 下载量: 列表页只要 15 KB 的队伍表 (队名映射要用); 三个模型 (局内、赛前、BP 后, 后者
 * 带选手-英雄表) 第一次打开看板或赛前页时才下载, 之后整个会话复用。
 *
 * 数据直连 lolesports: 两个主机都回 Access-Control-Allow-Origin: *, 流量摊在每个访客
 * 自己的 IP 上 —— 不像服务器版那样, 所有人的请求都从一台机器、一把 key 发出去。
 */
import type {
  BoardResponse,
  MatchListResponse,
  PredictRequestBody,
  PredictResponse,
  TimelinePoint,
} from "../api/types.ts";
import type { Models } from "./board.ts";
import { buildBoard, fineTimeline, liveList, upcomingList } from "./board.ts";
import type { ExplainFile } from "./explain.ts";
import { Feed } from "./feed.ts";
import type { IngameModelFile } from "./ingame.ts";
import { loadIngame } from "./ingame.ts";
import type { Stage1File } from "./stage1.ts";
import { loadStage1, predictResponse } from "./stage1.ts";
import type { Stage2File } from "./stage2.ts";
import { loadStage2 } from "./stage2.ts";
import type { TeamsFile } from "./teams.ts";
import { localToday } from "./teams.ts";
import { EN } from "../i18n.ts";
import type { ViewCtx } from "./player.ts";
import { LivePlayer, liveView } from "./player.ts";

async function getJSON<T>(name: string): Promise<T> {
  const r = await fetch(`${import.meta.env.BASE_URL}web/${name}`);
  if (!r.ok) throw new Error(`模型文件 ${name} 下载失败: HTTP ${r.status}`);
  return (await r.json()) as T;
}

// 失败了要能重试, 所以出错时把缓存的 Promise 清掉 —— 否则一次网络抖动会让整个会话都坏着
let teamsP: Promise<TeamsFile> | null = null;
let modelsP: Promise<Models> | null = null;
let feedP: Promise<Feed> | null = null;
/** 逐秒走势补到了哪: 按 game_id, end 是已补到的帧时刻 (毫秒), done = 这局打完且补齐了 */
const fineState = new Map<string, { pts: TimelinePoint[]; end: number; done: boolean }>();
const fineBusy = new Set<string>();
/** 逐秒走势的胜率记忆, 按 game_id (见 board.fineTimeline) */
const fineMemo = new Map<string, Map<string, { p: unknown } | null>>();

function teams(): Promise<TeamsFile> {
  teamsP ??= getJSON<TeamsFile>("teams.json").catch((e) => {
    teamsP = null;
    throw e;
  });
  return teamsP;
}

function models(): Promise<Models> {
  modelsP ??= Promise.all([
    getJSON<IngameModelFile>("ingame_live.json"),
    getJSON<Stage1File>("stage1_pre.json"),
    getJSON<ExplainFile>("explain.json"),
    getJSON<Stage2File>("stage2_post.json"),
  ])
    .then(([ig, s1, ex, s2]) => ({ ingame: loadIngame(ig), s1: loadStage1(s1), s2: loadStage2(s2), ex }))
    .catch((e) => {
      modelsP = null;
      throw e;
    });
  return modelsP;
}

/** 整个会话一个 Feed —— 自适应 lag、开局零点、暂停观测这些状态要跨轮询保留 */
function feed(): Promise<Feed> {
  feedP ??= teams()
    .then((t) => new Feed({ teams: t, runeLocale: EN ? "en_US" : "zh_CN" }))
    .catch((e) => {
      feedP = null;
      throw e;
    });
  return feedP;
}

export const staticApi = {
  async live(): Promise<MatchListResponse> {
    return (await liveList(await feed(), await teams())) as unknown as MatchListResponse;
  },

  async upcoming(limit: number, pastHours: number): Promise<MatchListResponse> {
    return (await upcomingList(await feed(), await teams(), limit, pastHours)) as unknown as MatchListResponse;
  },

  async board(matchId: string, gameId: string | null, points: number, curves: boolean): Promise<BoardResponse> {
    const [f, t, m] = await Promise.all([feed(), teams(), models()]);
    return (await buildBoard(f, t, m, localToday(), matchId, gameId, curves, points)) as unknown as BoardResponse;
  },

  /**
   * 直播回放 (见 player.ts): 每回放到一帧, 用它改写最新的看板再交给界面。
   * getBoard 取的是界面手里最新的一份看板 —— 轮询还在照常刷新它 (暂停、完整曲线靠它)。
   * 返回停止函数。
   */
  async play(
    gameId: string,
    getBoard: () => BoardResponse | null,
    onView: (v: BoardResponse) => void,
  ): Promise<() => void> {
    const [f, t, m] = await Promise.all([feed(), teams(), models()]);
    const [start, trinkets] = await Promise.all([f.gameStart(gameId), f.trinketIds()]);
    const ctx: ViewCtx = { models: m, teams: t, today: localToday(), start, trinkets, trail: [] };
    const player = new LivePlayer(f, gameId, (pf) => {
      const b = getBoard();
      if (!b?.live || b.live.game_id !== gameId || b.prediction?.probability_blue == null) return;
      try {
        onView(liveView(b, pf, ctx));
      } catch (e) {
        console.warn(`回放帧失败: ${e}`);
      }
    });
    void player.start();
    return () => player.stop();
  },

  /**
   * 逐秒走势 (见 board.fineTimeline)。直播中的局由 useFineTimeline 定时再调, 打完的局补齐后不再取
   * (局号标签来回切不重取)。同一局同时只跑一份。
   *
   * 每次都**整局重算**, 不做增量拼接: 暂停区间要等暂停过去一阵才认得出来, 认出来之前算的那批点
   * 局内时间是偏大的 —— 增量拼接会把它们永久留下, 之后正确的点反而被当成"比末尾还早"丢掉,
   * 曲线断一段。整局重算不贵: 窗口缓存 1 小时 (发布后不变), 只有新窗口真取; 胜率有 memo, 只算新点。
   * 第一次补到哪交到哪; 之后算完整条再交, 免得中途那一刻只剩最近几分钟。
   */
  async fine(board: BoardResponse, onPts: (pts: TimelinePoint[]) => void): Promise<void> {
    const lv = board.live;
    const m = board.match;
    const blue = m?.teams[0]?.model_name;
    const red = m?.teams[1]?.model_name;
    if (!lv || !m || !blue || !red || !lv.frame_time) return;
    const gid = lv.game_id;
    const upto = Date.parse(lv.frame_time);
    const st = fineState.get(gid);
    if (st && (st.done || upto - st.end < 5_000)) {
      onPts(st.pts);
      return;
    }
    if (fineBusy.has(gid)) return;
    fineBusy.add(gid);
    try {
      const [f, t, mo] = await Promise.all([feed(), teams(), models()]);
      let memo = fineMemo.get(gid);
      if (!memo) fineMemo.set(gid, (memo = new Map()));
      let all: TimelinePoint[] = [];
      await fineTimeline(
        f,
        t,
        mo,
        localToday(),
        { gameId: gid, blue, red, league: m.league, upto },
        (pts) => {
          all = pts as unknown as TimelinePoint[];
          if (!st) onPts(all);
        },
        memo,
      );
      if (st) onPts(all);
      fineState.set(gid, { pts: all, end: upto, done: lv.game_state === "finished" });
    } finally {
      fineBusy.delete(gid);
    }
  },

  async predict(body: PredictRequestBody): Promise<PredictResponse> {
    const [t, m] = await Promise.all([teams(), models()]);
    return predictResponse(m.s1, t, m.ex, body, localToday()) as unknown as PredictResponse;
  },
};

export type StaticApi = typeof staticApi;
