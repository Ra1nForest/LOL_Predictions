/**
 * 静态站的"后端" —— 在浏览器里代替 api.py。
 *
 * 只有 `vite build --mode pages` 构建出来的版本 (GitHub Pages) 会加载这个模块;
 * 服务器版照旧请求 api.py, 包里也不带它 (client.ts 里是动态 import)。
 * 输出和服务器同形, 组件感知不到差别。
 *
 * 下载量: 列表页只要 15 KB 的队伍表 (队名映射要用); 两个模型 (~1.4 MB, gzip 后约
 * 450 KB) 第一次打开看板或赛前页时才下载, 之后整个会话复用。
 *
 * 数据直连 lolesports: 两个主机都回 Access-Control-Allow-Origin: *, 流量摊在每个访客
 * 自己的 IP 上 —— 不像服务器版那样, 所有人的请求都从一台机器、一把 key 发出去。
 */
import type { BoardResponse, MatchListResponse, PredictRequestBody, PredictResponse } from "../api/types.ts";
import type { Models } from "./board.ts";
import { buildBoard, liveList, upcomingList } from "./board.ts";
import type { ExplainFile } from "./explain.ts";
import { Feed } from "./feed.ts";
import type { IngameModelFile } from "./ingame.ts";
import { loadIngame } from "./ingame.ts";
import type { Stage1File } from "./stage1.ts";
import { loadStage1, predictResponse } from "./stage1.ts";
import type { TeamsFile } from "./teams.ts";
import { localToday } from "./teams.ts";

async function getJSON<T>(name: string): Promise<T> {
  const r = await fetch(`${import.meta.env.BASE_URL}web/${name}`);
  if (!r.ok) throw new Error(`模型文件 ${name} 下载失败: HTTP ${r.status}`);
  return (await r.json()) as T;
}

// 失败了要能重试, 所以出错时把缓存的 Promise 清掉 —— 否则一次网络抖动会让整个会话都坏着
let teamsP: Promise<TeamsFile> | null = null;
let modelsP: Promise<Models> | null = null;
let feedP: Promise<Feed> | null = null;

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
  ])
    .then(([ig, s1, ex]) => ({ ingame: loadIngame(ig), s1: loadStage1(s1), ex }))
    .catch((e) => {
      modelsP = null;
      throw e;
    });
  return modelsP;
}

/** 整个会话一个 Feed —— 自适应 lag、开局零点、暂停观测这些状态要跨轮询保留 */
function feed(): Promise<Feed> {
  feedP ??= teams()
    .then((t) => new Feed({ teams: t }))
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

  async predict(body: PredictRequestBody): Promise<PredictResponse> {
    const [t, m] = await Promise.all([teams(), models()]);
    return predictResponse(m.s1, t, m.ex, body, localToday()) as unknown as PredictResponse;
  },
};

export type StaticApi = typeof staticApi;
