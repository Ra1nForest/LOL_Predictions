import { useEffect, useRef, useState } from "react";
import { IS_STATIC } from "../api/client";
import type { BoardResponse, TimelinePoint } from "../api/types";
import { localize } from "../i18n";

/** 服务器版构建里 IS_STATIC 是常量 false, 这一行连同回放器整个被摇掉 */
const loadStatic = IS_STATIC ? () => import("../web/static-api").then((m) => m.staticApi) : null;

/**
 * 直播回放 (web/player.ts): 比赛进行中时, 用逐帧回放改写看板, 让记分板、胜率、走势图末端
 * 每秒更新约 5 次, 而不是每次轮询跳一下。只在静态版里有; 其余时候原样返回 null,
 * 界面照常用轮询拿到的那份看板。
 */
export function useLivePlayback(data: BoardResponse | null): BoardResponse | null {
  const [view, setView] = useState<BoardResponse | null>(null);
  const ref = useRef(data);
  ref.current = data;

  const lv = data?.live;
  const pr = data?.prediction;
  const gid =
    loadStatic && lv && lv.game_state === "in_game" && !lv.stalled && pr?.probability_blue != null && pr.model
      ? lv.game_id
      : null;

  useEffect(() => {
    if (!gid || !loadStatic) {
      setView(null);
      return;
    }
    let alive = true;
    let stop: (() => void) | null = null;
    void loadStatic()
      .then((s) => s.play(gid, () => ref.current, (v) => alive && setView(localize(v))))
      .then((fn) => {
        if (alive) stop = fn;
        else fn();
      })
      .catch((e) => console.warn(`回放启动失败: ${e}`));
    return () => {
      alive = false;
      stop?.();
      setView(null);
    };
  }, [gid]);

  return gid && view?.live?.game_id === gid ? view : null;
}

/**
 * 逐秒的历史走势 (web/board.fineTimeline)。看板自带的走势是每分钟一个点; 这里在后台把整局
 * 每个 10 秒窗口都取回来, 每秒一个点, 从最近往前一批批补, 补到哪交到哪。
 * 直播和打完的局都做 —— 否则打开旧局是每分钟一个点、看直播是每秒一个点, 两种图不统一。
 * 只在静态版里有; 其余时候返回 null, 图表照常用看板的每分钟点。
 */
export function useFineTimeline(data: BoardResponse | null): TimelinePoint[] | null {
  const [fine, setFine] = useState<{ gid: string; pts: TimelinePoint[] } | null>(null);
  const ref = useRef(data);
  ref.current = data;
  const lv = data?.live;
  // 看板自己画得出曲线 (有 timeline) 才补 —— 画不出来的局 (没开到 3 分钟、队伍对不上) 补了也没处画
  const gid = loadStatic && lv && data?.match && data.timeline.length > 0 ? lv.game_id : null;

  useEffect(() => {
    if (!gid || !loadStatic) return;
    const load = loadStatic;
    let alive = true;
    const run = () => {
      const board = ref.current;
      if (!board || board.live?.game_id !== gid) return;
      void load()
        .then((s) => s.fine(board, (pts) => alive && setFine({ gid, pts })))
        .catch((e) => console.warn(`逐秒走势补全失败: ${e}`));
    };
    run();
    // 直播中的局要一直往后续补, 不能只在打开时补一次: 实时回放攒下的逐秒轨迹会断 ——
    // 暂停时回放停掉、轨迹丢了; 标签页在后台时浏览器把定时器降到每秒甚至每分钟一次, 回放跳着走,
    // 轨迹稀成一分钟一个点。缺的这些段由这里从数据源补回来: 每 30 秒一次, 切回标签页时立刻一次
    // (打完的局 fine() 直接交缓存, 不联网)
    const iv = setInterval(run, 30_000);
    const onVis = () => {
      if (document.visibilityState === "visible") run();
    };
    document.addEventListener("visibilitychange", onVis);
    return () => {
      alive = false;
      clearInterval(iv);
      document.removeEventListener("visibilitychange", onVis);
    };
  }, [gid]);

  return gid && fine?.gid === gid ? fine.pts : null;
}
