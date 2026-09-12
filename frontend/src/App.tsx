import { useCallback, useEffect, useState } from "react";
import { api, ApiError, prefetchBoard } from "./api/client";
import type { Match } from "./api/types";
import { MatchList } from "./components/MatchList";
import { Board } from "./components/Board";
import { ThemeToggle } from "./components/ThemeToggle";
import { T } from "./i18n";

const LIST_POLL_MS = 30_000;

/** 当前看的是哪场 —— 放进 URL, 刷新不丢, 链接可分享 */
function readMatchFromUrl(): string | null {
  return new URLSearchParams(window.location.search).get("match");
}

export default function App() {
  const [live, setLive] = useState<Match[]>([]);
  const [upcoming, setUpcoming] = useState<Match[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [openId, setOpenId] = useState<string | null>(readMatchFromUrl);
  const [openMatch, setOpenMatch] = useState<Match | null>(null);

  // 预热正在进行的比赛。
  //
  // 观赛板的冷路径要 5–11 秒, 全花在 window(precise_minute=True) 的暂停扫描上
  // (~33 个上游请求 x 548–681ms 往返, 已经是 8 线程并行)。那是到
  // feed.lolesports.com 的固有延迟, 挤不出来 —— 但用户读列表的这几秒本来就
  // 是空的, 拿来预热正好。命中之后点开只要 ~50ms。
  //
  // 只预热**可预测**的场次; 上限 3 场是为了不给那个公共 API key 添无谓的负担。
  //
  // 这里**不再按 match.state 过滤**: live 数组本身就是 /esports/live 用帧判过
  // 的结果, 再拿赛程的 state 筛一遍只会把它筛坏 —— 实测 2026-09-04 LPL
  // LGD vs AL 正打到第 3 局, state 却是 completed, 于是**最该预热的那场反而
  // 被排除**, 点开要等十几秒。state 过期时无声无息, 这个 filter 也就无声无息
  // 地失效。
  useEffect(() => {
    if (openId) return;              // 已经在看某一场了, 别再抢带宽
    live
      .filter((m) => m.predictable)
      .slice(0, 3)
      .forEach((m) => prefetchBoard(m.match_id));
  }, [live, openId]);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const [l, u] = await Promise.all([api.live(), api.upcoming(24, 5)]);
      const liveIds = new Set(l.matches.map((m) => m.match_id));
      setLive(l.matches);
      // 正在进行的不要在"即将开赛"里重复出现
      setUpcoming(u.matches.filter((m) => !liveIds.has(m.match_id)));
      setError(null);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  // 看板打开时不轮询列表 —— 免得和 board 抢上游配额
  useEffect(() => {
    if (openId) return;
    const t = window.setInterval(() => void load(), LIST_POLL_MS);
    return () => clearInterval(t);
  }, [openId, load]);

  // 回到这个标签页时立刻刷新一次。
  // 浏览器会限制后台标签页的定时器 (Chrome 降到每分钟一次, 挂久了直接冻结),
  // 所以切走再切回来时列表很可能停在几分钟前 —— 光靠 setInterval 补不回来。
  useEffect(() => {
    if (openId) return;
    const wake = () => {
      if (document.visibilityState === "visible") void load();
    };
    document.addEventListener("visibilitychange", wake);
    window.addEventListener("focus", wake);
    return () => {
      document.removeEventListener("visibilitychange", wake);
      window.removeEventListener("focus", wake);
    };
  }, [openId, load]);

  // 浏览器前进/后退
  useEffect(() => {
    const onPop = () => setOpenId(readMatchFromUrl());
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  const openBoard = (m: Match) => {
    setOpenMatch(m);
    setOpenId(m.match_id);
    history.pushState(null, "", `?match=${encodeURIComponent(m.match_id)}`);
  };

  const back = () => {
    setOpenId(null);
    setOpenMatch(null);
    history.pushState(null, "", window.location.pathname);
  };

  if (openId) {
    return (
      <div className="wrap">
        <Board matchId={openId} initial={openMatch} onBack={back} />
      </div>
    );
  }

  return (
    <div className="wrap">
      <div className="topbar rise">
        <span className="spacer" />
        <ThemeToggle />
      </div>
      <header className="masthead rise">
        <h1>{T("胜率预测")}</h1>
        <p>{T("四大赛区比赛的实时胜率预测。")}</p>
      </header>
      <MatchList
        title={T("正在进行")}
        matches={live}
        kind="live"
        loading={loading}
        error={error}
        emptyText={T("当前没有进行中的比赛")}
        onOpen={openBoard}
        onRefresh={() => void load()}
      />
      <MatchList
        title={T("即将开始")}
        matches={upcoming}
        kind="upcoming"
        loading={loading}
        error={null}
        emptyText={T("近期没有排期")}
        onOpen={openBoard}
        onRefresh={() => void load()}
      />
    </div>
  );
}
