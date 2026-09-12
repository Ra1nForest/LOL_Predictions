# -*- coding: utf-8 -*-
"""上游到底提前多久发布帧 —— 只能在**直播进行中**跑。

背景: window 的窗口实测是以 startingTime 为中心的 (帧覆盖 T-6s..T+3s,
步长 0.2 秒), 所以 `_lagged(lag)` 里那个 lag 有多大, 画面就落后多少。
代码里的 60 秒是当年测出来"能稳定命中"的保守值, 不是上游的真实发布延迟。

esports_feed 现在会自己往下试探 (见 window() 的"自适应 lag"), 这个脚本
是用来**核对它收敛得对不对**的: 同一时刻把各档一起问一遍, 看最小可用档
在哪, 以及那一档拿回来的末帧实际有多旧。

    python tools/probe_feed_lag.py                # 自动找正在直播的那一局
    python tools/probe_feed_lag.py --game <id>    # 指定一局
    python tools/probe_feed_lag.py --rounds 5     # 多测几轮看稳不稳

没有直播时会直接说没有, 不会拿已结束的比赛凑数 —— 已结束的局任何更晚的
startingTime 都返回同一批冻结帧 (模块顶部第 6 条), 测出来的"延迟"是假的。
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import esports_feed as EF  # noqa: E402

LAGS = [10, 20, 30, 40, 50, 60, 90, 120, 150]


def find_live(feed) -> list[tuple[str, str]]:
    """(gameId, 描述)。以帧为准, 不信 schedule 的 state。"""
    out = []
    for lg in EF.LEAGUE_IDS:
        try:
            ms = feed.schedule(lg)
        except Exception as e:
            print(f"  [{lg}] 赛程拉取失败: {e}")
            continue
        for m in ms:
            need = ((m.best_of or 1) // 2) + 1
            if any((t.game_wins or 0) >= need for t in m.teams):
                continue
            try:
                games = feed.games(m.match_id)
            except Exception:
                continue
            for g in games:
                if g.get("state") != "inProgress":
                    continue
                names = " vs ".join(t.api_name for t in m.teams)
                out.append((g["id"], f"[{lg}] {names} 第 {g.get('number')} 局"))
    return out


def probe(feed, gid: str) -> None:
    now = datetime.now(timezone.utc)

    def one(lag):
        p = feed._window(gid, EF.EsportsFeed._lagged(lag, now), ttl=0)
        frames = (p or {}).get("frames") or []
        if not frames:
            return lag, None, None
        last = feed._ts(frames[-1])
        return lag, len(frames), (datetime.now(timezone.utc) - last).total_seconds()

    with ThreadPoolExecutor(max_workers=len(LAGS)) as ex:
        rows = list(ex.map(one, LAGS))

    hits = [r for r in rows if r[1]]
    print(f"  {now:%H:%M:%S}  ", end="")
    if not hits:
        print("所有档位都没有帧 (可能正在暂停或转播中断)")
        return
    for lag, n, age in rows:
        mark = "·" if n else "×"
        print(f"{mark}{lag}", end=" ")
    best = min(hits, key=lambda r: r[0])
    print(f"  -> 最小可用档 {best[0]}s, 该档末帧实际落后 {best[2]:.0f}s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--interval", type=float, default=12.0)
    a = ap.parse_args()

    feed = EF.EsportsFeed()
    if a.game:
        targets = [(a.game, "指定")]
    else:
        print("找正在直播的对局…")
        targets = find_live(feed)
        if not targets:
            print("没有正在进行的比赛 —— 这个脚本必须在直播时跑, "
                  "已结束的比赛测出来的延迟是假的 (返回的是冻结帧)。")
            return 1
    for gid, desc in targets:
        print(f"\n{desc}  game_id={gid}")
        print(f"  档位: {'  '.join(str(l) for l in LAGS)}   (· 有帧  × 空)")
        for i in range(a.rounds):
            probe(feed, gid)
            if i < a.rounds - 1:
                time.sleep(a.interval)
        print(f"  自适应当前停在: {feed._lag.get(gid, '未收敛')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
