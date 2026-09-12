"""延迟到底来自哪里?

我们自己在 _lagged() 里减了 60 秒。问题是: 这 60 秒是必需的吗?
扫一遍不同的 lag, 看 window 端点从哪一档开始有数据, 以及拿到的最新帧
距离"现在"有多远。

只有正在直播的局才说明问题 —— 已结束的局怎么退都有数据。
"""
import sys, time
from datetime import datetime, timezone, timedelta
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from esports_feed import EsportsFeed

f = EsportsFeed()


def find_live():
    # current_game() 返回 (game_dict, LiveState), 不是单个 dict
    try:
        for m in f.live():
            cg = f.current_game(m.match_id)
            if cg:
                return m, cg[0].get("id")
    except Exception as e:
        print("live() 失败:", e)
    cands = []
    for lg in ("LPL", "LCK", "LEC", "LCS"):
        try:
            cands += f.schedule(lg)
        except Exception:
            pass
    for m in f.live_by_frames(cands):
        cg = f.current_game(m.match_id)
        if cg:
            return m, cg[0].get("id")
    return None, None


m, gid = find_live()
if not gid:
    print("现在没有正在进行的局 —— 这个测量必须在直播时做。")
    sys.exit(0)

print(f"用: {m.league} {' vs '.join(t.api_name for t in m.teams)}  game={gid}\n")
print(f"{'请求lag':>8} {'帧数':>5} {'最新帧距now':>12} {'gameState':>10}")
print("-" * 42)

best = None
for lag in (0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 90):
    now = datetime.now(timezone.utc)
    at = f._lagged(lag)
    p = f._window(gid, at, ttl=1)
    frames = (p or {}).get("frames") or []
    if not frames:
        print(f"{lag:>7}s {0:>5} {'—':>12} {'—':>10}")
        continue
    newest = f._ts(frames[-1])
    behind = (now - newest).total_seconds()
    state = frames[-1].get("gameState", "")
    print(f"{lag:>7}s {len(frames):>5} {behind:>11.1f}s {state:>10}")
    if best is None:
        best = (lag, behind)
    time.sleep(0.4)

print()
if best:
    print(f"最小可用 lag = {best[0]}s, 此时最新帧仍比现在旧 {best[1]:.1f} 秒")
    print(f"我们代码里写死的是 60s")
    print(f"-> 其中 {best[1]:.1f}s 是上游自带的(转播延迟, 动不了),"
          f" {60 - best[0]}s 是我们多退的")
