"""
后期快照采集器
================
Oracle's Elixir 只提供 at10/15/20/25 四个时间点的快照。实测 T=25 外推到
后期的代价 (research/gate_late_game.py):

    25-30 分钟  96.7%      30-35 分钟  83.7%
    35-40 分钟  65.1%      40+  分钟  52.7%   ← 低于基线, 比瞎猜还差

40+ 分钟的比赛占 9.7%, 35+ 占 28.2%。要训练 T=30/35/40 的切片就得有那些
时刻的快照, 而唯一能提供的是 lolesports 的实时流 —— 只能边打边采。

本脚本每次运行采一轮:
    找出正在打的局 -> 取当前帧 -> 追加一条快照到 collected/{gameId}.jsonl
    局结束后, 用系列赛比分的增量判断这一局谁赢, 写 {gameId}.result.json

设计取舍
--------
· **只追加, 不改写**。每局一个文件, 中途崩了最多丢最后一条。
· **胜负靠 gameWins 增量**推断: 帧里没有"谁赢了"这个字段, getEventDetails
  的 games[] 也没有。但系列赛比分会在局结束后 +1, 对比前后即可。
· **幂等**: 同一时刻重复采不会写重复行 (按 rfc460Timestamp 去重)。
· 采集失败绝不影响别的东西 —— 它是纯旁路。

用法:
    python collect_live.py            采一轮
    python collect_live.py --status   看已经攒了多少
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
from esports_feed import EsportsFeed, FeedError   # noqa: E402

OUT = _HERE / "collected"
STATE = OUT / "_series_state.json"


def log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


def load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_state(s: dict):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")


def seen_stamps(path: Path) -> set:
    """已经写过的帧时刻, 用来去重。"""
    if not path.exists():
        return set()
    out = set()
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            try:
                out.add(json.loads(line).get("t"))
            except Exception:
                continue
    return out


def snapshot(feed: EsportsFeed, game_id: str, meta: dict) -> int:
    """采一条当前快照。返回新写入的条数 (0 或 1)。

    minute 要精确: 这一条是要进训练集的, 而 minute 就是它的 T。帧的时间戳
    是真实世界时刻, 暂停期间照走 —— 不扣的话, 一局停过 7 分钟的比赛会往
    训练集里塞一批"标着 T=25 其实是 T=18"的样本, 且没有任何报错。
    采集是每两分钟一轮的定时任务, 多付一次扫描无所谓。
    """
    st = feed.window(game_id, precise_minute=True)
    if not st or st.minute is None:
        return 0
    path = OUT / f"{game_id}.jsonl"
    if st.frame_time in seen_stamps(path):
        return 0

    b = (st.teams or {}).get("blue", {})
    r = (st.teams or {}).get("red", {})
    row = {
        "t": st.frame_time, "minute": st.minute, "state": st.game_state,
        "golddiff": st.golddiff, "csdiff": st.csdiff,
        "blue_kills": st.blue_kills, "red_kills": st.red_kills,
        "blue_gold": b.get("gold"), "red_gold": r.get("gold"),
        "blue_towers": b.get("towers"), "red_towers": r.get("towers"),
        "blue_barons": b.get("barons"), "red_barons": r.get("barons"),
        "blue_dragons": b.get("dragons"), "red_dragons": r.get("dragons"),
        "blue_dragon_types": b.get("dragon_types"),
        "red_dragon_types": r.get("dragon_types"),
        "blue_inhibitors": b.get("inhibitors"), "red_inhibitors": r.get("inhibitors"),
        "blue_cs": b.get("cs"), "red_cs": r.get("cs"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 元数据只写一次
    mpath = OUT / f"{game_id}.meta.json"
    if not mpath.exists() and meta:
        mpath.write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                         encoding="utf-8")
    return 1


def collect(feed: EsportsFeed) -> dict:
    state = load_state()
    stats = {"matches": 0, "games": 0, "rows": 0, "finished": 0}

    try:
        live = feed.live()
    except Exception as e:
        log(f"取直播列表失败: {e}")
        return stats

    # 直播列表可能漏, 补上最近开赛的
    try:
        cands = []
        for lg in ("LCK", "LPL", "LEC", "LCS"):
            cands += feed.schedule(lg)
        seen = {m.match_id for m in live}
        for m in feed.live_by_frames(cands):
            if m.match_id not in seen:
                live.append(m)
    except Exception as e:
        log(f"补漏失败(不影响): {e}")

    for m in live:
        stats["matches"] += 1
        wins = {t.api_name: (t.game_wins or 0) for t in m.teams}
        prev = state.get(m.match_id, {})
        prev_wins = prev.get("wins") or {}

        try:
            games = feed.games(m.match_id)
        except Exception as e:
            log(f"  {m.match_id} 取对局失败: {e}")
            continue

        for g in games:
            gid = g.get("id")
            if not gid:
                continue
            try:
                st = feed.window(gid)
            except FeedError:
                continue
            if not st:
                continue

            # **判据是"这一局在打", 不是 st.live。**
            #
            # st.live 里含着一道判死逻辑: 帧超过 LIVE_STALE_SEC (600 秒)
            # 没更新就翻成 False。那道逻辑是给"挑当前局"用的, 用在这里是错的
            # —— 延迟大不等于帧无效, 那些数字是好的, 只是旧。
            #
            # 实测 2026-09-04 起 LPL 的帧按真实时间的 ~79% 发布 (采样区间
            # 31%-164%, 抖得厉害), 延迟在一局之内线性累积, 打到十几分钟就
            # 越过 600 秒。于是 09-04 至 09-06 每一局 LPL 的采集都停在第
            # 12-17 分钟, 而同期 LCK 采到第 30-41 分钟 —— **每局后半段全丢**,
            # 而 20 分钟以后正是 Stage 4 信息量最大的一段。
            #
            # 帧本身带 frame_time, 旧不旧读的人自己看得见; 这里的职责是
            # "别把有效数据扔了"。
            if st.game_state == "in_game":  # 正在打 -> 采一条
                # 蓝红必须按**这一局**的选边算, 不能用系列赛的队伍顺序 ——
                # BO 里每局换边, 实测 7 局里有 2 局被标反 (NAVI/MKOI g2、
                # EDG/TES g2)。蓝方是模型特征, 标反了整局的队伍级特征都会
                # 张冠李戴。id -> 名字 靠 match_teams(), 和 games() 同一个
                # URL, 走缓存不多花请求。
                # 选边以**帧**的 gameMetadata 为准, 不是 persisted 的
                # games[].teams[].side。这一行的 golddiff/blue_gold/blue_cs
                # 全部来自 feed.window() —— 也就是帧的分组。队名跟另一个源排,
                # 这条训练样本就是"A 队的数字配 B 队的名字"。
                # 两个源约 8% 的比赛对不上 (2026-08-20 抽查 13 场系列赛)。
                pers = {t.get("side"): str(t.get("id"))
                        for t in (g.get("teams") or []) if t.get("id")}
                names = {str(t.get("id")): t.get("name")
                         for t in feed.match_teams(m.match_id)}
                fs = {}
                try:
                    fs = feed.frame_sides(gid)
                except Exception:
                    pass
                blue = names.get(fs.get("blue"))
                red = names.get(fs.get("red"))
                src = "frames"
                if not (blue and red):      # 帧里换不出来 -> 退回 persisted
                    blue = names.get(pers.get("blue"))
                    red = names.get(pers.get("red"))
                    src = "per_game"
                if not (blue and red):      # 再换不出来才退回旧口径, 并标明
                    blue = next((t.api_name for t in m.teams), None)
                    red = next((t.api_name for t in reversed(m.teams)), None)
                    src = "series_order"
                # 两源不一致要记下来, 不能悄悄过去 —— 事后审计全靠这个字段
                conflict = bool(fs.get("blue") and pers.get("blue")
                                and fs["blue"] != pers["blue"])
                meta = {
                    "match_id": m.match_id, "game_id": gid,
                    "game_number": g.get("number"), "league": m.league,
                    "best_of": m.best_of,
                    "blue": blue, "red": red,
                    "side_source": src,
                    "side_conflict": conflict,
                    "sides": pers,
                    "frame_sides": fs,
                    "players": feed.game_metadata(gid),
                    "collected_from": datetime.now(timezone.utc).isoformat(),
                }
                n = snapshot(feed, gid, meta)
                stats["rows"] += n
                if n:
                    stats["games"] += 1
                    log(f"  {m.league} {meta['blue']} vs {meta['red']} "
                        f"g{g.get('number')} 第 {st.minute} 分钟 -> +1")

            elif st.game_state == "finished":
                # 结束了且还没判过胜负 -> 用系列赛比分增量判断
                rpath = OUT / f"{gid}.result.json"
                if rpath.exists() or not (OUT / f"{gid}.jsonl").exists():
                    continue
                winner = None
                for name, w in wins.items():
                    if w > prev_wins.get(name, 0):
                        winner = name
                        break
                if winner:
                    # 局长也要扣暂停。上面那个 st 是热循环里取的 (不扫暂停,
                    # 因为要对每一局都探一次), 这里一局只走一次, 付得起。
                    fin = feed.window(gid, precise_minute=True) or st
                    rpath.write_text(json.dumps({
                        "game_id": gid, "winner": winner,
                        "final_minute": fin.minute,
                        "wins_before": prev_wins, "wins_after": wins,
                        "decided_at": datetime.now(timezone.utc).isoformat(),
                    }, ensure_ascii=False, indent=1), encoding="utf-8")
                    stats["finished"] += 1
                    log(f"  g{g.get('number')} 结束, 胜者 {winner} "
                        f"(第 {fin.minute} 分钟)")

        state[m.match_id] = {"wins": wins,
                             "seen_at": datetime.now(timezone.utc).isoformat()}

    save_state(state)
    return stats


def reconcile(feed: EsportsFeed) -> int:
    """赛后对账: 把当时没抓到的胜负补回来。

    胜负本来只靠"系列赛比分 +1 的那一刻"来推 —— 采集器要正好在那两分钟里
    跑过, 而且前后两次都看到才行。实测 6 局只判出 3 局, 等于把本来就慢的
    积累速度又砍一半, 而这些局的快照已经采到手了, 丢标签就是白采。

    API 里没有逐局胜者字段 (getEventDetails 的 games[] 只有 state 和选边,
    result.gameWins 是系列赛总分)。但有两类情况可以**精确**推出来, 不是猜:

      · 横扫: 最终 2-0 / 3-0, 那么每一局都归同一队
      · 只剩一局未知: 总分减去已知的, 剩下那个就是它

    合起来能覆盖大多数漏掉的局。推不出来的 (比如 2-1 里两局都没抓到)
    就留着不动 —— 宁可少标, 不能标错。
    """
    filled = 0
    for mpath in sorted(OUT.glob("*.meta.json")):
        try:
            meta = json.loads(mpath.read_text(encoding="utf-8"))
        except Exception:
            continue
        mid = meta.get("match_id")
        if not mid or mid == "test":
            continue
        try:
            games = feed.games(mid)
        except Exception:
            continue
        # 系列赛没结束就别急着对账, 比分还会变
        if not games or any(g.get("state") not in ("completed", "unneeded")
                            for g in games):
            continue

        # 最终比分。schedule 里带 result.gameWins
        finals = {}
        for m in feed.schedule(meta.get("league") or ""):
            if m.match_id == mid:
                finals = {t.api_name: (t.game_wins or 0) for t in m.teams}
                break
        if not finals or sum(finals.values()) == 0:
            continue

        played = [g for g in games if g.get("state") == "completed"]
        known, unknown = {}, []
        for g in played:
            gid = g.get("id")
            if not gid or not (OUT / f"{gid}.jsonl").exists():
                continue          # 这一局没采到快照, 补了也没用
            rp = OUT / f"{gid}.result.json"
            if rp.exists():
                try:
                    known[gid] = json.loads(rp.read_text(encoding="utf-8"))["winner"]
                except Exception:
                    unknown.append(gid)
            else:
                unknown.append(gid)
        if not unknown:
            continue

        # 剩余待分配的胜场 = 最终比分 - 已经判过的
        rest = dict(finals)
        for w in known.values():
            if w in rest:
                rest[w] -= 1
        if any(v < 0 for v in rest.values()):
            log(f"  {mid} 比分对不上 (已知 {known} vs 最终 {finals}), 跳过")
            continue

        # 注意: unknown 只含"采到快照的局"; 没采到的局也占掉了胜场,
        # 所以只有在剩余胜场数正好等于 unknown 数时才能安全分配。
        pos = [t for t, v in rest.items() if v > 0]
        if sum(rest.values()) != len(unknown):
            continue
        if len(pos) == 1:                       # 横扫, 或剩下的全归一队
            winners = {gid: pos[0] for gid in unknown}
        elif len(unknown) == 1 and len(pos) == 1:
            winners = {unknown[0]: pos[0]}
        else:
            continue                            # 分不清是哪局归哪队, 不猜

        for gid, w in winners.items():
            (OUT / f"{gid}.result.json").write_text(json.dumps({
                "game_id": gid, "winner": w,
                "wins_final": finals, "known_before": known,
                "source": "reconcile",          # 区别于实时增量判定
                "decided_at": datetime.now(timezone.utc).isoformat(),
            }, ensure_ascii=False, indent=1), encoding="utf-8")
            filled += 1
            log(f"  对账补上 {gid} -> {w}")
    return filled


def fix_sides(feed: EsportsFeed) -> int:
    """把已经采下来的 meta 里标反的蓝红改正。

    改过两次, 两次的靶子不一样:

    一、早先用**系列赛队伍顺序**当蓝红, 而 BO 里每局换边 —— 实测 7 局错 2 局。
    二、然后改成用 persisted 的 games[].teams[].side, 仍然不对: 快照行里的
        golddiff/blue_gold/blue_cs 全部取自帧的 blueTeam/redTeam, 而这两个
        官方源约 8% 的比赛互相矛盾 (2026-08-20 抽查 13 场系列赛, 1 场不一致)。

    现在以**帧**为准。理由不是帧更权威, 而是内部一致性: 数值按哪个分组算出来,
    队名就得按哪个分组排。否则这条训练样本就是 A 队的数字配 B 队的名字。
    """
    fixed = 0
    for mpath in sorted(OUT.glob("*.meta.json")):
        try:
            meta = json.loads(mpath.read_text(encoding="utf-8"))
        except Exception:
            continue
        gid, mid = meta.get("game_id"), meta.get("match_id")
        if not gid or not mid or mid == "test":
            continue
        try:
            fs = feed.frame_sides(gid)
            names = {str(t.get("id")): t.get("name")
                     for t in feed.match_teams(mid)}
        except Exception:
            continue
        blue, red = names.get(fs.get("blue")), names.get(fs.get("red"))
        if not (blue and red):
            continue

        pers = meta.get("sides") or {}
        meta["frame_sides"] = fs
        meta["side_conflict"] = bool(
            pers.get("blue") and fs.get("blue")
            and str(pers["blue"]) != fs["blue"])
        if meta.get("blue") == blue and meta.get("red") == red:
            meta["side_source"] = "frames"        # 本来就是对的, 补个标记
            mpath.write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                             encoding="utf-8")
            continue
        log(f"  修正 {gid} g{meta.get('game_number')}: "
            f"蓝 {meta.get('blue')} -> {blue}"
            + ("  [两个官方源不一致]" if meta["side_conflict"] else ""))
        meta["blue"], meta["red"] = blue, red
        meta["side_source"] = "frames"
        meta["side_fixed_at"] = datetime.now(timezone.utc).isoformat()
        mpath.write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                         encoding="utf-8")
        fixed += 1
    return fixed


def status():
    if not OUT.exists():
        print("还没有采集到任何数据"); return
    games = sorted(OUT.glob("*.jsonl"))
    done = sorted(OUT.glob("*.result.json"))
    print(f"已采集 {len(games)} 局, 其中 {len(done)} 局已判定胜负\n")
    late = tot = 0
    for f in games:
        mins = []
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    mins.append(json.loads(line).get("minute") or 0)
                except Exception:
                    pass
        if not mins:
            continue
        tot += 1
        if max(mins) >= 30:
            late += 1
    print(f"  有 30 分钟后快照的: {late}/{tot} 局")
    print(f"  总快照数: {sum(1 for f in games for _ in open(f, encoding='utf-8'))}")
    print(f"\n目录: {OUT}")
    print("训练 T=30/35/40 需要几百局带胜负的后期样本 —— 一天十几场, 要攒几个月。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true", help="看已攒了多少")
    ap.add_argument("--reconcile", action="store_true",
                    help="只跑赛后对账, 不采集")
    ap.add_argument("--fix-sides", action="store_true",
                    help="修正历史 meta 里标反的蓝红 (一次性)")
    a = ap.parse_args()
    if a.status:
        status(); return 0
    if a.fix_sides:
        n = fix_sides(EsportsFeed())
        log(f"修正 {n} 局的蓝红")
        return 0
    if a.reconcile:
        n = reconcile(EsportsFeed())
        log(f"对账补上 {n} 局")
        return 0
    feed = EsportsFeed()
    log("采集一轮")
    s = collect(feed)
    # 每轮末尾对一次账 —— 只读已完赛的比赛, 都带缓存, 代价很小
    try:
        n = reconcile(feed)
    except Exception as e:
        n = 0
        log(f"对账失败(不影响采集): {type(e).__name__}: {e}")
    log(f"直播 {s['matches']} 场 · 新快照 {s['rows']} 条 · "
        f"判定胜负 {s['finished']} 局 · 对账补 {n} 局")
    return 0


if __name__ == "__main__":
    sys.exit(main())
