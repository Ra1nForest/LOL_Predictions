"""
回填后期快照 (T=30/35/40)
=========================

为什么需要
----------
Oracle's Elixir 只有 at10/15/20/25 四列, 30 分钟之后一个字段都没有。而一线
比赛 67% 会打过 30 分钟、30% 过 35 分钟 —— 也就是说模型在第 25 分钟之后
全是外推。research/gate_late_game.py 量过代价: 40+ 分钟只有 52.7%, 低于基线。

lolesports 的 window 接口对**已结束的比赛照样返回帧** (实测至少能追溯 94 天),
所以不用等直播慢慢攒, 可以直接往回扒。

怎么配对
--------
· **标签用 OE 的 result** —— API 里没有逐局胜者字段 (getEventDetails 的
  games[] 只有 state 和选边), 而 OE 每一行都有 result。这比靠系列赛比分
  增量去推可靠得多。
· **特征用 lolesports 的帧** —— 只有它有 25 分钟之后的数据。
· 两边靠 (赛区, 日期, 队名, 第几局) 对上。OE 的 gameid 是 Riot 赛事服的
  LOLTMNT05_xxxxx, 和 lolesports 的数字 id 不通用。

顺带做的事
----------
同时抓 T=25 并和 OE 的 golddiffat25 对照 —— 这是**训练-服务偏移**的直接
测量: 模型用 OE 的列训练, 线上喂的却是帧算出来的值, 两者有系统性偏差的话
线上预测会一直偏, 而离线评估永远看不见。

用法:
    python research/backfill_late.py --days 30 --limit 40      先试跑
    python research/backfill_late.py --days 95                 全量
    python research/backfill_late.py --report                  只看已攒的
"""
from __future__ import annotations
import argparse, json, sys, time
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
from esports_feed import EsportsFeed, FeedError, TEAM_ALIASES, LEAGUE_IDS
import sys

# Windows 计划任务里控制台是 GBK, 而本文件输出含 - (U+2212) / OK / WARN 这类
# GBK 编不了的字符 —— 不设这个会在 print 那一刻抛 UnicodeEncodeError, 表现为
# 任务"跑到一半消失"。2026-08-30 实测: LoL-Backfill 正好崩在 report() 的
# "帧 - OE 差" 那一行, 回填全做完了、报告打不出来, 整个任务算失败。
for _s in (sys.stdout, sys.stderr):
    if _s is not None and hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

# 回填要扫的联赛。**比 esports_feed.LEAGUE_IDS 宽** —— 那个只有四大赛区,
# 是给线上推理用的 (api.py 也只接受四大); 而模型从 2026-08-17 起就在用全部
# 联赛训练了 (research/gate_league_mix.py, t=+5.19, 五折全正), 所以回填也
# 该覆盖同样的范围, 否则训练集的两头补得不一致。
#
# 这些 id 来自 getLeagues, 且都在 Oracle's Elixir 里有数据 (能配上标签)。
# 不动 esports_feed.LEAGUE_IDS: 那会连带改变直播检测和赛程接口的行为。
EXTRA_LEAGUES = {
    "LCK Challengers": "98767991335774713",
    "NACL": "109511549831443335",
    "EMEA Masters": "100695891328981122",
    "LJL": "98767991349978712",
    "VCS": "107213827295848783",
    "PCS": "104366947889790212",
    "NLC": "105266098308571975",
    "TCL": "98767991343597634",
    "CBLOL": "98767991332355509",
    "LCP": "113476371197627891",
    "LLA": "101382741235120470",
    "LCO": "105709090213554609",
    "LRN": "110371976858004491",
    "LRS": "110372322609949919",
    "LES": "105266074488398661",
    "KeSPA Cup": "116929044967296666",
}
ALL_LEAGUES = {**LEAGUE_IDS, **EXTRA_LEAGUES}

# lolesports 的赛区名 -> Oracle's Elixir 的赛区名。只有对不上的才列。
# 名字对不上**不会报错**, 只会一局都配不上 —— 和队名那个坑同一类, 所以
# 同样显式列出、不做模糊匹配。
LEAGUE_TO_OE = {
    "LCK Challengers": "LCKC",
    "EMEA Masters": "EM",
}
# LTA North / South 故意不收: oe_index() 已经把 "LTA N" 并进 LCS, 再从 API
# 单独扒一遍只会和已有的 LCS 数据重复; LTA South 也只有 220 局, 不值得为它
# 引入第二套映射 —— 映射错了不报错, 只会静默配错标签。

OUT = _ROOT / "backfill"
ROWS = OUT / "snapshots.jsonl"
SEEN = OUT / "_done.json"
SLICES = [10, 15, 20, 25, 30, 35, 40, 45]
YEARS = (2025, 2026)

# 采集上界要在 OE 的局长之外多留这么多分钟, 给暂停用。
#
# 两个口径必须分清: gold_timeline 的 upto 是**墙钟**时刻, 而 OE 的
# gamelength 是**游戏内**时长, 两者差着这一局暂停了多久。按 glen+1 去截,
# 暂停过的局后段就整段采不到 —— 一局真打 41 分钟、中间停了 7 分钟, 帧要到
# 开局后 48 分钟才走完, 而 upto 停在 42 分钟。丢掉的恰好是回填最想要的
# T>25 那一段。
#
# 30 分钟是个够宽的余量 (停这么久的比赛通常已经改判重赛), 真正的过滤交给
# harvest 里的 mi > upto_min —— 那里的 mi 已经是扣掉暂停的游戏内分钟,
# 和 upto_min 同口径。多打的窗口有限: 赛后每个窗口返回的都是同一批冻结帧,
# 按时间戳去重后不产生任何多余样本。
PAUSE_MARGIN_MIN = 30


def log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


def oe_index() -> pd.DataFrame:
    """OE 的队伍行, 建成 (赛区, 日期, 队名, 局号) -> 结果 的查找表。"""
    dfs = []
    for y in YEARS:
        p = _ROOT / "data" / f"{y}_LoL_esports_match_data_from_OraclesElixir.csv"
        if p.exists():
            dfs.append(pd.read_csv(p, low_memory=False, usecols=[
                "gameid", "league", "date", "position", "side", "teamname",
                "game", "result", "gamelength", "golddiffat25"]))
    d = pd.concat(dfs, ignore_index=True)
    d = d[d.position == "team"].copy()
    d["league"] = d["league"].replace({"LTA N": "LCS"})
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"])
    # 把对手队名并到蓝方那一行上 —— 只按蓝方配对会在同一天有两场比赛时配错,
    # 必须两队都对上才算数。
    opp = (d[d.side == "Red"][["gameid", "teamname"]]
           .rename(columns={"teamname": "red_name"}))
    d = d[d.side == "Blue"].merge(opp, on="gameid", how="left")
    return d


def find_oe(oe: pd.DataFrame, league: str, when: datetime,
            blue: str, red: str, number: int, window_h: float = 12):
    """按 赛区 + **两队** + 局号 找 OE 行, 日期取最近的一条。

    不能用日期精确匹配: 赛程里的 startTime 是转播时间, 实测和开局差过
    67 分钟, 跨天的比赛还会差一整天。所以取最近且在 12 小时内的。

    **两队都要对上**: 只按蓝方 + 局号 + 最近日期, 同一天有两场比赛时会
    配到另一场去 —— 标签张冠李戴, 而且完全不报错。
    """
    c = oe[(oe.league == league) & (oe.teamname == blue)
           & (oe.red_name == red) & (oe.game == number)]
    if c.empty:
        return None
    gap = (c.date - when.replace(tzinfo=None)).abs()
    if gap.min() > timedelta(hours=window_h):
        return None
    return c.loc[gap.idxmin()]


def harvest(feed: EsportsFeed, gid: str, start: datetime, upto_min: float):
    """整局逐分钟取快照。返回 [{minute, ...}, ...]。

    为什么不按固定切片取
    --------------------
    局内模型里 **T (第几分钟) 本来就是一个特征**, 不是四个独立模型 —— 卡住
    它的只是训练数据只存在于 OE 给的 10/15/20/25 四个点。而帧是每分钟都有的,
    所以直接逐分钟采, T 就成了真正连续的变量, 第 5 分钟到终局中间不留断点。

    而且这样更省: gold_timeline() 本来就是逐分钟并行取整局 (画曲线用的就是
    它), 一次调用顶原来一个切片一次的 8 次请求。

    只收模型真正要的字段 (IngameState 就这几个)。塔/龙/大龙帧里也有, 但模型
    不吃, 为"以后也许用得上"多打几倍请求不划算 —— 要用的时候再回来采。
    """
    upto = start + timedelta(minutes=upto_min + PAUSE_MARGIN_MIN)
    rows = feed.gold_timeline(gid, upto=upto, step_sec=60)
    out = []
    for r in rows:
        mi = r.get("minute")
        # 5 分钟以前局内数据几乎没信息量, IngameState 下界也是 5;
        # 超过实际局长的是赛后空帧, 不要。
        if mi is None or mi < 5 or mi > upto_min:
            continue
        out.append({
            "minute": round(float(mi), 2),
            "golddiff": r.get("golddiff"),
            "csdiff": r.get("csdiff", 0),
            "blue_kills": r.get("blue_kills"), "red_kills": r.get("red_kills"),
            "gold_total": r.get("blue_gold"),
            "blue_gold": r.get("blue_gold"), "red_gold": r.get("red_gold"),
            "frame_t": r.get("t"),
            # 目标资源。上面那句"塔/龙/大龙帧里也有, 但模型不吃, 为'以后
            # 也许用得上'多打几倍请求不划算"说错了一半: 确实模型还不吃,
            # 但**不需要多打任何请求** —— gold_timeline 取回的同一批帧里
            # 就带着它们, 之前只是没往下传。
            #
            # 而这正是回填真正不可替代的地方: OE 的分时刻列全是经济/经验/
            # 补刀/KDA, 目标资源只有整局总数。想让局内模型看到"第 20 分钟
            # 谁拿了大龙", 除了帧没有第二个来源。
            "blue_towers": r.get("blue_towers"), "red_towers": r.get("red_towers"),
            "blue_inhibitors": r.get("blue_inhibitors"),
            "red_inhibitors": r.get("red_inhibitors"),
            "blue_barons": r.get("blue_barons"), "red_barons": r.get("red_barons"),
            "blue_dragons": r.get("blue_dragons"), "red_dragons": r.get("red_dragons"),
            "blue_dragon_types": r.get("blue_dragon_types"),
            "red_dragon_types": r.get("red_dragon_types"),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=95, help="往回扒多少天")
    ap.add_argument("--limit", type=int, default=0, help="最多处理几局 (0=不限)")
    ap.add_argument("--report", action="store_true", help="只汇报已攒的")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    if a.report:
        report()
        return 0

    # 断点续跑: 处理过的局记下来, 重跑不会重复打请求
    done = set(json.loads(SEEN.read_text(encoding="utf-8"))) if SEEN.exists() else set()
    log(f"已处理过 {len(done)} 局")

    log("加载 Oracle's Elixir 索引...")
    oe = oe_index()
    log(f"  {len(oe)} 局 (含 result)")

    feed = EsportsFeed()
    now = datetime.now(timezone.utc)
    todo = []
    for lg, lid in ALL_LEAGUES.items():
        try:
            sched = feed.schedule(lg, league_id=lid)
        except Exception as e:
            log(f"  {lg} 赛程取不到: {e}")
            continue
        for m in sched:
            try:
                ts = datetime.fromisoformat(m.start_time.replace("Z", "+00:00"))
            except Exception:
                continue
            if 0 <= (now - ts).days <= a.days:
                todo.append((ts, lg, m))
    todo.sort(key=lambda x: -x[0].timestamp())
    log(f"待处理比赛 {len(todo)} 场 (最近 {a.days} 天)")

    fh = open(ROWS, "a", encoding="utf-8")
    n_game = n_row = n_late = n_skip = 0
    t0 = time.time()
    stop = False
    for ts, lg, m in todo:
        if stop:
            break
        try:
            games = feed.games(m.match_id)
            id2n = {t.get("id"): t.get("name")
                    for t in feed.match_teams(m.match_id)}
        except Exception:
            continue
        for g in games:
            if a.limit and n_game >= a.limit:
                stop = True
                break
            if g.get("state") != "completed":
                continue
            gid = g.get("id")
            if not gid or gid in done:
                continue

            # 蓝方按**这一局**的选边算 —— BO 里每局换边
            sides = {t.get("side"): t.get("id") for t in (g.get("teams") or [])}
            blue_api = id2n.get(sides.get("blue"))
            red_api = id2n.get(sides.get("red"))
            blue = TEAM_ALIASES.get(blue_api, blue_api)
            red = TEAM_ALIASES.get(red_api, red_api)
            row = find_oe(oe, LEAGUE_TO_OE.get(lg, lg), ts,
                          blue, red, g.get("number"))
            if row is None:
                # **不记进 done**: 配不上多半是 OE 还没收录这场 (它的数据比
                # 现在晚几天)。记下来的话, 等 OE 补齐之后重跑也会跳过它 ——
                # 实测第一轮就有 793 局是这个原因被跳的, 全是能救回来的。
                # 代价只是每次重跑要为这些局多打两三个 (带缓存的) 请求。
                n_skip += 1
                continue
            try:
                start = feed._game_start(gid)
            except Exception:
                start = None
            if not start:
                n_skip += 1
                done.add(gid)
                continue

            glen = float(row["gamelength"])
            glen = glen / 60 if glen > 200 else glen
            snaps = harvest(feed, gid, start, glen)
            done.add(gid)
            if not snaps:
                n_skip += 1
                continue

            n_game += 1
            for snap in snaps:
                T = snap["minute"]
                rec = {"game_id": gid, "oe_gameid": row["gameid"],
                       "league": lg, "date": str(row["date"]),
                       "game_number": int(g.get("number") or 0),
                       "blue": blue, "red": red, "T": T,
                       "y": int(row["result"]),       # 蓝方是否获胜, 来自 OE
                       "gamelength_min": round(glen, 2), **snap}
                # 偏移校验: 只有整 25 分那个点能和 OE 的 golddiffat25 对照
                if abs(T - 25) < 1.0 and not pd.isna(row["golddiffat25"]):
                    rec["oe_golddiffat25"] = float(row["golddiffat25"])
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_row += 1
                if T >= 30:
                    n_late += 1

            if n_game % 10 == 0:
                fh.flush()
                SEEN.write_text(json.dumps(sorted(done)), encoding="utf-8")
                el = time.time() - t0
                log(f"  {n_game} 局 · {n_row} 条 (30 分钟后 {n_late} 条) "
                    f"· 跳过 {n_skip} · {el / max(n_game, 1):.1f}s/局")

    fh.close()
    SEEN.write_text(json.dumps(sorted(done)), encoding="utf-8")
    log(f"完成: {n_game} 局, {n_row} 条快照 (30 分钟后 {n_late} 条), 跳过 {n_skip}")
    report()
    return 0


def report():
    if not ROWS.exists():
        print("还没有回填数据")
        return
    # game_id / oe_gameid 必须按字符串读。lolesports 的 id 是 18 位数字,
    # 超出 float64 能精确表示的整数范围 (~9e15), 让 pandas 自己推断类型会把
    # 不同的局压成同一个值 —— 实测 1783 条被数成 461 条, 而且不报任何错。
    df = pd.read_json(ROWS, lines=True,
                      dtype={"game_id": str, "oe_gameid": str})
    df = df.drop_duplicates(["game_id", "T"])
    print(f"\n{'=' * 66}")
    print(f"  回填库: {df.game_id.nunique()} 局, {len(df)} 条快照")
    print("=" * 66)
    # 逐分钟采样后 T 是小数 (每局采样相位不同), 按整分钟归桶才看得出分布
    df["Tb"] = df["T"].round().astype(int)
    print("\n按分钟段:")
    for lo, hi in [(5, 10), (10, 15), (15, 20), (20, 25),
                   (25, 30), (30, 35), (35, 40), (40, 99)]:
        sub = df[(df.Tb >= lo) & (df.Tb < hi)]
        if not len(sub):
            continue
        mark = ("   ← OE 完全没有" if lo >= 25
                else "   ← OE 没有" if lo < 10 else "")
        print(f"  {lo:>2}-{hi:<3}分钟 {len(sub):>6} 条 "
              f"{sub.game_id.nunique():>4} 局{mark}")
    print("\n按赛区:")
    for lg, n in (df.groupby("league").game_id.nunique()
                    .sort_values(ascending=False).items()):
        print(f"  {lg:<5} {n:>5} 局")
    print(f"\n蓝方胜率 {df.drop_duplicates('game_id').y.mean():.1%}  "
          f"(健康值应在 50%~55%)")

    # 训练-服务偏移: 帧算的 golddiff vs OE 的 golddiffat25
    if "oe_golddiffat25" not in df.columns:
        return
    v = df[(df["T"] == 25) & df["oe_golddiffat25"].notna()]
    if not len(v):
        return
    d = v["golddiff"] - v["oe_golddiffat25"]
    print(f"\n{'-' * 66}")
    print(f"训练-服务偏移 (T=25, n={len(v)}):")
    print(f"  帧 − OE 经济差:  均值 {d.mean():+.0f}   中位 {d.median():+.0f}   "
          f"标准差 {d.std():.0f}")
    print(f"  |偏差| < 200 的占比 {(d.abs() < 200).mean():.1%}")
    print(f"  相关系数 {v['golddiff'].corr(v['oe_golddiffat25']):.4f}")
    # 判据要同时看均值和散布 —— 均值为零但每局差几千金币, 一样是坏的:
    # 线上每次预测都在读一个和训练时不同的量。
    ok_mean = abs(d.mean()) <= 300
    ok_sd = d.std() <= 800
    if ok_mean and ok_sd:
        print("  ✓ 无系统性偏移, 且逐局吻合")
    elif ok_mean:
        print(f"  ⚠ 均值没问题但逐局散布很大 (SD {d.std():.0f}) —— "
              f"多半是时刻没对齐, 不是量纲问题")
    else:
        print("  ⚠ 存在系统性偏移 —— 线上特征和训练特征不是一个量纲, 要查")


if __name__ == "__main__":
    sys.exit(main())
