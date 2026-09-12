"""
闸门: 开局前 10 分钟, 看板该显示 BP 后概率, 还是局内模型?
============================================================
看板现在的做法: 第 3 分钟以前显示 BP 后 (Stage 2) 的概率, 第 3 分钟起换成局内模型 (Stage 4
live 变体)。胜率走势图因此在前 3 分钟是一段平线, 第 3 分钟起才开始动。这里量两件事:

  1. 第 0-2 分钟: 局内模型是不是已经比 BP 后更准 (是 -> 局内模型应当从开局就用)
  2. 第 3-9 分钟: 局内模型是不是真的比 BP 后更准 (否 -> 3 分钟这个切换点本身就早了)

注意局内模型在第 12 分钟以前**一律按 T=10 评估** —— IngameModel.make_row 把分钟数归到最近的
训练切片 (10/15/20/25), 3 和 10 走的是同一片树。这里照线上原样算 (nearest_slice), 不假设连续 T。

考卷: research/backfill_late.py 从直播帧逐分钟重建的快照 (backfill/snapshots.jsonl, 带 OE 的真实
胜负), 只取四大赛区, 并且**同时落在两个模型的测试段里**: Stage 2 是 train.py 按日期留出的最后 20%,
Stage 4 是 ingame_model.split_by_game 按日期留出的最后 20%。用现在的数据重切一次得到的测试段只会
更靠后 (数据只增不减), 所以一定是两份线上模型都没见过的比赛。
快照从第 5 分钟才开始 (backfill_late 当时认为 5 分钟前没信息量), 0-4 分钟这里从帧补取,
缓存到 backfill/early_snapshots.jsonl, 重跑不再联网。

局内模型那一路的特征照 merge_backfill 的办法组: 一局之内不变的列 (赛前差值、阵容强势期、赛区)
按 oe_gameid 从 ingame_model.build() 借, 派生列用 merge_backfill.derive (公式照抄 build())。
BP 后那一路直接用 build_training_matrix 的测试行喂 Stage("post_draft") —— gate_live_names 验证过
这些行和线上 make_row 逐列相等。

判据: 分数取负损失, t > 0 表示局内模型更好。逐局配对 t, 另把考卷按日期切 5 段按段配对,
和 harness.gate 同一套 |t| > 2.5。

    python research/gate_early_minutes.py            (第一次要联网补 0-4 分钟的帧, 约十来分钟)
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

import api                                              # noqa: E402  Stage 和 DATA
import ingame_model as IM                               # noqa: E402
from esports_feed import EsportsFeed                    # noqa: E402
from feature_store import build_training_matrix         # noqa: E402
from harness import T_ACCEPT, paired_t                  # noqa: E402
from ingame_service import IngameModel                  # noqa: E402
from merge_backfill import borrow_cols, derive          # noqa: E402

SNAP = ROOT / "backfill" / "snapshots.jsonl"
EARLY = ROOT / "backfill" / "early_snapshots.jsonl"
MAJORS = {"LPL", "LCK", "LEC", "LCS"}
MAX_MIN = 10
TRAIN_FRAC, CAL_FRAC = 0.60, 0.20      # 和 train.py / ingame_model.split_by_game 一致
N_BLOCKS = 5
KEEP = ("minute", "golddiff", "csdiff", "blue_kills", "red_kills", "gold_total", "blue_gold", "red_gold")


def fetch_early(feed: EsportsFeed, gid: str, t5: datetime) -> list[dict]:
    """第 0-4 分钟的快照, 口径和 backfill_late.harvest 一样 (gold_timeline 每分钟一个窗口)。"""
    rows = feed.gold_timeline(gid, upto=t5, step_sec=60)
    out = []
    for r in rows:
        mi = r.get("minute")
        if mi is None or mi >= 5:
            continue
        out.append({"minute": round(float(mi), 2), "golddiff": r.get("golddiff"),
                    "csdiff": r.get("csdiff", 0), "blue_kills": r.get("blue_kills"),
                    "red_kills": r.get("red_kills"), "gold_total": r.get("blue_gold"),
                    "blue_gold": r.get("blue_gold"), "red_gold": r.get("red_gold")})
    return out


def load_early(need: dict[str, datetime], workers: int) -> dict[str, list[dict]]:
    have: dict[str, list[dict]] = {}
    if EARLY.exists():
        for line in open(EARLY, encoding="utf-8"):
            d = json.loads(line)
            have[d["game_id"]] = d["rows"]
    todo = {g: t for g, t in need.items() if g not in have}
    if todo:
        print(f"  补取 0-4 分钟: {len(todo)} 局 (已缓存 {len(need) - len(todo)})")
        feed = EsportsFeed()
        done = 0
        with ThreadPoolExecutor(workers) as ex, open(EARLY, "a", encoding="utf-8") as fo:
            futs = {ex.submit(fetch_early, feed, g, t): g for g, t in todo.items()}
            for fu in as_completed(futs):
                g = futs[fu]
                try:
                    rows = fu.result()
                except Exception as e:                 # 取不到的局记成空, 不拖垮整批; 下面会计数
                    print(f"    {g} 失败: {e}")
                    rows = []
                have[g] = rows
                fo.write(json.dumps({"game_id": g, "rows": rows}, ensure_ascii=False) + "\n")
                done += 1
                if done % 50 == 0:
                    print(f"    {done}/{len(todo)}")
    return have


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3, help="同时补取几局 (每局内部 gold_timeline 自己还有并发)")
    a = ap.parse_args()

    print("[1/5] 回溯快照")
    bf = pd.DataFrame([json.loads(l) for l in open(SNAP, encoding="utf-8")])
    bf = bf[bf.league.isin(MAJORS) & bf.oe_gameid.notna()]
    print(f"  四大赛区 {bf.game_id.nunique()} 局")

    print("[2/5] 两个模型的测试段 (现在的数据重切)")
    files = sorted(glob.glob(str(ROOT / "data" / "20*_from_OraclesElixir.csv")))
    mdf4, _ = IM.build(IM.load(files), verbose=False)
    _tr, _cal, te4 = IM.split_by_game(mdf4, TRAIN_FRAC, CAL_FRAC)
    const = mdf4.drop_duplicates("gameid").set_index("gameid")[borrow_cols(mdf4)]
    oe10 = mdf4[mdf4["T"] == 10].drop_duplicates("gameid").set_index("gameid")["golddiff"]
    mdf2 = build_training_matrix(api.DATA, verbose=False)
    test2 = mdf2.iloc[int(len(mdf2) * (TRAIN_FRAC + CAL_FRAC)):]
    rows2 = {r["gameid"]: r for r in test2.to_dict("records")}
    ok = set(te4["gameid"]) & set(rows2) & set(const.index)
    bf = bf[bf.oe_gameid.isin(ok)]
    print(f"  Stage 4 测试段 {te4.gameid.nunique()} 局, Stage 2 测试段 {len(rows2)} 局 "
          f"(自 {str(test2['date'].min())[:10]}); 和回溯交集 {bf.game_id.nunique()} 局")

    print("[3/5] 第 0-4 分钟")
    t5 = {}
    for gid, g in bf.groupby("game_id"):
        r0 = g.sort_values("minute").iloc[0]
        # 快照里最早那一行 (约第 5.1 分钟) 的帧时刻 ≈ 取到第 5 分钟为止
        t5[gid] = datetime.fromisoformat(str(r0["frame_t"]).replace("Z", "+00:00")) + timedelta(seconds=5)
    early = load_early(t5, a.workers)
    print(f"  有 0-4 分钟数据的 {sum(1 for g in t5 if early.get(g))}/{len(t5)} 局")

    print("[4/5] 逐局逐分钟算两个概率")
    live = IngameModel("_live")
    s1, s2 = api.Stage("pre_draft"), api.Stage("post_draft")
    recs, flipped, no_early = [], 0, 0
    for gid, g in bf.groupby("game_id"):
        oe_id = g.oe_gameid.iloc[0]
        snaps = [dict(zip(KEEP, (r[k] for k in KEEP))) for r in g.to_dict("records")]
        e = early.get(gid) or []
        if not e:
            no_early += 1
        snaps = sorted(e + snaps, key=lambda s: s["minute"])
        # 选边核对: 帧的蓝方经济差 vs OE 的 golddiffat10, 反了的整局丢掉 (不硬翻, 反了说明配错了局)
        s10 = next((s for s in snaps if math.floor(s["minute"]) == 10), None)
        if s10 is not None and oe_id in oe10.index:
            o = float(oe10[oe_id])
            if abs(o) > 300 and abs(s10["golddiff"] - o) > abs(s10["golddiff"] + o):
                flipped += 1
                continue
        c = const.loc[oe_id].to_dict()
        r2 = rows2[oe_id]
        pA = s2.predict(r2)[1]
        p1 = s1.predict(r2)[1]
        y = int(c["y"])
        seen = set()
        for s in snaps:
            m = math.floor(s["minute"])
            if m > MAX_MIN or m in seen or s["golddiff"] is None:
                continue
            seen.add(m)
            row = derive({**c, **s})
            row["T"] = live.nearest_slice(s["minute"])       # 线上就是这么做的, 见模块说明
            pB = live.predict(row)[1]
            recs.append({"game": gid, "date": r2["date"], "m": m, "y": y, "A": pA, "B": pB, "P1": p1})
    df = pd.DataFrame(recs)
    print(f"  {df.game.nunique()} 局 / {len(df)} 行; 选边对不上丢弃 {flipped} 局, 缺 0-4 分钟 {no_early} 局")
    miss = [f for f in live.features if f not in derive({**const.iloc[0].to_dict(), **dict.fromkeys(KEEP, 1.0)})
            and f != "T"]
    if miss:
        print(f"  ⚠ 局内模型要的特征缺: {miss}")

    print("[5/5] 结果  (A = BP 后, 看板前 3 分钟显示的;  B = 局内模型, 第 3 分钟起显示的)")
    eps = 1e-12
    df["bA"], df["bB"] = (df.A - df.y) ** 2, (df.B - df.y) ** 2
    ll = lambda p, y: -(y * np.log(np.clip(p, eps, 1)) + (1 - y) * np.log(np.clip(1 - p, eps, 1)))
    df["lA"], df["lB"] = ll(df.A, df.y), ll(df.B, df.y)
    df["aA"], df["aB"] = ((df.A > .5) == (df.y == 1)).astype(float), ((df.B > .5) == (df.y == 1)).astype(float)

    def compare(sub: pd.DataFrame, label: str):
        # 每局一个数 (窗口内几分钟取平均), 再逐局配对 —— 同一局的几分钟不是独立样本
        per = sub.groupby("game").agg(date=("date", "first"), bA=("bA", "mean"), bB=("bB", "mean"),
                                       lA=("lA", "mean"), lB=("lB", "mean"), aA=("aA", "mean"), aB=("aB", "mean"))
        per = per.sort_values("date")
        blocks = np.array_split(np.arange(len(per)), N_BLOCKS)
        out = []
        for name, ka, kb in (("Brier", "bA", "bB"), ("对数损失", "lA", "lB"), ("准确率", "aA", "aB")):
            sa = per[ka].values if name == "准确率" else -per[ka].values
            sb = per[kb].values if name == "准确率" else -per[kb].values
            _d, _sd, t_g, _ = paired_t(sa, sb)
            _d, _sd, t_b, _ = paired_t([sa[i].mean() for i in blocks], [sb[i].mean() for i in blocks])
            out.append((name, per[ka].mean(), per[kb].mean(), t_g, t_b))
        print(f"\n  {label}  ({len(per)} 局)")
        print(f"    {'':<8}{'A BP后':>9}{'B 局内':>9}{'逐局 t':>9}{'分段 t':>9}")
        for name, va, vb, tg, tb in out:
            fmt = (lambda v: f"{v:.1%}") if name == "准确率" else (lambda v: f"{v:.4f}")
            print(f"    {name:<8}{fmt(va):>9}{fmt(vb):>9}{tg:>+9.2f}{tb:>+9.2f}")
        return out

    print(f"\n  逐分钟 Brier (越小越好), t > 0 = 局内模型更好, |t| > {T_ACCEPT} 才算数:")
    print(f"    {'分钟':<6}{'局数':>6}{'A BP后':>9}{'B 局内':>9}{'逐局 t':>9}")
    for m in range(MAX_MIN + 1):
        sub = df[df.m == m]
        if len(sub) < 20:
            continue
        _d, _sd, t, _ = paired_t(-sub.bA.values, -sub.bB.values)
        print(f"    {m:<6}{len(sub):>6}{sub.bA.mean():>9.4f}{sub.bB.mean():>9.4f}{t:>+9.2f}")
    compare(df[df.m <= 2], "第 0-2 分钟 (看板现在显示 BP 后)")
    compare(df[(df.m >= 3) & (df.m <= 9)], "第 3-9 分钟 (看板现在显示局内模型)")
    compare(df[df.m == 10], "第 10 分钟 (参照: 局内模型训练过的最早切片)")
    print(f"\n  赛前模型 (参照) 整体 Brier {((df.drop_duplicates('game').P1 - df.drop_duplicates('game').y) ** 2).mean():.4f}")


if __name__ == "__main__":
    main()
