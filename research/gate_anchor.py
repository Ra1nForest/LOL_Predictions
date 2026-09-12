"""
闸门: 局内胜率锚定在 BP 后概率上, 三段 (赛前 / BP 后 / 局内) 接成一条线, 会不会变差?
=====================================================================================
用户 2026-09-13: 看板上 BP 后说 MKOI 六成多, 一到第 3 分钟切成局内模型, 马上给 VIT 八成 ——
经济差还是 0, 同一局的两个数自相矛盾。原因有两个:
  1. 局内模型自带一套"赛前判断" (diff_pre_* 队伍滚动统计 + 阵容强势期), 但看不到 BP 后模型
     用的英雄胜率、选手熟练度。两套先验打架时, 切换那一刻就跳。
  2. 线上 12 分钟以前一律按 T=10 评估 (nearest_slice): 第 3 分钟总经济很小, golddiff_norm
     (相对经济差) 被放大成第 10 分钟的大优势, 开头几分钟又高又抖。

候选 (都用线上的模型文件, 不重训):
  C0 现在的     第 3 分钟前 BP 后, 之后局内模型 (校准后)
  C1 锚定       logit p = logit(BP 后) + [局内 margin(实际局势) - 局内 margin(均势)]
                "均势" = 同一局同一时刻, 经济差/补刀差/人头差 (及由它们派生的交互项) 置 0, 其余不动。
                局内模型只贡献"局势比均势好多少", 先验交给 BP 后。开局局势持平 -> 就是 BP 后概率,
                从 0 分钟起全程这么算, 一条连续的线。margin 是 XGBoost 的原始对数几率 (树的输出
                在这个空间里本来就是相加的), 不走校准表 —— 校准表是分段常数, 差值会一跳一跳。
  C2 渐变       logit p = (1-w) logit(BP 后) + w logit(局内), w 从第 3 分钟的 0 线性升到第 15 分钟的 1

考卷和 gate_early_minutes 一样: 532 局左右四大赛区比赛, 同时落在 Stage 2 和 Stage 4 的按日期留出
测试段里, 两份线上模型都没见过; 0-4 分钟来自 backfill/early_snapshots.jsonl, 5 分钟起来自
backfill/snapshots.jsonl, 一直到终局。分钟段内每局取平均损失, 和 C0 逐局配对; 另把考卷按日期
切 5 段按段配对。|t| > 2.5 才算数。另报"第 2 -> 3 分钟"这一步平均跳多少, 量连续性。

    python research/gate_anchor.py      (0-4 分钟的帧先由 gate_early_minutes 补好)
"""
from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

import api                                              # noqa: E402
import ingame_model as IM                               # noqa: E402
from feature_store import build_training_matrix         # noqa: E402
from harness import T_ACCEPT, paired_t                  # noqa: E402
from ingame_service import IngameModel                  # noqa: E402
from merge_backfill import borrow_cols, derive          # noqa: E402

SNAP = ROOT / "backfill" / "snapshots.jsonl"
EARLY = ROOT / "backfill" / "early_snapshots.jsonl"
MAJORS = {"LPL", "LCK", "LEC", "LCS"}
TRAIN_FRAC, CAL_FRAC = 0.60, 0.20
N_BLOCKS = 5
KEEP = ("golddiff", "csdiff", "blue_kills", "red_kills", "gold_total")
BUCKETS = [("0-2", 0, 3), ("3-9", 3, 10), ("10-19", 10, 20), ("20-29", 20, 30), ("30+", 30, 999), ("全部", 0, 999)]
# 置 0 得到"均势": 有方向的局势量, 以及由它们乘出来的交互项。killsum / gold_total 是节奏, 不是谁领先, 不动
NEUTRAL = ("golddiff", "csdiff", "killdiff", "golddiff_norm", "x_gold_scaling", "x_goldnorm_scaling",
           "lead_by_early_comp")
RAMP = (3.0, 15.0)

logit = lambda p: math.log(p / (1 - p))
sig = lambda z: 1 / (1 + math.exp(-z))


def margin(live: IngameModel, row: dict) -> float:
    x = np.nan_to_num(np.array([[float(row.get(f, np.nan)) for f in live.features]]), nan=-999.0)
    return float(live.model.predict(x, output_margin=True)[0])


def main():
    print("[1/3] 考卷")
    bf = pd.DataFrame([json.loads(l) for l in open(SNAP, encoding="utf-8")])
    bf = bf[bf.league.isin(MAJORS) & bf.oe_gameid.notna()]
    early = {}
    for line in open(EARLY, encoding="utf-8"):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d["rows"]:
            early[d["game_id"]] = d["rows"]
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
    print(f"  {bf.game_id.nunique()} 局")

    print("[2/3] 逐局逐分钟算三种")
    live = IngameModel("_live")
    s2 = api.Stage("post_draft")
    recs, flipped = [], 0
    for gid, g in bf.groupby("game_id"):
        oe_id = g.oe_gameid.iloc[0]
        seq = sorted((early.get(gid) or []) + g.to_dict("records"), key=lambda s: s["minute"])
        s10 = next((s for s in seq if math.floor(s["minute"]) == 10), None)
        if s10 is not None and oe_id in oe10.index:
            o = float(oe10[oe_id])
            if abs(o) > 300 and abs(s10["golddiff"] - o) > abs(s10["golddiff"] + o):
                flipped += 1
                continue
        c = const.loc[oe_id].to_dict()
        r2 = rows2[oe_id]
        pA = s2.predict(r2)[1]
        seen = set()
        for s in seq:
            m = math.floor(s["minute"])
            if m in seen or s.get("golddiff") is None:
                continue
            seen.add(m)
            row = derive({**c, **{k: s.get(k) for k in KEEP}})
            row["T"] = live.nearest_slice(s["minute"])
            _raw, cal = live.predict(row)
            neu = {**row, **{k: 0.0 for k in NEUTRAL}}
            d = margin(live, row) - margin(live, neu)
            c0 = pA if m < 3 else cal
            c1 = sig(logit(pA) + d)
            w = min(1.0, max(0.0, (s["minute"] - RAMP[0]) / (RAMP[1] - RAMP[0])))
            c2 = sig((1 - w) * logit(pA) + w * logit(min(max(cal, 1e-6), 1 - 1e-6)))
            recs.append({"game": gid, "date": r2["date"], "m": m, "y": int(c["y"]),
                         "C0": c0, "C1": c1, "C2": c2})
    df = pd.DataFrame(recs)
    print(f"  {df.game.nunique()} 局 / {len(df)} 行, 选边对不上丢弃 {flipped} 局")

    print(f"[3/3] 结果  (t > 0 = 比 C0 好, |t| > {T_ACCEPT} 才算数)")
    names = {"C0": "C0 现在的", "C1": "C1 锚定", "C2": "C2 渐变"}
    eps = 1e-12
    for k in names:
        df[f"b{k}"] = (df[k] - df.y) ** 2
        df[f"l{k}"] = -(df.y * np.log(np.clip(df[k], eps, 1)) + (1 - df.y) * np.log(np.clip(1 - df[k], eps, 1)))
        df[f"a{k}"] = ((df[k] > .5) == (df.y == 1)).astype(float)
    for lab, lo, hi in BUCKETS:
        sub = df[(df.m >= lo) & (df.m < hi)]
        per = sub.groupby("game").agg(date=("date", "first"), **{
            f"{p}{k}": (f"{p}{k}", "mean") for k in names for p in "bla"}).sort_values("date")
        blocks = np.array_split(np.arange(len(per)), N_BLOCKS)
        print(f"\n  第 {lab} 分钟 ({len(per)} 局 / {len(sub)} 行)")
        print(f"    {'':<12}{'Brier':>8}{'对数损失':>9}{'准确率':>8}{'Brier 逐局t':>12}{'分段t':>8}{'对数损失 t':>11}")
        for k, nm in names.items():
            if k == "C0":
                tb = tbl = tl = float("nan")
            else:
                _, _, tb, _ = paired_t(-per["bC0"].values, -per[f"b{k}"].values)
                _, _, tl, _ = paired_t(-per["lC0"].values, -per[f"l{k}"].values)
                _, _, tbl, _ = paired_t([-per["bC0"].values[i].mean() for i in blocks],
                                        [-per[f"b{k}"].values[i].mean() for i in blocks])
            print(f"    {nm:<12}{per[f'b{k}'].mean():>8.4f}{per[f'l{k}'].mean():>9.4f}{per[f'a{k}'].mean():>8.1%}"
                  f"{tb:>+12.2f}{tbl:>+8.2f}{tl:>+11.2f}")

    # 连续性: 第 2 -> 3 分钟那一步 (现在切换模型的地方) 和之后每一步的平均跳动 (百分点)
    print("\n  连续性: 相邻两分钟胜率变化的平均幅度 (百分点)")
    wide = {k: df.pivot_table(index="game", columns="m", values=k) for k in names}
    for k, nm in names.items():
        w = wide[k]
        j23 = (w[3] - w[2]).abs().mean() * 100 if 2 in w and 3 in w else float("nan")
        steps = w.diff(axis=1).abs()
        later = steps.loc[:, [c for c in steps.columns if 4 <= c <= 30]].stack().mean() * 100
        big = (w[3] - w[2]).abs().gt(0.15).mean() if 2 in w and 3 in w else float("nan")
        print(f"    {nm:<12} 第2→3分钟 {j23:5.1f}   (跳 >15 个百分点的局 {big:.0%})   第4-30分钟每步 {later:4.1f}")


if __name__ == "__main__":
    main()
