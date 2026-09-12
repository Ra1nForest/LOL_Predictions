"""
闸门: 局内模型 (live 变体) 的训练数据 —— 只用 CSV 还是 CSV + 实时流, 四大还是全量
================================================================================
用户 2026-09-13 提的四种组合, 一次比完:

  V1  CSV · 全量       现在线上的 (ingame_model TARGET=None, OE 的 T=10/15/20/25)
  V2  CSV · 四大       2026-08-17 以前的做法 (gate_league_mix 量过全量更好, t=+5.19)
  V3  CSV+流 · 全量    OE + 从 lolesports 帧重建的逐分钟快照 (0 分钟起到终局)
  V4  CSV+流 · 四大    同上, 只用四大赛区

"流" = research/backfill_late.py 的逐分钟快照 (backfill/snapshots.jsonl, 5 分钟起) 加
gate_early_minutes 补的 0-4 分钟 (backfill/early_snapshots.jsonl); 胜负来自 OE。
为什么逐分钟而不是逐秒: 同一局相邻两秒几乎是同一个局面、同一个胜负标签 —— 多出来的 59 行
不带新信息, 只会让回溯局在训练里的权重放大 60 倍。

T (第几分钟) 的口径不同, 这是这次对比的一部分:
  V1/V2 只见过 10/15/20/25, 评估照线上原样 T = nearest_slice(分钟) —— 它们也只能这么用;
  V3/V4 见过每一分钟, 评估用真实分钟 —— 真要上线, make_row 也得跟着改成连续 T。
所以 V3/V4 若胜出, 胜出的是 "加流数据 + 连续 T" 这一整套, 不单是哪一项。

考卷: 回溯里的四大赛区局 (看板只预测四大), 按局随机分 k 折。回溯全在最近约 95 天, 不能按时间
分折 (见 gate_continuous 的说明)。每折: 训练 = OE 行 [+ 回溯行], 都排掉这一折的局, 再按变体
过滤赛区; 考 = 这一折的局在回溯里的每一分钟。每局只被没见过它的模型考一次。
按分钟段报 0-2 / 3-9 / 10-24 / 25+ / 全部: 每局在段内取平均损失, 和 V1 逐局配对, 另报按折配对。
|t| > 2.5 才算数。概率不做校准 (四种一样对待), 比的是相对好坏, 不是绝对的 Brier。

    python research/gate_train_mix.py [--folds 5] [--seeds 2]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

import ingame_model as IM                                  # noqa: E402
from harness import T_ACCEPT, paired_t                     # noqa: E402
from merge_backfill import borrow_cols, derive             # noqa: E402

SNAP = ROOT / "backfill" / "snapshots.jsonl"
EARLY = ROOT / "backfill" / "early_snapshots.jsonl"
MAJORS = {"LPL", "LCK", "LEC", "LCS"}
BUCKETS = [("0-2", 0, 3), ("3-9", 3, 10), ("10-24", 10, 25), ("25+", 25, 999), ("全部", 0, 999)]
VARIANTS = {                      # 名字: (含回溯流, 只用四大)
    "V1 CSV·全量": (False, False),
    "V2 CSV·四大": (False, True),
    "V3 CSV+流·全量": (True, False),
    "V4 CSV+流·四大": (True, True),
}
BASE = "V1 CSV·全量"
SNAP_KEYS = ("golddiff", "csdiff", "blue_kills", "red_kills", "gold_total")


def nearest(m: float) -> int:
    return min(IM.SLICES, key=lambda t: abs(t - m))


def stream_rows(const: pd.DataFrame, oe10: pd.Series) -> tuple[pd.DataFrame, int]:
    """回溯快照 -> 训练/评估行 (特征组法同 merge_backfill; T 用真实分钟, T_slice 是线上的归档)。"""
    early: dict[str, list] = {}
    if EARLY.exists():
        for line in open(EARLY, encoding="utf-8"):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:          # 补取还在往里写时, 最后一行可能只写了一半
                continue
            if d["rows"]:
                early[d["game_id"]] = d["rows"]
    snaps = pd.DataFrame([json.loads(l) for l in open(SNAP, encoding="utf-8")])
    snaps = snaps[snaps.oe_gameid.isin(const.index)]
    out, flipped = [], 0
    for gid, g in snaps.groupby("game_id"):
        oe_id = g.oe_gameid.iloc[0]
        seq = sorted(early.get(gid, []) + g.to_dict("records"), key=lambda s: s["minute"])
        # 选边核对 (同 gate_early_minutes): 帧的第 10 分钟经济差和 OE 的 golddiffat10 方向相反就整局丢掉
        s10 = next((s for s in seq if math.floor(s["minute"]) == 10), None)
        if s10 is not None and oe_id in oe10.index:
            o = float(oe10[oe_id])
            if abs(o) > 300 and abs(s10["golddiff"] - o) > abs(s10["golddiff"] + o):
                flipped += 1
                continue
        c = const.loc[oe_id].to_dict()
        seen = set()
        for s in seq:
            m = math.floor(s["minute"])
            if m in seen or s.get("golddiff") is None:
                continue
            seen.add(m)
            row = derive({**c, **{k: s.get(k) for k in SNAP_KEYS}})
            row.update(gameid=oe_id, minute=float(s["minute"]), T=float(s["minute"]),
                       T_slice=nearest(float(s["minute"])), source="bf")
            out.append(row)
    return pd.DataFrame(out), flipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=2)
    a = ap.parse_args()
    seeds = list(range(a.seeds))
    feats = json.loads((ROOT / "artifacts" / "calib_ingame_live.json").read_text(encoding="utf-8"))["features"]

    print("[1/4] OE 快照表 (全部赛区)")
    files = sorted(glob.glob(str(ROOT / "data" / "20*_from_OraclesElixir.csv")))
    mdf, _ = IM.build(IM.load(files), verbose=False)
    oe = mdf.copy()
    oe["source"] = "oe"
    const = mdf.drop_duplicates("gameid").set_index("gameid")[borrow_cols(mdf)]
    oe10 = mdf[mdf["T"] == 10].drop_duplicates("gameid").set_index("gameid")["golddiff"]
    print(f"  OE {oe.gameid.nunique()} 局 / {len(oe)} 行, 其中四大 {oe[oe.league.isin(MAJORS)].gameid.nunique()} 局")

    print("[2/4] 回溯流")
    bf, flipped = stream_rows(const, oe10)
    n_early = int((bf.minute < 5).sum())
    print(f"  {bf.gameid.nunique()} 局 / {len(bf)} 行 (0-4 分钟 {n_early} 行), 选边对不上丢弃 {flipped} 局; "
          f"四大 {bf[bf.league.isin(MAJORS)].gameid.nunique()} 局")

    test = bf[bf.league.isin(MAJORS)].reset_index(drop=True)
    games = np.array(sorted(test.gameid.unique()))
    np.random.default_rng(0).shuffle(games)
    folds = np.array_split(games, a.folds)
    pred = {v: np.full(len(test), np.nan) for v in VARIANTS}
    fold_of = np.full(len(test), -1)

    print(f"[3/4] {a.folds} 折 × {len(seeds)} 种子 × {len(VARIANTS)} 种训练集, 考卷 {len(games)} 局 / {len(test)} 行")
    for fi, te_g in enumerate(folds):
        te_g = set(te_g)
        idx = np.flatnonzero(test.gameid.isin(te_g).values)
        fold_of[idx] = fi
        te = test.iloc[idx]
        for v, (use_bf, majors) in VARIANTS.items():
            tr = oe[~oe.gameid.isin(te_g)]
            if use_bf:
                tr = pd.concat([tr, bf[~bf.gameid.isin(te_g)]], ignore_index=True)
            if majors:
                tr = tr[tr.league.isin(MAJORS)]
            X = tr[feats].astype(float).fillna(-999)
            y = tr["y"].astype(int)
            Xe = te.assign(T=te["T"] if use_bf else te["T_slice"])[feats].astype(float).fillna(-999)
            t0 = time.time()
            ps = []
            for s in seeds:
                m = XGBClassifier(**{**IM.XGB, "random_state": s})
                m.fit(X, y, verbose=False)
                ps.append(m.predict_proba(Xe)[:, 1])
            pred[v][idx] = np.mean(ps, axis=0)
            print(f"  折 {fi + 1} {v:<14} 训练 {len(tr):>6} 行  ({time.time() - t0:.0f}s)")

    print("[4/4] 结果  (t > 0 = 比 V1 好;  |t| > %.1f 才算数)" % T_ACCEPT)
    y = test["y"].astype(int).values
    eps = 1e-12
    for lab, lo, hi in BUCKETS:
        sel = (test.minute >= lo) & (test.minute < hi)
        if sel.sum() < 50:
            continue
        g = test.loc[sel, "gameid"].values
        f = pd.Series(fold_of[sel.values], index=g).groupby(level=0).first()
        print(f"\n  第 {lab} 分钟  ({len(set(g))} 局 / {int(sel.sum())} 行)")
        print(f"    {'':<16}{'Brier':>8}{'对数损失':>9}{'准确率':>8}{'Brier 逐局t':>12}{'按折t':>8}{'准确率 逐局t':>13}")
        per = {}
        for v in VARIANTS:
            p = pred[v][sel.values]
            yy = y[sel.values]
            d = pd.DataFrame({"g": g, "b": (p - yy) ** 2,
                              "l": -(yy * np.log(np.clip(p, eps, 1)) + (1 - yy) * np.log(np.clip(1 - p, eps, 1))),
                              "a": ((p > .5) == (yy == 1)).astype(float)}).groupby("g").mean()
            per[v] = d
        for v in VARIANTS:
            d = per[v]
            if v == BASE:
                tb = tf = ta = float("nan")
            else:
                b0 = per[BASE]
                _, _, tb, _ = paired_t(-b0.b.values, -d.b.values)
                _, _, ta, _ = paired_t(b0.a.values, d.a.values)
                fb0 = (-b0.b).groupby(f).mean()
                fb1 = (-d.b).groupby(f).mean()
                _, _, tf, _ = paired_t(fb0.values, fb1.values)
            print(f"    {v:<16}{d.b.mean():>8.4f}{d.l.mean():>9.4f}{d.a.mean():>8.1%}"
                  f"{tb:>+12.2f}{tf:>+8.2f}{ta:>+13.2f}")


if __name__ == "__main__":
    main()
