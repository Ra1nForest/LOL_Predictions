"""
全局 BP 的英雄池损耗, 按局号拆开看: 越往后打, 影响是不是越明显?
==============================================================
research/gate_fearless.py 把第 2-5 局混在一起进模型, 不过闸 (t≈0.7)。用户的直觉是效应随局号
增强 —— 第 4、5 局用掉的英雄最多, 选手最容易被逼到不擅长的英雄上。混在一起时后面的局样本少,
效应可能被前面的局稀释。这里不训练模型, 只做描述性检验:

  每一局比较两队的英雄池损耗 (同 gate_fearless 的两种口径):
    top5   每名选手近 365 天最常用 5 个英雄里已被本系列赛用掉的比例, 五人平均
    share  每名选手近 365 天的出场里落在已被用掉英雄上的比例, 五人平均
  损耗更少的一方赢了多少, 按局号 2/3/4/5 分开; 再按两队损耗差的大小分三档。
  z = (赢的局数 - n/2) / sqrt(n/4), 对 50% 的二项检验。|z| > 2.5 才算有信号 (和项目闸门同一口径,
  这里分组多, 更不能放宽)。

注意这是描述性的: 就算损耗少的一方赢得多, 也可能是强队本来就英雄池深、损耗少 —— 要进模型,
仍然得过 gate_fearless 那样的配对闸门。

    python research/fearless_by_game.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
from gate_fearless import load_rows, series_features  # noqa: E402


def z_of(k: int, n: int) -> float:
    return (k - n / 2) / math.sqrt(n / 4) if n else float("nan")


def main():
    pl, tm = load_rows()
    sf = series_features(pl, tm)
    winner = dict(zip(tm[tm["result"] == 1]["gameid"], tm[tm["result"] == 1]["teamname"].astype(str)))

    rows = []
    for gid, f in sf.items():
        if not f["fearless"] or f["game_no"] < 2 or len(f["team"]) != 2 or gid not in winner:
            continue
        (t1, (a5, as_)), (t2, (b5, bs)) = f["team"].items()
        w = winner[gid]
        for name, a, b in (("top5", a5, b5), ("share", as_, bs)):
            if np.isnan(a) or np.isnan(b) or a == b:
                continue
            lower = t1 if a < b else t2
            rows.append({"metric": name, "g": f["game_no"], "gap": abs(a - b), "win": int(w == lower)})
    df = pd.DataFrame(rows)

    for name in ("top5", "share"):
        d = df[df.metric == name]
        print(f"\n口径 {name}: 损耗更少的一方的胜率")
        print(f"  {'局号':<8}{'局数':>6}{'胜率':>8}{'z':>7}{'平均损耗差':>11}")
        for g in (2, 3, 4, 5):
            s = d[d.g == g]
            print(f"  第 {g} 局{len(s):>8}{s.win.mean():>8.1%}{z_of(s.win.sum(), len(s)):>+7.2f}{s.gap.mean():>11.3f}")
        s = d[d.g >= 4]
        print(f"  第 4-5 局{len(s):>6}{s.win.mean():>8.1%}{z_of(s.win.sum(), len(s)):>+7.2f}{s.gap.mean():>11.3f}")
        print(f"  全部{len(d):>10}{d.win.mean():>8.1%}{z_of(d.win.sum(), len(d)):>+7.2f}{d.gap.mean():>11.3f}")
        # 损耗差越大越该看得出来: 分三档
        q = d.gap.quantile([1 / 3, 2 / 3]).values
        print("  按两队损耗差分三档:")
        for lab, lo, hi in (("小", -1, q[0]), ("中", q[0], q[1]), ("大", q[1], 9)):
            s = d[(d.gap > lo) & (d.gap <= hi)]
            print(f"    差距{lab} ({s.gap.min():.2f}-{s.gap.max():.2f}){len(s):>6} 局  {s.win.mean():>6.1%}  z {z_of(s.win.sum(), len(s)):+.2f}")


if __name__ == "__main__":
    main()
