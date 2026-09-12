"""
闸门: 换掉 Platt, 用保序回归 (Isotonic) 校准 Stage 4?
======================================================

为什么重开这个案子
------------------
README 记着 Isotonic 输过一次, 理由是"校准数据不够, 它过拟合了自己"。
**那个前提现在不成立了** —— 当时的局内训练集远小于现在的 176322 条快照,
校准集有 3.5 万条, 对 Isotonic (自由度约 O(√n)) 绰绰有余。

而且实测显示问题出在**函数形式**, 不是数据量:

    金币差档      模型说    实际是
    < -6k         7.0%      2.7%     ← 两端不足
    -1k~1k       52.4%     51.1%
    1k~2k        78.0%     74.6%     ← 中间过冲
    > +6k        93.3%     98.0%     ← 两端不足

  >6k 领先的样本: XGBoost 原始输出平均 97.8% (实际 98.0%, 几乎完美),
  Platt 之后压到 93.3%。**模型判断是对的, 是校准层把它压扁了。**

中间过冲 + 两端塌陷正是单参数 sigmoid 拟合非 sigmoid 形状时的典型失配。
Isotonic 不假设形状, 只要求单调, 正好对症。

实验协议 (**先定死, 不事后挑**)
-------------------------------
· 模型固定不重训 —— 只比校准层, 否则模型方差会盖过要测的东西
· 校准/测试的划分按**局**做, 重复 N 折
· 三个候选全部报告: platt / isotonic / isotonic+裁剪
· 主判据 Brier, 次判据 ECE; 准确率一并报但**不作为判据** ——
  三者都是单调变换, 不改排序, 准确率只随决策阈值位置轻微浮动
· |t| > 2.5 采纳 (harness 口径)

用法: python research/gate_calibration.py [--folds 8] [--clip 0.01]
"""
from __future__ import annotations
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, brier_score_loss

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT)); sys.path.insert(0, str(_ROOT / "research"))
import ingame_model as IM
from harness import paired_t, T_ACCEPT


def ece(p, y, nb=10):
    edges = np.linspace(0, 1, nb + 1)
    out = 0.0
    for i in range(nb):
        m = (p >= edges[i]) & (p < edges[i + 1])
        if m.sum():
            out += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(out)


def fit_platt(pc, yc):
    lr = LogisticRegression(C=1e6).fit(pc.reshape(-1, 1), yc)
    return lambda r: lr.predict_proba(r.reshape(-1, 1))[:, 1]


def fit_isotonic(pc, yc, clip=None):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(pc, yc)
    if clip is None:
        return lambda r: iso.predict(r)
    return lambda r: np.clip(iso.predict(r), clip, 1 - clip)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--clip", type=float, default=0.01)
    a = ap.parse_args()

    files = sorted((_ROOT / "data").glob("20*_from_OraclesElixir.csv"))
    mdf, _sc = IM.build(IM.load([str(f) for f in files]), verbose=False)
    tr, cal, te = IM.split_by_game(mdf)

    cfg = json.loads((_ROOT / "artifacts/calib_ingame.json").read_text())
    feats = cfg["features"]
    m = XGBClassifier()
    m.load_model(str(_ROOT / "artifacts/model_ingame.json"))

    # 模型只在 tr 上训过, cal+te 都是它没见过的 —— 合起来做校准实验的池子
    pool = pd.concat([cal, te], ignore_index=True)
    pool["raw"] = m.predict_proba(pool[feats].astype(float).fillna(-999))[:, 1]
    print(f"校准实验池: {pool.gameid.nunique()} 局 / {len(pool)} 条 "
          f"(模型训练时没见过)")
    print(f"原始输出范围 [{pool.raw.min():.3f}, {pool.raw.max():.3f}]\n")

    games = pool[["gameid", "date"]].drop_duplicates("gameid") \
                                    .sort_values("date")["gameid"].values
    chunks = np.array_split(games, a.folds)

    res = {k: {"brier": [], "ece": [], "acc": [], "pmax": [], "pmin": []}
           for k in ("platt", "isotonic", "iso_clip")}

    for i, hold in enumerate(chunks):
        te_f = pool[pool.gameid.isin(hold)]
        cal_f = pool[~pool.gameid.isin(hold)]
        pc, yc = cal_f.raw.values, cal_f.y.astype(int).values
        pt, yt = te_f.raw.values, te_f.y.astype(int).values

        for name, fn in (("platt", fit_platt(pc, yc)),
                         ("isotonic", fit_isotonic(pc, yc)),
                         ("iso_clip", fit_isotonic(pc, yc, clip=a.clip))):
            p = np.clip(fn(pt), 0, 1)
            res[name]["brier"].append(-brier_score_loss(yt, p))
            res[name]["ece"].append(-ece(p, yt))
            res[name]["acc"].append(accuracy_score(yt, (p > .5).astype(int)))
            res[name]["pmax"].append(p.max())
            res[name]["pmin"].append(p.min())
        print(f"  fold {i+1}: 校准 {len(cal_f):>6} / 测试 {len(te_f):>6}   "
              + "   ".join(f"{k} Brier {-res[k]['brier'][-1]:.4f}"
                           for k in res))

    print("\n" + "=" * 72)
    print(f"{'方法':>10}{'Brier':>10}{'ECE':>9}{'准确率':>9}"
          f"{'概率下限':>10}{'概率上限':>10}")
    for k in res:
        r = res[k]
        print(f"{k:>10}{-np.mean(r['brier']):>10.4f}{-np.mean(r['ece']):>9.4f}"
              f"{np.mean(r['acc']):>9.4f}{np.mean(r['pmin']):>10.3f}"
              f"{np.mean(r['pmax']):>10.3f}")

    print("\n" + "=" * 72)
    print("配对检验 (基线 = platt, 判据 |t| > 2.5)")
    for cand in ("isotonic", "iso_clip"):
        print(f"\n  {cand}  vs  platt")
        for metric, label in (("brier", "Brier(负)"), ("ece", "ECE(负)")):
            d, sd, t, _ = paired_t(np.array(res["platt"][metric]),
                                   np.array(res[cand][metric]))
            verdict = ("采纳" if abs(t) > T_ACCEPT and d > 0
                       else "拒绝 (变差)" if abs(t) > T_ACCEPT
                       else "噪声内, 不采纳")
            print(f"    {label:<10} Δ={d:+.5f}  配对SD={sd:.5f}  "
                  f"t={t:+.2f} -> {verdict}")
        d, sd, t, _ = paired_t(np.array(res["platt"]["acc"]),
                               np.array(res[cand]["acc"]))
        print(f"    {'准确率':<9} Δ={d:+.5f}  t={t:+.2f}  "
              f"(**不作为判据**: 单调变换不改排序)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
