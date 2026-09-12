"""
闸门: Stage 1/2 (赛前 / BP 后) 该不该也换成 Isotonic?
=====================================================

局内模型 (Stage 4) 换 Isotonic 已经过闸: Brier t=+6.91, ECE t=+7.12, 8 折全正。
但**不能顺手把这两段一起改** —— 当年 Isotonic 输给 Platt 的理由是
"校准数据不够, 它会过拟合自己", 而这个理由在两处的成立程度完全不同:

    Stage 4    校准集约 35000 条快照   -> 那个理由早就不成立
    Stage 1/2  校准集     1716 条比赛  -> 很可能仍然成立

实验设计上的一个坑
------------------
照搬 Stage 4 的做法 (把 cal+test 合起来重新分折) 会把校准集撑到 3000+ 条,
比线上真实的 1716 大一倍 —— 那是在给 Isotonic 一个它上线后享受不到的优惠,
测出来的优势也就不能兑现。

所以这里每折都**重新训练模型**, 并把校准集锁死在 CAL_N=1716,
和 train.py 的 60/20/20 时序切分保持一致。切分点沿时间轴滑动, 得到多折。

判据同前: 主看 Brier 和 ECE, |t| > 2.5; 准确率一并报但不作为判据。

用法: python research/gate_calibration_pre.py [--folds 6]
"""
from __future__ import annotations
import argparse, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, brier_score_loss

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT)); sys.path.insert(0, str(_ROOT / "research"))
from feature_store import build_training_matrix, select_features
from harness import paired_t, T_ACCEPT

# 照抄 train.py 的设置, 否则测的就不是线上那个模型
XGB = dict(n_estimators=400, max_depth=5, learning_rate=0.04,
           subsample=0.8, colsample_bytree=0.7,
           eval_metric="logloss", random_state=42, verbosity=0)
CAL_N = 1716          # 线上校准集的真实大小 (见 artifacts/summary.json)
TEST_N = 1200


def ece(p, y, nb=10):
    edges = np.linspace(0, 1, nb + 1)
    out = 0.0
    for i in range(nb):
        m = (p >= edges[i]) & (p < edges[i + 1])
        if m.sum():
            out += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(out)


def calibrators(pc, yc, clip=0.01):
    lr = LogisticRegression(C=1e6).fit(pc.reshape(-1, 1), yc)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(pc, yc)
    return {
        "platt": lambda r: lr.predict_proba(r.reshape(-1, 1))[:, 1],
        "isotonic": lambda r: iso.predict(r),
        "iso_clip": lambda r: np.clip(iso.predict(r), clip, 1 - clip),
    }


def run_stage(mdf, stage, folds):
    feats = select_features(mdf, stage)
    X = mdf[feats].astype(float).fillna(-999).reset_index(drop=True)
    y = mdf["y"].astype(int).reset_index(drop=True)
    n = len(X)
    lo, hi = int(n * 0.50), n - CAL_N - TEST_N
    if hi <= lo:
        print(f"  {stage}: 数据不足"); return None
    cuts = np.linspace(lo, hi, folds).astype(int)

    res = {k: {"brier": [], "ece": [], "acc": [], "pmin": [], "pmax": []}
           for k in ("platt", "isotonic", "iso_clip")}
    print(f"\n  === {stage}  ({len(feats)} 特征, 共 {n} 场) ===")
    for i, cut in enumerate(cuts):
        tr = slice(0, cut)
        ca = slice(cut, cut + CAL_N)
        te = slice(cut + CAL_N, cut + CAL_N + TEST_N)
        m = XGBClassifier(**XGB)
        m.fit(X.iloc[tr], y.iloc[tr], verbose=False)
        pc = m.predict_proba(X.iloc[ca])[:, 1]
        pt = m.predict_proba(X.iloc[te])[:, 1]
        yc, yt = y.iloc[ca].values, y.iloc[te].values

        line = []
        for name, fn in calibrators(pc, yc).items():
            p = np.clip(fn(pt), 0, 1)
            res[name]["brier"].append(-brier_score_loss(yt, p))
            res[name]["ece"].append(-ece(p, yt))
            res[name]["acc"].append(accuracy_score(yt, (p > .5).astype(int)))
            res[name]["pmin"].append(p.min()); res[name]["pmax"].append(p.max())
            line.append(f"{name} {-res[name]['brier'][-1]:.4f}")
        print(f"    fold {i+1}: 训 {cut:>5} / 校准 {CAL_N} / 测 {TEST_N}   "
              + "  ".join(line))

    print(f"    {'方法':>10}{'Brier':>10}{'ECE':>9}{'准确率':>9}"
          f"{'下限':>8}{'上限':>8}")
    for k in res:
        r = res[k]
        print(f"    {k:>10}{-np.mean(r['brier']):>10.4f}{-np.mean(r['ece']):>9.4f}"
              f"{np.mean(r['acc']):>9.4f}{np.mean(r['pmin']):>8.3f}"
              f"{np.mean(r['pmax']):>8.3f}")
    for cand in ("isotonic", "iso_clip"):
        for metric, label in (("brier", "Brier"), ("ece", "ECE")):
            d, sd, t, _ = paired_t(np.array(res["platt"][metric]),
                                   np.array(res[cand][metric]))
            verdict = ("采纳" if abs(t) > T_ACCEPT and d > 0
                       else "拒绝 (变差)" if abs(t) > T_ACCEPT
                       else "噪声内, 不采纳")
            print(f"    {cand:>10} vs platt  {label:<6} Δ={d:+.5f}  "
                  f"t={t:+.2f} -> {verdict}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=6)
    a = ap.parse_args()

    files = [str(_ROOT / "data" /
                 f"{y}_LoL_esports_match_data_from_OraclesElixir.csv")
             for y in (2022, 2023, 2024, 2025, 2026)]
    print("构建训练矩阵…")
    mdf = build_training_matrix(files, verbose=False)
    print(f"  {len(mdf)} 场")
    for stage in ("pre_draft", "post_draft"):
        run_stage(mdf, stage, a.folds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
