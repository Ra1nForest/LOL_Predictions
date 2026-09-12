"""
闸门: BP 信息 (Stage 2) 到底值不值这 33 个特征?
================================================

当前 artifacts 上的留出集数字:

    pre_draft   准确率 0.5966   Brier 0.2391   ECE 0.0426    79 特征
    post_draft  准确率 0.5972   Brier 0.2365   ECE 0.0455   112 特征

准确率只差 +0.0006 —— 1703 局里约 1 局, 大约 0.05 个标准误。单次留出集
比不出来, 所以这里走配对前进检验: 同一批 fold, 两套特征各训一遍, 逐 fold
相减, 消掉「这段时期有多难」。

三个口径分开判, 因为它们可能给出不同答案:
    · 归一化 lift (准确率口径) —— walk_forward 的默认分数
    · Brier      —— 概率质量, BP 可能只在这里有用
    · 对数损失

判据统一用 harness.paired_t 的 |t| > 2.5。

用法: python research/gate_draft_lift.py [--folds 6] [--seeds 3]
"""
from __future__ import annotations
import argparse, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT)); sys.path.insert(0, str(_ROOT / "research"))
from feature_store import build_training_matrix, select_features
from harness import from_folds, gate, T_ACCEPT

YEARS = (2022, 2023, 2024, 2025, 2026)
XGB_P = dict(n_estimators=400, max_depth=5, learning_rate=0.04,
             subsample=0.8, colsample_bytree=0.7,
             eval_metric="logloss", verbosity=0)


def run(mdf, feats, cuts, val_size, seeds):
    """返回 {口径: 逐 fold 分数}。"""
    X = mdf[feats].astype(float).fillna(-999)
    y = mdf["y"].astype(int)
    out = {"lift": [], "brier": [], "logloss": []}
    for cut in cuts:
        Xtr, ytr = X.iloc[:cut], y.iloc[:cut]
        Xva, yva = X.iloc[cut:cut + val_size], y.iloc[cut:cut + val_size]
        accs, brs, lls = [], [], []
        for s in seeds:
            m = XGBClassifier(random_state=s, **XGB_P)
            m.fit(Xtr, ytr, verbose=False)
            p = m.predict_proba(Xva)[:, 1]
            base = max(yva.mean(), 1 - yva.mean())
            accs.append((accuracy_score(yva, (p >= .5).astype(int)) - base) / (1 - base))
            brs.append(-brier_score_loss(yva, p))      # 取负, 越大越好
            lls.append(-log_loss(yva, p, labels=[0, 1]))
        out["lift"].append(np.mean(accs))
        out["brier"].append(np.mean(brs))
        out["logloss"].append(np.mean(lls))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    seeds = list(range(a.seeds))

    print("=" * 72)
    print("  闸门: BP 信息 (Stage 2) 相对 Stage 1 的增量")
    print("=" * 72)

    files = [str(_ROOT / "data" / f"{y}_LoL_esports_match_data_from_OraclesElixir.csv")
             for y in YEARS]
    print("\n[1/2] 构建训练矩阵")
    mdf = build_training_matrix(files, verbose=False)
    mdf = mdf.sort_values("date").reset_index(drop=True) if "date" in mdf.columns else mdf
    f_pre = select_features(mdf, "pre_draft")
    f_post = select_features(mdf, "post_draft")
    added = [c for c in f_post if c not in f_pre]
    print(f"      {len(mdf)} 局   pre_draft {len(f_pre)} 特征   "
          f"post_draft {len(f_post)} 特征 (+{len(added)})")

    val_size = max(300, int(len(mdf) * 0.12))
    start, end = int(len(mdf) * 0.40), len(mdf) - val_size
    cuts = np.linspace(start, end, a.folds).astype(int)
    print(f"\n[2/2] {a.folds} 折 × {len(seeds)} 种子   验证窗口 {val_size} 局")

    r_pre = run(mdf, f_pre, cuts, val_size, seeds)
    r_post = run(mdf, f_post, cuts, val_size, seeds)

    NAMES = {"lift": "归一化 lift (准确率口径)",
             "brier": "负 Brier (概率质量)",
             "logloss": "负对数损失"}
    print()
    accepted = []
    for k in ("lift", "brier", "logloss"):
        print("-" * 72)
        print(f"  口径: {NAMES[k]}")
        ok = gate(from_folds(r_pre[k], "pre_draft"),
                  from_folds(r_post[k], "post_draft"),
                  name=f"加入 BP 特征 (+{len(added)} 个)")
        accepted.append((k, ok))
    print("=" * 72)
    hit = [NAMES[k] for k, ok in accepted if ok]
    if hit:
        print(f"  结论: BP 特征在 {', '.join(hit)} 上通过闸门。")
    else:
        print("  结论: BP 特征在所有口径上都没能通过闸门 —— 33 个特征、"
              "整条选人输入链路, 换不来可测量的提升。")


if __name__ == "__main__":
    sys.exit(main())
