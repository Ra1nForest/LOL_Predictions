"""
训练两个模型 + 校准器
=======================
  pre_draft   仅队伍滚动统计        — BP 之前可用
  post_draft  队伍统计 + draft 特征  — BP 之后可用

各自独立训练和校准, 保存到 artifacts/。
"""
import numpy as np, pandas as pd, json, os, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from feature_store import build_training_matrix, select_features

_ROOT = Path(__file__).parent
DATA = [str(_ROOT / "data" / f"{y}_LoL_esports_match_data_from_OraclesElixir.csv")
        for y in (2022, 2023, 2024, 2025, 2026)]
# 无人值守的流水线会把 LOL_ARTIFACTS 指到暂存目录, 先训出来比指标,
# 通过了再换掉线上的 artifacts/。见 daily_update.py。
OUT = Path(os.environ.get("LOL_ARTIFACTS") or (_ROOT / "artifacts"))
OUT.mkdir(parents=True, exist_ok=True)

XGB = dict(n_estimators=400, max_depth=5, learning_rate=0.04,
           subsample=0.8, colsample_bytree=0.7,
           eval_metric="logloss", random_state=42, verbosity=0)
TRAIN_FRAC, CAL_FRAC = 0.60, 0.20


def ece(p, y, nb=10):
    e, ed = 0.0, np.linspace(0, 1, nb + 1)
    for i in range(nb):
        m = (p > ed[i]) & (p <= ed[i+1]) if i else (p >= 0) & (p <= ed[1])
        if m.sum(): e += m.sum()/len(p) * abs(p[m].mean() - y[m].mean())
    return e


def train_stage(mdf, stage):
    feats = select_features(mdf, stage)
    X = mdf[feats].astype(float).fillna(-999)
    y = mdf["y"].astype(int)
    n = len(mdf); i_tr, i_cal = int(n*TRAIN_FRAC), int(n*(TRAIN_FRAC+CAL_FRAC))

    model = XGBClassifier(**XGB)
    model.fit(X.iloc[:i_tr], y.iloc[:i_tr], verbose=False)
    p_cal = model.predict_proba(X.iloc[i_tr:i_cal])[:, 1]
    calib = LogisticRegression(C=1e6).fit(p_cal.reshape(-1,1), y.iloc[i_tr:i_cal])

    Xte, yte = X.iloc[i_cal:], y.iloc[i_cal:].values
    raw = model.predict_proba(Xte)[:, 1]
    cal = calib.predict_proba(raw.reshape(-1,1))[:, 1]
    base = max(yte.mean(), 1-yte.mean())
    acc = accuracy_score(yte, (cal > .5).astype(int))

    metrics = {
        "stage": stage, "n_features": len(feats),
        "n_train": int(i_tr), "n_cal": int(i_cal-i_tr), "n_test": int(n-i_cal),
        "accuracy": float(acc), "baseline": float(base),
        "lift": float((acc-base)/(1-base)),
        "brier": float(brier_score_loss(yte, cal)),
        "brier_constant": float(yte.mean()*(1-yte.mean())),
        "logloss": float(log_loss(yte, cal)),
        "ece_raw": float(ece(raw, yte)), "ece_cal": float(ece(cal, yte)),
        "p_min": float(cal.min()), "p_max": float(cal.max()),
    }

    model.save_model(str(OUT / f"model_{stage}.json"))
    with open(OUT / f"calib_{stage}.json", "w") as fh:
        json.dump({"coef": float(calib.coef_[0][0]),
                   "intercept": float(calib.intercept_[0]),
                   "features": feats,
                   "metrics": metrics}, fh, indent=2)
    return metrics


def main():
    print("="*66); print("  训练分段模型"); print("="*66)
    print("\n[1/2] 构建训练矩阵")
    mdf = build_training_matrix(DATA)

    print("\n[2/2] 训练两个阶段")
    results = {}
    for stage in ("pre_draft", "post_draft"):
        m = train_stage(mdf, stage)
        results[stage] = m
        print(f"\n  ── {stage} ──")
        print(f"     特征 {m['n_features']}   训练 {m['n_train']} / 校准 {m['n_cal']} / 测试 {m['n_test']}")
        print(f"     准确率 {m['accuracy']:.1%}  (基线 {m['baseline']:.1%}, lift {m['lift']:+.3f})")
        print(f"     Brier {m['brier']:.4f}  (常数 {m['brier_constant']:.4f})")
        print(f"     ECE   {m['ece_raw']:.4f} → {m['ece_cal']:.4f} (校准后)")
        print(f"     概率范围 [{m['p_min']:.3f}, {m['p_max']:.3f}]")

    d = results["post_draft"]["lift"] - results["pre_draft"]["lift"]
    print(f"\n{'='*66}")
    print(f"  draft 信息带来的增益: {d:+.4f} lift")
    print(f"  ⚠ 单次测试集对比, 未过 walk-forward 闸。参考用, 勿当结论。")
    print(f"{'='*66}")
    with open(OUT / "summary.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"  artifacts/ 已写入 4 个文件")


if __name__ == "__main__":
    main()
