"""
训练两个模型 + 校准器
=======================
  pre_draft   仅队伍滚动统计        — BP 之前可用
  post_draft  队伍统计 + draft 特征  — BP 之后可用

各自独立训练和校准, 保存到 artifacts/。

配方 (闸门 G1, 2026-10-17: research/gate_pregame_package.py)
--------------------------------------------------------------
预先登记的组合包 P 两个阶段都过闸, 按事先写死的简洁规则 (逐项剔除 t < 1.0 就不上线) 剩下 {a, d}:

  (a) 浅树: n_estimators=300, max_depth=3, learning_rate=0.03, min_child_weight=10, 其余照旧。
      约 5 千行的四大训练集上原来 400 棵 depth-5 的树, 原始输出的校准斜率只有 ~0.5, 典型的过拟合。
  (d) 赛区内 Elo 进校准层: Platt 的单输入 (原始概率) 换成 logistic 回归
      [logit(clip(p_raw)), elo_gap · ln(10)/400], C = 1e6, 在校准段上拟合。Elo 不进树
      (Elo 的定义只在 feature_store.py: elo_games / elo_replay / elo_rating)。
  这个组合是补充臂 B0+a+d: pre Brier t +6.55 (10/10 块), post t +4.67 (9/10), 准确率约 65%。
  次级联赛训练行 (b) 和选手-英雄胜率收缩 (c) 被简洁规则去掉了, 不上线。

没换的: 60/20/20 按时间顺序切分 (Stage 1/2 的"服务模型重训"没过闸, 不像 Stage 4 那样分两次拟合);
XGB 线程数不传 —— XGBoost 的随机流按线程切分, 闸门复核就是按默认线程数判的。
保序回归也不用: 它在 Stage 1/2 上过拟合 (t = -4.74, 见 CLAUDE.md)。

calib_<stage>.json
------------------
  calibrator  "logit_elo"。api.Stage 和 web/stage1.ts 按它选校准方式; 没有这个字段 (旧文件) 就是 Platt。
  logit_elo   {coef: [logit 系数, Elo 系数], intercept, logit_clip, elo_scale, elo_weight, elo_sd,
               elo: {column, k, init, regress}} —— 线上 σ((logit(clip(p))·c₁ + (elo_gap·scale)·c₂) + b)。
               elo 那几个常数是训练时 feature_store 的值; api.Stage 加载时核对, 对不上就拒绝加载
               (特征库按自己的常数算 elo_gap, 和训练时不同不会报错, 只会给出一个看着合理的错数)。
               elo_weight 是 Elo 在校准层里的份量 (|系数|×标准差的占比); elo_sd 是校准段上 elo_gap 的标准差
               (Elo 分)。top_evidence 给 Elo 那一条的权重是 elo_weight × min(1, |这一场的差| / elo_sd) ——
               只拿 elo_weight 当常数时, 两队差 7 分也排第一条理由 (见 api.Stage.top_evidence)。
  coef / intercept  同一个校准段上拟的**兼容 Platt** (只吃原始概率), 新代码不用 —— 留着是为了让旧版服务
               还能加载新文件 (合并后、下次重训前, 线上可能是新代码 + 旧文件, 也可能反过来; 两个方向都要能跑)。
  metrics     和以前同一口径 (测试段 = 最近 20%), 概率是 logit_elo 校准后的; 另带 compat_platt (同一批树、
              兼容 Platt 的测试段指标), 用来看 Elo 在同一批树上带来多少。daily_update 只读 accuracy / ece_cal / n_*。
"""
import numpy as np, pandas as pd, json, os, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from feature_store import (ELO_COL, ELO_INIT, ELO_K, ELO_REG, ELO_SCALE, LOGIT_CLIP,
                           build_training_matrix, logit_elo_inputs, select_features)

_ROOT = Path(__file__).parent
DATA = [str(_ROOT / "data" / f"{y}_LoL_esports_match_data_from_OraclesElixir.csv")
        for y in (2022, 2023, 2024, 2025, 2026)]
# 无人值守的流水线会把 LOL_ARTIFACTS 指到暂存目录, 先训出来比指标,
# 通过了再换掉线上的 artifacts/。见 daily_update.py。
OUT = Path(os.environ.get("LOL_ARTIFACTS") or (_ROOT / "artifacts"))
OUT.mkdir(parents=True, exist_ok=True)

# (a) 浅树 —— 闸门 G1 的 A_PARAMS 盖在原参数上, 其余 (subsample / colsample / 种子) 不变
XGB = dict(n_estimators=300, max_depth=3, learning_rate=0.03, min_child_weight=10,
           subsample=0.8, colsample_bytree=0.7,
           eval_metric="logloss", random_state=42, verbosity=0)
TRAIN_FRAC, CAL_FRAC = 0.60, 0.20
CALIBRATOR = "logit_elo"


def ece(p, y, nb=10):
    e, ed = 0.0, np.linspace(0, 1, nb + 1)
    for i in range(nb):
        m = (p > ed[i]) & (p <= ed[i+1]) if i else (p >= 0) & (p <= ed[1])
        if m.sum(): e += m.sum()/len(p) * abs(p[m].mean() - y[m].mean())
    return e


def fit_logit_elo(p_cal, e_cal, y_cal) -> LogisticRegression:
    """(d) 的校准层: logistic 回归 (C = 1e6, 等于不加正则) 吃 [logit(clip(p_raw)), elo_gap · ln10/400]。"""
    return LogisticRegression(C=1e6, max_iter=1000).fit(logit_elo_inputs(p_cal, e_cal), y_cal)


def elo_weight(lr: LogisticRegression, p_cal, e_cal) -> float:
    """Elo 在校准层里的份量: |c₂|·sd(Elo 输入) / (|c₁|·sd(logit 输入) + |c₂|·sd(Elo 输入)), 校准段上算。

    top_evidence 拿它当 Elo 那一条依据的满份量。树的特征重要性只描述树内部, Elo 不在树里, 没有现成的
    重要性可用; 这个占比说的是"校准后的对数几率里, 典型的一场有多少来自 Elo" —— 闸门的收益几乎全来自它,
    理由里不出现它反而是在误导。但它说的是**典型**的一场: 每一场按 |Elo 差| / elo_sd 缩放 (封顶 1), 见
    api.Stage.top_evidence; 不缩放时两队差几分也永远排第一。
    """
    Z = logit_elo_inputs(p_cal, e_cal)
    a = np.abs(lr.coef_[0]) * Z.std(axis=0)
    return float(a[1] / a.sum())


def _scores(y, p) -> dict:
    acc = accuracy_score(y, (p > .5).astype(int))
    return {"accuracy": float(acc), "brier": float(brier_score_loss(y, p)),
            "logloss": float(log_loss(y, p)), "ece_cal": float(ece(p, y))}


def train_stage(mdf, stage):
    feats = select_features(mdf, stage)
    if ELO_COL in feats:
        raise SystemExit(f"{ELO_COL} 进了树的特征 —— Elo 只该进校准层, 见 feature_store.select_features")
    X = mdf[feats].astype(float).fillna(-999)
    y = mdf["y"].astype(int)
    e = mdf[ELO_COL].astype(float).to_numpy()
    if np.isnan(e).any():
        raise SystemExit(f"{int(np.isnan(e).sum())} 行没有 {ELO_COL} —— Elo 重放没覆盖到训练行")
    n = len(mdf); i_tr, i_cal = int(n*TRAIN_FRAC), int(n*(TRAIN_FRAC+CAL_FRAC))

    model = XGBClassifier(**XGB)
    model.fit(X.iloc[:i_tr], y.iloc[:i_tr], verbose=False)
    p_cal = model.predict_proba(X.iloc[i_tr:i_cal])[:, 1]
    y_cal, e_cal = y.iloc[i_tr:i_cal].to_numpy(), e[i_tr:i_cal]
    lr = fit_logit_elo(p_cal, e_cal, y_cal)
    # 兼容 Platt: 旧版服务只认 coef / intercept (见文档); 同一个校准段, 和以前的拟法一字不差
    platt = LogisticRegression(C=1e6).fit(p_cal.reshape(-1, 1), y_cal)

    Xte, yte = X.iloc[i_cal:], y.iloc[i_cal:].values
    raw = model.predict_proba(Xte)[:, 1]
    cal = lr.predict_proba(logit_elo_inputs(raw, e[i_cal:]))[:, 1]
    cal_platt = platt.predict_proba(raw.reshape(-1, 1))[:, 1]
    base = max(yte.mean(), 1-yte.mean())
    acc = accuracy_score(yte, (cal > .5).astype(int))
    w = elo_weight(lr, p_cal, e_cal)

    metrics = {
        "stage": stage, "n_features": len(feats),
        "n_train": int(i_tr), "n_cal": int(i_cal-i_tr), "n_test": int(n-i_cal),
        "accuracy": float(acc), "baseline": float(base),
        "lift": float((acc-base)/(1-base)),
        "brier": float(brier_score_loss(yte, cal)),
        "brier_constant": float(yte.mean()*(1-yte.mean())),
        "logloss": float(log_loss(yte, cal)),
        "ece_raw": float(ece(raw, yte)), "ece_cal": float(ece(cal, yte)),
        # 测试段上校准后概率的两端 (和以前同一个口径)。logit_elo 没有硬上下限 —— Elo 差够大时能越过它们
        "p_min": float(cal.min()), "p_max": float(cal.max()),
        "calibrator": CALIBRATOR,
        # 同一批树换回只吃原始概率的 Platt: Elo 在同一批树上带来多少, 一眼看得出来
        "compat_platt": _scores(yte, cal_platt),
    }

    model.save_model(str(OUT / f"model_{stage}.json"))
    with open(OUT / f"calib_{stage}.json", "w") as fh:
        json.dump({"calibrator": CALIBRATOR,
                   "logit_elo": {"coef": [float(v) for v in lr.coef_[0]],
                                 "intercept": float(lr.intercept_[0]),
                                 "inputs": ["logit(clip(p_raw))", f"{ELO_COL} * elo_scale"],
                                 "logit_clip": LOGIT_CLIP, "elo_scale": ELO_SCALE,
                                 "elo_weight": w,
                                 # 校准段上 elo_gap 的标准差 (Elo 分, 和 elo_weight 同一段、同一个 np.std):
                                 # Elo 那条依据的权重按 |这一场的差| / elo_sd 缩放, 见 api.Stage.top_evidence
                                 "elo_sd": float(np.std(e_cal)),
                                 "elo": {"column": ELO_COL, "k": ELO_K, "init": ELO_INIT,
                                         "regress": ELO_REG}},
                   # 兼容 Platt —— 只给旧版服务读 (见文档)
                   "coef": float(platt.coef_[0][0]),
                   "intercept": float(platt.intercept_[0]),
                   "xgb": {k: v for k, v in XGB.items() if k != "verbosity"},
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
        cp = m["compat_platt"]
        print(f"\n  ── {stage} ──")
        print(f"     特征 {m['n_features']}   训练 {m['n_train']} / 校准 {m['n_cal']} / 测试 {m['n_test']}")
        print(f"     准确率 {m['accuracy']:.1%}  (基线 {m['baseline']:.1%}, lift {m['lift']:+.3f})")
        print(f"     Brier {m['brier']:.4f}  (常数 {m['brier_constant']:.4f})")
        print(f"     ECE   {m['ece_raw']:.4f} → {m['ece_cal']:.4f} (校准后)")
        print(f"     概率范围 [{m['p_min']:.3f}, {m['p_max']:.3f}]  (测试段两端, 不是硬上下限)")
        print(f"     同一批树只用 Platt: 准确率 {cp['accuracy']:.1%}  Brier {cp['brier']:.4f}  "
              f"ECE {cp['ece_cal']:.4f}  (Elo 进校准层之前)")

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
