"""
Walk-Forward Validation Harness
=================================
问题: 单次时序切分只给一个数, 无从知道它的方差。
      我们已经被一个 +3.1pp 的假发现骗过一次。

方法: 沿时间轴滚动多个 fold, 每个 fold 跑多个种子。
      得到 mean ± SD, 然后用 SD 当噪声地板去判断任何改动。

判据: 配对 t 检验, |t| > 2.5 才采纳。

      t = Δ / (配对 SD / √n)

      为什么不是 ratio = |Δ| / SD: 那是效果量, 分母不含 √n, 所以加 fold
      不会让它变大 —— 一个真实但微小的提升, 无论测多少次 ratio 都停在
      原地, 永远无法累积到「可以相信」。效果量回答「这个差异有多大」,
      决策规则要回答「我有多确信它不是零」, 是两个问题。
      ratio 仍然打印, 但只作参考。
"""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, r2_score


# ══════════════════════════════════════════════════════
#  核心: walk-forward 评估
# ══════════════════════════════════════════════════════

def walk_forward(X, y, dates, kind="clf",
                 n_folds=5, seeds=(0, 1, 2, 3, 4),
                 min_train_frac=0.40, val_frac=0.12,
                 params=None, sample_weight_fn=None,
                 verbose=False):
    """
    扩展窗口 walk-forward。

    fold k:  train = [0, t_k)   val = [t_k, t_k + val_size)
    t_k 从 min_train_frac 开始, 均匀推进到数据末尾。

    返回 每个 (fold, seed) 的分数数组。
    """
    n = len(X)
    val_size = max(80, int(n * val_frac))
    start = int(n * min_train_frac)
    end = n - val_size
    if end <= start:
        raise ValueError("数据不足以切出这么多 fold")
    cuts = np.linspace(start, end, n_folds).astype(int)

    if params is None:
        params = (dict(n_estimators=400, max_depth=5, learning_rate=0.04,
                       subsample=0.8, colsample_bytree=0.7,
                       eval_metric="logloss", verbosity=0)
                  if kind == "clf" else
                  dict(n_estimators=150, max_depth=2, learning_rate=0.03,
                       reg_lambda=10, min_child_weight=20,
                       subsample=0.8, colsample_bytree=0.7,
                       objective="reg:squarederror", verbosity=0))

    Model = XGBClassifier if kind == "clf" else XGBRegressor
    scores = np.zeros((n_folds, len(seeds)))

    for fi, cut in enumerate(cuts):
        Xtr, ytr = X.iloc[:cut], y.iloc[:cut]
        Xva, yva = X.iloc[cut:cut + val_size], y.iloc[cut:cut + val_size]

        w = None
        if sample_weight_fn is not None:
            w = sample_weight_fn(dates.iloc[:cut], dates.iloc[cut])

        for si, seed in enumerate(seeds):
            m = Model(random_state=seed, **params)
            m.fit(Xtr, ytr, sample_weight=w, verbose=False)
            if kind == "clf":
                acc = accuracy_score(yva, m.predict(Xva))
                base = max(yva.mean(), 1 - yva.mean())
                scores[fi, si] = (acc - base) / (1 - base)   # 归一化 lift
            else:
                scores[fi, si] = r2_score(yva, m.predict(Xva))

        if verbose:
            print(f"      fold {fi+1}: train={cut:>5}  "
                  f"val=[{dates.iloc[cut].date()} → {dates.iloc[cut+val_size-1].date()}]  "
                  f"score={scores[fi].mean():+.4f}")

    return scores


def summarize(scores, label=""):
    """fold 均值、fold 间标准差、种子间标准差。scores 形状 (n_folds, n_seeds)。"""
    fold_means = scores.mean(axis=1)
    return {
        "label": label,
        "mean": fold_means.mean(),
        "fold_sd": fold_means.std(ddof=1),
        "seed_sd": scores.std(axis=1).mean(),
        "folds": fold_means,
    }


def from_folds(fold_means, label=""):
    """已经按种子平均过的逐 fold 分数 → gate() 能吃的结构。

    给自带切分逻辑的调用方用: Stage 4 必须按 gameid 分组切分, 不能用
    walk_forward 的按行切分, 但判决规则必须和这里保持同一份。
    """
    f = np.asarray(fold_means, dtype=float)
    return {
        "label": label,
        "mean": f.mean(),
        "fold_sd": f.std(ddof=1),
        "seed_sd": float("nan"),   # 已在上游被平均掉, 无法还原
        "folds": f,
    }


T_ACCEPT = 2.5


def paired_t(baseline_folds, candidate_folds):
    """配对 t 检验的算术。返回 (Δ, 配对SD, t, 效果量)。

    这是全项目唯一一份。之前 ingame_model 和 gate_ingame_features 各自内联
    抄过一遍, 抄漏的都是同一个分支 (配对 SD 为零), 结果一致的稳定效应被判成
    噪声 —— 三份实现意味着修一处不会修好另外两处。
    """
    a = np.asarray(baseline_folds, dtype=float)
    b = np.asarray(candidate_folds, dtype=float)
    paired = b - a                      # 逐 fold 相减, 消掉「这段时期有多难」
    delta = paired.mean()
    paired_sd = paired.std(ddof=1)
    n = len(paired)

    # 容差而不是 > 0: 逐 fold 差值理论上相等时, 浮点残差会留下 ~1e-18 的
    # SD, 除出来是 1e15 量级的 t —— 方向对, 但打出来像 bug。真实的配对 SD
    # 在 1e-3 量级, 1e-12 远在其下。
    if paired_sd > 1e-12:
        t = delta / (paired_sd / np.sqrt(n))
        ratio = abs(delta) / paired_sd
    else:
        # 逐 fold 差值完全一致。此时不确定性为零:
        # Δ = 0 是「毫无影响」, Δ ≠ 0 是「每个 fold 都同向」, 证据拉满。
        t = 0.0 if abs(delta) <= 1e-12 else np.inf * np.sign(delta)
        ratio = np.inf if abs(delta) > 1e-12 else 0.0
    return delta, paired_sd, t, ratio


def gate(baseline, candidate, name=""):
    """配对 t 检验判定一个改动值不值得采纳。返回 True 表示采纳。"""
    paired = np.asarray(candidate["folds"]) - np.asarray(baseline["folds"])
    delta, paired_sd, t, ratio = paired_t(baseline["folds"], candidate["folds"])
    n = len(paired)

    # 判决只看 t。方向由 t 的符号给出, 不需要再单独判 delta。
    if t >= T_ACCEPT:
        verdict, mark = "采纳", "✓"
    elif t <= -T_ACCEPT:
        verdict, mark = f"拒绝 (稳定变差, t = {t:+.2f})", "✗"
    elif abs(t) >= 1.5:
        verdict, mark = f"存疑 (|t| = {abs(t):.2f} < {T_ACCEPT}, 需更多 fold)", "?"
    else:
        verdict, mark = "丢弃 (噪声内)", "✗"

    print(f"  {mark} {name}")
    print(f"      baseline  {baseline['mean']:+.4f}  (fold SD {baseline['fold_sd']:.4f})")
    print(f"      candidate {candidate['mean']:+.4f}  (fold SD {candidate['fold_sd']:.4f})")
    print(f"      Δ = {delta:+.4f}   配对 SD {paired_sd:.4f}   n = {n}")
    print(f"      t = {t:+.2f}   (判据 |t| > {T_ACCEPT})"
          f"   ·  效果量 {ratio:.2f}x SD, 仅供参考")
    print(f"      → {verdict}")
    print(f"      逐 fold Δ: {'  '.join(f'{d:+.3f}' for d in paired)}")
    return t >= T_ACCEPT


def report(res):
    print(f"  {res['label']}")
    print(f"      mean {res['mean']:+.4f}   fold SD {res['fold_sd']:.4f}   "
          f"seed SD {res['seed_sd']:.4f}")
    print(f"      folds: {'  '.join(f'{f:+.3f}' for f in res['folds'])}")
    print(f"      fold SD 是验证窗口难度的属性, 不是模型的 —— 加训练数据不会让它变小。")
    print(f"      单看它不构成判据: 改动值不值得采纳由 gate() 的配对 t 决定。")
