"""
闸门: champ_wr 用收缩 (shrinkage) 替掉 `len(p) >= 3` 的硬阈值?
================================================================

现状 (feature_store.py:169)::

    wr = float(np.mean(p)) if len(p) >= 3 else np.nan

两个毛病, 都是"不报错但给错权重"那一类:

1. **3 场和 300 场同权。** 选手用某英雄 3 战 2 胜是 0.667, 300 战 200 胜也是
   0.667。前者是噪声, 后者是水平, 模型看到的是同一个数。
2. **阈值是断崖。** 2 场 -> NaN (下游 fillna(-999), 等于"无信息"哨兵),
   3 场 -> 0.667 (当成可信)。差一场比赛, 从一无所知跳到言之凿凿。

收缩把断崖抹平, 按样本量往先验拉::

    wr = (胜场 + K * 0.5) / (场次 + K)

**先验取 0.5 而不是全局均值, 是刻意的**: 每场职业比赛恰好一胜一负, 全局
选手-英雄胜率在构造上就是 0.5。用 0.5 既是理论正确值, 又完全不碰测试期数据,
不存在泄漏 —— 而"从全量数据算个全局均值"会把未来信息渗进先验。

K 控制拉多狠, K 越大越保守。三个候选 5 / 10 / 20。

多重比较的警告
--------------
一次试 3 个 K, 就是 3 次独立机会去撞 |t| > 2.5。单次阈值在这里**偏松**。
即使某个 K 过了闸, 也只能当"值得复核", 不能直接采纳 —— 正确的做法是拿过闸
的那个 K 单独再跑一次全新的折。脚本会把这句话打进结论里。

判据: 主看准确率与 Brier, |t| > 2.5 (harness.T_ACCEPT)。

设计说明
--------
* champ_wr 属于 DRAFT_PREFIXES, `pre_draft` 阶段本来就看不到它 —— 所以只
  测 `post_draft`, 省掉一半算力。
* 两臂下游处理完全一致 (都不校准), 保证比的是特征本身。
* **每折算完立刻落盘**。这个实验是在一台随时会被回收的机器上跑的, 中途断掉
  也要留下能用的配对折。

用法::

    python research/gate_champ_shrinkage.py --folds 8 --seeds 3
    python research/gate_champ_shrinkage.py --ks 10        # 只复核单个 K
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
from sklearn.metrics import accuracy_score, brier_score_loss
from xgboost import XGBClassifier

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "research"))

import feature_store as fs
from feature_store import build_training_matrix, select_features
from harness import T_ACCEPT, paired_t

# 照抄 train.py, 否则测的不是线上那个模型
XGB = dict(n_estimators=400, max_depth=5, learning_rate=0.04,
           subsample=0.8, colsample_bytree=0.7,
           eval_metric="logloss", verbosity=0)
TEST_N = 1000
PRIOR = 0.5             # 见模块文档: 构造上的真值, 且不碰测试期数据

_ORIG = fs.FeatureStore.player_champ_stats

# 只有这 10 列来自 player_champ_stats。
# **命名陷阱**: `b_avg_champ_wr` 看着像这 5 个位置的平均, 其实完全不是 ——
# 它来自 store.champ_winrate() (英雄在该赛区的胜率), 跟选手-英雄胜率是两个
# 不相干的量, 本改动碰不到它。第一版验证脚本就查错了这一列, 差点把"生效了"
# 误判成"没生效"。
WR_COLS = [f"{s}_{r}_champ_wr" for s in ("b", "r") for r in fs.ROLES]


def _make_shrunk(k: float):
    """返回一个把 (胜场, 场次) 收缩到先验的 player_champ_stats。"""
    def player_champ_stats(self, player, champion, before=None):
        recs = self.player_champ.get((player, champion), [])
        p = [r for d, r in recs if before is None or d < before]
        n = len(p)
        # 注意 n = 0 也返回数值 (= PRIOR), 不再是 NaN —— 去掉哨兵是这个改动
        # 的一半意义所在。
        return (float(np.sum(p)) + k * PRIOR) / (n + k), n
    return player_champ_stats


def build(files, k, log):
    """k 为 None 表示基线 (原始硬阈值)。"""
    tag = "基线(>=3)" if k is None else f"收缩 K={k}"
    fs.FeatureStore.player_champ_stats = _ORIG if k is None else _make_shrunk(k)
    t0 = time.time()
    mdf = build_training_matrix(files, verbose=False)
    fs.FeatureStore.player_champ_stats = _ORIG          # 立刻还原, 别污染下一轮
    have = [c for c in WR_COLS if c in mdf.columns]
    miss = float(np.mean([mdf[c].isna().mean() for c in have]))
    log(f"  {tag:>12}: {len(mdf)} 场, 建矩阵 {time.time()-t0:.0f}s, "
        f"逐位置 champ_wr 缺失 {miss:.1%} ({len(have)} 列)")
    return mdf


def ece(p, y, nb=10):
    edges = np.linspace(0, 1, nb + 1)
    out = 0.0
    for i in range(nb):
        m = (p >= edges[i]) & (p < edges[i + 1])
        if m.sum():
            out += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(out)


def eval_fold(mdf, feats, cut, seeds):
    """一折: 训 [0,cut), 测 [cut,cut+TEST_N)。多种子取平均压掉 XGB 自身抖动。"""
    X = mdf[feats].astype(float).fillna(-999).reset_index(drop=True)
    y = mdf["y"].astype(int).reset_index(drop=True)
    tr, te = slice(0, cut), slice(cut, cut + TEST_N)
    accs, briers, eces = [], [], []
    for s in range(seeds):
        m = XGBClassifier(**XGB, random_state=42 + s)
        m.fit(X.iloc[tr], y.iloc[tr], verbose=False)
        p = m.predict_proba(X.iloc[te])[:, 1]
        yt = y.iloc[te].values
        accs.append(accuracy_score(yt, (p > .5).astype(int)))
        briers.append(-brier_score_loss(yt, p))     # 取负, 统一"越大越好"
        eces.append(-ece(p, yt))
    return dict(acc=float(np.mean(accs)), brier=float(np.mean(briers)),
                ece=float(np.mean(eces)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--ks", type=float, nargs="*", default=[5, 10, 20])
    ap.add_argument("--out", default=str(_ROOT / "research" / "champ_shrinkage_result.json"))
    a = ap.parse_args()

    out_path = Path(a.out)
    logf = out_path.with_suffix(".log").open("w", encoding="utf-8")

    def log(msg=""):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log("闸门: champ_wr 收缩 vs >=3 硬阈值")
    log(f"  折数 {a.folds}  种子 {a.seeds}  K {a.ks}  判据 |t| > {T_ACCEPT}")
    log()

    files = [str(_ROOT / "data" /
                 f"{y}_LoL_esports_match_data_from_OraclesElixir.csv")
             for y in (2022, 2023, 2024, 2025, 2026)]

    log("构建训练矩阵 (每个变体一次)…")
    arms = {"baseline": build(files, None, log)}
    for k in a.ks:
        arms[f"K={k:g}"] = build(files, k, log)

    n = len(arms["baseline"])
    feats = select_features(arms["baseline"], "post_draft")
    lo, hi = int(n * 0.50), n - TEST_N
    cuts = np.linspace(lo, hi, a.folds).astype(int)
    log(f"\npost_draft: {len(feats)} 特征, {n} 场, "
        f"切分点 {cuts[0]}…{cuts[-1]}, 每折测 {TEST_N} 场\n")

    # 每折算完立刻写盘 —— 机器随时可能被回收
    res = {name: {m: [] for m in ("acc", "brier", "ece")} for name in arms}
    state = {"folds_done": 0, "cuts": cuts.tolist(), "n": n,
             "feats": len(feats), "ks": a.ks, "seeds": a.seeds, "res": res}

    for i, cut in enumerate(cuts):
        t0 = time.time()
        line = []
        for name, mdf in arms.items():
            r = eval_fold(mdf, feats, int(cut), a.seeds)
            for m in ("acc", "brier", "ece"):
                res[name][m].append(r[m])
            line.append(f"{name} {r['acc']:.4f}")
        state["folds_done"] = i + 1
        out_path.write_text(json.dumps(state, ensure_ascii=False, indent=1),
                            encoding="utf-8")
        log(f"  fold {i+1}/{a.folds} (训 {cut:>5}, {time.time()-t0:.0f}s)  "
            + "  ".join(line))

    log("\n" + "=" * 64)
    log(f"{'变体':>12}{'准确率':>10}{'Brier':>10}{'ECE':>10}")
    for name in arms:
        r = res[name]
        log(f"{name:>12}{np.mean(r['acc']):>10.4f}"
            f"{-np.mean(r['brier']):>10.4f}{-np.mean(r['ece']):>10.4f}")

    log("\n配对 t 检验 (对基线):")
    passed = []
    for name in arms:
        if name == "baseline":
            continue
        for metric, label in (("acc", "准确率"), ("brier", "Brier"), ("ece", "ECE")):
            d, sd, t, _ = paired_t(res["baseline"][metric], res[name][metric])
            if t >= T_ACCEPT:
                verdict = "采纳(但见多重比较警告)"
                passed.append((name, label))
            elif t <= -T_ACCEPT:
                verdict = "拒绝(稳定变差)"
            elif abs(t) >= 1.5:
                verdict = "存疑"
            else:
                verdict = "噪声内"
            log(f"  {name:>8} {label:<6} Δ={d:+.5f}  配对SD={sd:.5f}  "
                f"t={t:+.2f}  -> {verdict}")

    log("\n" + "=" * 64)
    if passed:
        log("⚠ 多重比较警告: 本次同时检验了 "
            f"{len(a.ks)} 个 K × 3 个指标 = {len(a.ks)*3} 次比较。")
        log("  单次 |t| > 2.5 的阈值在这种情况下**偏松**, 上面的'采纳'只应")
        log("  当作『值得复核』。正确做法是拿过闸的 K 单独重跑全新的折:")
        for name, label in passed:
            k = name.split("=")[1]
            log(f"      python research/gate_champ_shrinkage.py --ks {k} --folds 10")
    else:
        log("没有任何 K 过闸 —— 收缩在当前数据量下不值得采纳。")
        log("这是有价值的负结果: 硬阈值虽然粗糙, 但没有证据表明换掉它能提升。")

    log(f"\n结果已落盘: {out_path}")
    logf.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
