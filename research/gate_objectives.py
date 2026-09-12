"""
闸门: 目标资源特征值不值得进 Stage 4?
========================================

问题
----
模型现在 76.7% 的决策来自五个"谁领先多少"的数字 (golddiff_norm 0.345 +
golddiff 0.247 + xpdiff 0.091 + killdiff 0.062 + csdiff 0.022), 而**塔、龙、
龙魂、大龙、水晶一个都看不到**。

这些不是金币的线性函数:
  · 大龙是**时限** buff, 金币表达不了"还剩两分钟"
  · 龙魂是**永久**跃迁, 四条龙和三条龙差一个台阶, 金币上只差几百
  · 破水晶 = 持续兵线压力, 金币上是滞后的
  · 2k 落后但拿到龙魂, 和 2k 落后没龙魂, 是两种局面

为什么只能在回填子集里测
------------------------
Oracle's Elixir 的 60 个分时刻列 (at10/15/20/25) 全是经济/经验/补刀/KDA,
目标资源只有整局总数 —— 结构上就给不了"第 20 分钟谁拿了大龙"。
所以这些列只有回填样本有。混进 OE+回填的主表去训, "有值 / NaN" 会直接
变成 "回填 / OE" 的来源标签, 模型学到的是数据来自哪里, 不是局势 ——
这正是本项目最怕的那类静默污染 (见 README 的 zombie columns)。

于是: **只用回填子集**, 同一批样本、同一批折, 只改特征集。

判据
----
配对 t 检验, |t| > 2.5 (research/harness.py 的口径)。按**局**分折, 避免
同一局既训又测。同时报准确率和 Brier —— 校准是单调变换不改排序, 但
Brier 会动, 两个一起看才知道到底是"排得更对"还是只是"更敢下注"。

用法: python research/gate_objectives.py [--folds 5] [--seeds 3]
"""
from __future__ import annotations
import argparse, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, brier_score_loss

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT)); sys.path.insert(0, str(_ROOT / "research"))
from harness import paired_t, T_ACCEPT
from merge_backfill import OBJECTIVE_COLS

MERGED = _ROOT / "backfill" / "merged.csv.gz"
XGB_P = dict(n_estimators=400, max_depth=5, learning_rate=0.04,
             subsample=0.8, colsample_bytree=0.7,
             eval_metric="logloss", verbosity=0)
# 回填没有 XP, 整个对比在 live 变体的特征集上做 —— 否则"加了目标资源"和
# "少了经验差"两件事会混在一起。
DROP = {"xpdiff", "x_xp_scaling"}
NOT_FEATURE = {"gameid", "game_id", "oe_gameid", "date", "league", "y",
               "blue_team", "red_team", "source", "Tr", "frame_t",
               "blue", "red", "minute", "gamelength_min", "game_number"}


def feature_sets(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    base = [c for c in df.columns
            if c not in NOT_FEATURE and c not in DROP
            and c not in OBJECTIVE_COLS
            and not c.startswith(("blue_", "red_"))
            and pd.api.types.is_numeric_dtype(df[c])]
    obj = [c for c in OBJECTIVE_COLS if c in df.columns]
    return sorted(base), obj


def folds_by_game(df: pd.DataFrame, n_folds: int):
    """按局分折, 且按日期排序 —— 时序上靠后的局做验证, 不打乱时间。"""
    games = (df[["gameid", "date"]].drop_duplicates("gameid")
             .sort_values("date")["gameid"].values)
    return np.array_split(games, n_folds)


def score(tr, te, feats, seeds):
    X, y = tr[feats].astype(float).fillna(-999), tr["y"].astype(int)
    Xe = te[feats].astype(float).fillna(-999)
    ye = te["y"].astype(int).values
    accs, brs = [], []
    for s in seeds:
        m = XGBClassifier(random_state=s, **XGB_P)
        m.fit(X, y, verbose=False)
        p = m.predict_proba(Xe)[:, 1]
        base = max(ye.mean(), 1 - ye.mean())
        accs.append((accuracy_score(ye, (p >= .5).astype(int)) - base) / (1 - base))
        brs.append(-brier_score_loss(ye, p))
    return float(np.mean(accs)), float(np.mean(brs))


def run(df, base_f, cand_f, folds, seeds, label):
    a_acc, b_acc, a_br, b_br = [], [], [], []
    for i, hold in enumerate(folds):
        te = df[df.gameid.isin(hold)]
        tr = df[~df.gameid.isin(hold)]
        if len(te) < 50 or len(tr) < 200:
            continue
        aa, ab = score(tr, te, base_f, seeds)
        ba, bb = score(tr, te, cand_f, seeds)
        a_acc.append(aa); b_acc.append(ba)
        a_br.append(ab); b_br.append(bb)
        print(f"    fold {i+1}: {len(tr):>6} 训 / {len(te):>5} 测   "
              f"lift {aa:+.4f} -> {ba:+.4f}   Brier {-ab:.4f} -> {-bb:.4f}")
    if len(a_acc) < 3:
        print("    折数不足, 无法判定"); return
    for name, A, B in (("归一化 lift", a_acc, b_acc), ("Brier(负)", a_br, b_br)):
        # paired_t 返回 (Δ, 配对SD, t, 效果量) —— 顺序照 harness.py:147
        d, sd, t, _ratio = paired_t(np.array(A), np.array(B))
        verdict = "采纳" if abs(t) > T_ACCEPT and d > 0 else (
            "拒绝 (变差)" if abs(t) > T_ACCEPT else "噪声内, 不采纳")
        print(f"  {label} · {name}: Δ={d:+.4f}  配对SD={sd:.4f}  "
              f"t={t:+.2f}  (判据 {T_ACCEPT}) -> {verdict}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    seeds = tuple(range(a.seeds))

    if not MERGED.exists():
        print("没有 merged.csv.gz, 先跑 research/merge_backfill.py"); return 1
    df = pd.read_csv(MERGED, low_memory=False)
    df = df[df.source == "backfill"].copy()
    if not len(df):
        print("合表里没有回填样本"); return 1
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["y"])

    base_f, obj_f = feature_sets(df)
    if not obj_f:
        print("合表里没有目标资源列 —— 回填数据是旧版采的, 先重采"); return 1
    print(f"回填子集: {df.gameid.nunique()} 局 / {len(df)} 条快照")
    print(f"基线特征 {len(base_f)} 个; 新增目标资源 {len(obj_f)} 个: "
          f"{', '.join(obj_f)}\n")

    # 目标资源的取值分布 —— 全是 0 的话这个实验没有意义
    print("  目标资源的分布:")
    for c in obj_f:
        v = df[c]
        print(f"    {c:<16} 非零 {100*(v != 0).mean():>5.1f}%   "
              f"范围 [{v.min():g}, {v.max():g}]")
    print()

    folds = folds_by_game(df, a.folds)
    print("  [A] 全部快照")
    run(df, base_f, base_f + obj_f, folds, seeds, "全部")

    late = df[df["T"] >= 25]
    if late.gameid.nunique() > 40:
        print("\n  [B] 只看 T>=25 (目标资源最该起作用的阶段)")
        run(late, base_f, base_f + obj_f, folds_by_game(late, a.folds),
            seeds, "T>=25")

    early = df[df["T"] <= 15]
    if early.gameid.nunique() > 40:
        print("\n  [C] 只看 T<=15 (模型最弱的阶段, acc 0.71)")
        run(early, base_f, base_f + obj_f, folds_by_game(early, a.folds),
            seeds, "T<=15")
    return 0


if __name__ == "__main__":
    sys.exit(main())
