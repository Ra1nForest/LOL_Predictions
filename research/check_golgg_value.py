"""
gol.gg 部分数据的价值评估
==========================
gol.gg 对 LPL 有 GD@15, 但没有 CSD@15 / XPD@15 / 各时间点的击杀数。
问题: 只用这一个字段, 局内模型还剩多少预测力?

方法: 在 LCK/LEC/LCS 上 (那里两套字段都全) 训三个 T=15 模型:
    A  完整特征        —— 现在的 Stage 4
    B  只有 golddiff   —— gol.gg 能给的
    C  赛前特征 + golddiff —— B 加上队伍历史统计

如果 B/C 掉得不多, 抓 gol.gg 值得。掉一半, 就别花那几天。

用法: python check_golgg_value.py
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
from ingame_model import load, build, split_by_game, ece, XGB

YEARS = (2022, 2023, 2024, 2025, 2026)
T = 15


def run(mdf, feats, label):
    tr, cal, te = split_by_game(mdf)
    Xtr, ytr = tr[feats].astype(float).fillna(-999), tr["y"].astype(int)
    Xc, yc = cal[feats].astype(float).fillna(-999), cal["y"].astype(int)
    Xte, yte = te[feats].astype(float).fillna(-999), te["y"].astype(int).values
    m = XGBClassifier(**XGB); m.fit(Xtr, ytr, verbose=False)
    pc = m.predict_proba(Xc)[:, 1]
    cb = LogisticRegression(C=1e6).fit(pc.reshape(-1, 1), yc)
    p = cb.predict_proba(m.predict_proba(Xte)[:, 1].reshape(-1, 1))[:, 1]
    base = max(yte.mean(), 1 - yte.mean())
    acc = accuracy_score(yte, (p > .5).astype(int))
    return dict(label=label, n_feat=len(feats), acc=acc, base=base,
                lift=(acc - base) / (1 - base),
                brier=brier_score_loss(yte, p), ece=ece(p, yte),
                p_min=p.min(), p_max=p.max(), n_test=len(te))


def main():
    print("=" * 70)
    print(f"  gol.gg 部分数据的价值评估  (T={T})")
    print("=" * 70)

    print("\n[1/3] 加载")
    files = [str(_ROOT / "data" / f"{y}_LoL_esports_match_data_from_OraclesElixir.csv")
             for y in YEARS]
    mdf, _ = build(load(files), verbose=False)
    mdf = mdf[mdf["T"] == T].reset_index(drop=True)
    print(f"    T={T} 的快照 {len(mdf)} 个 ({mdf['gameid'].nunique()} 场)")

    allf = [c for c in mdf.columns
            if c not in ("gameid", "date", "league", "y", "blue_team", "red_team")
            and mdf[c].dtype in ("float64", "int64", "float32", "int32")]

    # gol.gg 能提供的: 只有队伍级 golddiff (五个选手的 GD@15 求和)
    golgg = ["golddiff", "golddiff_norm"]
    golgg = [c for c in golgg if c in mdf.columns]
    # 赛前特征 —— 这些不依赖分钟级数据, LPL 本来就有
    pre = [c for c in allf if c.startswith("diff_pre_") or c.startswith("is_")
           or "scaling" in c]

    cfgs = [
        ("A 完整特征 (现状)", allf),
        ("B 只有 golddiff", golgg),
        ("C 赛前 + golddiff", sorted(set(pre + golgg))),
        ("D 赛前特征 (对照)", pre),
    ]

    print("\n[2/3] 训练四个配置")
    res = []
    for label, f in cfgs:
        if not f:
            print(f"    {label:<22} 特征为空, 跳过"); continue
        r = run(mdf, f, label); res.append(r)
        print(f"    {label:<22} {r['n_feat']:>3} 特征   准确率 {r['acc']:.1%}   "
              f"lift {r['lift']:+.3f}")

    print(f"\n{'=' * 70}")
    print(f"  结果  (测试集 {res[0]['n_test']} 个快照, 基线 {res[0]['base']:.1%})")
    print(f"{'=' * 70}")
    print(f"  {'配置':<24}{'特征':>5}{'准确率':>9}{'lift':>9}{'Brier':>9}{'概率范围':>16}")
    print(f"  {'─' * 72}")
    for r in res:
        rng = f"[{r['p_min']:.2f},{r['p_max']:.2f}]"
        print(f"  {r['label']:<24}{r['n_feat']:>5}{r['acc']:>9.1%}"
              f"{r['lift']:>+9.3f}{r['brier']:>9.4f}{rng:>16}")

    A = res[0]
    print(f"\n{'=' * 70}")
    print(f"  值不值得抓 gol.gg")
    print(f"{'=' * 70}")
    for r in res[1:]:
        keep = r["lift"] / A["lift"] if A["lift"] > 0 else 0
        print(f"\n  {r['label']}")
        print(f"    保留了完整模型 {keep:.0%} 的预测力 "
              f"(lift {r['lift']:+.3f} vs {A['lift']:+.3f})")
        print(f"    准确率损失 {A['acc']-r['acc']:+.1%}")

    C = next((r for r in res if r["label"].startswith("C")), None)
    D = next((r for r in res if r["label"].startswith("D")), None)
    if C and D:
        gain = C["lift"] - D["lift"]
        print(f"\n  ★ 关键对比: 「赛前+golddiff」 vs 「只有赛前」")
        print(f"    加上 GD@15 带来 {gain:+.3f} lift "
              f"({C['acc']-D['acc']:+.1%} 准确率)")
        print()
        if gain > 0.08:
            print(f"    → 值得抓。单靠一个 GD@15 就能显著提升 LPL 的局内预测。")
        elif gain > 0.03:
            print(f"    → 边际。提升真实但不大, 看你愿不愿意为此维护一个爬虫。")
        else:
            print(f"    → 不值得。只有 GD@15 这一个字段, 相比纯赛前模型没什么增量。")
        print(f"\n    注意: 这是在 LCK/LEC/LCS 上测的。LPL 的规律未必一样 ——")
        print(f"    而这恰恰是无法验证的, 因为 LPL 根本没有局内数据可供对照。")

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
