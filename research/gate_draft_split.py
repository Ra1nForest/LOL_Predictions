"""
Draft 特征拆解 — 计数 vs 胜率
================================
draft 特征整体过闸了 (t = -4.50), 但它混了两类东西:

  计数类 (19 个)   *_champ_games, *_comfort
                   精确值。Viper 用 EZ 打了 25 场就是 25 场, 无测量误差。

  胜率类 (14 个)   *_champ_wr
                   估计量。分半信度测出来只有 r ≈ 0.25。

假设: 起作用的是计数类, 不是胜率类。

依据: 信度 0.25 的特征在训练集上学到的模式, 到验证集上大部分不成立。
      而计数是直接观测, 没有这个问题。

四个配置各自过闸, 直接看是谁在干活:
    A  无 draft
    B  只加计数类
    C  只加胜率类
    D  两类都加  (= 现状)

如果 B ≈ D 而 C ≈ A, 假设成立。

需要 multiyear_gate.py 在同目录。
用法: python gate_draft_split.py
"""
import numpy as np, pandas as pd, warnings, sys
from pathlib import Path
warnings.filterwarnings("ignore")
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from multiyear_gate import (CANDIDATE_STATS, XGB, load_year, data_path,
                            audit_pool, build, feat_cols)

YEARS = (2022, 2023, 2024, 2025, 2026)
N_FOLDS, SEEDS, VAL_FRAC = 8, (0, 1, 2), 0.15


def classify(cols):
    """把 draft 特征分成计数类和胜率类。"""
    counts = [c for c in cols if "champ_games" in c or "comfort" in c]
    rates  = [c for c in cols if "champ_wr" in c]
    base   = [c for c in cols if c not in counts and c not in rates]
    return base, counts, rates


def run(mdf, feats, folds, seeds=SEEDS):
    sc = np.zeros((len(folds), len(seeds)))
    for fi, (t0, t1) in enumerate(folds):
        tr = mdf[mdf["date"] < t0]
        va = mdf[(mdf["date"] >= t0) & (mdf["date"] <= t1)]
        Xtr, ytr = tr[feats].astype(float).fillna(-999), tr["y"].astype(int)
        Xva = va[feats].astype(float).fillna(-999)
        yv = va["y"].astype(int).values
        base = max(yv.mean(), 1 - yv.mean())
        for si, sd in enumerate(seeds):
            m = XGBClassifier(random_state=sd, **XGB)
            m.fit(Xtr, ytr, verbose=False)
            a = accuracy_score(yv, (m.predict_proba(Xva)[:, 1] > .5).astype(int))
            sc[fi, si] = (a - base) / (1 - base)
    return sc.mean(axis=1)


def compare(ref, cand, name):
    pr = cand - ref
    d, sd = pr.mean(), pr.std(ddof=1)
    t = d / (sd / np.sqrt(len(pr))) if sd > 0 else 0.0
    mark = "✓" if t > 2.5 else ("?" if t > 1.5 else "✗")
    print(f"  {mark} {name:<26} Δ {d:>+8.4f}   配对SD {sd:.4f}   "
          f"t {t:>+6.2f}   {sum(1 for x in pr if x>0)}/{len(pr)}")
    return d, t


def main():
    print("=" * 74)
    print("  Draft 特征拆解 — 计数 vs 胜率")
    print("=" * 74)

    print("\n[1/4] 加载")
    by_year = {}
    for y in YEARS:
        d = load_year(data_path(y))
        if d is None:
            print(f"    {y} 缺失"); continue
        by_year[y] = d
        print(f"    {y}  {d[d['position']=='team']['gameid'].nunique():>5} 场")
    if len(by_year) < 2:
        print("  数据不足"); sys.exit(1)

    print("\n[2/4] 构建特征")
    usable, _ = audit_pool(by_year, CANDIDATE_STATS)
    df = pd.concat([by_year[y] for y in sorted(by_year)], ignore_index=True)
    mdf = build(df.sort_values("date"), usable)
    all_f = feat_cols(mdf)
    base_f, count_f, rate_f = classify(all_f)
    print(f"    {len(mdf)} 场")
    print(f"    基础 {len(base_f)}  |  计数类 {len(count_f)}  |  胜率类 {len(rate_f)}")
    print(f"\n    计数类: {', '.join(count_f[:4])} ...")
    print(f"    胜率类: {', '.join(rate_f[:4])} ...")

    print("\n[3/4] 验证窗口")
    n = len(mdf); val_n = int(n * VAL_FRAC)
    cuts = np.linspace(int(n * 0.40), n - val_n, N_FOLDS).astype(int)
    folds = [(mdf["date"].iloc[c], mdf["date"].iloc[min(c + val_n - 1, n - 1)])
             for c in cuts]
    print(f"    {N_FOLDS} 个 x {val_n} 场")

    print(f"\n[4/4] 四个配置  ({N_FOLDS}x{len(SEEDS)}x4 = "
          f"{N_FOLDS*len(SEEDS)*4} 次训练, 会比较慢)")
    cfgs = {
        "A 无 draft":     base_f,
        "B 只加计数类":    base_f + count_f,
        "C 只加胜率类":    base_f + rate_f,
        "D 两类都加":      all_f,
    }
    res = {}
    for name, fs in cfgs.items():
        res[name] = run(mdf, fs, folds)
        print(f"    {name:<16} {len(fs):>4} 特征   lift {res[name].mean():+.4f}   "
              f"fold SD {res[name].std(ddof=1):.4f}")

    A = res["A 无 draft"]
    print(f"\n{'=' * 74}")
    print(f"  相对「无 draft」的提升")
    print(f"{'=' * 74}")
    dB, tB = compare(A, res["B 只加计数类"], "只加计数类")
    dC, tC = compare(A, res["C 只加胜率类"], "只加胜率类")
    dD, tD = compare(A, res["D 两类都加"],   "两类都加 (现状)")

    print(f"\n  互相比较")
    print(f"  {'─' * 60}")
    compare(res["B 只加计数类"], res["D 两类都加"], "在计数类基础上加胜率")
    compare(res["C 只加胜率类"], res["D 两类都加"], "在胜率类基础上加计数")

    print(f"\n{'=' * 74}")
    print(f"  判决")
    print(f"{'=' * 74}")
    if tB > 2.5 and tC < 1.5:
        print(f"  ✓ 假设成立: 起作用的是计数类 (熟练度场次), 不是胜率估计。")
        print(f"\n    含义: 模型真正在用的信息是「这个选手对这个英雄有多熟」,")
        print(f"    而不是「他用这个英雄赢过多少」。前者是精确观测, 后者是")
        print(f"    信度只有 0.25 的噪声估计。")
        print(f"\n    可以考虑直接删掉 {len(rate_f)} 个胜率类特征 —— 更少的特征、")
        print(f"    同样的性能、更容易解释。")
    elif tC > 2.5 and tB < 1.5:
        print(f"  ✗ 假设反了: 起作用的是胜率类。")
        print(f"    这说明尽管信度只有 0.25, 它携带的信息仍然有用 ——")
        print(f"    低信度不等于零价值, 只是需要更多样本才能利用。")
    elif tB > 2.5 and tC > 2.5:
        print(f"  两类都有独立贡献, 应当都保留。")
    elif max(tB, tC, tD) < 1.5:
        print(f"  三个配置都没过闸 —— 与之前 t=-4.50 的结果不一致, 检查配置。")
    else:
        print(f"  信号混杂 (tB={tB:+.2f}, tC={tC:+.2f}, tD={tD:+.2f})。")
        print(f"  两类可能高度冗余: 各自单独加都有用, 但加了一个之后")
        print(f"  另一个就没有增量了。这种情况保留计数类即可 (更少噪声)。")
    print(f"\n{'=' * 74}")


if __name__ == "__main__":
    main()
