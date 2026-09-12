"""
局内模型特征过闸 —— 2 个特征够不够?
=====================================
单次切分上看到:

    A  完整 28 特征     69.2%   lift +0.333
    B  只有 golddiff    69.9%   lift +0.348   ← 2 个特征反而更好
    C  赛前 + golddiff  69.0%   lift +0.329
    D  只有赛前         66.5%   lift +0.274

但差异只有 0.7pp, 而单次切分的噪声地板通常在这个量级。必须过闸。

这个结论有两个用途:
  1. 决定要不要抓 gol.gg —— 如果 golddiff 真的够, 爬虫只需要抓一个字段,
     比抓十几个简单一个数量级, 也不容易因为网站改版而崩
  2. 如果 B 真的不输 A, Stage 4 在 T=15 上可以从 28 个特征砍到 2 个

按「比赛」切分, 不按「快照」—— 同一场比赛的多个时间点若分落两边,
模型就见过这场比赛了。

用法: python gate_ingame_features.py
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from ingame_model import load, build, XGB
from harness import paired_t, T_ACCEPT

YEARS = (2022, 2023, 2024, 2025, 2026)
SLICES_TO_TEST = [10, 15, 20, 25]
N_FOLDS, SEEDS, VAL_FRAC = 8, (0, 1, 2), 0.15


def make_folds(mdf, n_folds=N_FOLDS, val_frac=VAL_FRAC):
    """按比赛的首次出现时间排序切分"""
    order = mdf.groupby("gameid")["date"].min().sort_values()
    gids = order.index.to_numpy()
    n = len(gids)
    val_g = max(60, int(n * val_frac))
    cuts = np.linspace(int(n * 0.40), n - val_g, n_folds).astype(int)
    return [(set(gids[:c]), set(gids[c:c + val_g])) for c in cuts]


def run(mdf, feats, folds, seeds=SEEDS):
    sc = np.zeros((len(folds), len(seeds)))
    for fi, (tr_g, va_g) in enumerate(folds):
        tr = mdf[mdf["gameid"].isin(tr_g)]
        va = mdf[mdf["gameid"].isin(va_g)]
        Xtr = tr[feats].astype(float).fillna(-999)
        ytr = tr["y"].astype(int)
        Xva = va[feats].astype(float).fillna(-999)
        yv = va["y"].astype(int).values
        base = max(yv.mean(), 1 - yv.mean())
        for si, sd in enumerate(seeds):
            m = XGBClassifier(**{**XGB, "random_state": sd})
            m.fit(Xtr, ytr, verbose=False)
            a = accuracy_score(yv, (m.predict_proba(Xva)[:, 1] > .5).astype(int))
            sc[fi, si] = (a - base) / (1 - base)
    return sc


def compare(ref, cand, name, ref_name="A 完整"):
    # 判决算术走 harness, 这里只管排版 —— 见 harness.paired_t 的注释
    pr = cand - ref
    d, sd, t, _ = paired_t(ref, cand)
    mark = "✓" if abs(t) > T_ACCEPT else ("?" if abs(t) > 1.5 else "≈")
    print(f"  {mark} {name:<22} Δ {d:>+8.4f}  配对SD {sd:.4f}  "
          f"t {t:>+6.2f}  {sum(1 for x in pr if x > 0)}/{len(pr)}")
    return d, t


def main():
    print("=" * 74)
    print("  局内模型特征过闸")
    print("=" * 74)

    print("\n[1/3] 加载")
    files = [str(_ROOT / "data" / f"{y}_LoL_esports_match_data_from_OraclesElixir.csv")
             for y in YEARS]
    full, _ = build(load(files), verbose=False)
    print(f"    {full['gameid'].nunique()} 场 → {len(full)} 个快照")

    for T in SLICES_TO_TEST:
        mdf = full[full["T"] == T].reset_index(drop=True)
        if len(mdf) < 500:
            print(f"\n  T={T}: 样本不足, 跳过"); continue

        allf = [c for c in mdf.columns
                if c not in ("gameid", "date", "league", "y",
                             "blue_team", "red_team")
                and mdf[c].dtype in ("float64", "int64", "float32", "int32")]
        gold = [c for c in ("golddiff", "golddiff_norm") if c in mdf.columns]
        pre = [c for c in allf if c.startswith("diff_pre_")
               or c.startswith("is_") or "scaling" in c]
        live = [c for c in allf if c in ("golddiff", "golddiff_norm", "xpdiff",
                                         "csdiff", "killdiff", "killsum")]

        # F: lolesports 的 livestats 帧里没有 XP 字段 (只有每人的 level,
        # 和训练用的 xpat10/15/20 原始经验值量纲不同, 不能顶替)。所以这一组
        # 就是「实时接入实际喂得进去的全部特征」。缺这一项的代价必须单独测,
        # 用 C (还扣掉了 cs/人头, 而那些 API 给得出) 去推会高估。
        #
        # 必须连 x_xp_scaling 一起去掉 —— 它是 xpdiff * scaling_diff
        # (ingame_model.py:270)。第一版只删了 xpdiff, 模型仍能从这个交互项
        # 拿到经验差信息, 测出来的代价是偏小的。
        XP_DERIVED = {"xpdiff", "x_xp_scaling"}
        no_xp = [c for c in allf if c not in XP_DERIVED]

        cfgs = {
            "A 完整": allf,
            "B 只有 golddiff": gold,
            "C 赛前+golddiff": sorted(set(pre + gold)),
            "D 只有赛前": pre,
            "E 全部局内": live,
            "F 完整−xpdiff": no_xp,
        }

        print(f"\n{'=' * 74}")
        print(f"  T = {T}   ({len(mdf)} 场)")
        print(f"{'=' * 74}")

        folds = make_folds(mdf)
        res = {}
        for name, f in cfgs.items():
            if not f:
                continue
            s = run(mdf, f, folds)
            res[name] = s.mean(axis=1)
            print(f"    {name:<20}{len(f):>4} 特征   "
                  f"lift {res[name].mean():+.4f}   "
                  f"fold SD {res[name].std(ddof=1):.4f}")

        A = res["A 完整"]
        print(f"\n  相对「完整特征」")
        print(f"  {'─' * 62}")
        verdicts = {}
        for name in cfgs:
            if name == "A 完整" or name not in res:
                continue
            verdicts[name] = compare(A, res[name], name)

        # 判决
        B = verdicts.get("B 只有 golddiff")
        C = verdicts.get("C 赛前+golddiff")
        D = verdicts.get("D 只有赛前")
        print(f"\n  判决")
        print(f"  {'─' * 62}")
        if B:
            d, t = B
            if abs(t) <= 1.5:
                print(f"  · 2 个特征和 28 个特征无法区分 (t={t:+.2f})")
                print(f"    → T={T} 的模型可以砍到只剩 golddiff")
            elif t < -2.5:
                print(f"  · 只用 golddiff 显著更差 (t={t:+.2f}), 需要完整特征")
            elif t > 2.5:
                print(f"  · 只用 golddiff 显著更好 (t={t:+.2f}) —— "
                      f"其余特征在稀释信号")
            else:
                print(f"  · 方向明确但未达标 (t={t:+.2f})")
        if C and D:
            gain = res["C 赛前+golddiff"] - res["D 只有赛前"]
            g, gsd = gain.mean(), gain.std(ddof=1)
            gt = g / (gsd / np.sqrt(len(gain))) if gsd > 0 else 0
            print(f"  · GD@{T} 相对纯赛前模型的增量: {g:+.4f} (t={gt:+.2f})")
            if gt > 2.5:
                print(f"    → 这一个字段确实有独立贡献")
            else:
                print(f"    → 增量不显著")

    print(f"\n{'=' * 74}")
    print(f"  对 gol.gg 的含义")
    print(f"{'=' * 74}")
    print("""
  gol.gg 对 LPL 只有 GD@15, 没有 T=10/20/25, 没有 xp/cs 差。

  若 T=15 的判决是「golddiff 够了」→ 爬虫只需抓一个字段, 简单且稳健。
  若判决是「需要完整特征」        → gol.gg 给不了, 别抓。

  但无论哪种, 都只能覆盖 T=15 这一个时间点。而模型恰恰是在后期
  才有把握 (T=20 约 77%, T=25 约 80%), 那两个时间点 gol.gg 没有。

  另外这里所有验证都在 LCK/LEC/LCS 上做 —— LPL 的规律是否一致
  无法检验, 因为 LPL 根本没有局内数据可供对照。
    """)


if __name__ == "__main__":
    main()
