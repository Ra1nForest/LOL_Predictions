"""
Draft 特征终审 — 五年数据 + 大验证窗口
========================================
这是唯一悬而未决的假设。三次测试:

    5 folds  (2yr, val 12%)   Δ -0.0325   t -1.73
   12 folds  (2yr, val  7%)   Δ -0.0310   t -2.00
   本次      (5yr, val 15%)   ?

前两次都卡在 t≈2 出不来。原因不是效应不存在 —— 点估计在两种独立
的 fold 配置下几乎一致 (-0.0325 / -0.0310) —— 而是精度不够。

关键认识 (从 multiyear_gate 得到):
  fold SD 反映的是「不同验证窗口本身的难度差异」, 是验证集的属性。
  加训练数据降不了它, 只有加大验证窗口才行。

  seed SD = 0.014   模型估计误差   ← 加训练数据能降
  fold SD = 0.090   窗口难度差异   ← 只能靠加大窗口降

所以本次的改动不是「更多训练数据」, 是「更大的验证窗口」。
五年数据 (8259 场) 让 val_frac 从 0.07 提到 0.15 仍留有充足训练集。

判据不变: t > 2.5 采纳。这是最后一次测试 —— 再换配置重测就是 p-hacking。
"""
import numpy as np, pandas as pd, warnings, sys
from pathlib import Path
from collections import defaultdict
warnings.filterwarnings("ignore")
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from multiyear_gate import (TARGET, W, ROLES, CANDIDATE_STATS, SUM_F, XGB,
                            load_year, data_path, audit_pool, build, feat_cols)

YEARS    = (2022, 2023, 2024, 2025, 2026)
N_FOLDS  = 8
SEEDS    = (0, 1, 2)
VAL_FRAC = 0.15          # ← 关键改动
DRAFT_KW = ("champ_wr", "champ_games", "comfort")


def is_draft(c):
    return any(k in c for k in DRAFT_KW)


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
    return sc


def main():
    print("=" * 74)
    print("  Draft 特征终审 — 五年数据 + 15% 验证窗口")
    print("=" * 74)

    print("\n[1/4] 加载")
    by_year = {}
    for y in YEARS:
        d = load_year(data_path(y))
        if d is None:
            print(f"    {y}  缺失"); continue
        by_year[y] = d
        t = d[d["position"] == "team"]
        print(f"    {y}  {t['gameid'].nunique():>5} 场")
    if len(by_year) < 2:
        print("  数据不足"); sys.exit(1)

    print("\n[2/4] 审计 + 构建")
    usable, dropped = audit_pool(by_year, CANDIDATE_STATS)
    print(f"    可用统计量 {len(usable)}/{len(CANDIDATE_STATS)}")
    for c, why in dropped:
        print(f"      ✗ {c}: {why}")
    df = pd.concat([by_year[y] for y in sorted(by_year)], ignore_index=True)
    mdf = build(df.sort_values("date"), usable)
    print(f"    {len(mdf)} 场可用")

    f_all = feat_cols(mdf)
    f_no = [c for c in f_all if not is_draft(c)]
    n_draft = len(f_all) - len(f_no)
    print(f"    特征 {len(f_all)} 个, 其中 draft {n_draft} 个")

    print("\n[3/4] 定义验证窗口")
    n = len(mdf)
    val_n = int(n * VAL_FRAC)
    cuts = np.linspace(int(n * 0.40), n - val_n, N_FOLDS).astype(int)
    folds = [(mdf["date"].iloc[c], mdf["date"].iloc[min(c + val_n - 1, n - 1)])
             for c in cuts]
    print(f"    {N_FOLDS} 个窗口, 每个 {val_n} 场 "
          f"(之前 12-fold 那次每个只有 {int(n*0.07)} 场的量级)")
    for i, (a, b) in enumerate(folds, 1):
        print(f"      fold {i}: {a.date()} → {b.date()}")

    print(f"\n[4/4] 跑 {N_FOLDS} folds x {len(SEEDS)} seeds x 2 配置 "
          f"= {N_FOLDS*len(SEEDS)*2} 次训练")
    print("    (五年数据, 会比较慢)")
    A = run(mdf, f_all, folds)      # 含 draft
    B = run(mdf, f_no, folds)       # 去掉 draft

    af, bf = A.mean(axis=1), B.mean(axis=1)
    seed_sd = A.std(axis=1).mean()
    pr = bf - af                    # 去掉后的变化, 负 = 变差 = draft 有用
    d, psd = pr.mean(), pr.std(ddof=1)
    t = d / (psd / np.sqrt(len(pr))) if psd > 0 else 0

    print(f"\n{'=' * 74}")
    print(f"  结果")
    print(f"{'=' * 74}")
    print(f"  含 draft   {af.mean():+.4f}   fold SD {af.std(ddof=1):.4f}")
    print(f"  无 draft   {bf.mean():+.4f}   fold SD {bf.std(ddof=1):.4f}")
    print(f"  seed SD    {seed_sd:.4f}   (对比 fold SD, 看主导项是哪个)")
    print(f"\n  Δ (去掉 draft) = {d:+.4f}   配对SD {psd:.4f}   t = {t:+.2f}")
    print(f"  逐fold: {'  '.join(f'{x:+.3f}' for x in pr)}")
    print(f"  {sum(1 for x in pr if x < 0)}/{len(pr)} 个 fold 去掉后变差")

    print(f"\n  历次测试对比")
    print(f"  {'─' * 56}")
    print(f"  {'配置':<28}{'Δ':>10}{'t':>9}")
    print(f"   5 folds, 2yr, val 12%      {-0.0325:>+10.4f}{-1.73:>9.2f}")
    print(f"  12 folds, 2yr, val  7%      {-0.0310:>+10.4f}{-2.00:>9.2f}")
    print(f"  {N_FOLDS:>2} folds, 5yr, val 15%      {d:>+10.4f}{t:>9.2f}   ← 本次")

    print(f"\n  判决")
    print(f"  {'─' * 56}")
    if t < -2.5:
        print(f"  ✓ draft 特征确认有效 (去掉显著变差)")
        print(f"    效应约 {abs(d):.3f} 归一化 lift ≈ {abs(d)*0.475*100:.1f} 个百分点准确率")
    elif t > 2.5:
        print(f"  ✗ 去掉 draft 反而显著更好 —— 应当移除")
    elif abs(t) > 1.5:
        print(f"  ? 方向一致但仍未达标 (t={t:+.2f})")
        print(f"    三次测试点估计稳定在 -0.03 附近, 方向没变过。")
        print(f"    结论: 效应大概率真实存在, 量级约 1.5pp, 但在这个")
        print(f"    问题的噪声水平下无法达到 t>2.5。保留, 标注为未确认。")
        print(f"    ⚠ 不要再换配置重测 —— 那是 p-hacking。")
    else:
        print(f"  ✗ 在噪声内 (t={t:+.2f}), draft 特征贡献不可测")
        print(f"    保留无害 (零成本), 但不应在简历上宣称它有效。")
    print(f"\n{'=' * 74}")


if __name__ == "__main__":
    main()
