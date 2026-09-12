"""
英雄协同特征 — 过闸测试
=========================
假设: 某些英雄两两搭配的胜率高于各自单独水平的预期。

构造
----
在 logit 空间做「无交互」基线:

    synergy(A,B) = logit(WR_AB) − logit(WR_A) − logit(WR_B)

logit 加法是「没有交互」的自然零假设 —— 如果 A 和 B 各自的效应
简单叠加, 这个差就是 0。正值 = 配合起来超出预期。

收缩: synergy × n/(n+K)。中位数 pair 只有 21 场, 不收缩全是噪声。

防泄漏: 所有 pair 和单英雄的胜率都只用该场比赛之前的数据。

预期
----
大概率过不了闸, 而且有结构性原因:

  职业 BP 是对抗的。真正强的组合会被 ban。你在数据里观测到的
  「A+B 同时出场」的比赛, 恰恰是对手认为不值得 ban 的那些。
  选择是内生的, 观测胜率被推回 50%。

  这和「公开的正向信号会被交易到消失」是同一件事。

如果真是这样, 负面结果 + 这个解释本身就是有价值的产出。

需要 multiyear_gate.py 在同目录。
"""
import numpy as np, pandas as pd, warnings, sys, itertools
from pathlib import Path
from collections import defaultdict
warnings.filterwarnings("ignore")
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from multiyear_gate import (TARGET, ROLES, CANDIDATE_STATS, XGB,
                            load_year, data_path, audit_pool, build, feat_cols)

YEARS      = (2022, 2023, 2024, 2025, 2026)
MIN_PAIR   = 30      # pair 最少场次
MIN_CHAMP  = 50      # 单英雄最少场次
SHRINK_K   = 120     # 收缩强度
N_FOLDS, SEEDS, VAL_FRAC = 8, (0, 1, 2), 0.15


def logit(p, eps=1e-3):
    p = min(max(p, eps), 1 - eps)
    return np.log(p / (1 - p))


class SynergyStore:
    """前缀和 + bisect, 支持「某日期之前」的胜率查询。"""

    def __init__(self, players):
        champ = defaultdict(list)          # champion -> [(date, result)]
        pair  = defaultdict(list)          # (A,B) -> [(date, result)]

        for (gid, side), g in players.groupby(["gameid", "side"]):
            cs = sorted(g["champion"].dropna().unique())
            if len(cs) != 5:
                continue
            date = g["date"].iloc[0]
            res  = int(g["result"].iloc[0])
            for c in cs:
                champ[c].append((date, res))
            for a, b in itertools.combinations(cs, 2):
                pair[(a, b)].append((date, res))

        def pack(dd):
            out = {}
            for k, v in dd.items():
                v.sort(key=lambda x: x[0])
                dates = np.array([d for d, _ in v], dtype="datetime64[ns]")
                wins  = np.concatenate([[0], np.cumsum([r for _, r in v])])
                out[k] = (dates, wins.astype(float))
            return out

        self.champ = pack(champ)
        self.pair  = pack(pair)
        self.n_pairs_total = len(pair)

    def _wr(self, tbl, key, before, min_n):
        e = tbl.get(key)
        if e is None:
            return None, 0
        dates, wins = e
        i = int(np.searchsorted(dates, np.datetime64(before), side="left"))
        if i < min_n:
            return None, i
        return wins[i] / i, i

    def synergy(self, a, b, before):
        """返回 (收缩后的协同分, pair 样本量)。数据不足返回 (nan, n)。"""
        key = (a, b) if a <= b else (b, a)
        wr_ab, n = self._wr(self.pair, key, before, MIN_PAIR)
        if wr_ab is None:
            return np.nan, n
        wr_a, _ = self._wr(self.champ, a, before, MIN_CHAMP)
        wr_b, _ = self._wr(self.champ, b, before, MIN_CHAMP)
        if wr_a is None or wr_b is None:
            return np.nan, n
        raw = logit(wr_ab) - logit(wr_a) - logit(wr_b)
        return raw * (n / (n + SHRINK_K)), n

    def team_features(self, champs, before):
        vals, ns = [], []
        for a, b in itertools.combinations(sorted(champs), 2):
            v, n = self.synergy(a, b, before)
            ns.append(n)
            if not np.isnan(v):
                vals.append(v)
        if not vals:
            return dict(syn_sum=np.nan, syn_mean=np.nan, syn_max=np.nan,
                        syn_min=np.nan, syn_cover=0, syn_n_med=int(np.median(ns)) if ns else 0)
        return dict(syn_sum=float(np.sum(vals)), syn_mean=float(np.mean(vals)),
                    syn_max=float(np.max(vals)), syn_min=float(np.min(vals)),
                    syn_cover=len(vals), syn_n_med=int(np.median(ns)) if ns else 0)


SYN_COLS = ["b_syn_sum", "b_syn_mean", "b_syn_max", "b_syn_min", "b_syn_cover",
            "r_syn_sum", "r_syn_mean", "r_syn_max", "r_syn_min", "r_syn_cover",
            "diff_syn_sum", "diff_syn_mean", "diff_syn_max", "diff_syn_cover"]


def add_synergy(mdf, players, store, verbose=True):
    by_game = {}
    for (gid, side), g in players.groupby(["gameid", "side"]):
        cs = sorted(g["champion"].dropna().unique())
        if len(cs) == 5:
            by_game[(gid, side)] = cs

    rows = []
    for r in mdf.itertuples(index=False):
        gid, dt = r.gameid, r.date
        bc = by_game.get((gid, "Blue")); rc = by_game.get((gid, "Red"))
        if bc is None or rc is None:
            rows.append({c: np.nan for c in SYN_COLS}); continue
        bf = store.team_features(bc, dt)
        rf = store.team_features(rc, dt)
        d = {}
        for k, v in bf.items():
            if k == "syn_n_med": continue
            d[f"b_{k}"] = v
        for k, v in rf.items():
            if k == "syn_n_med": continue
            d[f"r_{k}"] = v
        for k in ("syn_sum", "syn_mean", "syn_max", "syn_cover"):
            d[f"diff_{k}"] = (bf[k] - rf[k]) if (not np.isnan(bf[k]) and not np.isnan(rf[k])) else np.nan
        rows.append(d)

    S = pd.DataFrame(rows, index=mdf.index)
    out = pd.concat([mdf, S], axis=1)
    if verbose:
        cov = out["b_syn_cover"].fillna(0)
        print(f"    每队 10 个 pair 中有数据的: 中位 {cov.median():.0f}, "
              f"均值 {cov.mean():.1f}")
        print(f"    协同分分布: p5 {out['b_syn_mean'].quantile(.05):+.4f}  "
              f"中位 {out['b_syn_mean'].median():+.4f}  "
              f"p95 {out['b_syn_mean'].quantile(.95):+.4f}")
    return out


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
    print("=" * 76)
    print("  英雄协同特征 — 过闸")
    print("=" * 76)

    print("\n[1/5] 加载")
    by_year = {}
    for y in YEARS:
        d = load_year(data_path(y))
        if d is None:
            print(f"    {y} 缺失"); continue
        by_year[y] = d
        print(f"    {y}  {d[d['position']=='team']['gameid'].nunique():>5} 场")
    if len(by_year) < 2:
        print("  数据不足"); sys.exit(1)

    df = pd.concat([by_year[y] for y in sorted(by_year)], ignore_index=True).sort_values("date")
    players = df[df["position"].isin(ROLES)].copy()
    players["result"] = players["result"].astype(int)

    print("\n[2/5] 基础特征")
    usable, _ = audit_pool(by_year, CANDIDATE_STATS)
    mdf = build(df, usable)
    print(f"    {len(mdf)} 场, {len(feat_cols(mdf))} 特征")

    print(f"\n[3/5] 协同分 (MIN_PAIR={MIN_PAIR}, MIN_CHAMP={MIN_CHAMP}, K={SHRINK_K})")
    store = SynergyStore(players)
    print(f"    观测到的 pair: {store.n_pairs_total:,}")
    mdf = add_synergy(mdf, players, store)

    # 合理性检查: 最强/最弱的 pair
    end = mdf["date"].max()
    scored = []
    for key in store.pair:
        v, n = store.synergy(key[0], key[1], end)
        if not np.isnan(v) and n >= 80:
            scored.append((key, v, n))
    scored.sort(key=lambda x: -x[1])
    if scored:
        print(f"\n    协同分最高/最低的组合 (>= 80 场)")
        print(f"    {'最强':<38}{'最弱':<38}")
        print(f"    {'─'*74}")
        for i in range(min(8, len(scored) // 2)):
            (a1, b1), v1, n1 = scored[i]
            (a2, b2), v2, n2 = scored[-(i + 1)]
            L = f"{a1}+{b1}"[:24]
            R = f"{a2}+{b2}"[:24]
            print(f"    {L:<24}{v1:>+7.3f} n={n1:<4}{R:<24}{v2:>+7.3f} n={n2}")
        print(f"\n    ↑ 如果最强的那几组是你认得的经典组合, 说明特征抓到了东西;")
        print(f"      如果看着像随机组合, 那就是噪声。")

    print("\n[4/5] 定义验证窗口")
    n = len(mdf); val_n = int(n * VAL_FRAC)
    cuts = np.linspace(int(n * 0.40), n - val_n, N_FOLDS).astype(int)
    folds = [(mdf["date"].iloc[c], mdf["date"].iloc[min(c + val_n - 1, n - 1)]) for c in cuts]
    print(f"    {N_FOLDS} 个窗口 x {val_n} 场")

    print(f"\n[5/5] 过闸  ({N_FOLDS}x{len(SEEDS)}x2 = {N_FOLDS*len(SEEDS)*2} 次训练)")
    f_base = [c for c in feat_cols(mdf) if c not in SYN_COLS]
    f_syn  = f_base + [c for c in SYN_COLS if c in mdf.columns]
    A = run(mdf, f_base, folds)
    B = run(mdf, f_syn, folds)

    af, bf = A.mean(axis=1), B.mean(axis=1)
    pr = bf - af
    d, psd = pr.mean(), pr.std(ddof=1)
    t = d / (psd / np.sqrt(len(pr))) if psd > 0 else 0

    print(f"\n{'=' * 76}")
    print(f"  结果")
    print(f"{'=' * 76}")
    print(f"  无协同   {af.mean():+.4f}   fold SD {af.std(ddof=1):.4f}")
    print(f"  含协同   {bf.mean():+.4f}   fold SD {bf.std(ddof=1):.4f}")
    print(f"\n  Δ = {d:+.4f}   配对SD {psd:.4f}   t = {t:+.2f}   "
          f"{sum(1 for x in pr if x>0)}/{len(pr)} fold 改善")
    print(f"  逐fold: {'  '.join(f'{x:+.3f}' for x in pr)}")

    print(f"\n  判决")
    print(f"  {'─' * 60}")
    if t > 2.5:
        print(f"  ✓ 协同特征有效, 采纳")
    elif t < -2.5:
        print(f"  ✗ 加了反而显著变差, 移除")
    else:
        print(f"  ✗ 在噪声内 (t={t:+.2f}), 协同特征无可测贡献")
        print(f"\n    结构性解释: 职业 BP 是对抗的。真正强的组合会被 ban,")
        print(f"    所以你观测到的「A+B 同时出场」的比赛, 恰恰是对手认为")
        print(f"    不值得 ban 的那些。选择内生, 观测胜率被推回 50%。")
        print(f"\n    真正的协同优势应该体现在 ban 位上, 而不是 pick 的胜率上。")
        print(f"    数据里有 ban1-5, 那是另一个方向。")
    print(f"\n{'=' * 76}")


if __name__ == "__main__":
    main()
