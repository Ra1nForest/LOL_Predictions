"""
逐年加数据 — 审计 + 过闸
=========================
问题: 加历史数据到底有没有用? 加到哪一年为止?

加数据的目的不是「让模型更准」, 而是「降低噪声地板」。
噪声地板按 1/√n 缩, 所以数据 ×3 → 精度 ×1.73。
draft 特征那个 0.77x SD 的悬案, 需要地板降到 ±0.026 才有机会判定。

方法
----
验证窗口固定在最近的数据上, 只改变训练池:

    配置 A: 2025+2026        (现状)
    配置 B: +2024
    配置 C: +2023
    配置 D: +2022

同一批 fold、同一批验证比赛, 只有训练数据不同。
这样测出来的差异只反映「老数据有没有用」。

每加一年看两件事:
  1. 平均 lift 有没有掉  → 掉了说明老数据的规律已不适用
  2. fold SD 有没有降    → 这才是加数据的目的

外加: 每加一年先跑列语义审计。只保留在整个池子里都可用的列 ——
一个 2024 有 2022 没有的字段, 混在一起会变成年份标签。
"""
import numpy as np, pandas as pd, warnings, sys, re
from pathlib import Path
from collections import defaultdict
warnings.filterwarnings("ignore")

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

TARGET = ["LPL", "LCK", "LEC", "LCS"]
W      = 10
ROLES  = ["top", "jng", "mid", "bot", "sup"]
N_FOLDS, SEEDS, VAL_FRAC = 6, (0, 1), 0.10

CANDIDATE_STATS = [
    "kills", "deaths", "assists", "totalgold", "damagetochampions",
    "wardsplaced", "wardskilled", "dragons", "barons", "towers",
    "cspm", "golddiffat10", "golddiffat15",
    "firstblood", "firstdragon", "firsttower", "ckpm", "team kpm",
]
SUM_F = ["avg_kills", "avg_deaths", "avg_gamelen", "avg_game_kills",
         "avg_ckpm", "avg_team_kpm", "avg_towers", "avg_dragons"]

XGB = dict(n_estimators=400, max_depth=5, learning_rate=0.04,
           subsample=0.8, colsample_bytree=0.7,
           eval_metric="logloss", verbosity=0)


# ══════════════════════════════════════════════════════════
#  加载
# ══════════════════════════════════════════════════════════

# CSV 的位置只在这里写一次。多个闸门脚本共用 load_year, 各自拼路径的话
# 漏改一处就是静默少加载一年 —— 而 load_year 对缺失文件只返回 None。
DATA_DIR = Path(__file__).parent.parent / "data"


def data_path(year) -> str:
    return str(DATA_DIR / f"{year}_LoL_esports_match_data_from_OraclesElixir.csv")


def load_year(path):
    if not (Path(path).exists() or str(path).startswith("/mnt")):
        return None
    d = pd.read_csv(path, low_memory=False)
    d["league"] = d["league"].replace({"LTA N": "LCS", "LTA S": "CBLOL"})
    d = d[d["league"].isin(TARGET)].copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    return d.sort_values("date").reset_index(drop=True)


# ══════════════════════════════════════════════════════════
#  审计: 找出在整个池子里都可用的列
# ══════════════════════════════════════════════════════════

def audit_pool(by_year, cols):
    """
    对每个候选列, 检查它在池子里每一年是否都「可用」:
      · 存在
      · 填充率 >= 50%
      · 非常数 (标准差 > 0)
      · 均值跨年漂移 < 3 个合并标准差 (量纲断裂)
    """
    usable, dropped = [], []
    for c in cols:
        prof = {}
        ok = True
        for yr, d in sorted(by_year.items()):
            t = d[d["position"] == "team"]
            if c not in t.columns:
                dropped.append((c, f"{yr} 缺列")); ok = False; break
            v = pd.to_numeric(t[c], errors="coerce")
            fill = v.notna().mean()
            if fill < 0.50:
                dropped.append((c, f"{yr} 填充率 {fill:.0%}")); ok = False; break
            sd = v.std()
            if not sd or sd == 0:
                dropped.append((c, f"{yr} 常数")); ok = False; break
            prof[yr] = (v.mean(), sd)
        if not ok:
            continue
        # 量纲断裂检测: 任意两年均值差 > 3 个合并 SD
        ys = sorted(prof)
        brk = None
        for i in range(len(ys) - 1):
            m1, s1 = prof[ys[i]]; m2, s2 = prof[ys[i + 1]]
            pooled = (s1 + s2) / 2
            if pooled > 0 and abs(m2 - m1) / pooled > 3.0:
                brk = f"{ys[i]}→{ys[i+1]} 均值跳 {abs(m2-m1)/pooled:.1f}SD"
                break
        if brk:
            dropped.append((c, brk))
        else:
            usable.append(c)
    return usable, dropped


# ══════════════════════════════════════════════════════════
#  特征构建 (groupby 优化, 9000 场也能跑)
# ══════════════════════════════════════════════════════════

def build(df, stats):
    teams = df[df["position"] == "team"].copy()
    teams["result"] = teams["result"].astype(int)
    gl = pd.to_numeric(teams["gamelength"], errors="coerce")
    teams["gamelength_min"] = gl / 60 if gl.max() > 200 else gl
    teams["game_total_kills"] = (pd.to_numeric(teams["teamkills"], errors="coerce")
                                 + pd.to_numeric(teams["teamdeaths"], errors="coerce"))
    players = df[df["position"].isin(ROLES)].copy()
    players["result"] = players["result"].astype(int)

    # 队伍滚动
    hist_rows = []
    for tm, t in teams.groupby("teamname"):
        t = t.sort_values("date").copy()
        for c in stats:
            t[f"avg_{c.replace(' ','_')}"] = (pd.to_numeric(t[c], errors="coerce")
                                              .rolling(W, min_periods=3).mean().shift(1))
        t["rolling_wr"] = t["result"].rolling(W, min_periods=3).mean().shift(1)
        s, st = 0, []
        for r in t["result"]:
            st.append(s); s = (s + 1 if s > 0 else 1) if r == 1 else (s - 1 if s < 0 else -1)
        t["streak"] = st
        t["avg_gamelen"] = t["gamelength_min"].rolling(W, min_periods=3).mean().shift(1)
        t["avg_game_kills"] = t["game_total_kills"].rolling(W, min_periods=3).mean().shift(1)
        hist_rows.append(t)
    H = pd.concat(hist_rows, ignore_index=True)
    fcols = [c for c in H.columns if c.startswith("avg_") or c in ("rolling_wr", "streak")]
    Hidx = {(r["gameid"], r["teamname"]): r for _, r in H.iterrows()}

    # 英雄 / 选手历史 (前缀和 + bisect)
    champ, pchamp = defaultdict(list), defaultdict(list)
    for r in players.itertuples(index=False):
        champ[(r.champion, r.league)].append((r.date, r.result))
        pchamp[(r.playername, r.champion)].append((r.date, r.result))
    for k in champ:  champ[k].sort(key=lambda x: x[0])
    for k in pchamp: pchamp[k].sort(key=lambda x: x[0])

    def prior(recs, before, mg):
        i = np.searchsorted([d for d, _ in recs], before, side="left")
        if i < mg: return np.nan, i
        vals = [v for _, v in recs[:i]]
        return float(np.mean(vals)), i

    pbg = {g: d for g, d in players.groupby("gameid")}
    rows = []
    for gid, grp in teams.groupby("gameid"):
        if len(grp) != 2: continue
        r = grp.sort_values("participantid")
        bT, rT = r.iloc[0], r.iloc[1]
        bt, rt, lg, gd = bT["teamname"], rT["teamname"], bT["league"], bT["date"]
        bh, rh = Hidx.get((gid, bt)), Hidx.get((gid, rt))
        if bh is None or rh is None: continue

        m = {"gameid": gid, "date": gd, "league": lg, "y": int(bT["result"]),
             "is_lpl": int(lg == "LPL"), "is_lck": int(lg == "LCK"),
             "is_lec": int(lg == "LEC"), "is_lcs": int(lg == "LCS"),
             "playoffs": int(pd.to_numeric(bT.get("playoffs", 0), errors="coerce") or 0)}
        for c in fcols:
            try:
                bv, rv = float(bh.get(c, np.nan)), float(rh.get(c, np.nan))
                m[f"diff_{c}"], m[f"b_{c}"], m[f"r_{c}"] = bv - rv, bv, rv
                if c in SUM_F: m[f"sum_{c}"] = bv + rv
            except (TypeError, ValueError):
                pass

        gp = pbg.get(gid)
        if gp is not None:
            bp, rp = gp[gp["side"] == "Blue"], gp[gp["side"] == "Red"]
            if len(bp) >= 5 and len(rp) >= 5:
                bw, rw, bc, rc = [], [], [], []
                for side, sub, aw, ac in (("b", bp, bw, bc), ("r", rp, rw, rc)):
                    for x in sub.itertuples(index=False):
                        cv, _ = prior(champ.get((x.champion, lg), []), gd, 5)
                        if not np.isnan(cv): aw.append(cv)
                        pv, n = prior(pchamp.get((x.playername, x.champion), []), gd, 3)
                        ac.append(n)
                        m[f"{side}_{x.position}_champ_wr"] = pv
                        m[f"{side}_{x.position}_champ_games"] = n
                m["b_avg_champ_wr"] = float(np.mean(bw)) if bw else np.nan
                m["r_avg_champ_wr"] = float(np.mean(rw)) if rw else np.nan
                if bw and rw:
                    m["diff_avg_champ_wr"] = m["b_avg_champ_wr"] - m["r_avg_champ_wr"]
                    m["sum_avg_champ_wr"] = m["b_avg_champ_wr"] + m["r_avg_champ_wr"]
                m["b_comfort"], m["r_comfort"] = sum(bc), sum(rc)
                m["diff_comfort"], m["sum_comfort"] = sum(bc) - sum(rc), sum(bc) + sum(rc)
                for role in ROLES:
                    m[f"diff_{role}_comfort"] = ((m.get(f"b_{role}_champ_games") or 0)
                                                 - (m.get(f"r_{role}_champ_games") or 0))
        rows.append(m)

    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    core = [c for c in out.columns if c.startswith("diff_avg_")][:5]
    return out.dropna(subset=core + ["y"]).reset_index(drop=True)


def feat_cols(mdf):
    return [c for c in mdf.columns
            if (c.startswith(("diff_", "b_", "r_", "sum_", "is_")) or c == "playoffs")
            and mdf[c].dtype in ("float64", "int64", "float32", "int32")]


# ══════════════════════════════════════════════════════════
#  评估: 固定验证窗口, 只变训练池
# ══════════════════════════════════════════════════════════

def make_folds(mdf_recent, n_folds=N_FOLDS, val_frac=VAL_FRAC):
    """
    基于最新的那份数据定义验证窗口 (日期区间)。
    所有配置共用这批窗口, 保证可比。
    """
    n = len(mdf_recent)
    val_n = max(80, int(n * val_frac))
    cuts = np.linspace(int(n * 0.40), n - val_n, n_folds).astype(int)
    folds = []
    for c in cuts:
        folds.append((mdf_recent["date"].iloc[c],
                      mdf_recent["date"].iloc[min(c + val_n - 1, n - 1)]))
    return folds


def eval_pool(mdf, folds, feats, seeds=SEEDS):
    """对每个 fold: 训练 = 窗口开始之前的全部, 验证 = 窗口内。"""
    sc = np.zeros((len(folds), len(seeds)))
    sizes = []
    for fi, (t0, t1) in enumerate(folds):
        tr = mdf[mdf["date"] < t0]
        va = mdf[(mdf["date"] >= t0) & (mdf["date"] <= t1)]
        if len(tr) < 200 or len(va) < 40:
            sc[fi, :] = np.nan; sizes.append((len(tr), len(va))); continue
        Xtr, ytr = tr[feats].astype(float).fillna(-999), tr["y"].astype(int)
        Xva = va[feats].astype(float).fillna(-999)
        yv = va["y"].astype(int).values
        base = max(yv.mean(), 1 - yv.mean())
        for si, sd in enumerate(seeds):
            m = XGBClassifier(random_state=sd, **XGB)
            m.fit(Xtr, ytr, verbose=False)
            a = accuracy_score(yv, (m.predict_proba(Xva)[:, 1] > .5).astype(int))
            sc[fi, si] = (a - base) / (1 - base)
        sizes.append((len(tr), len(va)))
    return np.nanmean(sc, axis=1), sizes


# ══════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════

FILES = {y: data_path(y) for y in (2022, 2023, 2024, 2025, 2026)}


def main():
    print("=" * 78)
    print("  逐年加数据 — 审计 + 过闸")
    print("=" * 78)

    print("\n[1/4] 加载")
    by_year = {}
    for y, f in FILES.items():
        d = load_year(f)
        if d is None:
            print(f"    {y}  缺失, 跳过"); continue
        by_year[y] = d
        t = d[d["position"] == "team"]
        print(f"    {y}  {t['gameid'].nunique():>5} 场   "
              f"{t['date'].min().date()} → {t['date'].max().date()}")
    if 2026 not in by_year or 2025 not in by_year:
        print("\n  至少需要 2025 和 2026"); sys.exit(1)

    years_desc = sorted(by_year, reverse=True)
    configs = []
    for i in range(2, len(years_desc) + 1):
        configs.append(sorted(years_desc[:i]))

    print(f"\n[2/4] 列语义审计 (只保留在整个池子里都可用的列)")
    pool_cols = {}
    for cfg in configs:
        sub = {y: by_year[y] for y in cfg}
        usable, dropped = audit_pool(sub, CANDIDATE_STATS)
        pool_cols[tuple(cfg)] = usable
        name = f"{cfg[0]}–{cfg[-1]}"
        print(f"\n  {name}   可用 {len(usable)}/{len(CANDIDATE_STATS)}")
        if dropped:
            for c, why in dropped:
                print(f"      ✗ {c:<22} {why}")

    print(f"\n[3/4] 构建特征 + 定义共用验证窗口")
    mdfs = {}
    for cfg in configs:
        key = tuple(cfg)
        df = pd.concat([by_year[y] for y in cfg], ignore_index=True).sort_values("date")
        mdfs[key] = build(df, pool_cols[key])
        print(f"    {cfg[0]}–{cfg[-1]}   {len(mdfs[key])} 场可用   "
              f"{len(pool_cols[key])} 个统计量")

    base_key = tuple(configs[0])
    folds = make_folds(mdfs[base_key])
    print(f"\n    验证窗口 ({len(folds)} 个, 所有配置共用):")
    for i, (a, b) in enumerate(folds, 1):
        print(f"      fold {i}: {a.date()} → {b.date()}")

    print(f"\n[4/4] 逐配置评估")
    results = {}
    for cfg in configs:
        key = tuple(cfg)
        mdf = mdfs[key]
        # 特征取所有配置的交集, 保证可比
        results[key] = None
    common = set(feat_cols(mdfs[base_key]))
    for key in mdfs:
        common &= set(feat_cols(mdfs[key]))
    common = sorted(common)
    print(f"    共用特征 {len(common)} 个 (各配置交集)\n")

    for cfg in configs:
        key = tuple(cfg)
        lifts, sizes = eval_pool(mdfs[key], folds, common)
        results[key] = lifts
        tr_med = int(np.median([s[0] for s in sizes]))
        print(f"    {cfg[0]}–{cfg[-1]:<6} 训练中位 {tr_med:>5} 场   "
              f"lift {np.nanmean(lifts):+.4f}   fold SD {np.nanstd(lifts, ddof=1):.4f}")

    print(f"\n{'=' * 78}")
    print(f"  结果")
    print(f"{'=' * 78}")
    print(f"  {'配置':<14}{'训练场次':>9}{'平均 lift':>11}{'fold SD':>10}"
          f"{'vs 基线 Δ':>11}{'t':>8}")
    print(f"  {'─' * 66}")
    base_lift = results[base_key]
    for cfg in configs:
        key = tuple(cfg)
        L = results[key]
        name = f"{cfg[0]}–{cfg[-1]}"
        n_tr = len(mdfs[key])
        sd = np.nanstd(L, ddof=1)
        if key == base_key:
            print(f"  {name:<14}{n_tr:>9}{np.nanmean(L):>+11.4f}{sd:>10.4f}"
                  f"{'—':>11}{'—':>8}   ← 基线")
        else:
            pr = L - base_lift
            d = np.nanmean(pr)
            psd = np.nanstd(pr, ddof=1)
            t = d / (psd / np.sqrt(np.sum(~np.isnan(pr)))) if psd > 0 else 0
            print(f"  {name:<14}{n_tr:>9}{np.nanmean(L):>+11.4f}{sd:>10.4f}"
                  f"{d:>+11.4f}{t:>8.2f}")

    print(f"\n  读法")
    print(f"  {'─' * 66}")
    print(f"  · fold SD 是噪声地板。加数据的目的就是把它降下来。")
    print(f"  · 平均 lift 掉了 → 老数据的规律已不适用, 停在上一年。")
    print(f"  · lift 持平但 SD 降 → 这就是你要的, 保留。")
    print(f"  · |t| > 2.5 且 Δ 为负 → 该年数据有害, 别加。")

    best = min(results, key=lambda k: np.nanstd(results[k], ddof=1))
    print(f"\n  噪声地板最低: {best[0]}–{best[-1]}  "
          f"(±{np.nanstd(results[best], ddof=1):.4f})")
    b0 = np.nanstd(base_lift, ddof=1)
    bb = np.nanstd(results[best], ddof=1)
    if bb < b0:
        print(f"  相对基线降低 {(1-bb/b0)*100:.0f}%  →  "
              f"之前 0.77x SD 的 draft 悬案现在约 {0.033/bb:.2f}x SD")
    print(f"\n{'=' * 78}")


if __name__ == "__main__":
    main()
