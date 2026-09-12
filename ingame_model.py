"""
Stage 4 — 局内预测模型
========================
目标: 敢下判断, 且能识别「领先但阵容吃亏」这类翻盘隐患。

核心思路
--------
1. 英雄强势期指数 (scaling index)
     index(C) = 平均(赢的局时长) - 平均(输的局时长)
   正 = 后期英雄 (赢的都是长局)   负 = 前期英雄 (赢的都是快局)
   只用该场比赛之前的数据计算, 防泄漏。

2. 阵容强势期 = 五个英雄指数的均值, 双方相减得 scaling_diff

3. 交互项 —— 这是「翻盘隐患」的量化
     golddiff × scaling_diff
   领先 4k + 对面后期阵容  → 负 → 压低胜率
   领先 4k + 对面前期阵容  → 正 → 抬高胜率
   同样的经济差, 在不同阵容下含义相反。

4. 时间切片 T ∈ {10,15,20,25} 作为特征, 一个模型覆盖全部时间点
   3000 场 × 4 切片 ≈ 12000 样本, 是赛前模型的四倍

防泄漏的关键
------------
按「比赛」切分训练/测试, 不是按「快照」。同一场比赛的 T=10 和 T=25
如果分别落进训练集和测试集, 模型就见过这场比赛了。

关于条件化 (不是 survivorship bias)
-----------------------------------
T=25 的模型只在打过 25 分钟的比赛上训练。这是正确的 —— 你在第 25 分钟
做预测时, 面对的本来就是一场打过 25 分钟的比赛。训练条件和预测条件一致。
真实影响只是: 巨大领先的局常提前结束, 所以 T=25 时极端领先的样本偏少,
模型在那个区域会更不确定 —— 这是对的行为。
"""
import numpy as np, pandas as pd, warnings, sys, bisect, json
from pathlib import Path
warnings.filterwarnings("ignore")

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

# 训练用的赛区。**不只是四大** —— research/gate_league_mix.py 在 2026-08-17
# 量过: 把次级/青训联赛加进训练集, 对四大赛区自身的预测稳定变好。
#     仅四大 +0.4195  →  含次级 +0.4598
#     Δ=+0.0402  配对SD 0.0173  t=+5.19 (判据 2.5)  五折全正
# 训练量 20311 -> 175188 条快照。次级联赛只进训练侧, 验证窗口始终只有四大。
#
# 直觉上"杂牌数据会污染一线预测"是错的: 局内胜负的规律 (经济差怎么转化成
# 胜率、什么时候还能翻) 跨级别是通用的, 而样本量翻 8.6 倍带来的方差下降
# 压过了分布差异。
#
# 推理侧不受影响: api.py 只接受四大赛区, 这里扩的是模型见过多少局面。
TARGET   = None          # None = 不按赛区过滤; 见上方闸门结论
SLICES   = [10, 15, 20, 25]
W        = 10
ROLES    = ["top", "jng", "mid", "bot", "sup"]
MIN_CHAMP_GAMES = 60          # 算 scaling index 所需的最少场次
SHRINK_K        = 80          # 收缩强度: n/(n+K), 小样本往 0 拉

PRE_STATS = ["kills", "deaths", "totalgold", "towers", "dragons",
             "golddiffat15", "cspm", "ckpm"]

XGB = dict(n_estimators=500, max_depth=5, learning_rate=0.04,
           subsample=0.8, colsample_bytree=0.7,
           eval_metric="logloss", random_state=42, verbosity=0)


# ══════════════════════════════════════════════════════════
#  1. 加载
# ══════════════════════════════════════════════════════════

def load(files):
    dfs = []
    for f in files:
        if Path(f).exists() or f.startswith("/mnt"):
            print(f"    {f.split('/')[-1]}")
            dfs.append(pd.read_csv(f, low_memory=False))
    if not dfs:
        print("  找不到 CSV"); sys.exit(1)
    raw = pd.concat(dfs, ignore_index=True)
    raw["league"] = raw["league"].replace({"LTA N": "LCS", "LTA S": "CBLOL"})
    d = (raw if TARGET is None
         else raw[raw["league"].isin(TARGET)]).copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    out = d.sort_values("date").reset_index(drop=True)
    # 用旧数据跑出来的结果看着完全正常, 所以每次加载都报一下新鲜度
    try:
        from data_freshness import check_frame
        check_frame(out, label="    Oracle's Elixir ")
    except Exception:
        pass
    return out


# ══════════════════════════════════════════════════════════
#  2. 英雄强势期指数 (prior-only)
# ══════════════════════════════════════════════════════════

class ScalingIndex:
    """
    强势期指数 = 胜率对局时长的斜率, 而不是「赢的时长 − 输的时长」。

    为什么换
    --------
    旧构造 mean(len|win) − mean(len|loss) 是两个子集均值的差, 每个子集
    都只有一半样本, 方差是单个均值的两倍。冷门英雄二十场就能算出 ±4 分钟
    的极端值 —— 大嘴排「最前期」、盖伦排「最后期」就是这么来的。

    新构造用线性概率模型的斜率:
        slope = Cov(result, length) / Var(length)
    它在一个回归里用掉全部样本, 而且把「这个英雄整体强不强」(截距)
    和「它随时间变强还是变弱」(斜率) 分开了 —— 后者才是我们要的。

    单位: 每 10 分钟的胜率变化。+0.05 = 局时每长 10 分钟, 胜率涨 5pp。

    收缩
    ----
    slope × n/(n+K)。样本少的往 0 拉, 而不是硬性截断。这比设一个门槛
    然后一刀切更合理 —— 30 场的英雄有信息, 只是应该少信一点。

    全部用前缀和 + bisect, 查询 O(log n), 8000 场也不慢。
    """
    def __init__(self, players):
        self.tbl = {}
        p = players.dropna(subset=["champion", "gamelength_min", "result"])
        for ch, g in p.groupby("champion"):
            g = g.sort_values("date")
            x = g["gamelength_min"].values.astype(float)     # 局时长
            y = g["result"].values.astype(float)             # 胜负
            z = np.zeros(1)
            self.tbl[ch] = dict(
                dates=g["date"].values,
                n  =np.arange(len(x) + 1, dtype=float),
                sx =np.concatenate([z, np.cumsum(x)]),
                sy =np.concatenate([z, np.cumsum(y)]),
                sxy=np.concatenate([z, np.cumsum(x * y)]),
                sxx=np.concatenate([z, np.cumsum(x * x)]),
            )

    def get(self, champ, before):
        e = self.tbl.get(champ)
        if e is None:
            return np.nan
        i = int(np.searchsorted(e["dates"], np.datetime64(before), side="left"))
        n = e["n"][i]
        if n < MIN_CHAMP_GAMES:
            return np.nan
        sx, sy, sxy, sxx = e["sx"][i], e["sy"][i], e["sxy"][i], e["sxx"][i]
        denom = n * sxx - sx * sx
        if denom <= 0:
            return np.nan
        slope = (n * sxy - sx * sy) / denom        # 胜率 / 分钟
        shrunk = slope * (n / (n + SHRINK_K))      # 小样本往 0 收缩
        return float(shrunk * 10.0)                # 换算成「每 10 分钟」

    def sample_size(self, champ, before):
        e = self.tbl.get(champ)
        if e is None:
            return 0
        return int(np.searchsorted(e["dates"], np.datetime64(before), side="left"))

    def snapshot(self, before, top_n=12):
        out = []
        for ch in self.tbl:
            v = self.get(ch, before)
            if not np.isnan(v):
                out.append((ch, v, self.sample_size(ch, before)))
        out.sort(key=lambda x: -x[1])
        return out[:top_n], out[-top_n:]


# ══════════════════════════════════════════════════════════
#  3. 构建局内快照数据集
# ══════════════════════════════════════════════════════════

def build(df, verbose=True):
    teams = df[df["position"] == "team"].copy()
    teams["result"] = teams["result"].astype(int)
    gl = pd.to_numeric(teams["gamelength"], errors="coerce")
    teams["gamelength_min"] = gl / 60 if gl.max() > 200 else gl

    players = df[df["position"].isin(ROLES)].copy()
    players["result"] = players["result"].astype(int)
    pgl = pd.to_numeric(players["gamelength"], errors="coerce")
    players["gamelength_min"] = pgl / 60 if pgl.max() > 200 else pgl

    if verbose:
        print(f"    {teams['gameid'].nunique()} 场, {players['champion'].nunique()} 个英雄")
        print("    构建英雄强势期指数...")
    sc = ScalingIndex(players)

    # 赛前队伍滚动统计 (轻量版)
    avail = [c for c in PRE_STATS if c in teams.columns]
    hist = {}
    for tm in teams["teamname"].unique():
        t = teams[teams["teamname"] == tm].sort_values("date").copy()
        for c in avail:
            t[f"pre_{c.replace(' ','_')}"] = (pd.to_numeric(t[c], errors="coerce")
                                              .rolling(W, min_periods=3).mean().shift(1))
        t["pre_wr"] = t["result"].rolling(W, min_periods=3).mean().shift(1)
        hist[tm] = t
    pre_cols = [c for c in list(hist.values())[0].columns
                if c.startswith("pre_")]

    pbg = {g: d for g, d in players.groupby("gameid")}
    rows = []
    if verbose: print("    构建快照...")

    for gid, grp in teams.groupby("gameid"):
        if len(grp) != 2:
            continue
        r = grp.sort_values("participantid")
        bT, rT = r.iloc[0], r.iloc[1]
        bt, rt = bT["teamname"], rT["teamname"]
        if bt not in hist or rt not in hist:
            continue
        bh = hist[bt][hist[bt]["gameid"] == gid]
        rh = hist[rt][hist[rt]["gameid"] == gid]
        if bh.empty or rh.empty:
            continue
        bh, rh = bh.iloc[0], rh.iloc[0]

        gd, lg = bT["date"], bT["league"]
        length = bT["gamelength_min"]
        y = int(bT["result"])

        # 阵容强势期
        gp = pbg.get(gid)
        b_sc = r_sc = np.nan
        if gp is not None:
            bp = gp[gp["side"] == "Blue"]["champion"].tolist()
            rp = gp[gp["side"] == "Red"]["champion"].tolist()
            if len(bp) == 5 and len(rp) == 5:
                bv = [sc.get(c, gd) for c in bp]
                rv = [sc.get(c, gd) for c in rp]
                bv = [x for x in bv if not np.isnan(x)]
                rv = [x for x in rv if not np.isnan(x)]
                if len(bv) >= 3 and len(rv) >= 3:
                    b_sc, r_sc = float(np.mean(bv)), float(np.mean(rv))
        if np.isnan(b_sc) or np.isnan(r_sc):
            continue
        sc_diff = b_sc - r_sc

        # 赛前基础
        base = {"gameid": gid, "date": gd, "league": lg, "y": y,
                "blue_team": bt, "red_team": rt,
                "is_lpl": int(lg == "LPL"), "is_lck": int(lg == "LCK"),
                "is_lec": int(lg == "LEC"), "is_lcs": int(lg == "LCS"),
                "b_scaling": b_sc, "r_scaling": r_sc, "scaling_diff": sc_diff,
                "scaling_sum": b_sc + r_sc}
        for c in pre_cols:
            try:
                bvv, rvv = float(bh.get(c, np.nan)), float(rh.get(c, np.nan))
                base[f"diff_{c}"] = bvv - rvv
            except (TypeError, ValueError):
                pass

        # 每个时间切片一行
        for T in SLICES:
            if length <= T:
                continue
            need = [f"golddiffat{T}", f"xpdiffat{T}", f"csdiffat{T}",
                    f"killsat{T}", f"opp_killsat{T}"]
            if not all(c in bT.index for c in need):
                continue
            vals = [pd.to_numeric(bT.get(c), errors="coerce") for c in need]
            if any(pd.isna(v) for v in vals):
                continue
            gdif, xdif, cdif, bk, rk = [float(v) for v in vals]
            gold_total = pd.to_numeric(bT.get(f"goldat{T}"), errors="coerce")
            gold_total = float(gold_total) if pd.notna(gold_total) else np.nan

            m = dict(base)
            m["T"] = T
            m["golddiff"] = gdif
            m["xpdiff"] = xdif
            m["csdiff"] = cdif
            m["killdiff"] = bk - rk
            m["killsum"] = bk + rk
            # 归一化领先 —— 同样 4k 在 10 分钟和 25 分钟含义不同
            m["golddiff_norm"] = gdif / gold_total if gold_total and gold_total > 0 else np.nan

            # ★ 交互项: 翻盘隐患
            m["x_gold_scaling"] = gdif * sc_diff
            m["x_goldnorm_scaling"] = (m["golddiff_norm"] * sc_diff
                                       if pd.notna(m["golddiff_norm"]) else np.nan)
            m["x_xp_scaling"] = xdif * sc_diff
            # 领先方是否是前期阵容 (领先但吃亏的信号)
            m["lead_by_early_comp"] = gdif * (-sc_diff)

            rows.append(m)

    out = pd.DataFrame(rows)
    if verbose:
        print(f"    {out['gameid'].nunique()} 场 → {len(out)} 个快照")
        for T in SLICES:
            print(f"      T={T:>2}: {(out['T']==T).sum():>5}")
    return out, sc


INTERACTIONS = ["x_gold_scaling", "x_goldnorm_scaling",
                "x_xp_scaling", "lead_by_early_comp"]


def features(mdf, with_interactions=True):
    f = [c for c in mdf.columns
         if c not in ("gameid", "date", "league", "y", "blue_team", "red_team")
         and mdf[c].dtype in ("float64", "int64", "float32", "int32")]
    if not with_interactions:
        f = [c for c in f if c not in INTERACTIONS]
    return f


# ══════════════════════════════════════════════════════════
#  4. 按「比赛」切分 (不是按快照)
# ══════════════════════════════════════════════════════════

def split_by_game(mdf, frac_tr=0.60, frac_cal=0.20):
    order = (mdf.groupby("gameid")["date"].min().sort_values())
    gids = order.index.to_numpy()
    n = len(gids)
    a, b = int(n * frac_tr), int(n * (frac_tr + frac_cal))
    s_tr, s_cal, s_te = set(gids[:a]), set(gids[a:b]), set(gids[b:])
    return (mdf[mdf["gameid"].isin(s_tr)],
            mdf[mdf["gameid"].isin(s_cal)],
            mdf[mdf["gameid"].isin(s_te)])


def ece(p, y, nb=10):
    e, ed = 0.0, np.linspace(0, 1, nb + 1)
    for i in range(nb):
        m = (p > ed[i]) & (p <= ed[i + 1]) if i else (p >= 0) & (p <= ed[1])
        if m.sum():
            e += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
    return e


def fit_eval(mdf, feats, label="", verbose=True):
    tr, cal, te = split_by_game(mdf)
    Xtr, ytr = tr[feats].astype(float).fillna(-999), tr["y"].astype(int)
    Xc,  yc  = cal[feats].astype(float).fillna(-999), cal["y"].astype(int)
    Xte, yte = te[feats].astype(float).fillna(-999), te["y"].astype(int).values

    m = XGBClassifier(**XGB); m.fit(Xtr, ytr, verbose=False)
    pc = m.predict_proba(Xc)[:, 1]
    cb = LogisticRegression(C=1e6).fit(pc.reshape(-1, 1), yc)
    raw = m.predict_proba(Xte)[:, 1]
    p = cb.predict_proba(raw.reshape(-1, 1))[:, 1]

    base = max(yte.mean(), 1 - yte.mean())
    acc = accuracy_score(yte, (p > .5).astype(int))
    res = dict(label=label, model=m, calib=cb, feats=feats,
               n_train=len(tr), n_test=len(te),
               acc=acc, base=base, lift=(acc - base) / (1 - base),
               brier=brier_score_loss(yte, p),
               brier_const=yte.mean() * (1 - yte.mean()),
               logloss=log_loss(yte, p), ece=ece(p, yte),
               p_min=float(p.min()), p_max=float(p.max()),
               p=p, yte=yte, te=te)
    return res


# ══════════════════════════════════════════════════════════
#  5. Walk-forward 闸 (按比赛切, 复用 harness 的思路)
# ══════════════════════════════════════════════════════════

def wf_gate(mdf, feats_a, feats_b, n_folds=6, seeds=(0, 1, 2), val_frac=0.12):
    """
    比较两组特征。按比赛滚动切分, 返回逐 fold 的归一化 lift。
    """
    order = mdf.groupby("gameid")["date"].min().sort_values()
    gids = order.index.to_numpy()
    n = len(gids)
    val_g = max(60, int(n * val_frac))
    cuts = np.linspace(int(n * 0.40), n - val_g, n_folds).astype(int)

    out = {"A": np.zeros((n_folds, len(seeds))), "B": np.zeros((n_folds, len(seeds)))}
    for fi, cut in enumerate(cuts):
        tr_g = set(gids[:cut]); va_g = set(gids[cut:cut + val_g])
        tr = mdf[mdf["gameid"].isin(tr_g)]
        va = mdf[mdf["gameid"].isin(va_g)]
        yv = va["y"].astype(int).values
        base = max(yv.mean(), 1 - yv.mean())
        for tag, fs in (("A", feats_a), ("B", feats_b)):
            Xtr = tr[fs].astype(float).fillna(-999); ytr = tr["y"].astype(int)
            Xva = va[fs].astype(float).fillna(-999)
            for si, sd in enumerate(seeds):
                mm = XGBClassifier(**{**XGB, "random_state": sd})
                mm.fit(Xtr, ytr, verbose=False)
                a = accuracy_score(yv, (mm.predict_proba(Xva)[:, 1] > .5).astype(int))
                out[tag][fi, si] = (a - base) / (1 - base)
    return out["A"].mean(axis=1), out["B"].mean(axis=1)


# ══════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════

def main(files):
    print("=" * 74)
    print("  Stage 4 — 局内预测模型")
    print("=" * 74)

    print("\n[1/5] 加载")
    df = load(files)

    print("\n[2/5] 时间切片跨年审计")
    df["_yr"] = df["date"].dt.year
    t_all = df[df["position"] == "team"]
    bad_T = []
    print(f"    {'年份':<8}" + "".join(f"T={T:<8}" for T in SLICES) + "场均时长")
    print(f"    {'─'*54}")
    for yr, g in t_all.groupby("_yr"):
        gl = pd.to_numeric(g["gamelength"], errors="coerce")
        gl = gl / 60 if gl.max() > 200 else gl
        line = f"    {int(yr):<8}"
        for T in SLICES:
            cols = [f"golddiffat{T}", f"xpdiffat{T}", f"csdiffat{T}",
                    f"killsat{T}", f"opp_killsat{T}"]
            have = [c for c in cols if c in g.columns]
            fill = g[have].notna().all(axis=1).mean() if len(have) == len(cols) else 0.0
            line += f"{fill:>6.0%}   "
            if fill < 0.30:
                bad_T.append((int(yr), T, fill))
        line += f"{gl.median():>6.1f}"
        print(line)
    if bad_T:
        print(f"\n    ⚠ 以下年份/切片填充率过低, 这些年的该切片会自动缺失:")
        for yr, T, f_ in bad_T:
            print(f"        {yr} T={T}: {f_:.0%}")
    print(f"\n    注: 场均时长跨年若有明显下降, 英雄强势期指数 (基于局时长) 的")
    print(f"        绝对值会随之漂移。它是个差值, 相对稳健, 但值得留意。")

    print("\n[2b/5] 构建快照数据集")
    mdf, sc = build(df)
    if len(mdf) < 500:
        print("  样本太少"); sys.exit(1)

    # 强势期指数合理性检查
    late, early = sc.snapshot(mdf["date"].max())
    print(f"\n  英雄强势期指数 (胜率对局时长的斜率, 每 10 分钟的胜率变化)")
    print(f"  正 = 越拖越强   负 = 越快越强   已按样本量收缩")
    print(f"  {'最后期 (scaling)':<34}{'最前期 (early)':<34}")
    print(f"  {'─'*68}")
    for i in range(min(10, len(late), len(early))):
        L = f"{late[i][0]:<16}{late[i][1]:>+7.3f} (n={late[i][2]})"
        E = f"{early[-(i+1)][0]:<16}{early[-(i+1)][1]:>+7.3f} (n={early[-(i+1)][2]})"
        print(f"  {L:<34}{E:<34}")
    n_ok = sum(1 for ch in sc.tbl if not np.isnan(sc.get(ch, mdf["date"].max())))
    print(f"\n  {n_ok}/{len(sc.tbl)} 个英雄样本足够 (>= {MIN_CHAMP_GAMES} 场)")
    print(f"\n  ↑ 拿这个对照你的游戏理解 —— 如果排序明显荒谬, 说明指数构造有问题")

    print("\n[3/5] 训练局内模型")
    f_all = features(mdf, True)
    f_no  = features(mdf, False)
    r = fit_eval(mdf, f_all, "含交互项")
    print(f"    训练 {r['n_train']} 快照 / 测试 {r['n_test']} 快照, {len(f_all)} 特征")

    print(f"\n{'=' * 74}")
    print(f"  局内模型 vs 赛前模型")
    print(f"{'=' * 74}")
    print(f"  {'':<22}{'赛前 (Stage 2)':>16}{'局内 (Stage 4)':>16}")
    print(f"  {'─' * 54}")
    print(f"  {'准确率':<20}{'64.3%':>16}{r['acc']:>16.1%}")
    print(f"  {'归一化 lift':<19}{'+0.219':>16}{r['lift']:>+16.3f}")
    print(f"  {'Brier':<20}{'0.2276':>16}{r['brier']:>16.4f}")
    print(f"  {'ECE':<20}{'0.046':>16}{r['ece']:>16.4f}")
    rng = "[%.2f, %.2f]" % (r["p_min"], r["p_max"])
    print(f"  {'概率范围':<20}{'[0.27, 0.70]':>16}{rng:>16}")
    width_pre, width_in = 0.70 - 0.27, r["p_max"] - r["p_min"]
    print(f"\n  概率范围宽度 {width_pre:.2f} → {width_in:.2f}  "
          f"({width_in/width_pre:.1f}x)")

    # 分时间点
    print(f"\n  分时间点 (测试集)")
    print(f"  {'T':>4}{'快照数':>8}{'准确率':>9}{'p 范围':>18}{'高置信占比':>11}")
    print(f"  {'─' * 50}")
    te = r["te"].copy(); te["p"] = r["p"]
    for T in SLICES:
        s = te[te["T"] == T]
        if len(s) < 20: continue
        acc = accuracy_score(s["y"].astype(int), (s["p"] > .5).astype(int))
        conf = ((s["p"] > 0.8) | (s["p"] < 0.2)).mean()
        rng_t = "[%.2f, %.2f]" % (s["p"].min(), s["p"].max())
        print(f"  {T:>4}{len(s):>8}{acc:>9.1%}{rng_t:>18}{conf:>11.1%}")

    print("\n[4/5] 交互项过闸")
    # 判决规则只有一份, 在 research/harness.py。这里曾经内联抄过一遍, 抄漏了
    # 配对 SD 为零的分支 (t 记作 0), 于是「每个 fold 都同向」这种证据最强的
    # 情况反而被判成噪声。切分逻辑不共用是有意的: 见 wf_gate 的 gameid 分组。
    sys.path.insert(0, str(Path(__file__).parent / "research"))
    from harness import from_folds, gate

    A, B = wf_gate(mdf, f_no, f_all)
    n_better = int((B - A > 0).sum())
    print(f"    {n_better}/{len(A)} fold 改善")
    gate(from_folds(A, "无交互项"), from_folds(B, "含交互项"), "局内交互项")

    print("\n[5/5] 翻盘隐患检验")
    print("  同样的经济领先, 阵容优劣不同时的实际胜率")
    print("  ⚠ 按「比赛」计数, 不按快照 —— 同一场比赛的 4 个快照结果相同, 不是独立样本")

    lead = te[(te["golddiff"] > 1500) & (te["golddiff"] < 6000)].copy()
    if len(lead) > 40:
        # 每场比赛取一个代表快照 (领先幅度中位的那个), 避免重复计数
        lead = lead.sort_values("golddiff").groupby("gameid", as_index=False).nth(
            len(lead) // max(1, lead["gameid"].nunique()) // 2 if False else 0)
        g = lead.groupby("gameid").agg(
            golddiff=("golddiff", "mean"), scaling_diff=("scaling_diff", "first"),
            p=("p", "mean"), y=("y", "first")).reset_index()

        n_games = len(g)
        print(f"\n    领先 1.5k–6k 的比赛 {n_games} 场 (原始快照 {len(te[(te['golddiff']>1500)&(te['golddiff']<6000)])} 个)")

        if n_games < 30:
            print(f"    ⚠ 只有 {n_games} 场, 样本不足以得出结论。两年数据会好很多。")

        q = g["scaling_diff"].quantile([0.33, 0.67])
        good = g[g["scaling_diff"] >= q.iloc[1]]
        bad  = g[g["scaling_diff"] <= q.iloc[0]]

        def ci(k, n):
            """胜率的 Wilson 95% 区间"""
            if n == 0: return (0, 0)
            p_ = k / n; z = 1.96
            d = 1 + z*z/n
            c = (p_ + z*z/(2*n)) / d
            h = z*np.sqrt(p_*(1-p_)/n + z*z/(4*n*n)) / d
            return (max(0, c-h), min(1, c+h))

        for nm, s in (("阵容也占优", good), ("阵容吃亏  ", bad)):
            if len(s) == 0: continue
            k = int(s["y"].sum()); n_ = len(s)
            lo, hi = ci(k, n_)
            print(f"      {nm}  预测均值 {s['p'].mean():.3f}   "
                  f"实际 {k}/{n_} = {k/n_:.3f}  [95% CI {lo:.2f}–{hi:.2f}]")

        if len(good) and len(bad):
            gap_p = good["p"].mean() - bad["p"].mean()
            gap_y = good["y"].mean() - bad["y"].mean()
            # 两个比例差的合并标准误
            se = np.sqrt(good["y"].var(ddof=1)/len(good) + bad["y"].var(ddof=1)/len(bad)) \
                 if len(good) > 1 and len(bad) > 1 else np.nan
            print(f"\n      模型给出的差距 {gap_p:+.3f}   实际差距 {gap_y:+.3f}", end="")
            if se and not np.isnan(se) and se > 0:
                print(f"  (± {1.96*se:.3f})")
                if abs(gap_y) < 1.96 * se:
                    print(f"      → 实际差距在噪声内, 无法判定翻盘隐患是否存在")
                elif gap_y > 0 and gap_p > 0:
                    print(f"      ✓ 现象存在且模型方向正确")
                elif gap_y > 0:
                    print(f"      ⚠ 现象存在但模型没捕捉到")
                else:
                    print(f"      ✗ 方向与假设相反 —— 领先方阵容偏前期时反而赢得更多")
                    print(f"        (合理解释: 前期阵容拿到领先会滚雪球提前结束比赛,")
                    print(f"         没滚起来的那些局根本不会出现在「领先」这个筛选里)")
            else:
                print()

    imp = pd.Series(r["model"].feature_importances_, index=f_all).sort_values(ascending=False)
    print(f"\n  Top 15 特征")
    print(f"  {'─' * 50}")
    for i, (f, v) in enumerate(imp.head(15).items(), 1):
        tag = " ★交互" if f in INTERACTIONS else ""
        print(f"  {i:>2}. {f:<34}{v:>8.4f}{tag}")

    print(f"\n{'=' * 74}")


if __name__ == "__main__":
    FILES = [str(Path(__file__).parent / "data" /
                 f"{y}_LoL_esports_match_data_from_OraclesElixir.csv")
             for y in (2022, 2023, 2024, 2025, 2026)]
    main(FILES)
