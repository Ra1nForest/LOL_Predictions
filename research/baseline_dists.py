"""
总人头 / 时长 的基线分布
==========================
为什么需要这个: 现有的四个阶段**只输出 P(蓝方获胜)**。拿这一个数去对
"总人头大小 30.5"或"时长大小 30.5 分钟"的盘口, 是无从下手的 —— 那两个
市场问的是另一个随机变量的分布, 不是胜负。

所以在碰盘口之前, 先得有一个能回答"P(总人头 > x)"的东西。本文件建两级:

    M0  联赛常数   —— 只用该联赛的历史均值和残差分布
    M1  队伍滚动   —— 加上两队近 10 场的自身节奏

**M1 必须过闸才算数**, 判据和全项目一致 (harness.paired_t, |t| > 2.5)。
过不了就用 M0 —— 一个诚实的常数模型也能定价, 只是价定得宽。

━━━ 两个刻意的选择 ━━━

**残差用经验分布, 不假设正态。** 时长和人头都右偏 (打满 45 分钟的有,
打 18 分钟结束的少)。正态会把右尾压扁, 而 O/U 盘口卖的恰恰就是尾部 ——
在盘口价格上, 尾部错 2 个百分点就是全部的利润空间。同时也报一份正态的
分数, 好知道这个选择值多少。

**评分用 CRPS 而不是 RMSE。** RMSE 只管点估计准不准; 但我们要卖的是
"超过某条线的概率", 那是整条分布的事。一个 RMSE 更好但方差估歪的模型,
在盘口上会稳定亏钱。CRPS 同时罚点估计和离散度, 正是我们要的。

    python research/baseline_dists.py                 # 两个目标都跑
    python research/baseline_dists.py --target kills --folds 8
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from harness import from_folds, gate  # noqa: E402
from multiyear_gate import data_path, load_year  # noqa: E402

YEARS = (2022, 2023, 2024, 2025, 2026)
# 近 W 场。10 场大约是一个赛季的三分之一, 短到能跟上阵容变动, 长到
# 不会被一场 50 人头的乱局带偏。和 multiyear_gate 用的是同一个数。
W = 10
# 每折至少要这么多场, 否则分位数估计本身就是噪声
MIN_FOLD = 120


# ══════════════════════════════════════════════════════════
#  逐局表
# ══════════════════════════════════════════════════════════

def build_games():
    """把队伍行折成逐局一行, 并算出每队**赛前**可知的滚动特征。

    滚动量一律 shift(1) —— 不 shift 就把本局的结果算进了本局的特征里,
    而这种泄漏从分数上看就是"模型突然变神", 不会报任何错。
    """
    frames = []
    for y in YEARS:
        d = load_year(data_path(y))
        if d is None:
            print(f"  (缺 {y} 的 CSV, 跳过)")
            continue
        frames.append(d)
    if not frames:
        raise SystemExit("data/ 里一个 CSV 都没有。")
    t = pd.concat(frames, ignore_index=True)
    t = t[t["position"] == "team"].copy()
    t["date"] = pd.to_datetime(t["date"], errors="coerce")
    t = t.dropna(subset=["date", "gameid", "teamname", "kills", "gamelength"])
    t = t.sort_values("date").reset_index(drop=True)

    # 本局的两个目标量, 挂到每一条队伍行上 (两行相同)
    tot = t.groupby("gameid")["kills"].transform("sum")
    t["game_kills"] = tot
    t["minutes"] = t["gamelength"] / 60.0

    # 逐队滚动: 用的是**这支队伍此前打过的局**的节奏
    g = t.groupby("teamname", group_keys=False)
    t["r_kills"] = g["game_kills"].apply(lambda s: s.shift(1).rolling(W, min_periods=3).mean())
    t["r_min"] = g["minutes"].apply(lambda s: s.shift(1).rolling(W, min_periods=3).mean())
    # 自己打出多少人头 vs 全场多少人头 —— 一支"自己猛但对面也猛"的队
    # 和一支"自己猛且能压死对面"的队, 对总人头的影响方向相反。
    t["r_own"] = g["kills"].apply(lambda s: s.shift(1).rolling(W, min_periods=3).mean())

    # 折成逐局一行: 两队的滚动量分别当特征
    rows = []
    for gid, grp in t.groupby("gameid", sort=False):
        if len(grp) != 2:
            continue
        a, b = grp.iloc[0], grp.iloc[1]
        if pd.isna(a["r_kills"]) or pd.isna(b["r_kills"]):
            continue
        rows.append({
            "gameid": gid, "date": a["date"], "league": a["league"],
            "kills": float(a["game_kills"]), "minutes": float(a["minutes"]),
            "rk": (a["r_kills"] + b["r_kills"]) / 2,
            "rm": (a["r_min"] + b["r_min"]) / 2,
            # 两队节奏的**差异**: 快节奏打慢节奏, 结果既不是快也不是慢
            "rk_gap": abs(a["r_kills"] - b["r_kills"]),
            "rm_gap": abs(a["r_min"] - b["r_min"]),
            "ro_sum": a["r_own"] + b["r_own"],
        })
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════
#  评分
# ══════════════════════════════════════════════════════════

def crps_sample(y, draws):
    """经验分布的 CRPS, 用样本算。

    CRPS = E|X - y| - 0.5 * E|X - X'|
    第二项是**离散度的罚**: 分布铺得越宽, 减掉的越多 —— 所以一个把方差
    吹大来"保平安"的模型不会在这里占便宜。这正是 RMSE 给不了的东西。
    """
    d = np.sort(np.asarray(draws, dtype=float))
    n = len(d)
    term1 = np.abs(d - y).mean()
    # E|X-X'| 的 O(n log n) 算法: 排完序后 = 2/n^2 * Σ (2i-n+1) d_i
    i = np.arange(n)
    term2 = 2.0 * np.sum((2 * i - n + 1) * d) / (n * n)
    return term1 - 0.5 * term2


def brier_at(p_over, y, line):
    return (p_over - (1.0 if y > line else 0.0)) ** 2


# ══════════════════════════════════════════════════════════
#  两个模型
# ══════════════════════════════════════════════════════════

def fit_predict(train, test, target, feats):
    """返回 (M0 的残差样本+均值, M1 的残差样本+均值)。

    两个模型共用**同一套经验残差**的做法是错的 —— M1 如果真的解释了一部分
    方差, 它的残差就该更窄。所以各留各的。
    """
    ytr = train[target].to_numpy(float)

    # M0: 联赛常数。每个联赛一个均值 + 一套自己的残差
    lg_mean = train.groupby("league")[target].mean()
    glob = ytr.mean()
    mu0_tr = train["league"].map(lg_mean).fillna(glob).to_numpy(float)
    res0 = ytr - mu0_tr
    mu0_te = test["league"].map(lg_mean).fillna(glob).to_numpy(float)

    # M1: 在 M0 之上做线性回归。用最小二乘而不是 XGB —— 八千场、五个特征,
    # 树模型在这里主要是在拟合噪声, 而且它的残差分布会因为叶子分箱而带
    # 阶梯, 直接影响尾部概率。线性 + 经验残差在这个量级上更稳。
    Xtr = np.column_stack([np.ones(len(train))] + [train[f].to_numpy(float) for f in feats])
    Xte = np.column_stack([np.ones(len(test))] + [test[f].to_numpy(float) for f in feats])
    # 加一点岭正则, 免得两个高度相关的滚动量把系数顶到天上
    lam = 1e-3 * len(train)
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
    coef = np.linalg.solve(A, Xtr.T @ ytr)
    mu1_tr = Xtr @ coef
    res1 = ytr - mu1_tr
    mu1_te = Xte @ coef

    return (mu0_te, res0), (mu1_te, res1)


def mu_noise(train, test, target, feats, seed=0):
    """我们给出的那个均值, 本身有多不稳?

    做法: 把训练集**随机**对半分, 各拟合一次 M1, 让它们预测同一批比赛,
    看两份预测差多少。随机而不是按时间分 —— 按时间分会把"赛季漂移"混进来,
    那是真实的变化, 不是估计噪声。

    半份数据的估计方差约是全份的 2 倍, 所以
        Var(a - b) = 2 * Var(半份) ≈ 4 * Var(全份)
    → 全份拟合的均值标准误 ≈ sd(a - b) / 2。

    为什么非算这个不可: 下面那个"均值要差出多少才有 3% 边缘"的数, 只有
    和这个抖动放在一起才有意义。抖动比它还大, 就意味着每次"发现价值"都
    可能只是这一次拟合恰好偏了 —— 也就是自检 4 演示的那个陷阱, 换了个
    入口而已。
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(train))
    h = len(idx) // 2
    preds = []
    for half in (idx[:h], idx[h:]):
        sub = train.iloc[np.sort(half)]
        (_, _), (mu, _) = fit_predict(sub, test, target, feats)
        preds.append(mu)
    return float(np.std(preds[0] - preds[1], ddof=1) / 2.0)


def score_fold(mu_te, res, yte, line):
    """一折的 (CRPS, Brier@line, 正态 CRPS)。"""
    # 经验残差: 预测分布 = 点估计 + 训练残差的整条经验分布。
    # 抽 400 个而不是全部, 是为了 CRPS 的 O(n log n) 不至于在每条测试样本上
    # 都跑一遍上万个训练残差 —— 400 个分位点对分布形状已经足够。
    q = np.quantile(res, np.linspace(0.002, 0.998, 400))
    sd = res.std(ddof=1)
    c_emp, c_norm, b = [], [], []
    for mu, y in zip(mu_te, yte):
        draws = mu + q
        c_emp.append(crps_sample(y, draws))
        # 正态那一份只是对照, 用同样的点估计和残差 SD
        c_norm.append(_crps_norm(mu, sd, y))
        p_over = float((draws > line).mean())
        b.append(brier_at(p_over, y, line))
    return float(np.mean(c_emp)), float(np.mean(b)), float(np.mean(c_norm)), float(sd)


def _crps_norm(mu, sd, y):
    """正态 CRPS 的闭式解。"""
    from math import erf, exp, pi, sqrt
    z = (y - mu) / sd
    cdf = 0.5 * (1 + erf(z / sqrt(2)))
    pdf = exp(-0.5 * z * z) / sqrt(2 * pi)
    return sd * (z * (2 * cdf - 1) + 2 * pdf - 1 / sqrt(pi))


# ══════════════════════════════════════════════════════════

FEATS = {
    "kills": ["rk", "rm", "rk_gap", "ro_sum"],
    "minutes": ["rm", "rk", "rm_gap"],
}
CN = {"kills": "总人头", "minutes": "时长(分)"}


def run(df, target, n_folds):
    feats = FEATS[target]
    n = len(df)
    edges = [round(n * k / n_folds) for k in range(n_folds + 1)]
    # 第一折没有训练数据可用, 所以从第二折开始评; 训练池是它之前的全部
    f0, f1, b0, b1, fn0, fn1, sds, wob = [], [], [], [], [], [], [], []
    lines = []
    for k in range(1, n_folds):
        train = df.iloc[:edges[k]]
        test = df.iloc[edges[k]:edges[k + 1]]
        if len(test) < MIN_FOLD:
            continue
        # 盘口线取**训练集中位数**: 真实盘口就是开在两边差不多的地方,
        # 拿一条固定的线 (比如永远 30.5) 会让某几折全部倒向一边, 那时
        # Brier 比的其实是"猜得准不准"而不是"概率给得好不好"。
        line = float(train[target].median()) + 0.5
        lines.append(line)
        (mu0, res0), (mu1, res1) = fit_predict(train, test, target, feats)
        yte = test[target].to_numpy(float)
        c0, x0, nn0, _ = score_fold(mu0, res0, yte, line)
        c1, x1, nn1, sd1 = score_fold(mu1, res1, yte, line)
        sds.append(sd1)
        wob.append(mu_noise(train, test, target, feats, seed=k))
        f0.append(c0); f1.append(c1)
        b0.append(x0); b1.append(x1)
        fn0.append(nn0); fn1.append(nn1)

    print(f"\n  折数 {len(f0)}   盘口线 {min(lines):.1f} ~ {max(lines):.1f}")
    print(f"  M0 CRPS {np.mean(f0):.3f}   M1 CRPS {np.mean(f1):.3f}")
    print(f"  M0 Brier {np.mean(b0):.4f}   M1 Brier {np.mean(b1):.4f}")

    # CRPS 和 Brier 都是越小越好, gate() 认越大越好 → 取负。
    print()
    ok_c = gate(from_folds([-x for x in f0], "M0"),
                from_folds([-x for x in f1], "M1"),
                f"{CN[target]}: 队伍滚动特征是否优于联赛常数 (CRPS)")
    print()
    ok_b = gate(from_folds([-x for x in b0], "M0"),
                from_folds([-x for x in b1], "M1"),
                f"{CN[target]}: 同上 (盘口线上的 Brier)")

    print()
    e = np.mean(fn1) - np.mean(f1)
    print(f"  经验残差 vs 正态残差 (都用 M1 的点估计): CRPS {np.mean(f1):.3f} vs "
          f"{np.mean(fn1):.3f}  差 {e:+.3f}")
    print(f"  正数表示经验分布更好 —— 也就是这个量确实偏得够多, 值得不假设正态。")

    # ── 这个市场到底玩不玩得动 ──
    # 过不过闸只说明"M1 比 M0 好"; 能不能定价看的是**残差有多宽**。
    # 残差 SD = sigma 时, 我们对"超过线 L"的概率估计是
    #     P = 1 - Phi((L - mu) / sigma)
    # 于是盘口的线每偏离我们的均值 1 个单位, 概率才动 phi(0)/sigma ≈ 0.4/sigma。
    # 要吃到 3% 的 EV, 概率至少得比市场高 3 个点 (还没算抽水), 换算过来
    # 就是"我们的均值必须比市场的线稳定地差出这么多"。
    from math import sqrt, pi
    sig = float(np.mean(sds))
    per_unit = 0.3989 / sig          # 线动一个单位, P(over) 动多少
    need = 0.03 / per_unit           # 要 3 个百分点的概率差, 均值得差多少
    print()
    print(f"  ── 定价精度 ──")
    print(f"  M1 残差 SD {sig:.2f} ({CN[target]}的单位)")
    print(f"  盘口线每偏 1 个单位, 我们给出的 P(大分) 只动 {per_unit:.3f}")
    print(f"  → 想拿到 3 个百分点的概率优势, 我们的均值必须比市场的线")
    print(f"     差出 **{need:.2f}** 个单位。")

    w = float(np.mean(wob))
    print()
    print(f"  而我们自己那个均值的标准误是 **{w:.2f}** 个单位 (半分法测得)。")
    if w >= need:
        print(f"  ⚠ 抖动 ({w:.2f}) >= 需要的差距 ({need:.2f})。")
        print(f"    也就是说单看一场, '模型说 3% 边缘' 和 '这次拟合恰好偏了'")
        print(f"    根本分不开 —— 逐场挑价值在这个市场上是没有意义的。")
        print(f"    唯一还站得住的用法: 在**大量比赛上聚合**看有没有系统性偏移,")
        print(f"    而不是拿它去选某一场的盘。")
    else:
        print(f"  抖动 ({w:.2f}) < 需要的差距 ({need:.2f})，比值 {need/w:.1f}x。")
        print(f"    信噪比勉强够, 但这只是必要条件 —— 市场的线本身也不是均值,")
        print(f"    它已经包含了我们没有的信息 (伤病、轮换、赛程)。")
    return ok_c, ok_b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["kills", "minutes", "both"], default="both")
    ap.add_argument("--folds", type=int, default=8)
    a = ap.parse_args()

    print("加载 …")
    df = build_games()
    print(f"  可用对局 {len(df)}   {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  联赛 {sorted(df['league'].unique())}")
    for tgt in ["kills", "minutes"]:
        print(f"\n  {CN[tgt]}: 均值 {df[tgt].mean():.1f}  SD {df[tgt].std():.1f}  "
              f"偏度 {df[tgt].skew():+.2f}")

    for tgt in (["kills", "minutes"] if a.target == "both" else [a.target]):
        print("\n" + "=" * 66)
        print(f"目标: {CN[tgt]}")
        print("=" * 66)
        run(df, tgt, a.folds)

    print("\n" + "=" * 66)
    print("提醒: 这里比的是 M1 vs M0, **不是 M1 vs 盘口**。")
    print("能不能定价要等有了赔率之后跑 gate_vs_market.py 才知道 ——")
    print("残差 SD 就是我们的定价精度, 而盘口的定价精度通常比它窄得多。")
    print("=" * 66)


if __name__ == "__main__":
    main()
