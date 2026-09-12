"""
分半信度 — 这些量本身可不可重复测量
=====================================
过闸测的是「加进模型有没有用」。信度测的是更根本的问题:
**你测的这个东西, 换一半数据再测一遍还是同一个值吗?**

方法
----
把比赛分成两半, 各自独立算一遍, 然后求相关。

  真实的属性  → 两半的估计应该一致, r 明显 > 0
  纯噪声      → r 接近 0

两种切法各有含义:
  随机切半   测「可测量性」—— 排除时间因素, 这个量能不能稳定估出来
  按时间切   测「时间稳定性」—— 会混入版本更迭和 meta 漂移

参照系
------
同时测一个我们已知真实的量: **单英雄胜率**。
它给出「一个真实但有噪声的英雄属性, 在这个数据量下信度大概多少」。

如果单英雄胜率 r=0.5 而协同 r=0.05, 结论就很清楚了。

用法: python reliability.py
"""
import numpy as np, pandas as pd, warnings, sys, itertools
from pathlib import Path
from collections import defaultdict
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from multiyear_gate import TARGET, ROLES, load_year, data_path

YEARS      = (2022, 2023, 2024, 2025, 2026)
MIN_PAIR   = 30
MIN_CHAMP  = 50
MIN_SCALE  = 60
SHRINK_K   = 120


def logit(p, eps=1e-3):
    p = min(max(p, eps), 1 - eps)
    return np.log(p / (1 - p))


def collect(teamsides):
    """teamsides: list of (champs, result, gamelength)"""
    champ_w, champ_n = defaultdict(float), defaultdict(int)
    pair_w,  pair_n  = defaultdict(float), defaultdict(int)
    # 强势期用的累积量
    sc = defaultdict(lambda: dict(n=0., sx=0., sy=0., sxy=0., sxx=0.))
    for champs, res, glen in teamsides:
        for c in champs:
            champ_w[c] += res; champ_n[c] += 1
            e = sc[c]
            e["n"] += 1; e["sx"] += glen; e["sy"] += res
            e["sxy"] += glen * res; e["sxx"] += glen * glen
        for a, b in itertools.combinations(sorted(champs), 2):
            pair_w[(a, b)] += res; pair_n[(a, b)] += 1
    return champ_w, champ_n, pair_w, pair_n, sc


def champ_wr(cw, cn, min_n=MIN_CHAMP):
    return {c: cw[c] / cn[c] for c in cn if cn[c] >= min_n}


def pair_syn(cw, cn, pw, pn, min_p=MIN_PAIR, min_c=MIN_CHAMP, K=SHRINK_K):
    out = {}
    for k, n in pn.items():
        if n < min_p: continue
        a, b = k
        if cn.get(a, 0) < min_c or cn.get(b, 0) < min_c: continue
        raw = logit(pw[k] / n) - logit(cw[a] / cn[a]) - logit(cw[b] / cn[b])
        out[k] = raw * (n / (n + K))
    return out


def scaling(sc, min_n=MIN_SCALE, K=80):
    out = {}
    for c, e in sc.items():
        n = e["n"]
        if n < min_n: continue
        den = n * e["sxx"] - e["sx"] ** 2
        if den <= 0: continue
        slope = (n * e["sxy"] - e["sx"] * e["sy"]) / den
        out[c] = slope * (n / (n + K)) * 10.0
    return out


def corr(d1, d2, label, min_overlap=30):
    keys = sorted(set(d1) & set(d2))
    if len(keys) < min_overlap:
        print(f"  {label:<24} 重叠只有 {len(keys)} 项, 样本不足")
        return None
    x = np.array([d1[k] for k in keys])
    y = np.array([d2[k] for k in keys])
    if x.std() == 0 or y.std() == 0:
        print(f"  {label:<24} 方差为零"); return None
    r = float(np.corrcoef(x, y)[0, 1])
    rx = pd.Series(x).rank().values; ry = pd.Series(y).rank().values
    rho = float(np.corrcoef(rx, ry)[0, 1])
    # 相关系数的近似 95% CI (Fisher z)
    n = len(keys)
    z = 0.5 * np.log((1 + r) / (1 - r)) if abs(r) < 0.999 else 0
    se = 1 / np.sqrt(max(1, n - 3))
    lo = np.tanh(z - 1.96 * se); hi = np.tanh(z + 1.96 * se)
    verdict = ("可重复" if r > 0.35 else "弱" if r > 0.15 else "≈ 噪声")
    print(f"  {label:<24} n={n:<5} r={r:>+6.3f} [{lo:+.2f},{hi:+.2f}]  "
          f"ρ={rho:>+6.3f}   {verdict}")
    return r


def main():
    print("=" * 78)
    print("  分半信度 — 这些量本身可不可重复测量")
    print("=" * 78)

    print("\n[1/3] 加载")
    dfs = []
    for y in YEARS:
        d = load_year(data_path(y))
        if d is None:
            print(f"    {y} 缺失"); continue
        dfs.append(d)
        print(f"    {y}  {d[d['position']=='team']['gameid'].nunique():>5} 场")
    if not dfs:
        print("  无数据"); sys.exit(1)
    df = pd.concat(dfs, ignore_index=True).sort_values("date")

    p = df[df["position"].isin(ROLES)].copy()
    p["result"] = p["result"].astype(int)
    gl = pd.to_numeric(p["gamelength"], errors="coerce")
    p["glen"] = gl / 60 if gl.max() > 200 else gl

    # 每个 (比赛, 阵营) 一条记录
    units = []
    for (gid, side), g in p.groupby(["gameid", "side"]):
        cs = sorted(g["champion"].dropna().unique())
        if len(cs) != 5: continue
        units.append((cs, int(g["result"].iloc[0]),
                      float(g["glen"].iloc[0]), g["date"].iloc[0], gid))
    units.sort(key=lambda u: u[3])
    print(f"\n    {len(units)} 个「队伍-比赛」单元")

    def split_stats(sel):
        return collect([(u[0], u[1], u[2]) for u in sel])

    for mode in ("随机", "时间"):
        print(f"\n{'=' * 78}")
        print(f"  {mode}切半")
        print(f"{'=' * 78}")
        if mode == "随机":
            rng = np.random.default_rng(0)
            idx = rng.permutation(len(units))
            print("  排除时间因素, 测「这个量能不能稳定估出来」")
        else:
            idx = np.arange(len(units))
            print("  前半 vs 后半, 会混入版本更迭与 meta 漂移")
        h = len(units) // 2
        A = [units[i] for i in idx[:h]]
        B = [units[i] for i in idx[h:]]

        cwA, cnA, pwA, pnA, scA = split_stats(A)
        cwB, cnB, pwB, pnB, scB = split_stats(B)

        print(f"\n  {'量':<24}{'重叠':<8}{'Pearson r':<24}{'Spearman':<10}")
        print(f"  {'─' * 74}")
        r_champ = corr(champ_wr(cwA, cnA), champ_wr(cwB, cnB), "单英雄胜率 (参照)")
        r_scale = corr(scaling(scA), scaling(scB), "强势期指数")
        r_syn   = corr(pair_syn(cwA, cnA, pwA, pnA),
                       pair_syn(cwB, cnB, pwB, pnB), "两英雄协同分")

        if r_champ is not None and r_syn is not None:
            print(f"\n  解读")
            print(f"  {'─' * 74}")
            print(f"  单英雄胜率是已知真实的属性, 它的 r={r_champ:+.3f} 就是")
            print(f"  「一个真实的英雄属性在这个数据量下能达到的信度上限」。")
            if r_syn < 0.15:
                print(f"\n  协同分 r={r_syn:+.3f} —— 换一半数据重测基本得到无关的值。")
                print(f"  这不是「效应小」, 是「测不出来」。加进模型不可能有稳定贡献。")
            elif r_syn < r_champ * 0.5:
                print(f"\n  协同分 r={r_syn:+.3f}, 明显低于参照系, 信噪比不足。")
            else:
                print(f"\n  协同分 r={r_syn:+.3f} 接近参照系, 说明它确实测到了东西 ——")
                print(f"  那么过闸失败的原因就不是「不存在」, 而是它的信息")
                print(f"  已经被其他特征覆盖了。")

    print(f"\n{'=' * 78}")
    print(f"  为什么这个检验比 t 检验更根本")
    print(f"{'=' * 78}")
    print("""
  t 检验问: 「加进模型有没有提升?」
  信度问:   「这个量本身可不可重复测量?」

  一个信度接近 0 的量, 无论怎么加进模型都不会有稳定贡献 —— 它在
  训练集上学到的模式在验证集上不成立。所以信度低时, t 检验的结果
  是可预测的, 不需要真的去跑。

  反过来, 信度高但过闸失败, 说明这个量是真的, 只是信息冗余 ——
  那是完全不同的结论, 对应完全不同的下一步。
    """)


if __name__ == "__main__":
    main()
