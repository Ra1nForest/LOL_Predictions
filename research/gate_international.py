"""
国际赛: 现役模型拿到跨赛区的比赛上, 给出的数还能不能信
========================================================
用户 2026-10-03 要把国际赛 (全球总决赛 / MSI / First Stand / 电竞世界杯, 以及正在打的德玛西亚杯
全球邀请赛 DCGI) 加进赛程和看板。在决定看板上显示什么之前, 先量清楚**现在线上的模型文件**
在历史国际赛上的表现 —— 不重训, 不改任何线上代码。

为什么不能想当然地照搬
----------------------
Stage 1/2 的特征全是"赛区内的相对量": 滚动胜率、场均经济、10/15 分钟经济差…… 都是一支队伍
**在自己赛区里**打出来的。LCS 打 7 成胜率的队和 LCK 打 5 成的队放在一起, 模型只看得见 0.7 对
0.5, 看不见两个赛区本身的差距。训练集里没有一场跨赛区的比赛 (feature_store.LEAGUES 只有四大,
国际赛被过滤掉), 所以这件事模型从来没机会学。预期是: 同赛区内战 (LPL 打 LPL) 大致照常,
跨赛区可能不比抛硬币强, 甚至自信地押错 —— 这里要的是实测, 不是预期。

Stage 4 不一样: 它在全部赛区上训练 (ingame_model.TARGET = None), 国际赛本来就在训练集里,
局内的规律 (经济差怎么变成胜率) 也不该随赛区变。要确认的是它在国际赛上没有变差。

做法
----
Stage 1/2 —— 和线上完全同一条路:
  · 特征库 = FeatureStore.from_frame(四大赛区), 和 api.py 启动时一样; 每局用
    make_row(..., as_of=比赛时间) 构造, 只用这局**之前**的四大赛区数据 (队伍滚动统计、英雄胜率、
    选手熟练度都按 as_of 截断), 这局本身和之后的比赛都看不到。
  · 模型 = api.Stage (artifacts/model_*.json + calib_*.json 的 Platt), 特征顺序从 calib 文件读。
  · playoffs 一律 0 —— 看板 (api.esports_board) 调 _predict_core 时就是传 False。
  · 可评估 = 两队在这局之前都有四大赛区历史, 且前五个 diff_avg_* 不缺 (和训练矩阵的 dropna 同一个
    口径), 且 OE 给齐了十个人的 BP。
  · league 参数两种都量 (线上最后定为 home, 且只用在同母赛区的对阵上, 见文末"结论"):
        home  传蓝方的母赛区 —— is_* 标志在训练分布内, 英雄胜率查母赛区的表
        intl  传赛事代码 (WLDs 等) —— is_* 全 0 (训练里从没出现过), 英雄胜率查不到 (全缺)
  · 母赛区 = 该队在这局之前最后一场四大赛区比赛的赛区; 两队相同 = 同赛区, 否则跨赛区。
  · 方向: 蓝方 = participantid 较小那一行 (和训练矩阵一致), 标签 = 蓝方胜负; 另核对 side 列。

Stage 4 —— 局内 live 变体 (看板实际用的那个):
  · 快照用 ingame_model.build 在全部赛区上重建, 用 split_by_game 复现线上的按日期切分;
    只评估测试段 (最后 20% 的比赛) 里的国际赛 —— 那些局没进过训练也没进过校准。
  · 三种赛前特征: 训练口径 (build 自带的、含次级联赛和此前国际赛的滚动统计)、线上口径
    (_pre_context: 四大赛区特征库按比赛时间截断; 有一方不是四大就整组缺失, 和线上一样)、
    整组置空 (线上遇到非四大队伍时的样子, 也是"跨赛区不信赛前统计"的候选做法)。
  · 按分钟切片 (T = 10/15/20/25) 报 Brier, 和同一测试段里的四大赛区内战对照。
  · 看板在 3-15 分钟把 BP 后概率渐变混进局内 (api._blend_prior)。T=10 时 BP 后还占 5/12 权重,
    跨赛区如果 Stage 2 是坏的, 这一步会把坏处带进局内曲线 —— 单独量一下。

判读
----
· 常数基线两个: 训练段的蓝方胜率 c (≈0.53) 和 0.5。模型连常数都比不过 = 这个数在误导人。
· 配对 t: 同一局上"常数的损失 − 模型的损失", 正 = 模型更好。逐局 t 会高估证据 —— 同一个系列赛
  的几局共享同一对队伍、几乎同一组赛前特征, 不是独立样本; 所以另报按系列赛聚合的 t,
  以及按赛事 (年 × 赛事) 分块的符号计数。|t| > 2.5 才算稳定 (harness.T_ACCEPT)。
· 校准斜率: 对 logit(p) 做 logistic 回归。1 = 刚好; < 1 = 过度自信 (该说 55% 的地方说了 65%);
  0 附近 = 没有信息; 负 = 方向反了。
· 样本小。国际赛每年两三百局, 跨赛区、两队都是四大的更少; 结论按这个量级读。
· 2022-2024 的国际赛落在 Stage 1/2 训练期内。模型没见过这些局 (国际赛不进训练集), 但拟合时
  见过同期之后的四大内战, 所以另报"测试期内"一行 —— 那段日期模型的训练和校准都没碰过。

结论 (2026-10-03, 数据截至 2026-10-02, 现役 artifacts)
------------------------------------------------------
国际赛可评估 837 局 (两队之前都有四大赛区历史、BP 齐全): 同母赛区 314、跨赛区 523, 跨赛区里
落在 Stage 1/2 测试期内 (>= 2026-01-17) 的 102 局。下面 Δ = 常数 c 的 Brier − 模型的 Brier,
正 = 模型更好; league 传母赛区 (传赛事代码结论相同, 数字差在千分位)。

· **跨赛区: Stage 1/2 没有预测力。** Stage 1 准确率 50.1%、Stage 2 48.8%; 校准斜率 −0.16±0.15 /
  −0.02±0.15 (国内参照 +0.77); 而自信程度和国内一样 (|p−.5| 0.121 / 0.122, 国内 0.124 / 0.134)。
  被看好的一方平均被给 62.2%, 实际只赢 48.8% (国内 63.4% 对 61.1%)。Δ = −0.0240 (t −4.04, 按系列赛
  聚合 −3.50) / −0.0203 (t −3.35, 系列赛 −1.96)。事后常识"强赛区一方赢" (LCK=LPL > LEC > LCS)
  在同一批局上对 71.7% (n = 325, 不含 LCK v LPL), Stage 2 只有 50.2% —— 它看不见赛区差距。
  审查更正: 这是"没有价值", 不是"处处比常数差"。按时期拆开, 显著差于常数的只有 2022-24
  (n = 301, Stage 1 Δ −0.0408 t −4.82, Stage 2 Δ −0.0266 t −3.15) —— 那几年落在训练期内, 赛区差距
  也更大; 2025 (n = 120) 和 2026 (n = 102, 完全在训练和校准之外) 约等于常数 (Stage 1 t −0.17 /
  −0.07, Stage 2 t −1.73 / −0.22)。不管哪种读法, 都没有理由把这个数显示出来。
· **同母赛区: 和国内差不多。** Stage 1 56.7% / Stage 2 57.6% (国内 58.6% / 61.1%), 校准斜率
  +0.46±0.21 / +0.57±0.19; 比常数好但 n = 314 下不显著 (Stage 2 Δ +0.0048, t +0.60, 系列赛 +1.50)。
  审查更正: 这批局大多是国际赛名义下的赛区内预选和内战 (2026 EWC 一届就 108 局), 不是"国际赛
  的一般情形"。
· **Stage 4 (live 变体) 在国际赛上照常。** 测试段 2026 年国际赛 360 局, 四个切片都显著好于常数
  (T=10/15/20/25: t +4.21 / +5.81 / +7.98 / +9.59), 和同期国内内战的 Brier 差 −0.003 ~ −0.009 (z 在
  ±0.7 以内)。跨赛区 102 局: t +2.81 / +2.72 / +2.66 / +3.56, 全部切片合起来 t = +3.43, 比国内差
  +0.0053 (z +0.27)。审查更正: n = 102 时比 ~0.055 Brier 小的变差测不出来 (R4 的 MDE), "没变差"
  只能读到这个精度。
· **局内模型的赛前特征对跨赛区无所谓。** 线上口径 → 整组置空: ΔBrier +0.0007, t = +0.15 (102 局);
  渐变 (api._blend_prior) 在 T=10 把跨赛区拉差 +0.0057 (t +0.85, 不显著, 方向不利)。

线上的做法 (api.pregame_league / esports_board, 浏览器版同一套):
  四大赛区不变; 国际赛两队同母赛区 → 赛前和 BP 后照常给, 按母赛区算 (league = 母赛区), 局内模型的
  league 仍是赛事本身 (is_* 全 0, 和训练时国际赛的编码一样); 跨赛区或有队不在四大赛区的数据里 →
  不给赛前和 BP 后, 局内模型不带赛前特征、不渐变, 第 3 分钟起直接用局内模型。

审查附加检查 (原 research/gate_international_review.py, 结论并进这一段后该脚本已删除; 编号沿用
那个脚本, 它里面没有 R6): R1 线上 make_row(as_of) 路径和训练矩阵在国内测试段 1769 局上给出同一个
概率 (Stage 1 max|Δp| 8.9e-8, Stage 2 1.2e-7, 只是 float32 噪声) —— 本脚本测的就是线上那条路。
R2 泄漏探针: 每年取几局跨赛区国际赛, 用截断到比赛时刻之前的特征库重算, 15 局 max|Δp| = 0 ——
as_of 截断没有漏进未来。R3 Stage 4 被评估的 360 局国际赛与它的训练 + 校准段重叠 0 局 (后者最晚
2025-11-22, 前者最早 2026-03-16)。R4 按所有赛区 (含次级/外卡) 的母赛区给 Stage 4 测试段的国际赛
分类: 同母赛区 184 局 (预选/内战, 其中 2026 EWC 169) t +5.57、跨赛区四大v四大 102 局 t +3.43
(MDE ≈ 0.055)、四大v外卡 70 局 t +6.08、外卡v外卡 4 局; 没有一类比国内差。R5 跨赛区四大v四大
把局内模型的 is_* 从全 0 改成蓝方母赛区: Brier 0.1781 → 0.1786 (t −0.37) —— 没有理由偏离训练时
的全 0 编码。R7 跨赛区 Stage 1/2 按时期拆开 (见上面"跨赛区"那条的审查更正); 同一批局上"强赛区
一方赢"在 2022-24 / 2025 / 2026 分别对 76.0% / 68.9% / 63.9%, 而 Stage 2 押强赛区一方的比例只有
40.2% / 58.1% / 44.4%。

    python research/gate_international.py       (从项目根目录跑; 约 10 分钟, 不联网)
"""
from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

import api                                                     # noqa: E402  Stage, DATA, _blend_prior
import ingame_model as IM                                      # noqa: E402
from feature_store import (FeatureStore, LEAGUES, ROLES,       # noqa: E402
                           build_training_matrix)
from harness import T_ACCEPT, paired_t                         # noqa: E402
from ingame_service import IngameModel                         # noqa: E402
from multiyear_gate import data_path                           # noqa: E402

YEARS = (2022, 2023, 2024, 2025, 2026)
FILES = [data_path(y) for y in YEARS]

# OE 的赛事代码 (2022-2026 五年 CSV 里实际出现的)。KeSPA 杯 / ASI / AC 这类以单一赛区为主、
# 偶有外卡的杯赛不算 —— 用户点名的是德玛西亚杯和 S 赛这一类。
INTL = {"WLDs": "全球总决赛", "MSI": "季中冠军赛", "FST": "First Stand", "EWC": "电竞世界杯"}
# 德玛西亚杯: 2022/2023/2025 都是 LPL + LDL 的国内杯赛, 在赛季结束、转会之后打 (12 月);
# 2026 的全球邀请赛 (DCGI) 正在打, 还没进 OE。单独报, 不混进国际赛的总数。
DCUP = "DCup"
EVENTS = list(INTL) + [DCUP]

TRAIN_FRAC, CAL_FRAC = 0.60, 0.20          # 和 train.py / ingame_model.split_by_game 一致
# 只用来做"强赛区方胜"这条参照规则 —— 这是事后的常识, 不是模型输入, 也不是可以拿去用的预测。
TIER = {"LCK": 1, "LPL": 1, "LEC": 2, "LCS": 3}
EPS = 1e-12


# ══════════════════════════════════════════════════════════
#  度量
# ══════════════════════════════════════════════════════════

def brier_v(p, y):
    return (np.asarray(p, float) - y) ** 2


def ll_v(p, y):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def calib_slope(p, y, iters=50):
    """y ~ a + b·logit(p) 的 logistic 回归 (牛顿法)。返回 (a, b, b 的标准误)。"""
    x = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6))
    X = np.column_stack([np.ones_like(x), x])
    w = np.zeros(2)
    for _ in range(iters):
        mu = 1 / (1 + np.exp(-(X @ w)))
        H = X.T @ (X * (mu * (1 - mu))[:, None])
        g = X.T @ (y - mu)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return np.nan, np.nan, np.nan
        w = w + step
        if np.abs(step).max() < 1e-10:
            break
    mu = 1 / (1 + np.exp(-(X @ w)))
    H = X.T @ (X * (mu * (1 - mu))[:, None])
    try:
        se = float(np.sqrt(np.linalg.inv(H)[1, 1]))
    except np.linalg.LinAlgError:
        se = np.nan
    return float(w[0]), float(w[1]), se


def group_t(score_a, score_b, groups):
    """先按组 (系列赛 / 赛事) 求平均, 再在组之间做配对 t。返回 (Δ, t, 组数, Δ>0 的组数)。"""
    df = pd.DataFrame({"a": score_a, "b": score_b, "g": groups})
    g = df.groupby("g")[["a", "b"]].mean()
    if len(g) < 2:
        return np.nan, np.nan, len(g), int((g["b"] - g["a"] > 0).sum())
    d, _sd, t, _ = paired_t(g["a"].values, g["b"].values)
    return d, t, len(g), int((g["b"] - g["a"] > 0).sum())


def evaluate(p, y, c, series=None, events=None):
    """一组预测的全部指标。所有 Δ 都是"常数的损失 − 模型的损失", 正 = 模型更好。"""
    p, y = np.asarray(p, float), np.asarray(y, float)
    out = {"n": len(y)}
    if len(y) == 0:
        return out
    bm, lm = brier_v(p, y), ll_v(p, y)
    out.update(acc=float(((p > 0.5) == (y == 1)).mean()),
               brier=float(bm.mean()), logloss=float(lm.mean()),
               ece=float(IM.ece(p, y.astype(int))) if len(y) else np.nan,
               conf=float(np.abs(p - 0.5).mean()),
               blue_rate=float(y.mean()))
    for tag, const in (("c", c), ("h", 0.5)):
        bc, lc = brier_v(np.full_like(p, const), y), ll_v(np.full_like(p, const), y)
        out[f"brier_{tag}"] = float(bc.mean())
        out[f"ll_{tag}"] = float(lc.mean())
        for nm, mod, con in (("b", bm, bc), ("l", lm, lc)):
            if len(y) >= 3:
                d, _sd, t, _ = paired_t(-con, -mod)
            else:
                d, t = float((con - mod).mean()), np.nan
            out[f"d{nm}_{tag}"], out[f"t{nm}_{tag}"] = float(d), float(t)
            if series is not None and len(y) >= 3:
                _d, ts, ns, _k = group_t(-con, -mod, series)
                out[f"ts{nm}_{tag}"], out["n_series"] = float(ts), ns
            if events is not None and len(y) >= 3:
                _d, te, ne, kpos = group_t(-con, -mod, events)
                out[f"te{nm}_{tag}"], out["n_events"], out[f"kpos{nm}_{tag}"] = float(te), ne, kpos
    if len(y) >= 20:
        out["cal_a"], out["cal_b"], out["cal_b_se"] = calib_slope(p, y)
    return out


def fmt_t(x):
    return "   —" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+5.2f}"


def print_table(title, rows):
    """rows: [(名字, evaluate() 的结果)]。"""
    print(f"\n  {title}")
    print(f"  {'子集':<22}{'n':>5}{'准确率':>8}{'Brier':>8}{'对数损失':>9}{'ECE':>7}"
          f"{'|p-.5|':>8}{'校准斜率':>14}")
    print(f"  {'─' * 81}")
    for nm, r in rows:
        if r["n"] == 0:
            print(f"  {nm:<22}{0:>5}   (无可评估的局)")
            continue
        cs = (f"{r['cal_b']:+.2f}±{r['cal_b_se']:.2f}" if "cal_b" in r and not np.isnan(r["cal_b"])
              else "—")
        print(f"  {nm:<22}{r['n']:>5}{r['acc']:>8.1%}{r['brier']:>8.4f}{r['logloss']:>9.4f}"
              f"{r['ece']:>7.3f}{r['conf']:>8.3f}{cs:>14}")
    print(f"\n  {'子集':<22}{'常数c Brier':>11}{'ΔBrier vs c':>12}{'t逐局':>7}{'t系列':>7}{'t赛事':>7}{'赛事+':>6}"
          f"{'ΔBrier vs .5':>13}{'t逐局':>7}{'Δ对数 vs c':>11}{'t':>7}{'Δ对数 vs .5':>12}{'t':>7}")
    print(f"  {'─' * 125}")
    for nm, r in rows:
        if r["n"] == 0:
            continue
        kp = (f"{r['kposb_c']}/{r['n_events']}" if "kposb_c" in r else "—")
        print(f"  {nm:<22}{r['brier_c']:>11.4f}{r['db_c']:>+12.4f}{fmt_t(r['tb_c']):>7}"
              f"{fmt_t(r.get('tsb_c')):>7}{fmt_t(r.get('teb_c')):>7}{kp:>6}"
              f"{r['db_h']:>+13.4f}{fmt_t(r['tb_h']):>7}"
              f"{r['dl_c']:>+11.4f}{fmt_t(r['tl_c']):>7}{r['dl_h']:>+12.4f}{fmt_t(r['tl_h']):>7}")


def reliability(p, y, title):
    """按"被看好的一方"折叠: q = max(p, 1-p), 看被看好的一方实际赢了多少。"""
    p, y = np.asarray(p, float), np.asarray(y, float)
    q = np.maximum(p, 1 - p)
    win = np.where(p >= 0.5, y, 1 - y)
    edges = [0.5, 0.55, 0.60, 0.65, 0.70, 1.0001]
    print(f"\n  {title}")
    print(f"  {'被看好方的概率':<16}{'局数':>6}{'平均预测':>10}{'实际胜率':>10}{'差 (实际−预测)':>16}{'±95%':>8}")
    print(f"  {'─' * 66}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (q >= lo) & (q < hi)
        if m.sum() == 0:
            continue
        k = int(m.sum())
        obs = win[m].mean()
        se = math.sqrt(max(obs * (1 - obs), 1e-9) / k)
        rng = f"[{lo:.2f}, {min(hi, 1):.2f})"
        print(f"  {rng:<16}{k:>6}{q[m].mean():>10.3f}{obs:>10.3f}{obs - q[m].mean():>+16.3f}{1.96 * se:>8.3f}")
    k = len(q)
    if k:
        print(f"  {'合计':<16}{k:>6}{q.mean():>10.3f}{win.mean():>10.3f}{win.mean() - q.mean():>+16.3f}"
              f"{1.96 * math.sqrt(max(win.mean() * (1 - win.mean()), 1e-9) / k):>8.3f}")


# ══════════════════════════════════════════════════════════
#  Stage 1/2: 国际赛逐局构造特征行
# ══════════════════════════════════════════════════════════

def home_league(store, team, before):
    t = store.team_history.get(team)
    if t is None:
        return None, None
    t = t[t["date"] < before]
    if t.empty:
        return None, None
    return t["league"].iloc[-1], t["date"].iloc[-1]


def event_rows(raw, store, s1, s2):
    """国际赛 (+ 德玛西亚杯) 的每一局: 能评估的给出两种 league 口径下的 Stage 1/2 概率。"""
    teams = raw[(raw["position"] == "team") & raw["league"].isin(EVENTS)]
    players = raw[raw["position"].isin(ROLES) & raw["league"].isin(EVENTS)]
    pbg = {g: d for g, d in players.groupby("gameid")}
    core = [f"diff_{c}" for c in store.roll_cols if c.startswith("avg_")][:5]

    out, skip = [], {"不是两行": 0, "蓝红对不上": 0, "队伍无四大历史": 0, "滚动统计不足": 0, "BP 不全": 0}
    for gid, grp in teams.groupby("gameid"):
        if len(grp) != 2:
            skip["不是两行"] += 1
            continue
        r = grp.sort_values("participantid")
        bT, rT = r.iloc[0], r.iloc[1]
        if str(bT.get("side")) != "Blue" or str(rT.get("side")) != "Red":
            skip["蓝红对不上"] += 1
            continue
        bt, rt, gd, ev = bT["teamname"], rT["teamname"], bT["date"], bT["league"]
        rec = {"gameid": gid, "date": gd, "event": ev, "year": int(gd.year),
               "blue": bt, "red": rt, "y": int(bT["result"]),
               "series": f"{ev}|{gd.date()}|{'|'.join(sorted([str(bt), str(rt)]))}"}
        hb, lb = home_league(store, bt, gd)
        hr, lr = home_league(store, rt, gd)
        rec.update(home_b=hb, home_r=hr)
        if hb is None or hr is None:
            skip["队伍无四大历史"] += 1
            rec["ok"] = False
            out.append(rec)
            continue
        rec["gap_days"] = max((gd - lb).days, (gd - lr).days)
        gp = pbg.get(gid)
        draft = {"blue": {}, "red": {}}
        if gp is not None:
            for x in gp.itertuples(index=False):
                side = "blue" if x.side == "Blue" else "red"
                draft[side][x.position] = (x.playername, x.champion)
        if sum(len(v) for v in draft.values()) != 10:
            skip["BP 不全"] += 1
            rec["ok"] = False
            out.append(rec)
            continue
        try:
            for tag, lg in (("home", hb), ("intl", ev)):
                row, _w = store.make_row(bt, rt, lg, draft=draft, playoffs=0, as_of=gd)
                if tag == "home" and any(np.isnan(float(row.get(c, np.nan))) for c in core):
                    raise KeyError("core")
                rec[f"p1_{tag}"] = s1.predict(row)[1]
                rec[f"p2_{tag}"] = s2.predict(row)[1]
        except KeyError:
            skip["滚动统计不足"] += 1
            rec["ok"] = False
            out.append(rec)
            continue
        except ValueError:
            skip["队伍无四大历史"] += 1
            rec["ok"] = False
            out.append(rec)
            continue
        rec["ok"] = True
        rec["same_region"] = hb == hr
        out.append(rec)
    return pd.DataFrame(out), skip


# ══════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════

def main():
    assert FILES == api.DATA, "训练/线上用的五年数据和 data_path() 不一致"
    print("=" * 100)
    print("  国际赛: 现役模型 (artifacts/) 在历史国际赛上的表现")
    print("=" * 100)

    print("\n[1/5] 加载五年 CSV (全部赛区)")
    raw = pd.concat([pd.read_csv(f, low_memory=False) for f in FILES if Path(f).exists()],
                    ignore_index=True)
    raw["league"] = raw["league"].replace({"LTA N": "LCS", "LTA S": "CBLOL"})
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.sort_values("date").reset_index(drop=True)
    t_ev = raw[(raw["position"] == "team") & raw["league"].isin(EVENTS)]
    for ev in EVENTS:
        s = t_ev[t_ev["league"] == ev]
        per = "  ".join(f"{y}:{g['gameid'].nunique()}" for y, g in s.groupby(s["date"].dt.year))
        print(f"    {ev:<5} {INTL.get(ev, '德玛西亚杯'):<8} {s['gameid'].nunique():>4} 局   {per}")

    # 特征库: 和 FeatureStore.from_csv 同一条路 (四大赛区, 按日期排序), 只是不重复读 CSV
    majors = raw[raw["league"].isin(LEAGUES)].copy().sort_values("date").reset_index(drop=True)
    store = FeatureStore.from_frame(majors, verbose=False)
    s1, s2 = api.Stage("pre_draft"), api.Stage("post_draft")

    # ── 国内参照: train.py 的测试段 ─────────────────────────
    print("\n[2/5] 国内参照 —— train.py 按日期留出的最后 20% (四大赛区内战)")
    mdf = build_training_matrix(FILES, verbose=False)
    n = len(mdf)
    i_tr, i_cal = int(n * TRAIN_FRAC), int(n * (TRAIN_FRAC + CAL_FRAC))
    c12 = float(mdf["y"].iloc[:i_tr].astype(int).mean())
    test = mdf.iloc[i_cal:].reset_index(drop=True)
    test_start = test["date"].min()
    yd = test["y"].astype(int).values
    dom = {}
    for nm, st in (("p1", s1), ("p2", s2)):
        X = test[st.features].astype(float).fillna(-999.0)
        rawp = st.model.predict_proba(X)[:, 1]
        dom[nm] = 1 / (1 + np.exp(-(st.a * rawp + st.b)))
        acc = float(((dom[nm] > 0.5) == (yd == 1)).mean())
        br = float(brier_v(dom[nm], yd).mean())
        print(f"    {st.name:<10} 复现: 准确率 {acc:.4f} (calib 文件 {st.metrics['accuracy']:.4f})   "
              f"Brier {br:.5f} ({st.metrics['brier']:.5f})")
    print(f"    矩阵 {n} 局, 测试段 {len(test)} 局, 从 {test_start.date()} 起;  "
          f"训练段蓝方胜率 c = {c12:.4f}")
    dom_series = (test["league"].astype(str) + "|" + test["date"].dt.date.astype(str) + "|"
                  + test[["blue_team", "red_team"]].astype(str).apply(lambda r: "|".join(sorted(r)), axis=1)).values

    # ── 国际赛 ──────────────────────────────────────────────
    print("\n[3/5] 国际赛逐局构造特征 (make_row, as_of = 比赛时间)")
    ev, skip = event_rows(raw, store, s1, s2)
    ok = ev[ev["ok"]].copy()
    ok["same_region"] = ok["same_region"].astype(bool)     # 整列混过 NaN 时是 object, ~ 会按整数取反
    print(f"    共 {len(ev)} 局, 可评估 {len(ok)} 局;  跳过: "
          + ", ".join(f"{k} {v}" for k, v in skip.items() if v))
    for e in EVENTS:
        a, b = (ev["event"] == e).sum(), (ok["event"] == e).sum()
        print(f"      {e:<5} {b:>4}/{a:<4}", end="")
    print()
    if len(ok):
        print(f"    两队离最后一场四大比赛的间隔 (取较大者): 中位 {ok['gap_days'].median():.0f} 天, "
              f"> 60 天的 {int((ok['gap_days'] > 60).sum())} 局")

    intl = ok[ok["event"].isin(INTL)]
    dcup = ok[ok["event"] == DCUP]
    blocks = {
        "国际赛 全部": intl,
        "  同赛区": intl[intl["same_region"]],
        "  跨赛区": intl[~intl["same_region"]],
        f"  跨赛区·测试期内": intl[(~intl["same_region"]) & (intl["date"] >= test_start)],
        "德玛西亚杯 (国内, 12月)": dcup,
    }
    by_ev = intl.groupby("event").size().to_dict()
    print(f"    国际赛可评估 {len(intl)} 局 ({', '.join(f'{k} {v}' for k, v in by_ev.items())}):  "
          f"同赛区 {int(intl['same_region'].sum())}, 跨赛区 {int((~intl['same_region']).sum())}, "
          f"跨赛区里测试期内 (≥ {test_start.date()}) "
          f"{int(((~intl['same_region']) & (intl['date'] >= test_start)).sum())}")
    print("    同赛区按赛区: " + ", ".join(f"{k} {v}" for k, v in
                                       intl[intl["same_region"]].groupby("home_b").size().items()))

    print("\n[4/5] Stage 1 / Stage 2 指标  (Δ = 常数的损失 − 模型的损失, 正 = 模型更好;"
          f" t 判据 |t| > {T_ACCEPT})")
    for stage, key in (("Stage 1 赛前", "p1"), ("Stage 2 BP 后", "p2")):
        for var, vdesc in (("home", "league = 蓝方母赛区"), ("intl", "league = 赛事代码 (is_* 全 0)")):
            rows = [("国内测试段 (参照)", evaluate(dom[key], yd, c12, series=dom_series))]
            for nm, sub in blocks.items():
                ev_lab = (sub["year"].astype(str) + " " + sub["event"]).values
                rows.append((nm, evaluate(sub[f"{key}_{var}"].values, sub["y"].values, c12,
                                          series=sub["series"].values, events=ev_lab)))
            print_table(f"── {stage}   {vdesc} ──", rows)

    # 跨赛区: 可靠性表 + 按赛区对 + 每个赛事块
    cross = intl[~intl["same_region"]].copy()
    for key, stage in (("p1", "Stage 1"), ("p2", "Stage 2")):
        reliability(cross[f"{key}_home"].values, cross["y"].values,
                    f"可靠性 —— 跨赛区, {stage} (league = 蓝方母赛区), 按被看好的一方折叠")
    reliability(dom["p2"], yd, "可靠性 —— 国内测试段, Stage 2 (参照)")

    if len(cross):
        print(f"\n  跨赛区按赛区对 (Stage 2, league = 蓝方母赛区)")
        print(f"  {'赛区对':<12}{'局数':>5}{'准确率':>8}{'Brier':>8}{'常数c':>8}{'模型押强赛区':>12}"
              f"{'强赛区实际胜':>12}")
        print(f"  {'─' * 65}")
        cross["pair"] = cross.apply(lambda r: " v ".join(sorted([r["home_b"], r["home_r"]],
                                                              key=lambda x: (TIER[x], x))), axis=1)
        tb = cross["home_b"].map(TIER)
        tr_ = cross["home_r"].map(TIER)
        cross["strong_is_blue"] = np.where(tb < tr_, 1, np.where(tb > tr_, 0, -1))
        for pair, g in cross.groupby("pair"):
            y, p = g["y"].values, g["p2_home"].values
            m = g["strong_is_blue"].values >= 0
            if m.any():
                fav = ((p[m] > 0.5).astype(int) == g["strong_is_blue"].values[m]).mean()
                won = (y[m] == g["strong_is_blue"].values[m]).mean()
                s_fav, s_won = f"{fav:.0%}", f"{won:.0%}"
            else:
                s_fav = s_won = "同级"
            print(f"  {pair:<12}{len(g):>5}{((p > 0.5) == (y == 1)).mean():>8.1%}"
                  f"{brier_v(p, y).mean():>8.4f}{brier_v(np.full_like(p, c12), y).mean():>8.4f}"
                  f"{s_fav:>12}{s_won:>12}")
        m = cross["strong_is_blue"].values >= 0
        if m.any():
            sb = cross["strong_is_blue"].values[m]
            print(f"  参照规则「强赛区一方赢」(LCK=LPL > LEC > LCS, 事后常识, 不是可用的预测): "
                  f"{(cross['y'].values[m] == sb).mean():.1%}  (n = {int(m.sum())}, 不含 LCK v LPL)")
            print(f"  Stage 2 在这些局上的准确率: "
                  f"{((cross['p2_home'].values[m] > 0.5).astype(int) == cross['y'].values[m]).mean():.1%}")

    # 同赛区里混着资格赛 (2026 的 EWC 四到五月是各赛区内部的预选, 2026 的 WLDs 是 LPL 的出线赛),
    # 那些其实就是内战; 按赛事块分开看
    for nm, sub in (("跨赛区", cross), ("同赛区", intl[intl["same_region"]])):
        if not len(sub):
            continue
        print(f"\n  {nm}按赛事块 (Stage 2, league = 蓝方母赛区; ΔBrier = 常数c − 模型, 正 = 模型更好)")
        print(f"  {'赛事':<12}{'局数':>5}{'准确率':>8}{'Brier':>8}{'ΔBrier vs c':>13}{'ΔBrier vs .5':>14}{'|p-.5|':>8}")
        print(f"  {'─' * 68}")
        for (yr, e), g in sub.groupby(["year", "event"]):
            y, p = g["y"].values, g["p2_home"].values
            bm = brier_v(p, y).mean()
            print(f"  {f'{yr} {e}':<12}{len(g):>5}{((p > 0.5) == (y == 1)).mean():>8.1%}{bm:>8.4f}"
                  f"{brier_v(np.full_like(p, c12), y).mean() - bm:>+13.4f}{0.25 - bm:>+14.4f}"
                  f"{np.abs(p - 0.5).mean():>8.3f}")

    # ── Stage 4 ────────────────────────────────────────────
    print("\n[5/5] Stage 4 局内 (live 变体) —— 只看按日期留出的测试段")
    mdf4, _sc = IM.build(raw, verbose=False)
    live = IngameModel("_live")
    end = mdf4["date"].max()
    if str(end.date()) != live.trained_through:
        # 数据比模型新: 按模型训练时的数据重切, 新增的局自然落在测试段之后, 同样没见过
        print(f"    ⚠ 数据截至 {end.date()}, 模型训练到 {live.trained_through}; 按后者复现切分")
        cut = pd.Timestamp(live.trained_through) + pd.Timedelta(days=1)
        _tr, _cal, te_old = IM.split_by_game(mdf4[mdf4["date"] < cut])
        t0 = te_old["date"].min()
        te = mdf4[mdf4["gameid"].isin(set(te_old["gameid"])) | (mdf4["date"] >= cut)].copy()
        c4 = float(_tr["y"].astype(int).mean())
    else:
        _tr, _cal, te = IM.split_by_game(mdf4)
        te = te.copy()
        t0 = te["date"].min()
        c4 = float(_tr["y"].astype(int).mean())
    X = te[live.features].astype(float).fillna(-999.0)
    te["raw"] = live.model.predict_proba(X)[:, 1]
    te["p"] = [live.calibrate(float(r)) for r in te["raw"]]
    acc_all = float(((te["p"] > 0.5) == (te["y"] == 1)).mean())
    print(f"    测试段 {te['gameid'].nunique()} 局 / {len(te)} 快照, 从 {t0.date()} 起;  "
          f"复现: 准确率 {acc_all:.4f} (calib 文件 {live.metrics['accuracy']:.4f}), "
          f"Brier {brier_v(te['p'], te['y']).mean():.5f} ({live.metrics['brier']:.5f});  训练段蓝方胜率 {c4:.4f}")

    # 线上口径的赛前特征: _pre_context —— 四大特征库, as_of = 比赛时间; 任一方不是四大 -> 整组缺失
    pre_cols = [f for f in live.features if f.startswith("diff_pre_")]
    keep = te["league"].isin(list(INTL) + [DCUP] + LEAGUES)
    sub4 = te[keep].copy()
    first = sub4.drop_duplicates("gameid").set_index("gameid")
    serv = {}
    for gid, r in first.iterrows():
        try:
            row, _w = store.make_row(r["blue_team"], r["red_team"], r["league"], as_of=r["date"])
            pre = {"diff_pre_" + k[len("diff_avg_"):]: v for k, v in row.items() if k.startswith("diff_avg_")}
            pre["diff_pre_wr"] = row.get("diff_rolling_wr", np.nan)
        except ValueError:
            pre = {}
        serv[gid] = pre
    for c in pre_cols:
        sub4[f"srv_{c}"] = [serv.get(g, {}).get(c, np.nan) for g in sub4["gameid"]]
    Xs = sub4[live.features].astype(float).copy()
    for c in pre_cols:
        Xs[c] = sub4[f"srv_{c}"].astype(float).values
    sub4["p_srv"] = [live.calibrate(float(r)) for r in
                     live.model.predict_proba(Xs.fillna(-999.0))[:, 1]]
    sub4["has_srv_pre"] = [bool(serv.get(g)) for g in sub4["gameid"]]
    # 第三种: 赛前特征整组置空 —— 线上碰到非四大队伍 (GAM、RED 这类) 时就是这样;
    # 也是"跨赛区不信赛前统计"这个可选做法的样子
    Xn = sub4[live.features].astype(float).copy()
    for c in pre_cols:
        Xn[c] = np.nan
    sub4["p_nopre"] = [live.calibrate(float(r)) for r in
                       live.model.predict_proba(Xn.fillna(-999.0))[:, 1]]

    is_int = sub4["league"].isin(INTL)
    is_dom = sub4["league"].isin(LEAGUES)
    is_dc = sub4["league"] == DCUP
    gi = sub4[is_int].drop_duplicates("gameid")
    print(f"    测试段里的国际赛 {gi['gameid'].nunique()} 局: "
          + ", ".join(f"{y} {e} {len(g)}" for (y, e), g in
                      gi.groupby([gi['date'].dt.year, 'league'])))
    print(f"      其中两队都有四大历史 (线上口径能拿到赛前特征) 的 {int(gi['has_srv_pre'].sum())} 局;"
          f"  德玛西亚杯 {sub4[is_dc]['gameid'].nunique()} 局 (OE 常不给 LPL 系赛事的分钟数据)")

    # 国际赛里的同/跨赛区, 用 Stage 1/2 那边算好的母赛区
    reg = ok.set_index("gameid")["same_region"].to_dict()
    sub4["same_region"] = sub4["gameid"].map(reg)

    def s4_line(nm, d, col, extra=""):
        y = d["y"].values.astype(float)
        p = d[col].values
        if len(y) == 0:
            print(f"  {nm:<26}{0:>6}")
            return
        bm, bc = brier_v(p, y), brier_v(np.full_like(p, c4), y)
        # 同一局的几个切片不独立: t 按局聚合 ("全部" 那行一局有多个快照)
        _d, t, _n, _k = group_t(-bc, -bm, d["gameid"].values) if len(y) >= 3 else (0, np.nan, 0, 0)
        print(f"  {nm:<26}{len(y):>6}{((p > 0.5) == (y == 1)).mean():>8.1%}{bm.mean():>8.4f}"
              f"{bc.mean():>9.4f}{fmt_t(t):>8}{ece_or(p, y):>7}{np.abs(p - 0.5).mean():>8.3f}{extra}")

    def ece_or(p, y):
        return f"{IM.ece(p, y.astype(int)):.3f}" if len(y) >= 20 else "  —"

    def vs_dom(mask_i, mask_d, col):
        """国际子集 − 国内的 Brier 差和 z。同一局多个快照不独立: 先按局平均再比。"""
        di, dd = sub4[mask_i], sub4[mask_d]
        if di["gameid"].nunique() < 3:
            return ""
        bi = pd.Series(brier_v(di[col].values, di["y"].values.astype(float)), index=di.index)
        bd = pd.Series(brier_v(dd[col].values, dd["y"].values.astype(float)), index=dd.index)
        gi_ = bi.groupby(di["gameid"]).mean().values
        gd_ = bd.groupby(dd["gameid"]).mean().values
        diff = gi_.mean() - gd_.mean()
        se = math.sqrt(gi_.var(ddof=1) / len(gi_) + gd_.var(ddof=1) / len(gd_))
        return f"{diff:>+16.4f}{diff / se:>+7.2f}"

    cross4 = sub4["same_region"] == False                                 # noqa: E712
    for col, desc in (("p_srv", "线上口径 (四大特征库给赛前特征)"), ("p", "训练口径 (build 自带的赛前特征)"),
                      ("p_nopre", "赛前特征整组置空")):
        print(f"\n  Stage 4 live 变体, 按分钟切片  —— {desc}")
        print(f"  {'切片 / 子集':<26}{'快照':>6}{'准确率':>8}{'Brier':>8}{'常数c':>9}{'t vs c':>8}{'ECE':>7}{'|p-.5|':>8}"
              f"{'减国内 ΔBrier':>16}{'z':>7}")
        print(f"  {'─' * 103}")
        for T in IM.SLICES + ["全部"]:
            mT = (sub4["T"] == T) if T != "全部" else pd.Series(True, index=sub4.index)
            s4_line(f"T={T} 四大内战", sub4[mT & is_dom], col)
            s4_line(f"T={T} 国际赛", sub4[mT & is_int], col, vs_dom(mT & is_int, mT & is_dom, col))
            if (mT & is_int & cross4).any():
                s4_line(f"T={T} 国际赛·跨赛区", sub4[mT & is_int & cross4], col,
                        vs_dom(mT & is_int & cross4, mT & is_dom, col))

    # 跨赛区上, 三种赛前口径逐局配对 (每局对各切片的 Brier 取平均)
    print(f"\n  跨赛区国际赛上, 赛前特征口径之间的逐局配对 (Brier, 正 = 后者更好)")
    d = sub4[is_int & cross4]
    if d["gameid"].nunique() >= 3:
        y = d["y"].values.astype(float)
        per = {c: pd.Series(brier_v(d[c].values, y), index=d.index).groupby(d["gameid"]).mean()
               for c in ("p_srv", "p", "p_nopre")}
        for a, b in (("p_srv", "p_nopre"), ("p_srv", "p")):
            dlt, _sd, t, _ = paired_t(-per[a].values, -per[b].values)
            print(f"    {a} → {b}:  ΔBrier {dlt:+.4f}   t = {t:+.2f}   ({len(per[a])} 局)")

    # 看板 3-15 分钟渐变: T=10 时 BP 后概率还占 5/12
    print(f"\n  看板渐变 (api._blend_prior) 在 T=10 的影响: 线上口径的局内概率 vs 渐变后 (BP 后权重 "
          f"{1 - (10 - api.BLEND_FROM) / (api.BLEND_TO - api.BLEND_FROM):.3f})")
    print(f"  {'子集':<26}{'局数':>6}{'Brier 局内':>11}{'Brier 渐变后':>13}{'Δ (渐变−局内)':>15}{'t':>7}")
    print(f"  {'─' * 78}")
    p2_int = ok.set_index("gameid")["p2_home"].to_dict()
    p2_dom = dict(zip(test["gameid"], dom["p2"]))
    for nm, mask, src in (("四大内战 (参照)", is_dom, p2_dom), ("国际赛 全部", is_int, p2_int),
                          ("国际赛·跨赛区", is_int & (sub4["same_region"] == False), p2_int)):  # noqa: E712
        d = sub4[mask & (sub4["T"] == 10)]
        d = d[d["gameid"].isin(src.keys())]
        if len(d) < 3:
            print(f"  {nm:<26}{len(d):>6}   (样本不足)")
            continue
        y = d["y"].values.astype(float)
        p4 = d["p_srv"].values
        pb = np.array([api._blend_prior(src[g], q, 10.0)[0] for g, q in zip(d["gameid"], p4)])
        b4, bb = brier_v(p4, y), brier_v(pb, y)
        t = paired_t(-bb, -b4)[2]
        print(f"  {nm:<26}{len(y):>6}{b4.mean():>11.4f}{bb.mean():>13.4f}{bb.mean() - b4.mean():>+15.4f}{fmt_t(t):>7}")
    print("  (Δ 为正 = 渐变让 T=10 变差; t 为正 = 纯局内更好)")

    print(f"\n{'=' * 100}")


if __name__ == "__main__":
    main()
