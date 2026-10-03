"""
跨赛区国际赛: 赛前 / BP 后模型的冻结协议闸 (DEV 复现 + HOLDOUT 只评一次)
======================================================================
用户 2026-10-03 问: 能不能训一个国际赛的模型, 不然只要是跨赛区的比赛就没有预测。现在看板上跨赛区的
对阵, 以及有一队不在四大赛区的对阵 (GAM、RED Kalunga = OE 的 RED Canids ……), 都不给赛前和 BP 后的数。
原因是 research/gate_international.py 量过: 线上 Stage 1/2 在 523 局跨赛区国际赛上准确率约 50%、校准斜率
约 0。它的特征全是"在自己赛区里"打出来的相对量, 看不见赛区之间的差距。

这个脚本是对那个问题的判决, 一次跑完, 可以复现:
  1. 从五年 CSV 重建局级表 (research/xregion_data.py)。
  2. 实现 DEV 上选出的两个候选 (从候选脚本逐行移植) 和协议的三个基线。
  3. 复现 DEV, 和候选报告的数逐项对账 (给了 --dev-preds-dir 就再逐局对账)。对不上就停, HOLDOUT 一局都不评。
  4. 在 HOLDOUT 上评一次, 打印全部表格。
候选的设计、特征和超参全在 DEV 上定, 这里一个都不改。

冻结的协议 (看任何结果之前定下)
------------------------------
· 单位: OE 的国际赛局。范围是 WLDs / MSI / FST / EWC, 加上按 xregion_data 的规则找出的 ASI 2025、LTA 2025、
  KeSPA Cup 2025、AC 2026; DCup 是 LPL 的国内杯赛, 排除。
  - 跨赛区 = 两队母赛区不同。母赛区 = 这局之前最后一场常规联赛所在的赛区, 范围是全部 OE 赛区。
  - 对字面规则的两处改动 (杯赛也跳过; 队名口径过期 365 天时按阵容认队) 见 xregion_data 的文档。
  - 同母赛区的国际赛局单独报, 不是目标。
· 时期: DEV = 2025-01-01 之前, HOLDOUT = 之后。赛前和 BP 后各只带一个候选去 HOLDOUT, 即 DEV 跨赛区 Brier
  最低的那个 (平局取简单的)。HOLDOUT 只评一次。
· 时间纪律: 每局的预测只用时间戳严格更早的信息。系数按赛事 (年 × 代码) walk-forward, 预测一个赛事时只在
  它第一局之前的全部跨赛区国际赛上拟合。线上 Stage 1/2 的产物在 HOLDOUT 上禁用, 因为它们用 2025 的国内
  比赛训过; 本脚本只把它们放在 [6] 里作对照。
· 指标: 逐局 Brier, 另报对数损失。三个基线:
  (a) 常数 = 切点前全部 OE 比赛的蓝方胜率;
  (b) 0.5;
  (c) 只用母赛区身份的 Bradley-Terry: 岭 logistic, 按赛事 walk-forward 在之前的跨赛区国际赛上拟合,
      λ=1 在 DEV 上挑。
· 通过的条件: HOLDOUT 跨赛区局上, 对常数的系列赛配对 t > 2.5 (同一系列赛的几局先平均), 且对赛区 BT 不显著
  更差 (系列赛 t > −2.5)。赛事块的符号计数一起报。

DEV 上试过的候选 (529 局 / 302 系列赛 / 7 赛事; 常数 0.2491, 赛区 BT 0.2075)
-----------------------------------------------------------------------
  赛前   C 分层 Elo                                Brier 0.1886  对常数 系列赛 t +8.13   ← 带去 HOLDOUT
         A 全赛区 Elo (调完退化为按赛区的在线 Elo)    0.1933        t +7.19
         B 赛区强度 + 赛区内状态 (Stage 1 按年重训)     0.2139        t +3.06 (只能评它覆盖的子集;
                                                    那个子集上常数 0.2504、赛区 BT 0.2120)
  BP 后  D = C + 英雄胜率 cw                        0.1833        t +8.50   ← 带去 HOLDOUT
         网格共 1020 个配置。D 对 C 的系列赛 t 只有 +2.03, 没过 2.5; 按协议原文 (Brier 最低) 仍带 D。
         s2d (Stage 2 按年重训的 BP 增量, 只能给四大v四大) 对 C 的 t +0.77; 选手-英雄特征 fam / pcw / pce
         打平或更差。
两个带去 HOLDOUT 的候选都不用 Stage 1/2 信号, 所以这里不需要做 Stage 1/2 的按年重训。

候选 C (赛前) 的结构:
  · 全球实力 S = o[母赛区] + r[队]。
  · r: 每个 OE 赛区一个 Elo 池, K=2, 由常规联赛和同池杯赛更新; 跨赛区国际赛另以 kint=24 更新两队的 r。
  · o: 只由跨赛区国际赛更新, eta=24, 零和。第一次出现时取先验: 四大 0, 其余 −600; 新赛区取迁入队伍原赛区
    偏移的均值。o 以半衰期 365 天向先验衰减。
  · 输出: p = σ(a + b_o·Δo/S + b_r·Δr/S), S = 400/ln10。(a, b_o, b_r) 按赛事 walk-forward 拟合, 岭先验朝
    (logit c, 1, 1), λ=30。
  · 评分在赛事内在线更新, 包括同一天、同一系列赛的前几局 (协议允许评分只用严格更早的结果)。
候选 D (BP 后): 在同一个 logistic 里加 β·cw/sd。cw = 英雄胜率表 (只累计四大常规联赛 + 国际赛, 半衰期 365 天,
  向 0.5 收缩 100 局) 的 logit 在五个位置上的平均, 蓝方减红方; β 的岭 λ_d=3。
参照行 (不参与判决): "C 次日 / D 次日" 是只用 OE 数据、结果隔天才生效时的版本, 参数是 C 在这个约束下 DEV 重调过的。

HOLDOUT 结论 (2026-10-03, 数据截至 2026-10-02; 跨赛区 450 局 / 177 系列赛 / 11 赛事; 只评了这一次)
--------------------------------------------------------------------------------------------
复现: DEV 的每个数都和候选报告一致 (逐局 max|Δp| = 1e-16)。在 HOLDOUT 上评分之前, 先只比了概率、没看胜负:
移植版和候选脚本自己的实现在 754 个 HOLDOUT 评估局上 max|Δp| = 0。

· **赛前 C 通过, 但只是刚过线。**
  - Brier 0.2231, 对数损失 0.6403, 准确率 65.3%, 校准斜率 0.72±0.11。常数 0.2478, 0.5 是 0.2500,
    赛区 BT 0.2274。
  - 对常数: Δ +0.0343, 系列赛 t **+2.87** (100/177 个系列赛更好), 逐局 t +2.77, 对数损失系列赛 t +2.53;
    但赛事块只有 7/11 (t +1.40)。
  - 对赛区 BT: Δ +0.0031, t +0.40, 打平。DEV 上的嵌套检查预告过这个结果 (2024 上 C 对赛区 BT t +0.08)。
    DEV 上对常数的优势 0.0816, 到 HOLDOUT 缩成 0.0343。
  - 分年: 2025 年 263 局, 对常数 t +1.52, 对赛区 BT t −0.91; 2026 年 187 局, 对常数 t +2.84, 对赛区 BT t +1.33。
  - 分对阵: 四大v四大 245 局, 对常数 t +1.51; 含非四大 205 局, 对常数 t +2.52。
  - 过度自信: 被看好方给 0.8-0.9 的 63 局实际只赢 76.2% (预测 84.2%); 给 0.9 以上的 11 局实际赢 72.7%
    (预测 91.4%)。
· **BP 后 D 不通过。**
  - Brier 0.2286; 对常数系列赛 t +2.47 (6/11), 差一点没到 2.5; 对赛区 BT t −0.17。
  - BP 项在 HOLDOUT 上是负的: D 对 C Δ −0.0044, 系列赛 t −1.64, 赛事块 3/11。2026 四大v四大 t −2.65。
  - DEV 上的 +2.03 没有延续。英雄胜率在跨赛区国际赛上不加信息。
  - 所以看板的 BP 后数应该继续显示 C 的赛前数, 不做 BP 调整 (候选 D 的服务说明第 6 条)。
· **赛事内实时重放有用。** 结果隔天才生效的 C 次日版: Brier 0.2265, 对在线 C 系列赛 t −2.25。看板要拿到 0.2231,
  需要按 C 的更新公式重放 lolesports 上已完成的局。
· **覆盖。** C / D 能给全部 450 局出数: 四大v四大 245 局, 含非四大 205 局。母赛区不明的 7 局 (国家队) 不出数。
  - 有 29 局的一方赛区此前没打过跨赛区国际赛, 偏移只能来自先验或迁入均值。这 29 局 C 很差: Brier 0.3024,
    常数 0.2561。
    · 其中 13 局是 LCP 在 2025 First Stand 的首秀: 迁入的 PCS/VCS 队给的先验是 −508, C 给 LCP 32%, 实际赢了 62%。
    · 另外 16 局是 LFL 在 2026 EWC EMEA 预选打 LEC: 先验 −62 (来自一支迁入的 LEC 队), 结果约 50/50。
  - 其余 421 局: C 0.2177, 常数 0.2473, 赛区 BT 0.2216。
  - 看板对"此赛区没有跨赛区国际赛记录"的对阵应当标出来或不给数。这一条是事后看到的, 不是闸的一部分。
· **对照: 线上 Stage 1/2 在同一批局上。** 它只能评四大v四大的 245 局。
  - Stage 1: Brier 0.2456, 准确率 56.3%, 校准斜率 0.50, 对常数 t −0.11。
  - Stage 2: Brier 0.2553, 准确率 51.0%, 校准斜率 0.22, 对常数 t −0.05。
  - 同样 245 局上 C 是 0.2284, 对线上 Stage 1 系列赛 t +1.46。
  - 含非四大的 205 局线上一局都给不了。
  - 注意 2025 年的局落在线上模型国内训练 / 校准的时间段里。
· 数据末尾 (2026-10-02) 的赛区偏移, 单位 Elo 分: LCK +30, LFL −68, LPL −73, LEC −89, LCS −140, LCP −422,
  CBLOL −445; PCS / VCS / LJL / LLA / LCO / TCL 约 −600 (已停办或没再打国际赛, 衰减回先验)。

    python research/gate_cross_region.py [--cache-dir D] [--refresh] [--dev-preds-dir D] [--dev-only]
                                         [--skip-production] [--out D]
    从项目根目录跑。约 40 秒 (重建局级表另加约 20 秒); 不联网, 不碰线上服务和 predictions/log.jsonl。
    结果 JSON 和 HOLDOUT 逐局预测写到 --out (默认 = 局级表的缓存目录, 不在仓库里)。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "research"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import xregion_data as XR                                                   # noqa: E402  共用局级表
from xregion_data import (target, walk_forward_events, group_t, brier,       # noqa: E402
                          logloss, LEAGUES, ROLES)
from harness import paired_t, T_ACCEPT                                      # noqa: E402  配对 t 全项目一份
from feature_store import champion_key                                      # noqa: E402  英雄名归一

S = 400 / math.log(10)            # Elo 刻度: 差 S 分 = logit 差 1
EPOCH = pd.Timestamp("1970-01-01")
LN2 = math.log(2)

# ══════════════════════════════════════════════════════════
#  冻结的候选 (DEV 上选定; 这里一个数都不许改)
# ══════════════════════════════════════════════════════════
# 赛前 = 候选 C 分层 Elo。DEV 上还调到了 reg=0 (不跨年回归)、rebrand=否 (不按阵容继承改名队的评分)
# —— 这两项在这个取值下什么都不做, 所以下面的实现里没有它们。
P_C = dict(K=2.0, knew=1.0, nnew=10, h=0.0, init=0.0, cup=True, intl_same=False,
           eta=24.0, kint=24.0, hl=365.0, dnm=600.0, lag="none")
LAM_C = 30.0                      # walk-forward logistic: Δo、Δr 分开, 岭先验朝 (logit c, 1, 1), λ = 30
# BP 后 = 候选 D: C + 英雄胜率 cw (四大常规联赛 + 国际赛合池, 半衰期 365 天, 向 0.5 收缩 100 局), β 的岭 λ_d = 3
D_CFG = dict(hl_c=365.0, scope="major", k_c=100.0, lam_d=3.0)
# 基线 (c) 赛区 BT 的岭强度, 也是 DEV 上挑的 (给基线它最好的一枪)
LAM_BT = 1.0
# 参照, 不是候选: 只用 OE 数据 (结果隔天才有) 时, C 在"次日生效"约束下 DEV 重调过的参数。
# 候选 D 的报告里 "C 次日 / D* 次日" 两行就是它; 它回答"看板不实时重放赛果会损失多少", 不参与判决。
P_C_NEXTDAY = dict(K=2.0, knew=2.0, nnew=10, h=20.0, init=0.0, cup=True, intl_same=False,
                   eta=16.0, kint=16.0, hl=1460.0, dnm=600.0, lag="nextday")

# 候选报告里的 DEV 数 —— 复现不出来就停, 不评 HOLDOUT (实现和被选中的那个不是同一个东西)。
# 精确值来自 cand_c_result.json / cand_d_result.json, 其余是报告里四位 / 两位小数的数。
EXPECTED_DEV = {
    "n_games": 529, "n_series": 302, "n_events": 7,
    "brier": {"C": (0.18855211470460995, 1e-9), "D": (0.18326580744256593, 1e-9),
              "常数 c": (0.2491, 5.1e-5), "赛区 BT": (0.2075, 5.1e-5),
              "C 次日 (参照)": (0.19090503292820535, 1e-9), "D 次日 (参照)": (0.18567811190082137, 1e-9)},
    "t_series": {("C", "常数 c"): 8.13, ("D", "常数 c"): 8.50, ("C", "赛区 BT"): 3.41,
                 ("D", "赛区 BT"): 3.80, ("D", "C"): 2.03},
    "events_better": {("C", "常数 c"): "6/7", ("D", "常数 c"): "7/7"},
}
CHAMP_COLS = [f"{s}_{r}_champ" for s in ("blue", "red") for r in ROLES]


# ══════════════════════════════════════════════════════════
#  度量
# ══════════════════════════════════════════════════════════

def ece(p, y, nb=10):
    """10 等宽箱, 和 train.ece / 候选脚本同一个算法。"""
    p, y = np.asarray(p, float), np.asarray(y, float)
    e, ed = 0.0, np.linspace(0, 1, nb + 1)
    for i in range(nb):
        m = (p > ed[i]) & (p <= ed[i + 1]) if i else (p >= 0) & (p <= ed[1])
        if m.sum():
            e += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
    return e


def calib_slope(p, y, iters=50):
    """y ~ a + b·logit(p) 的 logistic 回归 (牛顿法), 返回 (b, se)。1 = 刚好, < 1 = 过度自信, 0 = 没信息。"""
    p, y = np.asarray(p, float), np.asarray(y, float)
    x = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6))
    if np.std(x) < 1e-9:
        return np.nan, np.nan
    X = np.column_stack([np.ones_like(x), x])
    w = np.zeros(2)
    for _ in range(iters):
        mu = 1 / (1 + np.exp(-(X @ w)))
        H = X.T @ (X * (mu * (1 - mu))[:, None])
        try:
            step = np.linalg.solve(H, X.T @ (y - mu))
        except np.linalg.LinAlgError:
            return np.nan, np.nan
        w = w + step
        if np.abs(step).max() < 1e-10:
            break
    mu = 1 / (1 + np.exp(-(X @ w)))
    H = X.T @ (X * (mu * (1 - mu))[:, None])
    try:
        se = float(np.sqrt(np.linalg.inv(H)[1, 1]))
    except np.linalg.LinAlgError:
        se = np.nan
    return float(w[1]), se


def metrics(p, y) -> dict:
    p, y = np.asarray(p, float), np.asarray(y, float)
    if len(y) == 0:
        return {"n": 0}
    b, se = calib_slope(p, y) if len(y) >= 20 else (np.nan, np.nan)
    return {"n": len(y), "acc": float(((p > 0.5) == (y == 1)).mean()),
            "brier": float(brier(p, y).mean()), "logloss": float(logloss(p, y).mean()),
            "ece": float(ece(p, y)) if len(y) >= 20 else np.nan, "slope": b, "slope_se": se}


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


# ══════════════════════════════════════════════════════════
#  局级表 → 扁平数组 (Elo 主循环吃 Python 列表, 比逐行 pandas 快两个数量级)
# ══════════════════════════════════════════════════════════

def prepare(df: pd.DataFrame) -> dict:
    if not df["date"].is_monotonic_increasing:
        raise RuntimeError("局级表没按时间排序 —— Elo 和英雄表都假定按时间走")
    A = {"n": len(df)}
    # 天 (浮点)。不能用 astype("int64") / 86400e9: 缓存里的 date 是 datetime64[us] —— 候选 C 的第一版就这样
    # 差了 1000 倍, 半衰期整个失效, 没有任何报错
    A["t"] = ((df["date"] - EPOCH).dt.total_seconds() / 86400).tolist()
    ns = df["date"].astype("int64").tolist()                  # 只用来判断"同一时间戳", 单位无所谓
    A["year"] = df["year"].astype(int).tolist()
    A["ec"] = df["event_class"].map({"league": 0, "non_home": 1, "intl": 2}).astype(int).tolist()
    A["league"] = df["league"].astype(str).tolist()
    A["blue"] = df["blue"].astype(str).tolist()
    A["red"] = df["red"].astype(str).tolist()
    A["y"] = df["blue_win"].astype(float).tolist()
    A["ok"] = df["result_ok"].astype(bool).tolist()
    A["hb"] = [x if isinstance(x, str) else None for x in df["home_blue"].tolist()]
    A["hr"] = [x if isinstance(x, str) else None for x in df["home_red"].tolist()]
    A["status"] = df["region_status"].astype(str).tolist()
    # 同一时间戳的几局一组: 组内先全部取赛前值, 再统一更新 —— 同时开打的两局互相看不到
    starts = [0] + [i for i in range(1, len(ns)) if ns[i] != ns[i - 1]]
    A["groups"] = list(zip(starts, starts[1:] + [len(ns)]))
    day = [math.floor(x) for x in A["t"]]
    dstarts = [0] + [i for i in range(1, len(day)) if day[i] != day[i - 1]]
    A["day_groups"] = list(zip(dstarts, dstarts[1:] + [len(day)]))
    return A


# ══════════════════════════════════════════════════════════
#  候选 C: 分层 Elo
# ══════════════════════════════════════════════════════════

def run_elo(P: dict, A: dict) -> dict:
    """按时间走一遍全部 OE 比赛, 给每局记下赛前的 o_蓝 / o_红 (只有跨赛区国际赛才有) 和 r_蓝 / r_红。

    全球实力 S = o[母赛区] + r[队]:
      · r 每个 OE 赛区一个 Elo 池, 由常规联赛和同池杯赛更新 (K, 前 nnew 局 ×knew); 跨赛区国际赛另以 kint
        更新两队的 r。换赛区时: 新旧赛区都有偏移 → 保持全球实力不变; 迁入还没有偏移的新赛区 (LCP) → r 原样
        带过去, 并登记"迁入来源"; 原赛区没有偏移 (次级联赛升上来) → 重置为 init。
      · o 只由跨赛区国际赛更新 (学习率 eta, 零和), 第一次出现时取先验 (四大 0, 其余 −dnm; 新赛区取迁入
        队伍原赛区偏移的均值), 并以半衰期 hl 向先验指数衰减 (懒计算: 用到时才衰减)。
    时间纪律: 预测只用时间戳严格更早的结果 (lag="none", 协议允许评分这样用, 赛事内的更早局也算);
    lag="nextday" 是服务现实 —— 结果从下一个 UTC 日起才生效 (OE 隔天才有数据)。
    这是候选 C 脚本 run_elo 的逐行移植; main() 先在 DEV 上逐局对账, 对不上就停。
    """
    K, knew, nnew, h, init = P["K"], P["knew"], int(P["nnew"]), P["h"], P["init"]
    eta, kint, hl, dnm = P["eta"], P["kint"], P["hl"], P["dnm"]
    cup, intl_same = P["cup"], P["intl_same"]
    nextday = P["lag"] == "nextday"
    majors = set(LEAGUES)

    r: dict[str, float] = {}
    pool: dict[str, str] = {}
    ng: dict[str, int] = {}
    o: dict[str, float] = {}
    oprior: dict[str, float] = {}
    olast: dict[str, float] = {}
    migr: dict[str, list] = defaultdict(list)       # 还没有偏移的赛区 → 迁入队伍的原赛区
    pending: list = []                              # nextday: (生效时刻, 更新), 按时间排队
    stats = {"intl_pool_mismatch": 0}

    n = A["n"]
    ob, orr = np.full(n, np.nan), np.full(n, np.nan)
    rb, rr = np.full(n, np.nan), np.full(n, np.nan)
    tl, yl, ecl, lgl = A["t"], A["y"], A["ec"], A["league"]
    bl, rl, okl, hbl, hrl, stl = A["blue"], A["red"], A["ok"], A["hb"], A["hr"], A["status"]

    def enter(team, league):
        """常规联赛比赛: 确保队伍在这个池子里有 r (新队 / 换赛区)。"""
        old = pool.get(team)
        if old == league:
            return
        if old is None:
            r[team], ng[team], pool[team] = init, 0, league
            return
        if old in o and league in o:                    # 两个赛区都有偏移: 保持全球实力不变
            r[team] = o[old] + r[team] - o[league]
        elif old in o:                                  # 迁入还没有偏移的新赛区: r 带过去
            migr[league].append(old)
        else:                                           # 原赛区没有可比的零点: 当新队
            r[team] = init
            ng[team] = 0
        pool[team] = league
        ng.setdefault(team, 0)

    def off(L, t):
        """赛区偏移: 懒初始化 + 向先验的指数衰减。"""
        if L not in o:
            src = [o[x] for x in migr.get(L, []) if x in o]
            pri = float(np.mean(src)) if src else (0.0 if L in majors else -dnm)
            o[L] = oprior[L] = pri
            olast[L] = t
        elif hl != math.inf:
            dt = t - olast[L]
            if dt > 0:
                o[L] = oprior[L] + (o[L] - oprior[L]) * math.exp(-LN2 * dt / hl)
                olast[L] = t
        return o[L]

    def apply(u):
        if u[0] == "dom":
            _, i, b, rd, pe = u
            d = yl[i] - pe
            kb = K * (knew if ng.get(b, 0) < nnew else 1.0)
            kr = K * (knew if ng.get(rd, 0) < nnew else 1.0)
            r[b] += kb * d
            r[rd] -= kr * d
            ng[b] = ng.get(b, 0) + 1
            ng[rd] = ng.get(rd, 0) + 1
        else:
            _, i, b, rd, pe, hb, hr, okb, okr = u
            d = yl[i] - pe
            o[hb] += eta * d
            o[hr] -= eta * d
            if kint:
                if okb:
                    r[b] += kint * d
                if okr:
                    r[rd] -= kint * d

    pi = 0
    for g0, g1 in A["groups"]:
        if nextday:                                     # 到期的结果先生效 (按原来的时间顺序)
            while pi < len(pending) and pending[pi][0] <= tl[g0]:
                apply(pending[pi][1])
                pi += 1
        upd = []
        for i in range(g0, g1):
            ec, b, rd, t = ecl[i], bl[i], rl[i], tl[i]
            if ec == 0:                                 # 常规联赛
                L = lgl[i]
                enter(b, L)
                enter(rd, L)
                rb[i], rr[i] = r[b], r[rd]
                if okl[i]:
                    upd.append(("dom", i, b, rd, 1 / (1 + math.exp(-(r[b] + h - r[rd]) / S))))
            elif ec == 2:                               # 国际赛
                st = stl[i]
                if st == "unknown":
                    continue
                hb, hr = hbl[i], hrl[i]
                # 母赛区 (xregion_data 的口径, 可能按阵容认队) 和 Elo 池不一致的队, r 取 init 且不更新
                okb, okr = pool.get(b) == hb, pool.get(rd) == hr
                stats["intl_pool_mismatch"] += (not okb) + (not okr)
                vb = r[b] if okb else init
                vr = r[rd] if okr else init
                rb[i], rr[i] = vb, vr
                if st == "cross":
                    ob[i], orr[i] = off(hb, t), off(hr, t)
                    pe = 1 / (1 + math.exp(-(ob[i] + vb + h - orr[i] - vr) / S))
                    if okl[i]:
                        upd.append(("x", i, b, rd, pe, hb, hr, okb, okr))
                else:                                   # 同赛区: 偏移相同, 只看 r
                    ob[i] = orr[i] = o.get(hb, np.nan)
                    pe = 1 / (1 + math.exp(-(vb + h - vr) / S))
                    if intl_same and okb and okr and okl[i]:
                        upd.append(("dom", i, b, rd, pe))
            else:                                       # 杯赛 / 次级跨区赛: 同池两队才更新
                if cup and pool.get(b) is not None and pool.get(b) == pool.get(rd):
                    pe = 1 / (1 + math.exp(-(r[b] + h - r[rd]) / S))
                    if okl[i]:
                        upd.append(("dom", i, b, rd, pe))
        for u in upd:
            if nextday:
                pending.append((math.floor(tl[u[1]]) + 1.0, u))
            else:
                apply(u)
    return {"ob": ob, "or": orr, "rb": rb, "rr": rr, "o": dict(o), "oprior": dict(oprior),
            "olast": dict(olast), "migr": {k: list(v) for k, v in migr.items()}, "stats": stats,
            "r": dict(r), "pool": dict(pool)}


# ══════════════════════════════════════════════════════════
#  候选 D 的 BP 项: 英雄胜率表 (只累计四大常规联赛 + 国际赛)
# ══════════════════════════════════════════════════════════

def champ_table(df: pd.DataFrame, A: dict, hl_c: float, scope: str, lag: str = "none") -> dict:
    """每局每个位置取"赛前"的英雄衰减胜场 / 场次。候选 D 脚本 draft_pass 的英雄部分 (选手表 D 没用到)。

    lag="none": 同一时间戳一组, 组内先取值再统一更新 (严格更早, 含同一赛事里更早的局);
    lag="nextday": 同一个 UTC 日一组 —— 当天的局看不到当天的结果。
    scope="major": 只累计四大常规联赛 + 国际赛 (DEV 上全部赛区合池明显更差: 次级联赛的英雄胜率
    搬不到国际赛上)。
    """
    n, t, y, ok = A["n"], A["t"], A["y"], A["ok"]
    champs = [[(champion_key(c) if isinstance(c, str) else None) for c in row]
              for row in df[CHAMP_COLS].values.tolist()]
    if scope == "all":
        use_c = [True] * n
    else:
        use_c = ((df["league"].isin(LEAGUES) & (df["event_class"] == "league"))
                 | df["is_international"]).tolist()
    groups = A["groups"] if lag == "none" else A["day_groups"]
    cg, cw = np.zeros((n, 10)), np.zeros((n, 10))
    ctab: dict = {}                                     # 英雄 → [衰减胜场, 衰减场次, 上次更新时刻]
    decay = hl_c != math.inf
    for g0, g1 in groups:
        for i in range(g0, g1):
            ti = t[i]
            for s, c in enumerate(champs[i]):
                if c is None:
                    continue
                v = ctab.get(c)
                if v is not None:
                    f = math.exp(-LN2 * (ti - v[2]) / hl_c) if decay else 1.0
                    cw[i, s], cg[i, s] = v[0] * f, v[1] * f
        for i in range(g0, g1):
            if not ok[i] or not use_c[i]:
                continue
            ti = t[i]
            for s, c in enumerate(champs[i]):
                if c is None:
                    continue
                win = y[i] if s < 5 else 1.0 - y[i]
                v = ctab.get(c)
                if v is None:
                    ctab[c] = [win, 1.0, ti]
                else:
                    f = math.exp(-LN2 * (ti - v[2]) / hl_c) if decay else 1.0
                    v[0] = v[0] * f + win
                    v[1] = v[1] * f + 1.0
                    v[2] = ti
    miss = np.array([[c is None for c in row] for row in champs])
    return {"cg": cg, "cw": cw, "miss": miss, "table": ctab}


def cw_feature(T: dict, k_c: float) -> np.ndarray:
    """cw = 蓝方五个英雄 logit((w + k/2)/(g + k)) 的平均 − 红方的平均 (缺英雄的位置不算)。"""
    p = np.clip((T["cw"] + 0.5 * k_c) / (T["cg"] + k_c), 1e-6, 1 - 1e-6)
    v = np.where(T["miss"], np.nan, np.log(p / (1 - p)))
    return np.nanmean(v[:, :5], axis=1) - np.nanmean(v[:, 5:], axis=1)


# ══════════════════════════════════════════════════════════
#  walk-forward 的 logistic (岭先验) 和基线
# ══════════════════════════════════════════════════════════

def fit_ridge(X, y, prior, lam, pen=None, iters=60):
    """最小化 NLL + λ/2·Σ pen·(β − prior)²。没有训练行 (2022 MSI 之前) 就返回先验。"""
    prior = np.asarray(prior, float)
    if lam == math.inf or len(y) == 0:
        return prior.copy()
    pen = np.ones_like(prior) if pen is None else np.asarray(pen, float)
    beta = prior.copy()
    for _ in range(iters):
        p = sigmoid(X @ beta)
        g = X.T @ (y - p) - lam * pen * (beta - prior)
        H = X.T @ (X * (p * (1 - p))[:, None]) + lam * np.diag(pen) + 1e-9 * np.eye(len(beta))
        step = np.linalg.solve(H, g)
        beta = beta + step
        if np.abs(step).max() < 1e-10:
            break
    return beta


def const_rate(df, cut) -> float:
    """基线 (a): 切点之前全部 OE 比赛的蓝方胜率。"""
    m = (df["date"] < cut) & df["result_ok"]
    return float(df.loc[m, "blue_win"].mean())


def predict_wf(df, R, E, F=None, lam_d=None):
    """p = σ(a + b_o·Δo/S + b_r·Δr/S [+ β·F/sd])。系数按赛事 walk-forward, 只在切点之前的全部
    跨赛区国际赛上拟合; (a, b_o, b_r) 的岭先验朝 (logit c, 1, 1), λ = 30; β 先验 0, 岭 λ_d。
    sd = 切点前跨赛区国际赛上 F 的标准差 (只用特征值, 不用胜负)。返回 (概率, 每个赛事的系数)。"""
    x_o = (R["ob"] - R["or"]) / S
    x_r = (R["rb"] - R["rr"]) / S
    y = E["y"]
    dates = df["date"].values
    xmask = E["xmask"]
    p_out = pd.Series(np.nan, index=df.index)
    coefs = {}
    Fz = None if F is None else np.nan_to_num(F, nan=0.0)
    for name, cut, idx in E["ev_list"]:
        c = E["cut_consts"][name]
        tr = xmask & (dates < np.datetime64(cut))
        cols = [np.ones(tr.sum()), x_o[tr], x_r[tr]]
        prior, pen, sd = [logit(c), 1.0, 1.0], [1.0, 1.0, 1.0], 1.0
        if F is not None:
            have = tr & ~np.isnan(F)
            s_ = float(np.std(F[have])) if have.sum() >= 2 else 0.0
            sd = s_ if s_ > 1e-9 else 1.0
            cols.append(Fz[tr] / sd)
            prior.append(0.0)
            pen.append(lam_d / LAM_C)
        beta = fit_ridge(np.column_stack(cols), y[tr], prior, LAM_C, pen)
        ii = np.asarray(idx)
        xo = np.where(np.isnan(x_o[ii]), 0.0, x_o[ii])     # 同赛区局的偏移可能没初始化, 差为 0
        z = beta[0] + beta[1] * xo + beta[2] * x_r[ii]
        if F is not None:
            z = z + beta[3] * (Fz[ii] / sd)
        p_out.loc[ii] = sigmoid(z)
        coefs[name] = {"beta": [float(b) for b in beta], "sd": float(sd), "n_train": int(tr.sum()),
                       "c": float(c)}
    return p_out, coefs


def predict_region_bt(df, E, lam=LAM_BT):
    """基线 (c): 只用母赛区身份的 Bradley-Terry —— 母赛区 one-hot 之差 + 蓝方截距的岭 logistic,
    按赛事 walk-forward 在切点前的跨赛区国际赛上拟合; 训练里没出现过的赛区 (LCP、LFL) 系数取 0。"""
    y = E["y"]
    dates = df["date"].values
    hb, hr = df["home_blue"].values, df["home_red"].values
    p_out = pd.Series(np.nan, index=df.index)
    for name, cut, idx in E["ev_list"]:
        c = E["cut_consts"][name]
        ii = np.asarray(idx)
        tr = E["xmask"] & (dates < np.datetime64(cut))
        if tr.sum() == 0:
            p_out.loc[ii] = c
            continue
        leagues = sorted(set(hb[tr]) | set(hr[tr]))
        col = {L: k for k, L in enumerate(leagues)}
        X = np.zeros((tr.sum(), len(leagues) + 1))
        X[:, 0] = 1
        for j, (a, b_) in enumerate(zip(hb[tr], hr[tr])):
            X[j, 1 + col[a]] += 1
            X[j, 1 + col[b_]] -= 1
        prior = np.zeros(len(leagues) + 1)
        prior[0] = logit(c)
        pen = np.ones(len(leagues) + 1)
        pen[0] = 1e-3                                   # 截距几乎不罚
        beta = fit_ridge(X, y[tr], prior, lam, pen)
        z = np.full(len(ii), beta[0])
        for k, i in enumerate(ii):
            if hb[i] in col:
                z[k] += beta[1 + col[hb[i]]]
            if hr[i] in col:
                z[k] -= beta[1 + col[hr[i]]]
        p_out.loc[ii] = sigmoid(z)
    return p_out


# ══════════════════════════════════════════════════════════
#  一个时期 (DEV / HOLDOUT) 的全部预测
# ══════════════════════════════════════════════════════════

def setup_period(df: pd.DataFrame, period: str) -> dict:
    """评估行 = 该时期母赛区已知的国际赛 (跨赛区是目标, 同赛区单独报); 拟合行 = 切点前的全部跨赛区国际赛。"""
    intl = df["is_international"].values
    known = (df["region_status"] != "unknown").values
    emask = pd.Series(intl & (df["period"] == period).values & known, index=df.index)
    ev_list = list(walk_forward_events(df, emask))
    idx = np.concatenate([np.asarray(e[2]) for e in ev_list])
    E = {"period": period, "ev_list": ev_list, "idx": idx,
         "cut_consts": {name: const_rate(df, cut) for name, cut, _ in ev_list},
         "xmask": target(df).values, "y": df["blue_win"].values.astype(float)}
    E["cross"] = idx[df.loc[idx, "cross_region"].values]
    E["same"] = idx[(df.loc[idx, "region_status"] == "same").values]
    pt = df.loc[E["cross"], "pair_type"].values
    E["mvm"] = E["cross"][pt == "major_v_major"]
    E["nmj"] = E["cross"][pt == "involves_nonmajor"]
    return E


def period_preds(df, E, RC, RN, F_on, F_nd) -> tuple[dict, dict]:
    pC, coefC = predict_wf(df, RC, E)
    pD, coefD = predict_wf(df, RC, E, F_on, D_CFG["lam_d"])
    pCn, _ = predict_wf(df, RN, E)
    pDn, _ = predict_wf(df, RN, E, F_nd, D_CFG["lam_d"])
    pK = pd.Series(np.nan, index=df.index)
    for name, _cut, idx in E["ev_list"]:
        pK.loc[np.asarray(idx)] = E["cut_consts"][name]
    preds = {"C": pC, "D": pD, "常数 c": pK, "0.5": pd.Series(0.5, index=df.index),
             "赛区 BT": predict_region_bt(df, E), "C 次日 (参照)": pCn, "D 次日 (参照)": pDn}
    return preds, {"C": coefC, "D": coefD}


def tstat(df, E, preds, cand, base, idx=None, fn=brier):
    """(Δ, 系列赛 t, 系列赛数, 候选更好的系列赛数, 赛事块 t, 赛事数, 候选更好的赛事数, 逐局 t)。正 = cand 更好。"""
    idx = E["cross"] if idx is None else idx
    y = E["y"][idx]
    lb, lc = fn(preds[base].loc[idx].values, y), fn(preds[cand].loc[idx].values, y)
    d_s, t_s, n_s, k_s = group_t(lb, lc, df.loc[idx, "series"].values)
    _d, t_e, n_e, k_e = group_t(lb, lc, df.loc[idx, "event"].values)
    t_g = paired_t(-lb, -lc)[2] if len(idx) >= 3 else np.nan
    return {"d": float(d_s), "t_series": float(t_s), "n_series": int(n_s), "series_better": int(k_s),
            "t_event": float(t_e), "n_event": int(n_e), "event_better": int(k_e), "t_game": float(t_g)}


# ══════════════════════════════════════════════════════════
#  输出
# ══════════════════════════════════════════════════════════

def f4(x):
    return "—" if x is None or x != x else f"{x:.4f}"


def ft(x):
    return "—" if x is None or x != x else f"{x:+.2f}"


def md_metric_table(df, E, preds, names, subsets) -> list[str]:
    lines = ["| 子集 | 模型 | n | 准确率 | Brier | 对数损失 | ECE | 校准斜率 |", "|---|---|---|---|---|---|---|---|"]
    for lab, ii in subsets:
        if len(ii) == 0:
            continue
        y = E["y"][ii]
        for nm in names:
            m = metrics(preds[nm].loc[ii].values, y)
            sl = f"{m['slope']:+.2f}±{m['slope_se']:.2f}" if m["slope"] == m["slope"] else "—"
            ec = f"{m['ece']:.3f}" if m["ece"] == m["ece"] else "—"
            lines.append(f"| {lab} | {nm} | {m['n']} | {m['acc']:.3f} | {m['brier']:.4f} | {m['logloss']:.4f} | "
                         f"{ec} | {sl} |")
    return lines


def md_t_table(df, E, preds, pairs, idx=None) -> list[str]:
    lines = ["| 候选 vs 基线 | 指标 | Δ (基线−候选) | 系列赛 t | 候选更好的系列赛 | 赛事块 t | 候选更好的赛事 | 逐局 t |",
             "|---|---|---|---|---|---|---|---|"]
    for cand, base in pairs:
        for mname, fn in (("Brier", brier), ("对数损失", logloss)):
            r = tstat(df, E, preds, cand, base, idx, fn)
            lines.append(f"| {cand} vs {base} | {mname} | {r['d']:+.4f} | **{ft(r['t_series'])}** | "
                         f"{r['series_better']}/{r['n_series']} | {ft(r['t_event'])} | "
                         f"{r['event_better']}/{r['n_event']} | {ft(r['t_game'])} |")
    return lines


def md_event_table(df, E, preds, names) -> list[str]:
    lines = ["| 赛事 | 切点 | 跨赛区局 | 系列赛 | " + " | ".join(names) + " |",
             "|---|---|---|---|" + "---|" * len(names)]
    y = E["y"]
    for name, cut, idx in E["ev_list"]:
        ii = np.asarray(idx)
        ii = ii[df.loc[ii, "cross_region"].values]
        if len(ii) == 0:
            continue
        cells = [f4(float(brier(preds[nm].loc[ii].values, y[ii]).mean())) for nm in names]
        lines.append(f"| {name} | {pd.Timestamp(cut).date()} | {len(ii)} | {df.loc[ii, 'series'].nunique()} | "
                     + " | ".join(cells) + " |")
    return lines


def md_reliability(E, p, idx, title) -> list[str]:
    """按被看好的一方折叠: q = max(p, 1−p), 看被看好的一方实际赢了多少。"""
    p, y = p.loc[idx].values, E["y"][idx]
    q = np.maximum(p, 1 - p)
    win = np.where(p >= 0.5, y, 1 - y)
    edges = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
    lines = [f"**{title}**", "", "| 被看好方的概率 | 局数 | 平均预测 | 实际胜率 | 实际−预测 | ±95% |",
             "|---|---|---|---|---|---|"]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (q >= lo) & (q < hi)
        k = int(m.sum())
        if k == 0:
            continue
        obs = win[m].mean()
        se = math.sqrt(max(obs * (1 - obs), 1e-9) / k)
        lines.append(f"| [{lo:.1f}, {min(hi, 1):.1f}) | {k} | {q[m].mean():.3f} | {obs:.3f} | "
                     f"{obs - q[m].mean():+.3f} | {1.96 * se:.3f} |")
    k = len(q)
    lines.append(f"| 合计 | {k} | {q.mean():.3f} | {win.mean():.3f} | {win.mean() - q.mean():+.3f} | "
                 f"{1.96 * math.sqrt(max(win.mean() * (1 - win.mean()), 1e-9) / k):.3f} |")
    return lines


def say_lines(lines):
    print("\n".join("  " + x for x in lines))


# ══════════════════════════════════════════════════════════
#  DEV 对账
# ══════════════════════════════════════════════════════════

def check_dev(df, E, preds, dev_preds_dir) -> list[str]:
    """DEV 上的数必须和候选报告一致 (Brier 精确值到 1e-9, 报告里的四位 / 两位小数到舍入误差);
    给了候选的逐局 CSV 就再逐局比。返回不一致的项 (空 = 全对上)。"""
    bad = []
    cross = E["cross"]
    y = E["y"][cross]
    ns = df.loc[cross, "series"].nunique()
    ne = df.loc[cross, "event"].nunique()
    for k, v in (("n_games", len(cross)), ("n_series", ns), ("n_events", ne)):
        if v != EXPECTED_DEV[k]:
            bad.append(f"{k}: {v} ≠ {EXPECTED_DEV[k]}")
    for nm, (exp, tol) in EXPECTED_DEV["brier"].items():
        got = float(brier(preds[nm].loc[cross].values, y).mean())
        ok = abs(got - exp) <= tol
        print(f"    Brier {nm:<14}{got:.6f}  报告 {exp:.6f}  {'✓' if ok else '✗'}")
        if not ok:
            bad.append(f"Brier {nm}: {got:.6f} ≠ {exp}")
    for (c, b), exp in EXPECTED_DEV["t_series"].items():
        got = tstat(df, E, preds, c, b)["t_series"]
        ok = abs(got - exp) <= 0.0051
        print(f"    系列赛 t {c} vs {b:<8}{got:+.3f}  报告 {exp:+.2f}  {'✓' if ok else '✗'}")
        if not ok:
            bad.append(f"t {c} vs {b}: {got:+.3f} ≠ {exp:+.2f}")
    for (c, b), exp in EXPECTED_DEV["events_better"].items():
        r = tstat(df, E, preds, c, b)
        got = f"{r['event_better']}/{r['n_event']}"
        print(f"    赛事块 {c} vs {b:<10}{got}  报告 {exp}  {'✓' if got == exp else '✗'}")
        if got != exp:
            bad.append(f"赛事块 {c} vs {b}: {got} ≠ {exp}")
    if dev_preds_dir:
        d = Path(dev_preds_dir)
        gid = pd.Series(df.index, index=df["gameid"].values)
        for fname, cols in (("cand_c_dev_preds.csv", {"p_cand": "C", "p_const": "常数 c", "p_region_bt": "赛区 BT"}),
                            ("cand_d_dev_preds.csv", {"p_D": "D", "p_C": "C", "p_D_nextday": "D 次日 (参照)",
                                                      "p_C_nextday": "C 次日 (参照)"})):
            f = d / fname
            if not f.exists():
                print(f"    (没有 {f}, 跳过逐局对账)")
                continue
            ref = pd.read_csv(f)
            pos = gid.loc[ref["gameid"]].values
            for col, nm in cols.items():
                mx = float(np.abs(preds[nm].loc[pos].values - ref[col].values).max())
                ok = mx < 1e-9
                print(f"    逐局 {fname} {col:<12}→ {nm:<14} {len(ref)} 局 max|Δp| = {mx:.1e}  {'✓' if ok else '✗'}")
                if not ok:
                    bad.append(f"逐局 {fname}:{col} max|Δp| {mx:.2e}")
    return bad


# ══════════════════════════════════════════════════════════
#  对照: 线上 Stage 1/2 (artifacts/) 在同一批 HOLDOUT 局上
# ══════════════════════════════════════════════════════════

def production_stage12(df, idx) -> tuple[pd.Series, pd.Series, dict]:
    """和 gate_international 同一条路: 四大特征库 (FeatureStore.from_csv, 和 api.py 启动时一样) +
    make_row(as_of = 比赛时间, league = 蓝方母赛区, playoffs = 0) + api.Stage (artifacts 的 XGB + Platt)。
    只能给两队都是四大母赛区、且之前都有四大历史的局。**只作对照**: 这些模型用 2025 的国内比赛训过,
    协议禁止候选在 HOLDOUT 上用它们。"""
    import api                                          # 只取 Stage 类和 DATA, 不启动服务
    from feature_store import FeatureStore
    files = XR._files()
    if [Path(f).resolve() for f in files] != [Path(f).resolve() for f in api.DATA]:
        raise RuntimeError("线上用的五年数据和 data_path() 不一致")
    t0 = time.time()
    store = FeatureStore.from_csv(files, verbose=False)
    s1, s2 = api.Stage("pre_draft"), api.Stage("post_draft")
    core = [f"diff_{c}" for c in store.roll_cols if c.startswith("avg_")][:5]
    p1, p2 = pd.Series(np.nan, index=df.index), pd.Series(np.nan, index=df.index)
    why = defaultdict(int)
    for i in idx:
        r = df.loc[i]
        if not (r["home_major_blue"] and r["home_major_red"]):
            why["有一方不是四大母赛区"] += 1
            continue
        draft = {side: {role: (r[f"{side}_{role}_player"], r[f"{side}_{role}_champ"]) for role in ROLES}
                 for side in ("blue", "red")}
        try:
            row, _w = store.make_row(r["blue"], r["red"], r["home_blue"], draft=draft, playoffs=0,
                                     as_of=r["date"])
        except ValueError:
            why["无四大历史"] += 1
            continue
        if any(np.isnan(float(row.get(c, np.nan))) for c in core):
            why["滚动统计不足"] += 1
            continue
        p1.loc[i] = s1.predict(row)[1]
        p2.loc[i] = s2.predict(row)[1]
        why["可评估"] += 1
    info = {"why": dict(why), "secs": round(time.time() - t0),
            "s1_test_metrics": {k: s1.metrics.get(k) for k in ("n_train", "n_cal", "n_test", "accuracy", "brier")},
            "s2_test_metrics": {k: s2.metrics.get(k) for k in ("n_train", "n_cal", "n_test", "accuracy", "brier")}}
    return p1, p2, info


# ══════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="跨赛区国际赛: 赛前 / BP 后候选的冻结协议闸")
    ap.add_argument("--cache-dir", default=None, help="局级表缓存目录 (默认 LOL_XREGION_CACHE 或系统临时目录)")
    ap.add_argument("--refresh", action="store_true", help="强制从 CSV 重建局级表")
    ap.add_argument("--dev-preds-dir", default=None, help="候选脚本的逐局 DEV 预测 CSV 所在目录 (可选, 逐局对账)")
    ap.add_argument("--dev-only", action="store_true", help="只做 DEV 复现, 不碰 HOLDOUT")
    ap.add_argument("--skip-production", action="store_true", help="不算线上 Stage 1/2 的对照 (省 1-2 分钟)")
    ap.add_argument("--out", default=None, help="结果 JSON + HOLDOUT 逐局预测的输出目录 (默认 = 缓存目录)")
    a = ap.parse_args()
    T0 = time.time()
    print("=" * 100)
    print("  跨赛区国际赛: 赛前 (C 分层 Elo) / BP 后 (D = C + 英雄胜率) —— 冻结协议闸")
    print("=" * 100)

    # ── [1] 局级表 ──
    print("\n[1] 局级表 (research/xregion_data.py)")
    cache = Path(a.cache_dir) if a.cache_dir else XR.default_cache_dir()
    df = XR.load_games(cache, refresh=a.refresh, verbose=True)
    diag = df.attrs.get("diag", {})
    print(f"    {len(df)} 局, {df['date'].min().date()} … {df['date'].max().date()}; "
          f"国际赛 {int(df['is_international'].sum())}; 目标 (跨赛区国际赛) DEV "
          f"{int((target(df) & (df['period'] == 'DEV')).sum())} / HOLDOUT "
          f"{int((target(df) & (df['period'] == 'HOLDOUT')).sum())}")
    if diag.get("unlisted_lookalikes"):
        print(f"    ⚠ 没列入国际赛、却有跨赛区一级联赛对打的赛事: {diag['unlisted_lookalikes']}")

    # ── [2] 评分和英雄表: 全部 OE 比赛按时间走一遍 (只用严格更早的结果) ──
    print("\n[2] 分层 Elo + 英雄胜率表 (全部 OE 比赛, 按时间)")
    t = time.time()
    A = prepare(df)
    RC = run_elo(P_C, A)
    RN = run_elo(P_C_NEXTDAY, A)
    T_on = champ_table(df, A, D_CFG["hl_c"], D_CFG["scope"])
    T_nd = champ_table(df, A, D_CFG["hl_c"], D_CFG["scope"], lag="nextday")
    F_on, F_nd = cw_feature(T_on, D_CFG["k_c"]), cw_feature(T_nd, D_CFG["k_c"])
    print(f"    {time.time() - t:.1f}s; 母赛区和 Elo 池不一致的队次 (r 取 init): {RC['stats']['intl_pool_mismatch']}")

    # ── [3] DEV 复现 ──
    print("\n[3] DEV 复现 (2025-01-01 之前的国际赛; 必须和候选报告一致, 否则不评 HOLDOUT)")
    Ed = setup_period(df, "DEV")
    pdv, coef_dev = period_preds(df, Ed, RC, RN, F_on, F_nd)
    bad = check_dev(df, Ed, pdv, a.dev_preds_dir)
    if bad:
        print("\n  ✗ DEV 复现不出候选报告的数, 停在这里 (HOLDOUT 一局都没评):")
        for b in bad:
            print(f"      {b}")
        sys.exit(2)
    print("  ✓ DEV 全部对上")
    names = ["C", "D", "赛区 BT", "常数 c", "0.5", "C 次日 (参照)", "D 次日 (参照)"]
    dev_tab = md_metric_table(df, Ed, pdv, names, (("跨赛区 全部", Ed["cross"]), ("跨赛区 四大v四大", Ed["mvm"]),
                                                    ("跨赛区 含非四大", Ed["nmj"])))
    say_lines([""] + dev_tab)
    dev_t = md_t_table(df, Ed, pdv, (("C", "常数 c"), ("C", "赛区 BT"), ("D", "常数 c"), ("D", "赛区 BT"),
                                      ("D", "C")))
    say_lines([""] + dev_t)
    if a.dev_only:
        print(f"\n  --dev-only: 不评 HOLDOUT。用时 {time.time() - T0:.0f}s")
        return

    # ── [4] HOLDOUT (只评这一次) ──
    print("\n[4] HOLDOUT (2025-01-01 起的国际赛; 候选和全部参数都是冻结的)")
    Eh = setup_period(df, "HOLDOUT")
    ph, coef_h = period_preds(df, Eh, RC, RN, F_on, F_nd)
    cross, mvm, nmj, same = Eh["cross"], Eh["mvm"], Eh["nmj"], Eh["same"]
    yrs = df.loc[cross, "year"].values
    subsets = [("跨赛区 全部", cross), ("跨赛区 四大v四大", mvm), ("跨赛区 含非四大", nmj)]
    subsets += [(f"跨赛区 {y}", cross[yrs == y]) for y in sorted(set(yrs))]
    subsets += [(f"跨赛区 {y} 四大v四大", mvm[df.loc[mvm, 'year'].values == y]) for y in sorted(set(yrs))]
    subsets += [(f"跨赛区 {y} 含非四大", nmj[df.loc[nmj, 'year'].values == y]) for y in sorted(set(yrs))]
    subsets += [("同赛区 (不是目标)", same)]
    hold_tab = md_metric_table(df, Eh, ph, names, subsets)
    say_lines([""] + hold_tab)
    pairs = (("C", "常数 c"), ("C", "0.5"), ("C", "赛区 BT"), ("D", "常数 c"), ("D", "0.5"), ("D", "赛区 BT"),
             ("D", "C"), ("C 次日 (参照)", "C"), ("D 次日 (参照)", "D"))
    hold_t = md_t_table(df, Eh, ph, pairs)
    say_lines(["", "配对 t (先按系列赛 / 赛事求平均; 正 = 候选更好)", ""] + hold_t)
    sub_t = []
    for lab, ii in subsets[1:-1]:
        for cand in ("C", "D"):
            for base in ("常数 c", "赛区 BT"):
                r = tstat(df, Eh, ph, cand, base, ii)
                sub_t.append(f"| {lab} | {cand} vs {base} | {len(ii)} | {r['d']:+.4f} | {ft(r['t_series'])} | "
                             f"{r['series_better']}/{r['n_series']} | {r['event_better']}/{r['n_event']} |")
        r = tstat(df, Eh, ph, "D", "C", ii)
        sub_t.append(f"| {lab} | D vs C | {len(ii)} | {r['d']:+.4f} | {ft(r['t_series'])} | "
                     f"{r['series_better']}/{r['n_series']} | {r['event_better']}/{r['n_event']} |")
    sub_t = ["| 子集 | 比较 | 局数 | Δ Brier | 系列赛 t | 候选更好的系列赛 | 候选更好的赛事 |",
             "|---|---|---|---|---|---|---|"] + sub_t
    say_lines(["", "分子集的配对 t (Brier)", ""] + sub_t)
    ev_tab = md_event_table(df, Eh, ph, ["C", "D", "赛区 BT", "常数 c", "C 次日 (参照)"])
    say_lines(["", "逐赛事的跨赛区 Brier", ""] + ev_tab)
    rel = (md_reliability(Eh, ph["C"], cross, "可靠性 —— HOLDOUT 跨赛区, C (赛前)") + [""]
           + md_reliability(Eh, ph["D"], cross, "可靠性 —— HOLDOUT 跨赛区, D (BP 后)") + [""]
           + md_reliability(Eh, ph["C"], mvm, "可靠性 —— HOLDOUT 跨赛区四大v四大, C") + [""]
           + md_reliability(Eh, ph["C"], nmj, "可靠性 —— HOLDOUT 跨赛区含非四大, C"))
    say_lines([""] + rel)
    coef_lines = ["| 赛事 | 训练行 | 常数 c | C: a, b_o, b_r | D: a, b_o, b_r, β | cw 的 sd |", "|---|---|---|---|---|---|"]
    for name, _cut, _idx in Eh["ev_list"]:
        cc, dd = coef_h["C"][name], coef_h["D"][name]
        coef_lines.append(f"| {name} | {cc['n_train']} | {cc['c']:.4f} | "
                          + ", ".join(f"{x:+.3f}" for x in cc["beta"]) + " | "
                          + ", ".join(f"{x:+.3f}" for x in dd["beta"]) + f" | {dd['sd']:.4f} |")
    say_lines(["", "HOLDOUT 各赛事的 walk-forward 系数", ""] + coef_lines)

    # ── [5] 覆盖 (只看身份) ──
    print("\n[5] 覆盖")
    hold_all = df.index[target(df) & (df["period"] == "HOLDOUT")]
    unk = df.index[df["is_international"] & (df["period"] == "HOLDOUT") & (df["region_status"] == "unknown")]
    scored = {nm: int(ph[nm].loc[hold_all].notna().sum()) for nm in ("C", "D")}
    ev_first = df[df["is_international"]].groupby("event")["date"].min()
    xr = df[target(df)]
    seen_rows = []
    for i in hold_all:
        cut = ev_first[df.at[i, "event"]]
        before = xr[xr["date"] < cut]
        seen_l = set(before["home_blue"]) | set(before["home_red"])
        hb, hr = df.at[i, "home_blue"], df.at[i, "home_red"]
        seen_rows.append({"pair_type": df.at[i, "pair_type"], "both_seen": (hb in seen_l) and (hr in seen_l),
                          "unseen": "/".join(sorted(x for x in (hb, hr) if x not in seen_l)),
                          "roster": "roster" in (df.at[i, "home_source_blue"], df.at[i, "home_source_red"])})
    cov = pd.DataFrame(seen_rows, index=hold_all)
    cov_lines = ["| 对阵类型 | HOLDOUT 跨赛区局 | C 能出数 | D 能出数 | 两个母赛区此前都打过跨赛区国际赛 | 有一方此前没打过 (偏移来自先验 / 迁入均值) |",
                 "|---|---|---|---|---|---|"]
    for pt in ("major_v_major", "involves_nonmajor"):
        ii = cov.index[cov["pair_type"] == pt]
        cov_lines.append(f"| {pt} | {len(ii)} | {int(ph['C'].loc[ii].notna().sum())} | "
                         f"{int(ph['D'].loc[ii].notna().sum())} | {int(cov.loc[ii, 'both_seen'].sum())} | "
                         f"{int((~cov.loc[ii, 'both_seen']).sum())} |")
    cov_lines.append(f"| 合计 | {len(hold_all)} | {scored['C']} | {scored['D']} | {int(cov['both_seen'].sum())} | "
                     f"{int((~cov['both_seen']).sum())} |")
    unseen_ct = cov.loc[~cov["both_seen"], "unseen"].value_counts().to_dict()
    say_lines([""] + cov_lines)
    print(f"    此前没打过跨赛区国际赛的母赛区 (局数): {unseen_ct}; 母赛区按阵容认队的局 {int(cov['roster'].sum())}; "
          f"母赛区不明 (不出数) 的国际赛 {len(unk)} 局")
    # 有一方赛区此前没打过跨赛区国际赛的局, 单独看 (偏移完全来自先验 / 迁入均值)
    ii_un = cov.index[~cov["both_seen"]]
    unseen_tab = md_metric_table(df, Eh, ph, ["C", "D", "赛区 BT", "常数 c"],
                                 (("有一方赛区此前没打过跨赛区国际赛", np.asarray(ii_un)),
                                  ("两方都打过", np.asarray(cov.index[cov["both_seen"]]))))
    say_lines([""] + unseen_tab)

    # ── [6] 线上 Stage 1/2 的对照 ──
    prod_lines, prod_info = [], None
    if not a.skip_production:
        print("\n[6] 对照: 线上 Stage 1/2 (artifacts/) 在同一批 HOLDOUT 跨赛区局上 (只作对照, 候选不许用)")
        p1, p2, prod_info = production_stage12(df, cross)
        ph["线上 Stage 1"], ph["线上 Stage 2"] = p1, p2
        ok_idx = cross[p1.loc[cross].notna().values]
        print(f"    {prod_info['why']}  ({prod_info['secs']}s)")
        oy = df.loc[ok_idx, "year"].values
        prod_lines = md_metric_table(df, Eh, ph, ["线上 Stage 1", "线上 Stage 2", "C", "D", "赛区 BT", "常数 c"],
                                     [("线上能出数的跨赛区局", ok_idx)]
                                     + [(f"其中 {y}", ok_idx[oy == y]) for y in sorted(set(oy))])
        prod_lines += [""] + md_t_table(df, Eh, ph, (("线上 Stage 1", "常数 c"), ("线上 Stage 2", "常数 c"),
                                                    ("C", "线上 Stage 1"), ("D", "线上 Stage 2")), ok_idx)
        say_lines([""] + prod_lines)

    # ── [7] 判决 ──
    print("\n[7] 判决 (协议: HOLDOUT 跨赛区, 系列赛配对 t 对常数 > 2.5, 且对赛区 BT 不显著更差 (t > −2.5))")
    verdict = {}
    for kind, nm in (("pre", "C"), ("post", "D")):
        y = Eh["y"][cross]
        rk, rb = tstat(df, Eh, ph, nm, "常数 c"), tstat(df, Eh, ph, nm, "赛区 BT")
        v = {"candidate": nm, "n_games": int(len(cross)), "n_series": rk["n_series"],
             "brier": float(brier(ph[nm].loc[cross].values, y).mean()),
             "logloss": float(logloss(ph[nm].loc[cross].values, y).mean()),
             "constant_brier": float(brier(ph["常数 c"].loc[cross].values, y).mean()),
             "half_brier": float(brier(ph["0.5"].loc[cross].values, y).mean()),
             "regiononly_brier": float(brier(ph["赛区 BT"].loc[cross].values, y).mean()),
             "t_series_vs_constant": rk["t_series"], "t_game_vs_constant": rk["t_game"],
             "t_event_vs_constant": rk["t_event"],
             "event_blocks_positive": f"{rk['event_better']}/{rk['n_event']}",
             "series_better_vs_constant": f"{rk['series_better']}/{rk['n_series']}",
             "t_series_vs_regiononly": rb["t_series"],
             "event_blocks_positive_vs_regiononly": f"{rb['event_better']}/{rb['n_event']}"}
        v["passes"] = bool(v["t_series_vs_constant"] > T_ACCEPT and v["t_series_vs_regiononly"] > -T_ACCEPT)
        verdict[kind] = v
        print(f"    {kind:<5}{nm}: Brier {v['brier']:.4f} (常数 {v['constant_brier']:.4f}, 赛区 BT "
              f"{v['regiononly_brier']:.4f});  对常数 系列赛 t {v['t_series_vs_constant']:+.2f} "
              f"(赛事块 {v['event_blocks_positive']});  对赛区 BT t {v['t_series_vs_regiononly']:+.2f}  "
              f"→ {'通过' if v['passes'] else '不通过'}")
    rdc = tstat(df, Eh, ph, "D", "C")
    verdict["post_vs_pre"] = {"d": rdc["d"], "t_series": rdc["t_series"],
                              "event_blocks_positive": f"{rdc['event_better']}/{rdc['n_event']}"}
    print(f"    BP 项 (D vs C): Δ {rdc['d']:+.4f}, 系列赛 t {rdc['t_series']:+.2f} (赛事块 "
          f"{rdc['event_better']}/{rdc['n_event']}) —— 到 {T_ACCEPT} 才值得让 BP 后的数和赛前不同")

    # 数据末尾的赛区偏移 (按半衰期衰减到最后一局), 服务时就是这张表
    t_end = A["t"][-1]
    offs = {L: RC["oprior"][L] + (v - RC["oprior"][L]) * math.exp(-LN2 * (t_end - RC["olast"][L]) / P_C["hl"])
            for L, v in RC["o"].items()}
    print("    数据末尾的赛区偏移 (Elo 分): " + ", ".join(f"{L} {v:+.0f}" for L, v in
                                               sorted(offs.items(), key=lambda kv: -kv[1])))

    # ── 输出 ──
    out_dir = Path(a.out) if a.out else cache
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = np.concatenate([Eh["idx"], np.asarray(unk)])
    o = df.loc[keep, ["gameid", "date", "event", "series", "blue", "red", "home_blue", "home_red",
                      "region_status", "pair_type", "blue_win"]].copy()
    for nm, p in ph.items():
        o["p_" + nm] = p.loc[keep].values
    o.to_csv(out_dir / "gate_cross_region_holdout_preds.csv", index=False, encoding="utf-8")
    res = {"verdict": verdict, "dev_table": dev_tab, "dev_t": dev_t, "holdout_table": hold_tab,
           "holdout_t": hold_t, "holdout_sub_t": sub_t, "event_table": ev_tab, "reliability": rel,
           "coefs": coef_lines, "coverage": cov_lines, "coverage_unseen": unseen_ct,
           "coverage_unseen_table": unseen_tab, "production": prod_lines, "production_info": prod_info,
           "offsets_end": offs, "unknown_holdout_games": int(len(unk)),
           "data_through": str(df["date"].max().date())}
    (out_dir / "gate_cross_region_result.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\n  结果 {out_dir / 'gate_cross_region_result.json'}; 逐局预测 gate_cross_region_holdout_preds.csv")
    print(f"  总用时 {time.time() - T0:.0f}s")


if __name__ == "__main__":
    main()
