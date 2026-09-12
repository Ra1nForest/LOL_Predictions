"""
博彩市场的算术
==============
剥抽水、算期望值、算收盘线价值。**只有算术, 没有判决** —— 判决在
gate_vs_market.py 里, 和项目其余部分共用 harness 的那一套口径。

单独成文件是因为这几个函数会被三处用到 (取数、闸门、留档), 而它们各自
都有一个"看着对其实错"的写法, 抄三遍必然抄漏一处 —— 和 harness.paired_t
当初被内联抄三份是同一个教训。
"""
from __future__ import annotations


def implied(odds: float) -> float:
    """欧洲赔率 -> 隐含概率。1/odds, 没别的。"""
    if odds <= 1.0:
        raise ValueError(f"赔率必须大于 1, 收到 {odds}")
    return 1.0 / odds


def devig(odds: list[float]) -> list[float]:
    """剥掉抽水: 把一组互斥且穷尽的赔率还原成和为 1 的概率。

    **这一步不能省。** 博彩公司挂出的隐含概率加起来是 1.03-1.08, 多出来的
    就是抽成。直接拿 1/赔率 当市场预测去比 Brier, 市场会被系统性判成
    "过度自信" —— 而那是抽水造成的假象, 不是它预测得差。举个数:
    两边都挂 1.90, 隐含各 0.526, 合计 1.053; 不剥的话每一边都被高估 5.3%。

    用**乘法归一** (按比例摊掉)。这是最常见的做法, 但要知道它的偏差:
    它假设抽水在两边是等比例的, 而实测冷门那一侧通常被多抽一点
    (favorite-longshot bias)。所以剥完之后**冷门的真实概率会被略微高估**。
    要更准得用 Shin 或 power 方法, 那需要额外参数; 在"模型能不能打平市场"
    这个问题上, 乘法归一的偏差远小于我们要检测的效应。
    """
    ps = [implied(o) for o in odds]
    s = sum(ps)
    if s <= 0:
        raise ValueError("赔率无效")
    return [p / s for p in ps]


def overround(odds: list[float]) -> float:
    """抽水率。1.05 表示庄家留了 5% —— 拿来判断这家盘口值不值得信。"""
    return sum(implied(o) for o in odds) - 1.0


def ev(p_model: float, odds: float) -> float:
    """按模型概率算这一注的期望收益率 (每 1 元本金)。

    正数表示模型认为有价值。**注意这是"按模型算"** —— 模型错了它就是错的,
    正 EV 不等于真有优势, 这正是要拿 CLV 和长期盈亏去检验的东西。
    """
    return p_model * (odds - 1.0) - (1.0 - p_model)


def kelly(p_model: float, odds: float) -> float:
    """凯利比例。返回 0 表示不该下。

    留在这里是为了**算 EV 时能顺手看一眼仓位有多离谱** —— 凯利算出 40%
    仓位通常不是发现了金矿, 是模型概率错得离谱。实际记账一律用固定注,
    见 gate_vs_market 里的说明。
    """
    b = odds - 1.0
    f = (p_model * b - (1.0 - p_model)) / b
    return max(0.0, f)


def clv(odds_taken: float, odds_close: float) -> float:
    """收盘线价值。正数 = 你拿到的赔率比收盘时好。

    为什么它比盈亏重要: 一场比赛只给出一个 0/1 的结果, 信息量极小; 而
    "下注时的价格 vs 收盘价格"是个连续量, **收敛快一个数量级**。
    职业博彩圈拿它当"你到底有没有优势"的主指标, 就是这个原因。

    用去抽水后的概率差来表示, 而不是赔率比 —— 概率差可以直接和模型的
    校准误差放在同一个尺度上看。
    """
    return implied(odds_close) - implied(odds_taken)


def brier(p: float, y: int) -> float:
    """越小越好。"""
    return (p - y) ** 2


def logloss(p: float, y: int, eps: float = 1e-12) -> float:
    """越小越好。夹一下, 免得 p=0 而 y=1 时炸成 inf。"""
    p = min(1 - eps, max(eps, p))
    return -(y * __import__("math").log(p) + (1 - y) * __import__("math").log(1 - p))
