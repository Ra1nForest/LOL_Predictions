"""
模型 vs 盘口 —— 判决
======================
问题: 模型的胜率预测, 到底比博彩公司的盘口好还是差?

**这个问题不需要下注就能回答**, 而且这是全套里统计功效最高的一步:
把盘口去掉抽水换算成概率, 和模型的概率放在同一批比赛上比 Brier /
log-loss。几百场就能出结论; 而真金白银的盈亏要几千场才能把噪声压下去
—— 一场比赛只产出一个 0/1, 信息量太小。

所以顺序是死的:
    ① 比 Brier (本文件, 不下注)
    ② 比 CLV   (纸面, 见 account())
    ③ 才谈盈亏

判据沿用全项目那一套: research/harness.paired_t, |t| > 2.5。
不另立标准 —— 一旦这里放宽到 |t| > 2, 它就会变成"换个尺子直到过关"。

━━━ 两个必须做对的地方 ━━━

**抽水一定要剥。** 见 market.devig 的说明。不剥的话市场会被系统性判成
过度自信, 模型白捡一个不存在的胜利。

**时间必须对齐。** 拿第 20 分钟的局内概率去比赛前收盘线, 比的是两个不同
的信息集 —— 那当然是我们赢, 但赢的是"我们看到了 20 分钟的比赛"这件事,
不是模型。每条记录都带 t, 对齐由取数那一层负责保证, 这里只做校验并在
可疑时剔除。

━━━ 用法 ━━━

    python research/gate_vs_market.py --selftest        # 不需要任何数据
    python research/gate_vs_market.py --joined research/odds_joined.jsonl

joined 文件每行一个**盘口实例** (一场比赛的一个市场):

    {"t": "2026-08-20T11:00:00+00:00", "game_id": "...", "league": "LPL",
     "market": "moneyline", "outcomes": ["blue", "red"],
     "odds": [1.85, 1.95], "p_model": [0.56, 0.44], "y": 0,
     "t_model": "2026-08-20T10:58:00+00:00"}

y 是实际发生的那个 outcome 的下标。odds / p_model / outcomes 三者等长,
且这一组 outcome 必须**互斥且穷尽** —— devig 靠这个前提成立。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from harness import from_folds, gate  # noqa: E402
from market import brier, clv, devig, ev, logloss, overround  # noqa: E402

# 下注时刻和盘口时刻差超过这么久, 就不是同一个信息集了
MAX_ALIGN_SEC = 15 * 60
# 抽水高于这个数的盘口直接不要 —— 那种盘口的隐含概率噪声太大, 而且
# 没人会真去下。软欧盘常见 5-8%, 亚盘和 Pinnacle 在 2-3%。
MAX_OVERROUND = 0.12


# ══════════════════════════════════════════════════════════
#  读入与校验
# ══════════════════════════════════════════════════════════

def _ts(s):
    return datetime.fromisoformat(str(s).replace("Z", "+00:00"))


def load_joined(path):
    """读 joined 文件, 顺手把明显不能用的行剔掉并说明原因。

    宁可在这里吵, 也不要让一条错行悄悄进到 Brier 里 —— 这类错误从结果上
    看不出来, 只会让某个市场无缘无故显得好或差。
    """
    rows, drop = [], defaultdict(int)
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        o, p = r["odds"], r["p_model"]
        if len(o) != len(p) or len(o) < 2:
            drop["odds/p_model 长度不匹配"] += 1
            continue
        if not (0 <= r["y"] < len(o)):
            drop["y 越界"] += 1
            continue
        if abs(sum(p) - 1.0) > 1e-3:
            drop["模型概率不为 1 (这组 outcome 不互斥穷尽?)"] += 1
            continue
        ov = overround(o)
        if ov < -1e-9:
            drop["抽水为负 (套利? 更可能是赔率抄错)"] += 1
            continue
        if ov > MAX_OVERROUND:
            drop[f"抽水 > {MAX_OVERROUND:.0%}"] += 1
            continue
        if r.get("t_model"):
            gap = abs((_ts(r["t"]) - _ts(r["t_model"])).total_seconds())
            if gap > MAX_ALIGN_SEC:
                drop["模型时刻与盘口时刻相差过大"] += 1
                continue
        r["_t"] = _ts(r["t"])
        r["_q"] = devig(o)          # 去抽水后的市场概率
        rows.append(r)
    rows.sort(key=lambda r: r["_t"])
    return rows, dict(drop)


# ══════════════════════════════════════════════════════════
#  判决
# ══════════════════════════════════════════════════════════

def fold_scores(rows, n_folds, metric):
    """按时间切 fold, 每 fold 返回 (模型分, 市场分)。

    切分按**时间顺序**而不是随机: 盘口的准确度会随赛季变化 (阵容变动、
    版本、这个联赛有多少人下注), 随机切会把不同时期混在一起, 让配对
    失去它本来要消掉的那个东西 —— "这一段时期有多难预测"。
    """
    n = len(rows)
    if n < n_folds * 20:
        raise SystemExit(f"样本太少: {n} 行切 {n_folds} 折, 每折不到 20 个。"
                         f"先攒数据, 别在这个量级上做判决。")
    edges = [round(n * k / n_folds) for k in range(n_folds + 1)]
    fm, fq = [], []
    for k in range(n_folds):
        chunk = rows[edges[k]:edges[k + 1]]
        m = [metric(r["p_model"][r["y"]], 1) for r in chunk]
        q = [metric(r["_q"][r["y"]], 1) for r in chunk]
        fm.append(sum(m) / len(m))
        fq.append(sum(q) / len(q))
    return fm, fq


def judge(rows, name, n_folds=6):
    """对一个市场类型出判决。返回 True = 模型**打赢**了市场。

    注意方向: Brier 越小越好, 而 gate() 认的是"越大越好"。所以喂进去的是
    **负 Brier**。写反了会得到一个方向完全颠倒且毫无征兆的结论 —— 这是
    这个文件里最容易出的错, 所以取负这一步只在这里做一次。
    """
    ok = True
    for label, metric in (("Brier", brier), ("log-loss", logloss)):
        fm, fq = fold_scores(rows, n_folds, metric)
        base = from_folds([-x for x in fq], f"市场 ({label})")
        cand = from_folds([-x for x in fm], f"模型 ({label})")
        print(f"\n  ── {name} · {label} · n={len(rows)} ──")
        ok &= gate(base, cand, f"模型是否优于盘口 ({label})")
    return ok


# ══════════════════════════════════════════════════════════
#  记账: EV 与 CLV
# ══════════════════════════════════════════════════════════

def account(rows, name, ev_floor=0.03):
    """按**固定注**记模型自认为有价值的那些盘, 外加 CLV。

    为什么固定注而不是凯利: 凯利的仓位对概率误差极其敏感 —— 概率错 2 个
    百分点, 仓位能差一倍, 于是最后的盈亏曲线主要在反映"仓位算法放大了
    哪几场", 而不是"模型有没有优势"。固定注的曲线才读得懂。

    ev_floor 不是随便定的 3%: 它应该压过这个市场类型自身的校准误差, 否则
    筛出来的"价值"全是校准噪声。默认值只是占位, 真跑的时候要用该类型
    实测的 ECE 去定。
    """
    picks = []
    for r in rows:
        best = max(range(len(r["odds"])),
                   key=lambda i: ev(r["p_model"][i], r["odds"][i]))
        e = ev(r["p_model"][best], r["odds"][best])
        if e >= ev_floor:
            picks.append((r, best, e))
    if not picks:
        print(f"\n  {name}: EV >= {ev_floor:.0%} 的盘一个都没有 —— "
              f"模型和这家盘口基本一致。")
        return

    pnl = sum((r["odds"][i] - 1.0) if r["y"] == i else -1.0 for r, i, _ in picks)
    hit = sum(1 for r, i, _ in picks if r["y"] == i)
    have_close = [(r, i) for r, i, _ in picks if r.get("odds_close")]
    print(f"\n  {name}: 下注 {len(picks)} 笔 / 共 {len(rows)} 个盘"
          f"  命中 {hit}/{len(picks)} = {hit/len(picks):.1%}")
    print(f"      模型自认为的平均 EV {sum(e for _, _, e in picks)/len(picks):+.2%}")
    print(f"      固定注盈亏 {pnl:+.2f} 个单位  收益率 {pnl/len(picks):+.2%}")
    if have_close:
        c = [clv(r["odds"][i], r["odds_close"][i]) for r, i in have_close]
        m = sum(c) / len(c)
        pos = sum(1 for x in c if x > 0) / len(c)
        print(f"      CLV {m:+.4f} (概率单位)  为正的占 {pos:.1%}  n={len(c)}")
        print(f"      → CLV 才是主指标。盈亏这一行在几千笔之前基本是噪声。")
    else:
        print(f"      (没有收盘赔率 → 算不了 CLV。取数那层要把收盘价一起存,"
              f" 否则最有用的那个指标从一开始就缺了。)")


# ══════════════════════════════════════════════════════════
#  自检 —— 不需要任何数据, 验证的是算术和判决方向
# ══════════════════════════════════════════════════════════

def _synth(n, model_sd, market_sd, vig=0.045, seed=0):
    """造一批比赛: 真概率 p, 市场和模型各自带噪声地看到它。

    噪声小的那一方**应该**赢 —— 拿这个去验证判决方向有没有写反。
    """
    rng = random.Random(seed)
    t0 = datetime(2026, 1, 1)
    rows = []
    for k in range(n):
        p = min(0.92, max(0.08, rng.gauss(0.5, 0.16)))
        # 夹到 0.93 而不是 0.98: 加上抽水之后 1/(q*s) 必须仍然 > 1, 否则造出
        # 的是一个现实中不存在的赔率, implied() 会直接拒收。
        q = min(0.93, max(0.07, p + rng.gauss(0, market_sd)))
        pm = min(0.98, max(0.02, p + rng.gauss(0, model_sd)))
        # 把市场概率加上抽水还原成赔率 —— 反过来走一遍 devig
        s = 1.0 + vig
        o = [1.0 / (q * s), 1.0 / ((1 - q) * s)]
        y = 0 if rng.random() < p else 1
        rows.append({"t": (t0 + timedelta(hours=k)).isoformat(),
                     "game_id": f"g{k}", "league": "TEST",
                     "market": "moneyline", "outcomes": ["blue", "red"],
                     "odds": o, "p_model": [pm, 1 - pm], "y": y,
                     "_t": t0 + timedelta(hours=k), "_q": devig(o)})
    return rows


def selftest():
    print("=" * 66)
    print("自检 1 —— 剥抽水")
    print("=" * 66)
    o = [1.90, 1.90]
    print(f"  两边都 1.90: 隐含 {1/1.90:.4f} + {1/1.90:.4f} = {2/1.90:.4f}")
    print(f"  抽水 {overround(o):.2%}   剥完 {devig(o)[0]:.4f} / {devig(o)[1]:.4f}")
    assert abs(sum(devig(o)) - 1) < 1e-12
    assert abs(devig(o)[0] - 0.5) < 1e-12
    o = [1.50, 2.70]
    q = devig(o)
    print(f"  1.50 / 2.70: 抽水 {overround(o):.2%}  剥完 {q[0]:.4f} / {q[1]:.4f}")
    print(f"  不剥的话热门会被记成 {1/1.50:.4f} —— 高出 {1/1.50-q[0]:+.4f},")
    print(f"  而我们要检测的效应量本身也就这个量级。所以这一步不是讲究, 是必需。")
    assert abs(sum(q) - 1) < 1e-12

    print("\n" + "=" * 66)
    print("自检 2 —— 判决方向 (模型比市场差)")
    print("=" * 66)
    print("  造 900 场: 市场噪声 0.05, 模型噪声 0.12。模型应该被判**输**。")
    bad = judge(_synth(900, model_sd=0.12, market_sd=0.05, seed=1), "劣质模型")
    print(f"\n  → 判决 {'采纳(错!)' if bad else '未采纳 ✓ 方向正确'}")
    assert not bad, "判决方向写反了"

    print("\n" + "=" * 66)
    print("自检 3 —— 判决方向 (模型比市场好)")
    print("=" * 66)
    print("  同样 900 场, 噪声对调。模型应该被判**赢**。")
    good = judge(_synth(900, model_sd=0.05, market_sd=0.12, seed=1), "优质模型")
    print(f"\n  → 判决 {'采纳 ✓ 方向正确' if good else '未采纳(错!)'}")
    assert good, "真实优势没被检出"

    print("\n" + "=" * 66)
    print("自检 4 —— 挑最优的那个盘, 会凭空造出优势")
    print("=" * 66)
    selection_trap()

    print("\n" + "=" * 66)
    print("自检全部通过。算术和判决方向没问题。")
    print("=" * 66)


def selection_trap(n=1200, k_markets=4, model_sd=0.06, vig=0.045, reps=25):
    """**市场完全正确、模型纯噪声**时, "挑 EV 最高的那个盘"看起来什么样。

    这是这套东西最容易翻车的地方, 所以用数字摆出来而不是写句警告:
    模型对每个盘各犯一点独立的错, 挑 EV 最高的等于**专挑模型错得最有利
    的那个方向**。市场是完美的, 真实优势严格为零 (还倒欠抽水), 但屏幕上
    会显示一个漂亮的正 EV。

    README 里那个 +3.1pp 的假发现是同一类错误 —— 在候选里挑最好的一个,
    然后忘了自己挑过。

    跑 reps 轮而不是一轮: 单轮 1200 场的收益率标准误就有 3 个百分点上下,
    一轮跑出 +0.1% 会让人误以为"挑一挑也没亏" —— 那只是噪声压过了 -4.3%
    的真实亏损。把这个噪声本身量出来, 正好就是"先看 CLV 再看盈亏"的理由。
    """
    per_rep, ev_all, rate = [], [], []
    for rep in range(reps):
        rng = random.Random(1000 + rep)
        tot_ev, tot_pnl, bets = 0.0, 0.0, 0
        for _ in range(n):
            p = min(0.93, max(0.07, rng.gauss(0.5, 0.16)))
            s = 1.0 + vig
            o = [1.0 / (p * s), 1.0 / ((1 - p) * s)]
            cands = []
            for _m in range(k_markets):
                pm = min(0.98, max(0.02, p + rng.gauss(0, model_sd)))
                cands.append((ev(pm, o[0]), 0))
                cands.append((ev(1 - pm, o[1]), 1))
            e, side = max(cands)
            if e < 0.03:
                continue
            bets += 1
            tot_ev += e
            y = 0 if rng.random() < p else 1
            tot_pnl += (o[side] - 1.0) if y == side else -1.0
        if bets:
            per_rep.append(tot_pnl / bets)
            ev_all.append(tot_ev / bets)
            rate.append(bets / n)

    mean = sum(per_rep) / len(per_rep)
    sd = (sum((x - mean) ** 2 for x in per_rep) / (len(per_rep) - 1)) ** 0.5
    truth = 1.0 / (1.0 + vig) - 1.0
    print(f"  {k_markets} 个市场 × 2 边 = {k_markets*2} 个候选, 每场挑 EV 最高的一个")
    print(f"  模型概率 = 真概率 + N(0, {model_sd}) —— **零真实优势**")
    print(f"  {reps} 轮 × {n} 场, 每轮下注 {sum(rate)/len(rate):.0%}")
    print()
    print(f"  模型自认为的平均 EV  {sum(ev_all)/len(ev_all):+.2%}   ← 看着像座金矿")
    print(f"  实际收益率           {mean:+.2%}  (轮间 SD {sd:.2%})")
    print(f"  解析真值             {truth:+.2%}   ← 就是抽水, 一分不多一分不少")
    print()
    print(f"  顺带看清另一件事: 单轮 {n} 场的收益率标准误就有 {sd:.1%}。")
    print(f"  也就是说跑一个赛季的盘, 只看盈亏根本分不出 -4% 和 0% ——")
    print(f"  这就是为什么 CLV 必须排在盈亏前面。")
    print()
    print("  对策三条, 缺一不可:")
    print("   1. **分市场类型各记各的账**, 永远不要合成一条 [最佳选择] 曲线")
    print("   2. EV 门槛按该类型**实测的校准误差**定, 不是拍一个 3%")
    print("   3. 先看 CLV 再看盈亏 —— CLV 骗不了, 它对的是市场的最终价格")


# ══════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joined", help="joined 盘口文件 (jsonl)")
    ap.add_argument("--selftest", action="store_true", help="不用数据, 验算术和方向")
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--ev-floor", type=float, default=0.03)
    a = ap.parse_args()

    if a.selftest or not a.joined:
        if not a.joined:
            print("(没给 --joined, 跑自检)\n")
        selftest()
        return

    rows, drop = load_joined(a.joined)
    print(f"读入 {len(rows)} 行")
    for why, k in sorted(drop.items(), key=lambda x: -x[1]):
        print(f"  剔除 {k:>5} 行: {why}")
    if not rows:
        raise SystemExit("没有可用行。")

    by_market = defaultdict(list)
    for r in rows:
        by_market[r.get("market", "?")].append(r)

    # 分市场类型判, 不合并。合并等于把自检 4 演示的那个陷阱踩一遍。
    for name, sub in sorted(by_market.items()):
        print("\n" + "=" * 66)
        print(f"市场类型: {name}   ({len(sub)} 个盘)")
        print("=" * 66)
        try:
            judge(sub, name, a.folds)
        except SystemExit as e:
            print(f"  跳过判决: {e}")
        account(sub, name, a.ev_floor)


if __name__ == "__main__":
    main()
