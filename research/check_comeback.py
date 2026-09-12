"""
翻盘率校验 —— 用数据裁定 agent 的数字声明
==========================================
辩论中质疑者断言:
    "LEC 历史翻盘率在 20 分钟 +2800 金差时约 22–25%"
    "78.9% 隐含的翻盘率仅 21%, 已贴近下界"

它没有给任何来源 —— 这个数字是它编的。而协调者的 agent_errors 是空的,
没抓到。

但这个数字可以直接验证: 你手上有 8390 场带时间切片的数据。

本脚本回答三件事:
  1. 各时间点、各经济差档位的真实翻盘率是多少
  2. 20 分钟 +2800 附近的实测值, 和模型输出对不对得上
  3. 分赛区看有没有差异 (质疑者特指 LEC)

用法: python check_comeback.py
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
from ingame_model import load, build, SLICES

YEARS = (2022, 2023, 2024, 2025, 2026)
# 质疑者的声明
CLAIM_T, CLAIM_GD = 20, 2800
CLAIM_LO, CLAIM_HI = 0.22, 0.25
MODEL_P = 0.789          # 那次辩论里局内模型的输出


def wilson(k, n, z=1.96):
    """比例的 Wilson 95% 区间 —— 小样本下比正态近似可靠"""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    print("=" * 72)
    print("  翻盘率校验")
    print("=" * 72)

    print("\n[1/3] 加载")
    files = [str(_ROOT / "data" / f"{y}_LoL_esports_match_data_from_OraclesElixir.csv")
             for y in YEARS]
    df = load(files)
    mdf, _ = build(df, verbose=False)
    print(f"    {mdf['gameid'].nunique()} 场 → {len(mdf)} 个快照")

    # 统一到「领先方视角」: 领先方是否最终获胜
    m = mdf.copy()
    m["lead_gd"] = m["golddiff"].abs()
    m["lead_won"] = ((m["golddiff"] > 0) == (m["y"] == 1)).astype(int)
    m = m[m["golddiff"] != 0]

    print(f"\n{'=' * 72}")
    print(f"  [2/3] 领先方胜率 × 经济差档位 × 时间点")
    print(f"{'=' * 72}")
    bands = [(0, 500), (500, 1000), (1000, 1500), (1500, 2000),
             (2000, 2500), (2500, 3100), (3100, 4000),
             (4000, 6000), (6000, 99999)]
    print(f"  {'经济差':<14}" + "".join(f"T={T:<11}" for T in SLICES))
    print(f"  {'':14}" + "".join(f"{'胜率  n':<13}" for _ in SLICES))
    print(f"  {'─' * 66}")
    for lo, hi in bands:
        label = f"{lo/1000:.1f}–{hi/1000:.1f}k" if hi < 99999 else f"{lo/1000:.0f}k+"
        row = f"  {label:<14}"
        for T in SLICES:
            s = m[(m["T"] == T) & (m["lead_gd"] >= lo) & (m["lead_gd"] < hi)]
            if len(s) < 20:
                row += f"{'—':<13}"
            else:
                row += f"{s['lead_won'].mean():>5.1%} {len(s):>5}  "
        print(row)

    # ── 质疑者的具体声明 ──
    print(f"\n{'=' * 72}")
    print(f"  [3/3] 裁定: T={CLAIM_T}, 经济差 ≈ {CLAIM_GD}")
    print(f"{'=' * 72}")
    s = m[(m["T"] == CLAIM_T) & (m["lead_gd"].between(2500, 3100))]
    n, k = len(s), int(s["lead_won"].sum())
    if n < 20:
        print(f"  样本不足 (n={n})"); return
    win = k / n
    comeback = 1 - win
    lo, hi = wilson(n - k, n)

    print(f"  样本            {n} 场 (2500–3100 金币差)")
    print(f"  领先方胜率      {win:.1%}")
    print(f"  实测翻盘率      {comeback:.1%}   [95% CI {lo:.1%}–{hi:.1%}]")
    print()
    print(f"  质疑者声称      22.0%–25.0%")
    print(f"  模型隐含        {1-MODEL_P:.1%}   (输出 {MODEL_P:.1%})")
    print()
    claim_in = lo <= CLAIM_LO <= hi or lo <= CLAIM_HI <= hi or (CLAIM_LO <= lo and hi <= CLAIM_HI)
    model_in = lo <= (1 - MODEL_P) <= hi
    print(f"  质疑者的区间     {'✓ 落在置信区间内' if claim_in else '✗ 落在置信区间外'}")
    print(f"  模型的隐含值     {'✓ 落在置信区间内' if model_in else '✗ 落在置信区间外'}")
    print()
    if claim_in and model_in:
        print(f"  → 两者都在噪声范围内, 无法分辨。质疑者的「虚高 3-5pp」缺乏依据,")
        print(f"    但也不能说它错 —— 这个精度下判不了。")
    elif model_in and not claim_in:
        print(f"  → 模型准确, 质疑者的数字站不住。它编了一个没有来源的区间。")
    elif claim_in and not model_in:
        print(f"  → 质疑者对了, 模型在这个档位系统性高估。")
    else:
        print(f"  → 两者都偏离实测值。")

    # ── 分赛区 (质疑者特指 LEC) ──
    print(f"\n  分赛区 (T={CLAIM_T}, 2500–3100)")
    print(f"  {'─' * 52}")
    print(f"  {'赛区':<8}{'n':>6}{'领先方胜率':>12}{'翻盘率':>10}{'95% CI':>18}")
    for lg in ["LPL", "LCK", "LEC", "LCS"]:
        ls = s[s["league"] == lg]
        if len(ls) < 15:
            print(f"  {lg:<8}{len(ls):>6}   样本不足"); continue
        lk = int(ls["lead_won"].sum()); ln = len(ls)
        llo, lhi = wilson(ln - lk, ln)
        print(f"  {lg:<8}{ln:>6}{lk/ln:>12.1%}{1-lk/ln:>10.1%}"
              f"{f'{llo:.0%}–{lhi:.0%}':>18}")

    print(f"\n{'=' * 72}")
    print(f"  这个结果的用途")
    print(f"{'=' * 72}")
    print("""
  agent 在辩论里给出的每一个数字都应该能被这样检验。质疑者那句
  「22–25%」听起来很专业, 但它没有来源 —— 而协调者的 agent_errors
  是空的, 说明裁定层没有识别出「无来源的定量断言」这一类错误。

  可以做的改进: 在协调者的 prompt 里加一条 —— 任何带具体数字的
  断言, 若 agent 未标注来源, 一律列入 agent_errors。
    """)


if __name__ == "__main__":
    main()
