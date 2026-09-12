"""
把回填快照并进 Stage 4 的训练表
================================

Oracle's Elixir 只有 at10/15/20/25 四个时间点, 所以模型在第 10 分钟之前和
第 25 分钟之后全是外推。research/backfill_late.py 从 lolesports 的帧逐分钟
重建了 357 局, 覆盖第 5 分钟到终局 —— 这个脚本把两者合到一起。

怎么合
------
回填只采了局势数据 (经济/补刀/人头/总经济), **没有英雄阵容**, 而模型要用
阵容算强势期指数。解法是"借": 赛前特征、阵容强势期、赛区哑变量、标签这些
**在一局之内不随时间变化**, 按 oe_gameid 从 build() 的产出直接借过来即可。
实测 357 局里 339 局 (95%) 能对上, 21 个可借列全部验证为每局恒定。

    借自 build()  league y blue_team red_team is_* b_scaling r_scaling
                  scaling_diff scaling_sum diff_pre_*        (21 列)
    回填提供      T golddiff csdiff blue_kills red_kills gold_total
    现算          killdiff killsum golddiff_norm x_gold_scaling
                  x_goldnorm_scaling lead_by_early_comp
    拿不到        xpdiff x_xp_scaling  -> 只能训 live 变体

派生列的公式必须和 ingame_model.build() 逐字一致 (见该文件 265-285 行),
差一点就是训练-服务偏移。

用法:
    python research/merge_backfill.py                 生成并汇报
    python research/merge_backfill.py --out merged.parquet
"""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
import ingame_model as IM

BACKFILL = _ROOT / "backfill" / "snapshots.jsonl"

# 一局之内恒定、可以从 build() 借过来的列
BORROW_PREFIX = ("diff_pre_", "is_")
BORROW_EXACT = ("league", "y", "blue_team", "red_team",
                "b_scaling", "r_scaling", "scaling_diff", "scaling_sum")


def borrow_cols(mdf: pd.DataFrame) -> list[str]:
    return [c for c in mdf.columns
            if c.startswith(BORROW_PREFIX) or c in BORROW_EXACT]


# ── 目标资源 ──────────────────────────────────────────────────────────
# **只有回填有这些列, OE 结构上给不了** (它的 60 个分时刻列全是经济/经验/
# 补刀/KDA, 塔龙大龙只有整局总数)。所以它们不能进 OE+回填的混合训练 ——
# 那样 "有值 / NaN" 会直接变成 "回填 / OE" 的来源标签, 模型学到的是数据
# 来源而不是局势。想测它们的价值, 只能在回填子集内部做配对比较。
OBJECTIVE_COLS = ["towerdiff", "dragondiff", "barondiff", "inhibdiff",
                  "dragonsum", "souldiff", "baronbuffdiff"]

# 大龙 buff 的持续时间 (分钟)。**这是个版本相关的常数, 不是实测值** ——
# 采样每分钟一次, 所以这里只能分辨到分钟。大龙的价值几乎全在这个窗口里:
# 累计击杀数分不出"刚拿到"和"二十分钟前拿过早就没了", 而这两者对胜率的
# 含义完全不同。
BARON_BUFF_MIN = 3.0

SOUL_DRAGONS = 4          # 拿满 4 条小龙成龙魂


def add_objective_features(df: pd.DataFrame) -> pd.DataFrame:
    """目标资源的差值 + 龙魂 + 大龙 buff 是否在手。

    baronbuff 要按局的时间序列算: 帧只给累计击杀数, 得先找出这个数**跳增**
    的时刻, 才知道 buff 什么时候开始。
    """
    df = df.sort_values(["game_id", "minute"]).copy()
    for side in ("blue", "red"):
        for c in (f"{side}_towers", f"{side}_dragons", f"{side}_barons",
                  f"{side}_inhibitors"):
            df[c] = pd.to_numeric(df.get(c), errors="coerce").fillna(0)

    df["towerdiff"] = df.blue_towers - df.red_towers
    df["dragondiff"] = df.blue_dragons - df.red_dragons
    df["barondiff"] = df.blue_barons - df.red_barons
    df["inhibdiff"] = df.blue_inhibitors - df.red_inhibitors
    df["dragonsum"] = df.blue_dragons + df.red_dragons
    df["souldiff"] = ((df.blue_dragons >= SOUL_DRAGONS).astype(int)
                      - (df.red_dragons >= SOUL_DRAGONS).astype(int))

    for side in ("blue", "red"):
        col, out = f"{side}_barons", f"{side}_baronbuff"
        got = df.groupby("game_id")[col].diff().fillna(df[col]) > 0
        # 每次跳增记下当时的分钟, 向前填充 -> 每一行的"上一次大龙在几分钟前"
        when = df["minute"].where(got)
        when = when.groupby(df["game_id"]).ffill()
        age = df["minute"] - when
        df[out] = ((age >= 0) & (age < BARON_BUFF_MIN)).astype(int)
    df["baronbuffdiff"] = df.blue_baronbuff - df.red_baronbuff
    return df


def derive(m: dict) -> dict:
    """派生列。公式照抄 ingame_model.build(), 不能有任何出入。"""
    gdif = m["golddiff"]
    sc_diff = m.get("scaling_diff")
    gold_total = m.get("gold_total")

    m["killdiff"] = m["blue_kills"] - m["red_kills"]
    m["killsum"] = m["blue_kills"] + m["red_kills"]
    m["golddiff_norm"] = (gdif / gold_total
                          if gold_total and gold_total > 0 else np.nan)
    if pd.notna(sc_diff):
        m["x_gold_scaling"] = gdif * sc_diff
        m["x_goldnorm_scaling"] = (m["golddiff_norm"] * sc_diff
                                   if pd.notna(m["golddiff_norm"]) else np.nan)
        m["lead_by_early_comp"] = gdif * (-sc_diff)
    else:
        m["x_gold_scaling"] = m["x_goldnorm_scaling"] = np.nan
        m["lead_by_early_comp"] = np.nan
    # 帧里没有 XP —— 留 NaN, 只有 live 变体用得上这批数据
    m["xpdiff"] = np.nan
    m["x_xp_scaling"] = np.nan
    return m


def main():
    ap = argparse.ArgumentParser()
    # 用 csv.gz 而不是 parquet —— 不为一个中间产物给项目加 pyarrow 依赖
    ap.add_argument("--out", default=str(_ROOT / "backfill" / "merged.csv.gz"))
    a = ap.parse_args()

    if not BACKFILL.exists():
        print("没有回填数据, 先跑 research/backfill_late.py"); return 1

    print("[1/4] 读回填快照")
    bf = pd.DataFrame([json.loads(l) for l in
                       open(BACKFILL, encoding="utf-8")])
    bf = bf.drop_duplicates(["game_id", "T"])
    print(f"      {bf.game_id.nunique()} 局 / {len(bf)} 条")

    print("[2/4] 构建 OE 快照表 (拿可借的列)")
    files = sorted(glob.glob(str(_ROOT / "data" / "20*_from_OraclesElixir.csv")))
    mdf, _sc = IM.build(IM.load(files), verbose=False)
    cols = borrow_cols(mdf)
    # 每局取一行就够 —— 这些列在局内恒定 (backfill_late 里验证过)
    const = mdf.drop_duplicates("gameid").set_index("gameid")[cols]
    print(f"      {len(const)} 局, 可借 {len(cols)} 列")

    print("[3/4] 按 oe_gameid 借列 + 现算派生列")
    joined = bf.join(const, on="oe_gameid", how="inner", rsuffix="_oe")
    miss = bf.oe_gameid.nunique() - joined.oe_gameid.nunique()
    print(f"      对上 {joined.game_id.nunique()} 局, 丢弃 {miss} 局 (OE 里没有)")

    rows = [derive(r) for r in joined.to_dict("records")]
    out = pd.DataFrame(rows)
    # 和 build() 对齐列名: 它用 gameid, 回填用 game_id
    out["gameid"] = out["oe_gameid"]
    out["source"] = "backfill"
    have_obj = all(c in out.columns for c in
                   ("blue_towers", "blue_barons", "blue_dragons"))
    if have_obj:
        out = add_objective_features(out)
        print(f"      目标资源列: {', '.join(OBJECTIVE_COLS)}")
    else:
        print("      这批回填数据没有目标资源字段 (是旧版采的), 跳过那几列")
    keep = [c for c in mdf.columns if c in out.columns]
    extra = [c for c in OBJECTIVE_COLS if c in out.columns]
    out = out[keep + extra + ["source"]]

    print("[4/4] 和 OE 表拼接")
    base = mdf.copy()
    base["source"] = "oe"
    # 同一局同一时刻只留一份, 优先 OE (它的 at10/15/20/25 是官方口径)
    both = pd.concat([base, out], ignore_index=True)
    both["Tr"] = both["T"].round().astype(int)
    before = len(both)
    both = both.sort_values("source").drop_duplicates(["gameid", "Tr"], keep="last")
    print(f"      {before} -> {len(both)} 条 (去掉 {before-len(both)} 条重叠)")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    both.drop(columns=["Tr"]).to_csv(a.out, index=False, compression="gzip")

    print(f"\n{'=' * 62}")
    print(f"  合并结果: {both.gameid.nunique()} 局 / {len(both)} 条  ->  {a.out}")
    print("=" * 62)
    print("\n按分钟段 (只看回填带来的部分):")
    bfp = both[both.source == "backfill"]
    for lo, hi in [(3, 10), (10, 15), (15, 20), (20, 25),
                   (25, 30), (30, 35), (35, 40), (40, 99)]:
        n = ((bfp.Tr >= lo) & (bfp.Tr < hi)).sum()
        if n:
            mark = "   ← OE 完全没有" if (lo >= 25 or lo < 10) else ""
            print(f"  {lo:>2}-{hi:<3}分钟 {n:>6} 条{mark}")
    print(f"\n经验差覆盖: OE 部分 {base.xpdiff.notna().mean():.1%}, "
          f"回填部分 0% —— 所以这批只能训 ingame_live 变体")
    return 0


if __name__ == "__main__":
    sys.exit(main())
