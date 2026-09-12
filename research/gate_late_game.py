"""
后期比赛的代价 —— T=25 切片外推到 30/35/40 分钟, 到底掉多少?
=============================================================
Oracle's Elixir 只提供 at10/15/20/25 四个快照, 没有更晚的列。所以
「把训练点往后延」用现有数据做不到。但可以量出**外推的代价有多大**,
再决定值不值得为此自建数据源。

方法
----
拿 T=25 的快照训练 (这就是现在 Stage 4 在 T=25 做的事), 然后按**比赛
实际时长**分组看准确率:

    25-30 分钟   外推 0-5 分钟
    30-35 分钟   外推 5-10 分钟
    35-40 分钟   外推 10-15 分钟
    40+ 分钟     外推 15+ 分钟

如果准确率随时长单调下滑, 说明外推确实在伤害预测; 如果基本持平, 说明
25 分钟的局势已经足以决定结果, 延后训练点的收益有限。

用法: python research/gate_late_game.py
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, brier_score_loss

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from ingame_model import load, build, XGB
from harness import paired_t, T_ACCEPT

YEARS = (2022, 2023, 2024, 2025, 2026)
N_FOLDS, SEEDS, VAL_FRAC = 8, (0, 1, 2), 0.15
BUCKETS = [(25, 30), (30, 35), (35, 40), (40, 999)]


def main():
    print("=" * 74)
    print("  后期比赛: T=25 切片外推的代价")
    print("=" * 74)

    files = [str(_ROOT / "data" / f"{y}_LoL_esports_match_data_from_OraclesElixir.csv")
             for y in YEARS]
    print("\n[1/3] 加载")
    df = load(files)
    full, _ = build(df, verbose=False)

    # 每场比赛的时长
    print("\n[2/3] 取比赛时长")
    lens = {}
    for f in files:
        if not Path(f).exists():
            continue
        d = pd.read_csv(f, low_memory=False, usecols=["gameid", "position", "gamelength"])
        d = d[d["position"] == "team"].drop_duplicates("gameid")
        for gid, gl in zip(d["gameid"], d["gamelength"]):
            if pd.notna(gl):
                lens[gid] = gl / 60.0
    mdf = full[full["T"] == 25].reset_index(drop=True).copy()
    mdf["glen"] = mdf["gameid"].map(lens)
    mdf = mdf[mdf["glen"].notna()].reset_index(drop=True)
    print(f"    T=25 快照 {len(mdf)} 个, 有时长的 {mdf['glen'].notna().sum()}")
    for lo, hi in BUCKETS:
        k = ((mdf["glen"] > lo) & (mdf["glen"] <= hi)).sum()
        print(f"      {lo}-{hi if hi < 999 else '∞'} 分钟: {k:>5} 场")

    feats = [c for c in mdf.columns
             if c not in ("gameid", "date", "league", "y", "blue_team", "red_team", "glen")
             and mdf[c].dtype in ("float64", "int64", "float32", "int32")]

    print(f"\n[3/3] 按比赛切分做 {N_FOLDS} 折, 每折按时长分桶看准确率")
    order = mdf.groupby("gameid")["date"].min().sort_values()
    gids = order.index.to_numpy()
    n = len(gids)
    val_g = max(60, int(n * VAL_FRAC))
    cuts = np.linspace(int(n * 0.40), n - val_g, N_FOLDS).astype(int)

    per_bucket = {b: [] for b in BUCKETS}
    overall = []
    for fi, cut in enumerate(cuts):
        tr = mdf[mdf["gameid"].isin(set(gids[:cut]))]
        va = mdf[mdf["gameid"].isin(set(gids[cut:cut + val_g]))]
        Xtr = tr[feats].astype(float).fillna(-999); ytr = tr["y"].astype(int)
        ps = np.zeros(len(va))
        for sd in SEEDS:
            m = XGBClassifier(**{**XGB, "random_state": sd})
            m.fit(Xtr, ytr, verbose=False)
            ps += m.predict_proba(va[feats].astype(float).fillna(-999))[:, 1]
        ps /= len(SEEDS)
        yv = va["y"].astype(int).values
        overall.append(accuracy_score(yv, (ps > .5).astype(int)))
        for lo, hi in BUCKETS:
            sel = ((va["glen"] > lo) & (va["glen"] <= hi)).values
            if sel.sum() >= 20:
                per_bucket[(lo, hi)].append(
                    accuracy_score(yv[sel], (ps[sel] > .5).astype(int)))

    print(f"\n{'时长区间':<14}{'外推':<12}{'折数':>5}{'准确率':>10}{'相对全体':>11}")
    print("-" * 56)
    base = float(np.mean(overall))
    print(f"{'全体':<14}{'—':<12}{len(overall):>5}{base:>10.1%}{'':>11}")
    ref = per_bucket[(25, 30)]
    for (lo, hi) in BUCKETS:
        v = per_bucket[(lo, hi)]
        if not v:
            print(f"{f'{lo}-{hi}':<14}{'':<12}{'样本不足':>5}")
            continue
        a = float(np.mean(v))
        extra = f"{lo-25}-{min(hi,60)-25} 分钟"
        print(f"{f'{lo}-{hi if hi<999 else chr(8734)}':<14}{extra:<12}"
              f"{len(v):>5}{a:>10.1%}{a-base:>+10.1%}")

    # 最短桶 vs 最长桶做配对 t
    long_b = next((b for b in reversed(BUCKETS) if len(per_bucket[b]) == len(ref)), None)
    if ref and long_b and len(ref) == len(per_bucket[long_b]):
        d, sd, t, _ = paired_t(np.array(ref), np.array(per_bucket[long_b]))
        print(f"\n配对比较 25-30 分钟 vs {long_b[0]}+ 分钟:")
        print(f"  Δ = {d:+.4f}   配对SD {sd:.4f}   t = {t:+.2f}   (判据 |t| > {T_ACCEPT})")
        print(f"  → {'确认外推有代价' if t <= -T_ACCEPT else '差异未过闸, 外推代价不显著'}")

    print("\n" + "=" * 74)
    print("  说明: 数据源只有 at10/15/20/25, 无法直接训练更晚的切片。")
    print("  这里量的是「用 T=25 去覆盖后期」的代价, 用来判断值不值得")
    print("  为此自建数据源 (lolesports 实时流可以逐秒采, 但要攒几个月)。")
    print("=" * 74)


if __name__ == "__main__":
    main()
