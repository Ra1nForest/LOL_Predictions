"""
闸门: 把次级联赛加进 Stage 4 的训练集, 对四大赛区的预测是变好还是变差?
======================================================================

现状: ingame_model.load() 用 raw[league.isin(TARGET)] 只留四大赛区,
      得到 20311 条快照 (约 5078 局)。其中 LPL 只占 9.25%, 且全来自 2026 ——
      OE 在 2022-2025 根本没有 LPL 的 at10/15/20/25。

闸门外还躺着约 37500 局可用数据 (LCKC / NACL / LFL / VCS / PCS …)。
它们节奏和 BP 环境跟一线不同, 加进来可能帮忙 (样本翻七倍), 也可能拖累
(分布不一样)。没测过就不知道 —— 这个脚本就是来测的。

设计
----
· 验证窗口**永远只含四大赛区** —— 我们关心的是一线预测准不准,
  次级联赛只可能出现在训练侧。
· 逐 fold 配对: 同一个验证窗口, 分别用「仅四大」和「四大+次级」训练,
  两者相减消掉「这段时期有多难」。
· 判据用 research/harness.py 的 paired_t (|t| > 2.5), 全项目唯一一份算术。

用法: python research/gate_league_mix.py [--folds 5] [--seeds 3]
"""
from __future__ import annotations
import argparse, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "research"))
import ingame_model as IM
from harness import paired_t, T_ACCEPT

MAJORS = ["LPL", "LCK", "LEC", "LCS"]
YEARS = (2022, 2023, 2024, 2025, 2026)
XGB_P = dict(n_estimators=400, max_depth=5, learning_rate=0.04,
             subsample=0.8, colsample_bytree=0.7,
             eval_metric="logloss", verbosity=0)


def load_all(with_minors: bool) -> pd.DataFrame:
    """绕开 load() 的赛区过滤, 自己决定留哪些赛区。"""
    dfs = []
    for y in YEARS:
        p = _ROOT / "data" / f"{y}_LoL_esports_match_data_from_OraclesElixir.csv"
        if p.exists():
            dfs.append(pd.read_csv(p, low_memory=False))
    raw = pd.concat(dfs, ignore_index=True)
    raw["league"] = raw["league"].replace({"LTA N": "LCS", "LTA S": "CBLOL"})
    if not with_minors:
        raw = raw[raw["league"].isin(IM.TARGET)]
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    return raw.sort_values("date").reset_index(drop=True)


def snapshots(with_minors: bool):
    df = load_all(with_minors)
    mdf, _sc = IM.build(df, verbose=False)
    return mdf.sort_values("date").reset_index(drop=True)


def score(Xtr, ytr, Xva, yva, seeds):
    """归一化 lift, 和 harness.walk_forward 用的是同一个口径。"""
    out = []
    base = max(yva.mean(), 1 - yva.mean())
    for s in seeds:
        m = XGBClassifier(random_state=s, **XGB_P)
        m.fit(Xtr, ytr, verbose=False)
        acc = accuracy_score(yva, m.predict(Xva))
        out.append((acc - base) / (1 - base))
    return float(np.mean(out)), float(np.mean(out)) + 0 * base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    seeds = list(range(a.seeds))

    print("=" * 72)
    print("  闸门: 次级联赛数据该不该进 Stage 4 的训练集")
    print("=" * 72)

    print("\n[1/3] 构建快照 (仅四大赛区)")
    maj = snapshots(with_minors=False)
    print(f"      {len(maj)} 条快照, {maj.gameid.nunique()} 局")
    for lg in MAJORS:
        n = (maj.league == lg).sum()
        print(f"        {lg:<5} {n:>6} 条  {n/len(maj):>6.1%}")

    print("\n[2/3] 构建快照 (四大 + 次级)")
    allx = snapshots(with_minors=True)
    print(f"      {len(allx)} 条快照, {allx.gameid.nunique()} 局 "
          f"(比只用四大多 {len(allx)/len(maj):.1f} 倍)")

    feats = [c for c in IM.features(maj, with_interactions=True)
             if c in allx.columns]
    print(f"      共用特征 {len(feats)} 个")

    # 验证窗口只取四大赛区, 按时间推进
    maj_idx = np.arange(len(maj))
    val_size = max(400, int(len(maj) * 0.12))
    start, end = int(len(maj) * 0.40), len(maj) - val_size
    cuts = np.linspace(start, end, a.folds).astype(int)

    print(f"\n[3/3] {a.folds} 折 × {len(seeds)} 种子   验证窗口 {val_size} 条 (仅四大)")
    f_base, f_cand = [], []
    for fi, cut in enumerate(cuts):
        va = maj.iloc[cut:cut + val_size]
        t_end = va.date.min()                    # 训练只能用验证窗口之前的
        tr_maj = maj[maj.date < t_end]
        tr_all = allx[allx.date < t_end]

        Xva = va[feats].astype(float).fillna(-999)
        yva = va["y"].astype(int)
        s_base, _ = score(tr_maj[feats].astype(float).fillna(-999),
                          tr_maj["y"].astype(int), Xva, yva, seeds)
        s_cand, _ = score(tr_all[feats].astype(float).fillna(-999),
                          tr_all["y"].astype(int), Xva, yva, seeds)
        f_base.append(s_base); f_cand.append(s_cand)
        print(f"      fold {fi+1}: 验证 [{va.date.min().date()} → {va.date.max().date()}]"
              f"  训练 仅四大 {len(tr_maj):>6} / 含次级 {len(tr_all):>6}"
              f"   lift {s_base:+.4f} → {s_cand:+.4f}  ({s_cand-s_base:+.4f})")

    d, sd, t, ratio = paired_t(f_base, f_cand)
    print("\n" + "-" * 72)
    print(f"  仅四大   {np.mean(f_base):+.4f}")
    print(f"  含次级   {np.mean(f_cand):+.4f}")
    print(f"  Δ = {d:+.4f}   配对 SD {sd:.4f}   n = {len(f_base)}")
    print(f"  t = {t:+.2f}   (判据 |t| > {T_ACCEPT})   效果量 {ratio:.2f}x SD")
    if t >= T_ACCEPT:
        print("  → ✓ 采纳: 次级联赛数据能提升一线预测")
    elif t <= -T_ACCEPT:
        print("  → ✗ 拒绝: 加进来会稳定拖累一线预测")
    elif abs(t) >= 1.5:
        print("  → ? 存疑: 方向有迹象但证据不足, 需要更多 fold")
    else:
        print("  → ✗ 丢弃: 差异在噪声内")
    print(f"  逐 fold Δ: {'  '.join(f'{b-c:+.4f}'.replace('-','−') for c, b in zip(f_base, f_cand))}")


if __name__ == "__main__":
    sys.exit(main())
