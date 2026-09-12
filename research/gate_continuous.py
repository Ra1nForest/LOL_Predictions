"""
闸门: 回填数据值不值得进 Stage 4 的训练集?
============================================

要分开回答两个问题, 它们的答案可能不一样:

  A. **第 25 分钟之后**能不能预测准 (这才是回填的本来目的)
     现在模型最晚只在 T=25 训练过, 更晚全靠外推 ——
     research/gate_late_game.py 量过: 40+ 分钟只有 52.7%, 低于基线。
     测法: 在**留出局**的 T>=28 快照上比"只有 OE" vs "OE+回填"。

  B. 原有的 T=10/15/20/25 会不会被拖累
     回填是从帧重建的二手数据, 时刻对齐有约 15 秒残差 (见 esports_feed
     的 FIRST_INCOME_SEC 注释)。多喂这些点有可能反而添噪。
     测法: 在留出局的原有切片上比同样两版。

两问都用 harness.paired_t 判 (|t| > 2.5), 按**局**分折避免同一局既训又测。

用法: python research/gate_continuous.py [--folds 5] [--seeds 3]
"""
from __future__ import annotations
import argparse, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, brier_score_loss

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT)); sys.path.insert(0, str(_ROOT / "research"))
import ingame_model as IM
from harness import paired_t, T_ACCEPT

MERGED = _ROOT / "backfill" / "merged.csv.gz"
XGB_P = dict(n_estimators=400, max_depth=5, learning_rate=0.04,
             subsample=0.8, colsample_bytree=0.7,
             eval_metric="logloss", verbosity=0)
# 回填没有 XP, 所以整个对比都在 live 变体的特征集上做 —— 否则"加了回填"
# 和"少了经验差"两件事会混在一起, 分不清是谁的作用。
DROP = {"xpdiff", "x_xp_scaling"}


def league_weights(bf_path):
    """按各赛区实测的时刻对齐误差反比加权。

    对齐质量是**按赛区**分的, 不是按级别 (实测: LCK Challengers SD 574 最好,
    TCL SD 3615 最差, 而 LPL 1675 比 NACL 1359 还差)。所以权重取自实测的
    每赛区误差 SD, 不是拍脑袋按"一线/次级"分档。

    权重 = 一线中位 SD / 该赛区 SD, 夹在 [0.2, 1.0]:
    对得准的局说话响一点, 对得糊的局说话轻一点, 但谁也不至于被完全忽略。
    """
    import json as _j
    rows = [_j.loads(l) for l in open(bf_path, encoding="utf-8")]
    d = pd.DataFrame(rows)
    if "oe_golddiffat25" not in d:
        return {}
    v = d[d.oe_golddiffat25.notna()].copy()
    v["err"] = v.golddiff - v.oe_golddiffat25
    sd = v.groupby("league").err.std()
    sd = sd[v.groupby("league").size() >= 8]        # 样本太少的不参与, 用默认
    if sd.empty:
        return {}
    ref = sd.median()
    return {lg: float(np.clip(ref / s, 0.2, 1.0)) for lg, s in sd.items()}


def score(tr, te, feats, weights=None):
    X, y = tr[feats].astype(float).fillna(-999), tr["y"].astype(int)
    Xe, ye = te[feats].astype(float).fillna(-999), te["y"].astype(int).values
    w = None
    if weights:
        # OE 的行权重恒为 1 (它是官方口径); 只给回填的行按赛区打折
        w = np.where(tr["source"].values == "backfill",
                     tr["league"].map(weights).fillna(0.6).values, 1.0)
    accs, brs = [], []
    for s in SEEDS:
        m = XGBClassifier(random_state=s, **XGB_P)
        m.fit(X, y, sample_weight=w, verbose=False)
        p = m.predict_proba(Xe)[:, 1]
        base = max(ye.mean(), 1 - ye.mean())
        accs.append((accuracy_score(ye, (p >= .5).astype(int)) - base) / (1 - base))
        brs.append(-brier_score_loss(ye, p))
    return float(np.mean(accs)), float(np.mean(brs))


def main():
    global SEEDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--weighted", action="store_true",
                    help="按赛区对齐质量给回填行加权")
    a = ap.parse_args()
    SEEDS = list(range(a.seeds))

    print("=" * 70)
    print("  闸门: 回填数据该不该进 Stage 4")
    print("=" * 70)

    if not MERGED.exists():
        print("先跑 research/merge_backfill.py"); return 1
    df = pd.read_csv(MERGED, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    feats = [c for c in IM.features(df, with_interactions=True)
             if c not in DROP and c in df.columns]
    print(f"\n合并表 {df.gameid.nunique()} 局 / {len(df)} 条   特征 {len(feats)} 个")
    print(f"  其中回填 {(df.source=='backfill').sum()} 条, "
          f"OE {(df.source=='oe').sum()} 条")

    # 只有回填覆盖到的那些局才谈得上"晚期评估" —— 其余局根本没有 T>=28 的行
    bf_games = set(df.loc[df.source == "backfill", "gameid"])
    print(f"  有回填的局: {len(bf_games)}")

    # **不能按时间分折**: 回填的 339 局全部集中在最近 95 天, 按时间切会让
    # 它们整体落进最后一折的验证窗口 —— 前几折的晚期留出是 0 条, 最后一折
    # 的训练集里又一条回填都没有 (实测两边都是 154166 条, 完全相同)。
    # "有没有回填"和"是不是最近的比赛"完全重合, 时间分折分不开这两件事。
    #
    # 改成对**回填的那些局**做 k 折随机切分: OE 全量始终在训练侧, 只有回填
    # 部分在训练/验证之间轮换。按 gameid 隔离, 同一局不会既训又测。
    rng = np.random.default_rng(0)
    bf_list = np.array(sorted(bf_games))
    rng.shuffle(bf_list)
    folds_g = np.array_split(bf_list, a.folds)
    oe_all = df[df.source == "oe"]

    res = {k: {"oe": [], "both": []} for k in ("late_lift", "late_brier",
                                               "std_lift", "std_brier")}
    print(f"\n{a.folds} 折 × {len(SEEDS)} 种子, 按回填局随机切分")
    W = None
    if a.weighted:
        W = league_weights(_ROOT / "backfill" / "snapshots.jsonl")
        print("\n按赛区权重 (对齐越准权重越高):")
        for lg, w in sorted(W.items(), key=lambda x: -x[1]):
            print(f"    {lg:<18} {w:.2f}")

    for fi, te_g in enumerate(folds_g):
        te_g = set(te_g)
        tr_bf = df[(df.source == "backfill") & (~df.gameid.isin(te_g))]
        te = df[df.gameid.isin(te_g)]
        te_late = te[te["T"] >= 28]
        # 常规切片的留出用 OE 那部分, 同样只取这一折的局
        te_std = te[(te.source == "oe")]
        if len(te_late) < 40 or len(te_std) < 40:
            print(f"  fold {fi+1}: 留出太少 (晚期 {len(te_late)}, 常规 {len(te_std)}), 跳过")
            continue
        # 训练侧必须排掉验证局的 OE 行, 否则同一局既训又测
        tr_oe = oe_all[~oe_all.gameid.isin(te_g)]
        tr_both = pd.concat([tr_oe, tr_bf], ignore_index=True)

        for tag, tr in (("oe", tr_oe), ("both", tr_both)):
            ww = W if (tag == "both") else None
            l, b = score(tr, te_late, feats, ww)
            res["late_lift"][tag].append(l); res["late_brier"][tag].append(b)
            l2, b2 = score(tr, te_std, feats, ww)
            res["std_lift"][tag].append(l2); res["std_brier"][tag].append(b2)
        print(f"  fold {fi+1}: 训练 OE {len(tr_oe)} / +回填 {len(tr_both)}"
              f"   晚期留出 {len(te_late)} 条   常规留出 {len(te_std)} 条")

    NAMES = {"late_lift": "A. 第 28 分钟之后 · 准确率口径",
             "late_brier": "A. 第 28 分钟之后 · Brier",
             "std_lift": "B. 原有切片 · 准确率口径",
             "std_brier": "B. 原有切片 · Brier"}
    print()
    for k in ("late_lift", "late_brier", "std_lift", "std_brier"):
        base, cand = res[k]["oe"], res[k]["both"]
        if len(base) < 2:
            print(f"  {NAMES[k]}: fold 不足, 无法判定"); continue
        d, sd, t, ratio = paired_t(base, cand)
        mark = "✓" if t >= T_ACCEPT else ("✗" if t <= -T_ACCEPT else "?")
        print("-" * 70)
        print(f"  {mark} {NAMES[k]}")
        print(f"      仅 OE   {np.mean(base):+.4f}")
        print(f"      含回填  {np.mean(cand):+.4f}")
        print(f"      Δ = {d:+.4f}   配对 SD {sd:.4f}   t = {t:+.2f}"
              f"   (判据 |t| > {T_ACCEPT})")
        print(f"      逐 fold Δ: {'  '.join(f'{c-b:+.4f}' for b, c in zip(base, cand))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
