"""
Stage 4 训练器 — 局内模型
==========================
训练 + 校准 + 保存, 供 api.py 的 /predict/ingame 使用。

产出 artifacts/:
    model_ingame.json          完整模型 (手填经济/经验差时用)
    calib_ingame.json
    model_ingame_live.json     live 变体, 不含任何经验差特征
    calib_ingame_live.json

为什么要两个模型
----------------
lolesports 的实时帧里没有 XP 字段 (只有每人的 level, 和训练用的
xpat10/15/20 原始经验值量纲不同)。自动跟播时 xpdiff 拿不到。

给完整模型传一个编造的 xpdiff=0 有两重问题:
  · 模型在吃一个训练时没见过的系统性常量
  · explain 层会把它讲成「双方经验持平」—— 把缺失讲成了实测结论

所以实时路径走一个训练时就不含经验差的模型。代价由
research/gate_ingame_features.py 的配置 F 量出来。

注意 live 变体要排除的是 XP_DERIVED 两项, 不只是 xpdiff ——
x_xp_scaling = xpdiff * scaling_diff (ingame_model.py:270), 留着它
等于经验差信息还在。

英雄强势期表必须一起存 —— 线上推理时不能重算 (要扫全部历史)。

用法: python ingame_train.py
"""
import numpy as np, pandas as pd, json, os, warnings, sys
from pathlib import Path
warnings.filterwarnings("ignore")
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, brier_score_loss

sys.path.insert(0, str(Path(__file__).parent))
from ingame_model import (SLICES, ROLES, W, PRE_STATS, MIN_CHAMP_GAMES,
                          SHRINK_K, XGB, load, build, features,
                          split_by_game, ece, INTERACTIONS)

YEARS = (2022, 2023, 2024, 2025, 2026)
_ROOT = Path(__file__).parent
# 见 train.py 同处注释: 无人值守流水线用它训到暂存目录
OUT = Path(os.environ.get("LOL_ARTIFACTS") or (_ROOT / "artifacts"))
OUT.mkdir(parents=True, exist_ok=True)

# 经验差及其派生项。实时帧给不了 XP, 这两个必须一起去掉。
XP_DERIVED = {"xpdiff", "x_xp_scaling"}

# 校准用保序回归, 不是 Platt。**这一条只对局内模型成立**, Stage 1/2 仍用
# Platt —— research/gate_calibration{,_pre}.py 分别量过:
#
#   Stage 4  (校准集约 35000 条快照)  Brier t=+6.91  ECE t=+7.12  8 折全正 -> 换
#   Stage 1  (校准集     1716 场)     Brier t=-4.74  6 折全负            -> 不换
#   Stage 2  (校准集     1716 场)     Brier t=-1.97  方向也是变差        -> 不换
#
# README 里"Isotonic 输了, 因为校准数据不够会过拟合"那个结论没有错, 只是它
# 的前提 (数据量) 在局内模型上早已不成立: 那里有六万条校准样本。同一个方法
# 在两处的结论相反, 靠的是分别过闸, 不是推广。
#
# 为什么非换不可: Platt 只能画一条固定形状的 S 曲线, 实测它中间鼓两头塌 ——
# >6k 领先的样本实际胜率 98.0%, XGBoost 原始输出 97.8% (几乎分毫不差),
# 被 Platt 压成 93.3%。**模型判断是对的, 是校准层把它压扁了。**
CLIP = 0.01               # 保序回归会输出 0 和 1; 绝对确定既不真实也无必要


def train_variant(mdf, sc, feats, suffix, title):
    """训练 + 校准 + 存盘。suffix 为空是完整模型, "_live" 是实时变体。"""
    print(f"\n  ── {title}  ({len(feats)} 特征) ──")
    tr, cal, te = split_by_game(mdf)
    Xtr, ytr = tr[feats].astype(float).fillna(-999), tr["y"].astype(int)
    Xc, yc = cal[feats].astype(float).fillna(-999), cal["y"].astype(int)
    Xte, yte = te[feats].astype(float).fillna(-999), te["y"].astype(int).values

    m = XGBClassifier(**XGB); m.fit(Xtr, ytr, verbose=False)
    pc = m.predict_proba(Xc)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(pc, yc)
    # Platt 也照旧拟一份存进去。不是留着用, 是留着**能回滚**: 万一服务端
    # 退回旧版本, 旧代码只认 coef/intercept, 字段缺了会直接崩。
    cb = LogisticRegression(C=1e6).fit(pc.reshape(-1, 1), yc)
    raw = m.predict_proba(Xte)[:, 1]
    p = np.clip(iso.predict(raw), CLIP, 1 - CLIP)

    base = max(yte.mean(), 1 - yte.mean())
    acc = accuracy_score(yte, (p > .5).astype(int))
    print(f"    训练 {len(tr)} / 校准 {len(cal)} / 测试 {len(te)} 快照")
    print(f"    准确率 {acc:.1%} (基线 {base:.1%})   Brier {brier_score_loss(yte,p):.4f}"
          f"   ECE {ece(p, yte):.4f}")
    print(f"    概率范围 [{p.min():.2f}, {p.max():.2f}]")

    per_T = {}
    tt = te.copy(); tt["p"] = p
    print(f"    {'T':>6}{'快照':>7}{'准确率':>9}{'高置信':>9}")
    for T in SLICES:
        s = tt[tt["T"] == T]
        if len(s) < 20: continue
        a = accuracy_score(s["y"].astype(int), (s["p"] > .5).astype(int))
        conf = float(((s["p"] > .8) | (s["p"] < .2)).mean())
        per_T[str(T)] = {"accuracy": float(a), "n": int(len(s)),
                         "high_conf_share": conf}
        print(f"    {T:>6}{len(s):>7}{a:>9.1%}{conf:>9.1%}")

    m.save_model(str(OUT / f"model_ingame{suffix}.json"))

    # 英雄强势期表 —— 取最新日期的快照
    end = mdf["date"].max()
    scale_tbl = {}
    for ch in sc.tbl:
        v = sc.get(ch, end)
        if not np.isnan(v):
            scale_tbl[ch] = round(float(v), 5)

    cfg = {
        # 线上真正用的校准器。老 artifacts 没有这个字段, ingame_service
        # 读不到时默认回退成 platt —— 换代码和换模型文件可以分开做。
        "calibrator": "isotonic",
        "iso_x": [float(v) for v in iso.X_thresholds_],
        "iso_y": [float(v) for v in iso.y_thresholds_],
        "clip": CLIP,
        # 兼容字段, 见上面 cb 那句注释
        "coef": float(cb.coef_[0][0]),
        "intercept": float(cb.intercept_[0]),
        "features": feats,
        "slices": SLICES,
        "scaling_index": scale_tbl,
        "scaling_min_games": MIN_CHAMP_GAMES,
        "scaling_shrink_k": SHRINK_K,
        "interaction_cols": [c for c in INTERACTIONS if c in feats],
        "pre_stats": PRE_STATS,
        "roll_window": W,
        "trained_through": str(end.date()),
        "variant": "live" if suffix else "full",
        "has_xpdiff": "xpdiff" in feats,
        "metrics": {
            "accuracy": float(acc), "baseline": float(base),
            "lift": float((acc - base) / (1 - base)),
            "brier": float(brier_score_loss(yte, p)),
            "ece": float(ece(p, yte)),
            "p_min": float(p.min()), "p_max": float(p.max()),
            "per_T": per_T,
        },
    }
    with open(OUT / f"calib_ingame{suffix}.json", "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2, ensure_ascii=False)

    print(f"    -> model_ingame{suffix}.json  calib_ingame{suffix}.json")
    return acc


def main():
    print("=" * 70)
    print("  Stage 4 训练器")
    print("=" * 70)

    files = [str(_ROOT / "data" / f"{y}_LoL_esports_match_data_from_OraclesElixir.csv")
             for y in YEARS]
    print("\n[1/3] 加载")
    df = load(files)

    print("\n[2/3] 构建快照")
    mdf, sc = build(df)

    print("\n[3/3] 训练两个变体")
    feats_full = features(mdf, with_interactions=True)
    feats_live = [c for c in feats_full if c not in XP_DERIVED]
    dropped = [c for c in feats_full if c in XP_DERIVED]

    a_full = train_variant(mdf, sc, feats_full, "", "完整模型")
    a_live = train_variant(mdf, sc, feats_live, "_live",
                           f"live 变体 (去掉 {', '.join(dropped)})")

    print(f"\n{'=' * 70}")
    print(f"  完整 {a_full:.1%}   live {a_live:.1%}   "
          f"差 {(a_full - a_live) * 100:+.1f} 个百分点")
    print(f"  (单次留出对比, 未过闸。闸门结论见 research/gate_ingame_features.py)")
    print(f"\n  重启服务后可用 POST /predict/ingame")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
