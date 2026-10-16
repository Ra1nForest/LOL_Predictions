"""
闸门 G2: Stage 4 服务模型要不要"训到更近"?
=============================================

为什么要量
----------
ingame_train.train_variant 每次日更都做同一件事: 按局切 60/20/20 (split_by_game),
树只拟最早 60% 的局, 保序表拟随后 20%, 最近 20% 只当考卷算 metrics, 然后直接存盘
上线 —— 从不拿全部数据重拟。于是 2026-10 线上的树数据截至 2024-11-16 (28255 局,
其中 LPL 0 局: OE 在 2022-2025 没有 LPL 的 at10/15/20/25), 保序表截至 2026-01-15。
calib 里的 trained_through 只是强势期表的日期, 不是树的数据截止日。

两份探针 (idea-scan s4 / ingame-errors) 都看到"训到更近"在 Brier 上略好: 6 块滚动
Δ+0.0007, 逐局 t +3.03, 块 t +2.35 —— 方向一致但没过闸, 且切块方式是事后挑的。所以
先把协议冻结, 再正式量一次。这里比的只是**用哪段数据拟树和保序表**: 特征、超参数、
校准方法 (保序 + 裁剪到 [0.01, 0.99]) 一律不动。

冻结协议 (先定死, 不在评估数据上调任何东西)
-------------------------------------------
· 快照表: ingame_model.build(ingame_model.load(五年 CSV)), 全部赛区, T = 10/15/20/25,
  和线上训练同一份。CSV 路径只从 multiyear_gate.data_path() 来。
· 起点: 2026-03-01 … 2026-09-01 每月 1 日, 共 7 个 (fold)。
· 每个起点 O, 只用 date < O 的局 (按局的最早日期排序, 即 split_by_game 本身):
    CURRENT    今天的配方: 树 = 最早 60% 的局, 保序表 = 随后 20%, 最后 20% 不用
    CANDIDATE  树 = 最早 80% 的局, 保序表 = 最近 20%
  超参数 = ingame_model.XGB (只把 random_state 换成种子、n_jobs 换成本机线程数),
  保序 = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1) 再裁剪到 ingame_train.CLIP。
· 评估: O 之后一个自然月 [O, O + 1 月) 的全部快照 (全部赛区)。
· 3 个种子 (0, 1, 2), 两臂同种子配对。"种子平均" = 每个种子的模型各算各的损失再平均
  (估的是"线上随便一个种子的模型"的期望损失); 先平均概率再算 (等于 3 模型集成) 只作敏感性参考。
· 主判据: live 变体 (26 特征, 去掉 XP_DERIVED), 全部赛区, 7 个 fold 的逐月 Brier,
  harness.paired_t 配对 t, Δ = Brier(CURRENT) − Brier(CANDIDATE) (正 = CANDIDATE 更好)。
  **CANDIDATE 通过 ⇔ t > 2.5。**
· 一并报告 (不参与判决): 逐局 t (每局先对快照平均)、系列赛聚类 t、对数损失、准确率、
  ECE (ingame_model.ece, 10 档)、校准斜率 (logistic y ~ logit p)、四大赛区子集 (次判据)、
  full 变体 (28 特征)。
· Stage 1/2 的同类比较 (树 80% / Platt 20% vs 今天的 60/20, 同 7 个起点、同评估月、3 种子,
  train.py 的 XGB 与 Platt) 只作信息, 不作判决。

结论 (2026-10-17 运行, OE 数据截至 2026-10-16; 快照表 47092 局 / 183150 快照)
------------------------------------------------------------------------------
**CANDIDATE 通过。** live / 全部赛区 (7530 局, 29222 快照):
    Brier 0.15623 → 0.15477, Δ +0.00145, fold 配对 t = +6.54, 7/7 个月都更好
    逐局 t +4.56, 系列赛聚类 t +4.63 (3401 个系列赛), 对数损失 fold t +5.45 (7/7)
    准确率不变 (76.77% → 76.77%, t −0.38) —— 收益全在概率的刻度上:
    ECE 0.0188 → 0.0098 (fold t +2.13), 校准斜率 0.898 → 0.960 (更接近 1, 过度自信变少)
    逐月 Δ: +0.0007 +0.0011 +0.0011 +0.0013 +0.0020 +0.0017 +0.0024 (大体越晚越大, 和"树越来越旧"一致)
    四个切片都更好: T10 0.1906→0.1888, T15 0.1663→0.1648, T20 0.1422→0.1409, T25 0.1219→0.1207
次判据, 四大赛区 (1268 局): Δ +0.00225, fold t +2.72 (5/7), 逐局 t +2.19, 聚类 t +2.21 ——
    方向一致, 逐局层面单独不到 2.5 (样本只有全部的六分之一)。
full 变体 (28 特征): 全部赛区 Δ +0.00091, fold t +3.71 (6/7), 逐局 t +3.19; 四大 fold t +3.60, 逐局 t +1.89。
集成概率口径 (敏感性): live 全部赛区 fold t +6.34, 结论不变。
注意: 7 个起点下两臂的树里 LPL 都是 0 局 (OE 只有 2026 年起的 LPL 分钟数据, 80% 的切点仍在
2025 年) —— 收益不是"树见到了 LPL", 而是树和保序表的数据都比今天的配方近了约 10~12 个月。
本闸不拆"新树"和"新保序表"各占多少 (协议没有这两臂)。读缓存复跑一次, 所有数字逐位一致。

Stage 1/2 (只作信息, 四大, 1350 局): pre_draft Brier Δ +0.00274, fold t +1.78 (4/7), 逐局 t +1.07;
post_draft Δ +0.00163, fold t +0.97 (4/7), 逐局 t +1.00 —— 方向向好、校准斜率 0.68→0.90 / 0.74→0.88,
但远不到 2.5, 不支持据此改 train.py。

用法 (项目根目录, 项目 venv):
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe research/gate_ingame_refit.py \\
        [--cache 目录] [--json 结果.json] [--skip-stage12] [--jobs 16]
--cache 把快照表 / Stage 1/2 矩阵按 CSV 的大小+修改时间缓存 (数据变了自动重建)。
训练全在内存里做, 不写 artifacts/。import ingame_train / train 只为取 CLIP、XP_DERIVED、XGB 等常量
(单一来源); 它们导入时会 mkdir LOL_ARTIFACTS (默认 artifacts/, 已存在就什么也不写)。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "research"))
import ingame_model as IM  # noqa: E402
from harness import T_ACCEPT, paired_t  # noqa: E402
from multiyear_gate import data_path  # noqa: E402

YEARS = (2022, 2023, 2024, 2025, 2026)
MAJORS = ["LPL", "LCK", "LEC", "LCS"]
ORIGINS = list(pd.date_range("2026-03-01", "2026-09-01", freq="MS"))   # 冻结: 7 个起点
SEEDS = (0, 1, 2)                                                        # 冻结
ARMS = ("CURRENT", "CANDIDATE")
MONTH = pd.offsets.MonthBegin(1)


# ══════════════════════════════════════════════════════════
#  数据 (可选缓存; 签名 = CSV 的大小 + 修改时间)
# ══════════════════════════════════════════════════════════

def _files():
    return [data_path(y) for y in YEARS]


def _sig(paths):
    return [(Path(p).name, os.path.getsize(p), int(os.path.getmtime(p)))
            for p in paths if os.path.exists(p)]


def _cached(cache, name, builder):
    sig = _sig(_files())
    f = Path(cache) / f"{name}.pkl" if cache else None
    if f is not None and f.exists():
        with open(f, "rb") as fh:
            obj = pickle.load(fh)
        if obj.get("sig") == sig:
            print(f"  读缓存 {f}")
            return obj["data"]
        print(f"  缓存 {f.name} 的 CSV 签名不符, 重建")
    t0 = time.time()
    data = builder()
    print(f"  构建 {name} 用时 {time.time() - t0:.0f}s")
    if f is not None:
        f.parent.mkdir(parents=True, exist_ok=True)
        with open(f, "wb") as fh:
            pickle.dump({"sig": sig, "data": data}, fh)
    return data


def build_snapshots(cache):
    def b():
        m4, _sc = IM.build(IM.load(_files()), verbose=False)
        return m4
    m4 = _cached(cache, "g2_ingame_snapshots", b)
    m4["date"] = pd.to_datetime(m4["date"])
    return m4


def build_matrix12(cache):
    from feature_store import build_training_matrix
    from train import DATA
    # 和线上 train.py 读的必须是同一份五年 CSV (见 CLAUDE.md "Training and serving must read the same data")
    assert [Path(p).resolve() for p in DATA] == [Path(p).resolve() for p in _files()], "train.DATA 与 data_path 不一致"
    mdf = _cached(cache, "g2_stage12_matrix", lambda: build_training_matrix(_files(), verbose=False))
    mdf["date"] = pd.to_datetime(mdf["date"])
    return mdf


# ══════════════════════════════════════════════════════════
#  统计
# ══════════════════════════════════════════════════════════

def _ll(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def calib_slope(p, y):
    """logistic y ~ a + b·logit(p) 的 (b, a)。b = 1 完美, b < 1 过度自信。"""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def cluster_t(d, groups):
    """快照级平均差的聚类稳健 t (按系列赛聚类, CR1 小样本修正)。"""
    d = np.asarray(d, float)
    mu = d.mean()
    s = pd.Series(d - mu).groupby(np.asarray(groups)).sum().values
    G, N = len(s), len(d)
    se = math.sqrt((s ** 2).sum() * G / (G - 1)) / N
    return float(mu / se), int(G)


def series_key(df):
    d = df["date"].dt.normalize().astype(str).values
    b, r = df["blue_team"].astype(str).values, df["red_team"].astype(str).values
    lo, hi = np.where(b < r, b, r), np.where(b < r, r, b)
    return pd.Series(d).str.cat([df["league"].astype(str).values, lo, hi], sep="|").values


def evaluate(y, P, fold, games, series, n_fold, ece_fn):
    """P[arm]: (种子, 行) 概率。返回逐 fold 表 + 各种 t。正 Δ = CANDIDATE 更好。"""
    Pc, Pd = P["CURRENT"], P["CANDIDATE"]
    S = Pc.shape[0]
    Lc = ((Pc - y) ** 2).mean(0)            # 每行: 各种子 Brier 损失的平均
    Ld = ((Pd - y) ** 2).mean(0)
    LLc = np.mean([_ll(Pc[s], y) for s in range(S)], 0)
    LLd = np.mean([_ll(Pd[s], y) for s in range(S)], 0)
    Ec, Ed = Pc.mean(0), Pd.mean(0)          # 集成概率 (敏感性参考)

    folds = []
    for k in range(n_fold):
        m = fold == k
        if not m.any():
            continue
        r = {"fold": k, "n_snap": int(m.sum()), "n_games": int(pd.unique(games[m]).size)}
        for arm, PP, L, LL, E in (("cur", Pc, Lc, LLc, Ec), ("cand", Pd, Ld, LLd, Ed)):
            r[f"brier_{arm}"] = float(L[m].mean())
            r[f"ll_{arm}"] = float(LL[m].mean())
            r[f"acc_{arm}"] = float(np.mean([((PP[s, m] > .5) == y[m]).mean() for s in range(S)]))
            r[f"ece_{arm}"] = float(np.mean([ece_fn(PP[s, m], y[m]) for s in range(S)]))
            r[f"slope_{arm}"] = float(np.mean([calib_slope(PP[s, m], y[m])[0] for s in range(S)]))
            r[f"brier_ens_{arm}"] = float(((E[m] - y[m]) ** 2).mean())
        r["d_brier"] = r["brier_cur"] - r["brier_cand"]
        folds.append(r)
    fd = pd.DataFrame(folds)

    def ft(a, b):        # a, b: 损失 (越小越好) → 配对 t, Δ = a − b
        d, sd, t, _ = paired_t(-np.asarray(a), -np.asarray(b))
        return {"delta": float(d), "sd": float(sd), "t": float(t),
                "better": int((np.asarray(a) > np.asarray(b)).sum()), "n": int(len(a))}

    out = {"folds": folds,
           "fold_brier": ft(fd.brier_cur, fd.brier_cand),
           "fold_ll": ft(fd.ll_cur, fd.ll_cand),
           "fold_acc": ft(-fd.acc_cur, -fd.acc_cand),
           "fold_ece": ft(fd.ece_cur, fd.ece_cand),
           "fold_brier_ens": ft(fd.brier_ens_cur, fd.brier_ens_cand)}
    m = fold >= 0
    g = pd.DataFrame({"g": games[m], "d": (Lc - Ld)[m], "dl": (LLc - LLd)[m]}).groupby("g").mean()
    out["game"] = {"n_games": int(len(g)),
                   "delta": float(g.d.mean()), "t": float(g.d.mean() / (g.d.std(ddof=1) / math.sqrt(len(g)))),
                   "delta_ll": float(g.dl.mean()),
                   "t_ll": float(g.dl.mean() / (g.dl.std(ddof=1) / math.sqrt(len(g))))}
    ts, ns = cluster_t((Lc - Ld)[m], series[m])
    out["series"] = {"t": ts, "n_series": ns, "delta": float((Lc - Ld)[m].mean())}
    pooled = {}
    for arm, PP, L in (("cur", Pc, Lc), ("cand", Pd, Ld)):
        sl = [calib_slope(PP[s, m], y[m]) for s in range(S)]
        pooled[arm] = {"brier": float(L[m].mean()),
                       "ece": float(np.mean([ece_fn(PP[s, m], y[m]) for s in range(S)])),
                       "slope": float(np.mean([a for a, _ in sl])),
                       "intercept": float(np.mean([b for _, b in sl])),
                       "acc": float(np.mean([((PP[s, m] > .5) == y[m]).mean() for s in range(S)]))}
    out["pooled"] = pooled
    return out


def show(title, r, primary=False, unit="快照"):
    fb, g, s = r["fold_brier"], r["game"], r["series"]
    print(f"\n  ── {title} ──")
    print(f"    {'起点':<10}{'局':>6}{unit:>7}{'Brier 现':>10}{'Brier 候':>10}{'Δ':>9}"
          f"{'ECE 现':>8}{'ECE 候':>8}{'斜率 现':>8}{'斜率 候':>8}{'acc 现':>8}{'acc 候':>8}")
    for f in r["folds"]:
        print(f"    {str(ORIGINS[f['fold']].date()):<10}{f['n_games']:>6}{f['n_snap']:>7}"
              f"{f['brier_cur']:>10.4f}{f['brier_cand']:>10.4f}{f['d_brier']:>+9.5f}"
              f"{f['ece_cur']:>8.4f}{f['ece_cand']:>8.4f}{f['slope_cur']:>8.3f}{f['slope_cand']:>8.3f}"
              f"{f['acc_cur']:>8.4f}{f['acc_cand']:>8.4f}")
    pc, pd_ = r["pooled"]["cur"], r["pooled"]["cand"]
    print(f"    合计      Brier {pc['brier']:.5f} → {pd_['brier']:.5f}   ECE {pc['ece']:.4f} → {pd_['ece']:.4f}   "
          f"斜率 {pc['slope']:.3f} → {pd_['slope']:.3f}   截距 {pc['intercept']:+.3f} → {pd_['intercept']:+.3f}   "
          f"acc {pc['acc']:.4f} → {pd_['acc']:.4f}")
    mark = ""
    if primary:
        mark = "  ← 判据" + (" ✓ 通过" if fb["t"] > T_ACCEPT else " ✗ 未过")
    print(f"    Brier  fold t {fb['t']:+.2f} (Δ {fb['delta']:+.5f}, {fb['better']}/{fb['n']} 月更好){mark}")
    print(f"           逐局 t {g['t']:+.2f} ({g['n_games']} 局, Δ {g['delta']:+.5f})   "
          f"系列赛聚类 t {s['t']:+.2f} ({s['n_series']} 个系列赛)")
    for k, nm in (("fold_ll", "对数损失"), ("fold_acc", "准确率  "), ("fold_ece", "ECE     "),
                  ("fold_brier_ens", "Brier(集成概率)")):
        x = r[k]
        print(f"    {nm} fold t {x['t']:+.2f} (Δ {x['delta']:+.5f}, {x['better']}/{x['n']})")
    print(f"    对数损失 逐局 t {g['t_ll']:+.2f} (Δ {g['delta_ll']:+.5f})")


# ══════════════════════════════════════════════════════════
#  Stage 4
# ══════════════════════════════════════════════════════════

def run_stage4(m4, jobs):
    from ingame_train import CLIP, XP_DERIVED
    feats_full = IM.features(m4, with_interactions=True)
    feats_live = [c for c in feats_full if c not in XP_DERIVED]
    for name, fs in (("calib_ingame.json", feats_full), ("calib_ingame_live.json", feats_live)):
        f = _ROOT / "artifacts" / name                       # 只读, 核对特征列表和线上一致
        if f.exists():
            on = json.loads(f.read_text(encoding="utf-8"))["features"]
            print(f"  {name}: 特征 {len(fs)} 个, 与线上{'一致' if on == fs else '不一致 ⚠'}")
    variants = {"live": feats_live, "full": feats_full}

    end = ORIGINS[-1] + MONTH
    E = m4[(m4.date >= ORIGINS[0]) & (m4.date < end)].reset_index(drop=True)
    fold = np.full(len(E), -1)
    for i, o in enumerate(ORIGINS):
        fold[((E.date >= o) & (E.date < o + MONTH)).values] = i
    y = E.y.astype(int).values
    P = {v: {a: np.full((len(SEEDS), len(E)), np.nan) for a in ARMS} for v in variants}
    info = []
    t0 = time.time()
    for i, o in enumerate(ORIGINS):
        sub = m4[m4.date < o]
        c_tr, c_ca, c_un = IM.split_by_game(sub)                 # 今天的配方 (默认 0.6 / 0.2)
        d_tr, d_ca, d_un = IM.split_by_game(sub, 0.80, 0.20)
        assert d_un.empty
        ev = fold == i
        rec = {"origin": str(o.date()), "games_before": int(sub.gameid.nunique()),
               "cur_tree": [str(c_tr.date.min().date()), str(c_tr.date.max().date()), int(c_tr.gameid.nunique())],
               "cur_iso": [str(c_ca.date.min().date()), str(c_ca.date.max().date()), int(c_ca.gameid.nunique())],
               "cur_unused": [str(c_un.date.min().date()), str(c_un.date.max().date()), int(c_un.gameid.nunique())],
               "cand_tree": [str(d_tr.date.min().date()), str(d_tr.date.max().date()), int(d_tr.gameid.nunique())],
               "cand_iso": [str(d_ca.date.min().date()), str(d_ca.date.max().date()), int(d_ca.gameid.nunique())],
               "cur_tree_lpl_games": int(c_tr[c_tr.league == "LPL"].gameid.nunique()),
               "cand_tree_lpl_games": int(d_tr[d_tr.league == "LPL"].gameid.nunique()),
               "eval_games": int(E[ev].gameid.nunique()), "eval_snaps": int(ev.sum())}
        info.append(rec)
        for v, fs in variants.items():
            Xev = E.loc[ev, fs].astype(float).fillna(-999)
            for arm, tr, ca in (("CURRENT", c_tr, c_ca), ("CANDIDATE", d_tr, d_ca)):
                Xtr, ytr = tr[fs].astype(float).fillna(-999), tr["y"].astype(int)
                Xca, yca = ca[fs].astype(float).fillna(-999), ca["y"].astype(int)
                for si, sd in enumerate(SEEDS):
                    m = XGBClassifier(**{**IM.XGB, "random_state": sd, "n_jobs": jobs})
                    m.fit(Xtr, ytr, verbose=False)
                    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                    iso.fit(m.predict_proba(Xca)[:, 1], yca)
                    P[v][arm][si, ev] = np.clip(iso.predict(m.predict_proba(Xev)[:, 1]), CLIP, 1 - CLIP)
        print(f"  起点 {o.date()}: 之前 {rec['games_before']} 局 | 现: 树 →{rec['cur_tree'][1]} "
              f"保序 {rec['cur_iso'][0]}→{rec['cur_iso'][1]} | 候: 树 →{rec['cand_tree'][1]} "
              f"保序 {rec['cand_iso'][0]}→{rec['cand_iso'][1]} | 评估 {rec['eval_games']} 局 "
              f"{rec['eval_snaps']} 快照  [{time.time() - t0:.0f}s]", flush=True)
    assert all(not np.isnan(P[v][a]).any() for v in P for a in P[v])

    games, series = E.gameid.values, series_key(E)
    maj = E.league.isin(MAJORS).values
    res = {"info": info}
    for v in variants:
        for scope, msk in (("all", np.ones(len(E), bool)), ("majors", maj)):
            f2 = np.where(msk, fold, -1)
            res[f"{v}/{scope}"] = evaluate(y, P[v], f2, games, series, len(ORIGINS), IM.ece)
    # 按时间切片的合计 Brier (live, 全部赛区), 只看不判
    res["live/by_T"] = {}
    for T in IM.SLICES:
        mT = (E["T"] == T).values
        res["live/by_T"][str(T)] = {
            a: float(((P["live"][a][:, mT] - y[mT]) ** 2).mean()) for a in ARMS}
    return res


# ══════════════════════════════════════════════════════════
#  Stage 1/2 (只作信息)
# ══════════════════════════════════════════════════════════

def run_stage12(mdf, jobs):
    from feature_store import select_features
    from train import CAL_FRAC, TRAIN_FRAC, XGB, ece
    mdf = mdf.reset_index(drop=True)
    y = mdf.y.astype(int).values
    fold = np.full(len(mdf), -1)
    for i, o in enumerate(ORIGINS):
        fold[((mdf.date >= o) & (mdf.date < o + MONTH)).values] = i
    games, series = mdf.gameid.values, series_key(mdf)
    res = {}
    for st in ("pre_draft", "post_draft"):
        fs = select_features(mdf, st)
        X = mdf[fs].astype(float).fillna(-999)
        P = {a: np.full((len(SEEDS), len(mdf)), np.nan) for a in ARMS}
        info = []
        for i, o in enumerate(ORIGINS):
            pos = np.flatnonzero((mdf.date < o).values)
            if not np.array_equal(pos, np.arange(len(pos))):
                print(f"  ⚠ {st} 起点 {o.date()}: 起点前的行不是矩阵前缀 (矩阵没按日期排), 仍按行序切")
            n = len(pos)
            a, b = int(n * TRAIN_FRAC), int(n * (TRAIN_FRAC + CAL_FRAC))      # train.py 的切法
            ev = np.flatnonzero(fold == i)
            info.append({"origin": str(o.date()), "rows_before": n,
                         "cur_tree_end": str(mdf.date.iloc[pos[a - 1]].date()),
                         "cand_tree_end": str(mdf.date.iloc[pos[b - 1]].date()), "eval_games": int(len(ev))})
            for arm, tr, ca in (("CURRENT", pos[:a], pos[a:b]), ("CANDIDATE", pos[:b], pos[b:])):
                for si, sd in enumerate(SEEDS):
                    m = XGBClassifier(**{**XGB, "random_state": sd, "n_jobs": jobs})
                    m.fit(X.iloc[tr], y[tr], verbose=False)
                    cal = LogisticRegression(C=1e6).fit(m.predict_proba(X.iloc[ca])[:, 1].reshape(-1, 1), y[ca])
                    P[arm][si, ev] = cal.predict_proba(m.predict_proba(X.iloc[ev])[:, 1].reshape(-1, 1))[:, 1]
        r = evaluate(y, P, fold, games, series, len(ORIGINS), ece)
        r["info"] = info
        res[st] = r
        print(f"  {st}: {len(fs)} 特征; 起点 {info[0]['origin']} 现树 →{info[0]['cur_tree_end']} 候树 →{info[0]['cand_tree_end']}"
              f" … 起点 {info[-1]['origin']} 现树 →{info[-1]['cur_tree_end']} 候树 →{info[-1]['cand_tree_end']}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None, help="快照表 / 矩阵缓存目录 (放在项目外)")
    ap.add_argument("--json", default=None, help="结果写到这个 JSON")
    ap.add_argument("--skip-stage12", action="store_true")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    a = ap.parse_args()
    assert IM.SLICES == [10, 15, 20, 25]

    print("=" * 78)
    print("  闸门 G2: Stage 4 服务模型重拟 (树 80% + 保序 最近 20%  vs  今天的 60/20)")
    print("=" * 78)
    print(f"  起点 {', '.join(str(o.date()) for o in ORIGINS)}; 种子 {SEEDS}; 判据 fold 配对 t > {T_ACCEPT}")

    print("\n[1] Stage 4 快照表")
    m4 = build_snapshots(a.cache)
    print(f"  {m4.gameid.nunique()} 局 / {len(m4)} 快照, {m4.date.min().date()} → {m4.date.max().date()}")
    if m4.date.max() < ORIGINS[-1] + MONTH - pd.Timedelta(days=1):
        print("  ⚠ 数据没覆盖到最后一个评估月的月底")

    print("\n[2] Stage 4 滚动起点")
    r4 = run_stage4(m4, a.jobs)
    show("live 变体 (26 特征), 全部赛区 —— 主判据", r4["live/all"], primary=True)
    show("live 变体, 四大赛区 (次判据)", r4["live/majors"])
    show("full 变体 (28 特征), 全部赛区", r4["full/all"])
    show("full 变体, 四大赛区", r4["full/majors"])
    print("\n  live 全部赛区, 按切片的合计 Brier (现 → 候):  " + "   ".join(
        f"T{T} {v['CURRENT']:.4f}→{v['CANDIDATE']:.4f}" for T, v in r4["live/by_T"].items()))
    out = {"stage4": r4, "data_through": str(m4.date.max().date()), "seeds": list(SEEDS),
           "origins": [str(o.date()) for o in ORIGINS]}

    if not a.skip_stage12:
        print("\n[3] Stage 1/2 同类比较 (只作信息, 不判决)")
        mdf = build_matrix12(a.cache)
        print(f"  {len(mdf)} 局, {mdf.date.min().date()} → {mdf.date.max().date()}")
        r12 = run_stage12(mdf, a.jobs)
        show("Stage 1 pre_draft (树 80% / Platt 20% vs 60/20)", r12["pre_draft"], unit="行")
        show("Stage 2 post_draft (树 80% / Platt 20% vs 60/20)", r12["post_draft"], unit="行")
        out["stage12"] = r12

    t = r4["live/all"]["fold_brier"]["t"]
    print("\n" + "=" * 78)
    print(f"  判决: live / 全部赛区 fold 配对 t = {t:+.2f} → "
          + ("CANDIDATE 通过 (t > 2.5)" if t > T_ACCEPT else "CANDIDATE 未通过 (t ≤ 2.5), 维持今天的配方"))
    print("=" * 78)
    out["verdict"] = {"t": t, "passes": bool(t > T_ACCEPT)}
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
        print(f"  结果 → {a.json}")


if __name__ == "__main__":
    main()
