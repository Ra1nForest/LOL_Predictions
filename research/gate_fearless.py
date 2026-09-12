"""
闸门: 全局 BP (Fearless) 的"英雄池损耗"能不能改进赛前预测?
==========================================================
2025 年起四大赛区全是硬全局 BP: 同一系列赛里任何一方用过的英雄, 后面几局双方都不能再选。
数据核过 (2026-09-13): 2025-2026 共 1260 个系列赛, 英雄重复只有 2 例, 都在 2025-03 LPL 的
切换期; 2022-2024 年几乎每个系列赛都有重复。于是第 N 局 BP 之前就已经知道哪些英雄没了 ——
而赛前模型 (Stage 1) 把每局当独立比赛, 看不到这一层。

候选 A, 英雄池损耗 (全在 BP 之前可知, 不看本局阵容):
  fl_top5   每名选手近 365 天最常用的 5 个英雄里, 已在本系列赛被用掉的比例, 五人平均
  fl_share  每名选手近 365 天的出场里, 落在已被用掉的英雄上的比例, 五人平均
  各有 b_/r_/diff_ 三份。第 1 局、以及 2025 年以前的系列赛没有英雄被禁, 取值就是 0 ——
  这是真实取值, 不是缺失。
候选 B, 系列赛上下文 (第几局、之前的比分差、上一局谁赢): 同样赛前可知、现在也没用。
  单独过闸, 免得把它的效果算到全局 BP 头上。

选手用本局实际上场的五人 (首发名单赛前就公布); 英雄池只看系列赛当天之前的比赛。

判据: 和 train.py 一样的 XGBoost + Platt。测试段取 2025 年起 (全局 BP 时代) 按时间切成
不重叠的 N 段, 每段之前一段做校准、再之前全部做训练, 每段重新训练。看 Brier / 对数损失 /
准确率, 按段配对 t, |t| > 2.5 才算。另报"受影响的局" (全局 BP 的第 2 局及以后) 上的指标 ——
整个测试段里受影响的只占一部分, 效果会被稀释。

    python research/gate_fearless.py [--folds 6] [--seeds 3]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from xgboost import XGBClassifier

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "research"))
from feature_store import LEAGUES, ROLES, build_training_matrix, select_features  # noqa: E402
from harness import T_ACCEPT, paired_t  # noqa: E402
from train import CAL_FRAC, DATA, XGB  # noqa: E402  训练用的同一份数据和参数

FEARLESS_FROM = pd.Timestamp("2025-01-01")
LOOKBACK_DAYS = 365
FL = ["b_fl_top5", "r_fl_top5", "diff_fl_top5", "b_fl_share", "r_fl_share", "diff_fl_share"]
CTX = ["ctx_game_no", "ctx_wins_diff", "ctx_prev"]


def load_rows():
    use = ["gameid", "league", "date", "game", "teamname", "position", "champion", "playername", "result"]
    df = pd.concat([pd.read_csv(f, usecols=use, low_memory=False) for f in DATA if Path(f).exists()],
                   ignore_index=True)
    df["league"] = df["league"].replace({"LTA N": "LCS", "LTA S": "CBLOL"})
    df = df[df["league"].isin(LEAGUES)].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df[df["position"].isin(ROLES)].copy(), df[df["position"] == "team"].copy()


def series_features(pl: pd.DataFrame, tm: pd.DataFrame) -> dict:
    games = {}
    for gid, d in pl.groupby("gameid"):
        teams = sorted(d["teamname"].astype(str).unique())
        if len(d) != 10 or len(teams) != 2 or pd.isna(d["game"].iloc[0]):
            continue
        r = d.iloc[0]
        games[gid] = dict(day=r.date.normalize(), game=int(r.game), key=(r.league, r.date.normalize(), tuple(teams)),
                          champs=set(d["champion"].astype(str)),
                          players={t: list(d[d.teamname == t]["playername"].astype(str)) for t in teams})
    winner = dict(zip(tm[tm["result"] == 1]["gameid"], tm[tm["result"] == 1]["teamname"].astype(str)))

    hist: dict = defaultdict(list)
    for p, c, dt in pl[["playername", "champion", "date"]].itertuples(index=False):
        hist[str(p)].append((dt, str(c)))
    for p in hist:
        hist[p].sort()
    hdates = {p: np.array([d for d, _ in v], dtype="datetime64[ns]") for p, v in hist.items()}

    def pool(player, day):
        v = hist.get(player)
        if not v:
            return None
        arr = hdates[player]
        hi = np.searchsorted(arr, np.datetime64(day), side="left")          # 当天之前, 不看未来
        lo = np.searchsorted(arr, np.datetime64(day - pd.Timedelta(days=LOOKBACK_DAYS)), side="left")
        cnt = Counter(c for _, c in v[lo:hi])
        return cnt or None

    ser = defaultdict(list)
    for gid, g in games.items():
        ser[g["key"]].append((g["game"], gid))

    out, repeats = {}, Counter()
    for key, lst in ser.items():
        lst.sort()
        day, teams = key[1], key[2]
        fearless = day >= FEARLESS_FROM
        used: set = set()
        wins = {t: 0 for t in teams}
        prev = None
        pools: dict = {}
        for gno, gid in lst:
            g = games[gid]
            repeats[(fearless, bool(g["champs"] & used))] += 1
            ban = used if fearless else set()
            team = {}
            for t in teams:
                l5, ls = [], []
                for p in g["players"][t]:
                    if p not in pools:
                        pools[p] = pool(p, day)
                    cnt = pools[p]
                    if not cnt:
                        continue
                    top = [c for c, _ in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[:5]]
                    l5.append(sum(c in ban for c in top) / len(top))
                    ls.append(sum(n for c, n in cnt.items() if c in ban) / sum(cnt.values()))
                team[t] = (float(np.mean(l5)) if l5 else np.nan, float(np.mean(ls)) if ls else np.nan)
            out[gid] = dict(team=team, game_no=gno, wins=dict(wins), prev=prev, fearless=fearless)
            used |= g["champs"]
            w = winner.get(gid)
            if w in wins:
                wins[w] += 1
            prev = w
    print(f"  系列赛 {len(ser)} 个 (含 BO1), 局 {len(out)}")
    print(f"  同一系列赛英雄重复的局: 全局 BP 时代 {repeats[(True, True)]} / {repeats[(True, True)] + repeats[(True, False)]}, "
          f"之前 {repeats[(False, True)]} / {repeats[(False, True)] + repeats[(False, False)]}")
    return out


def add_features(mdf: pd.DataFrame, sf: dict) -> pd.DataFrame:
    rows = []
    for r in mdf[["gameid", "blue_team", "red_team"]].itertuples(index=False):
        f = sf.get(r.gameid)
        if f is None:                                    # 系列赛拼不出来的: 当第 1 局处理
            rows.append((0.0, 0.0, 0.0, 0.0, 1, 0, 0, False))
            continue
        b = f["team"].get(r.blue_team, (np.nan, np.nan))
        d = f["team"].get(r.red_team, (np.nan, np.nan))
        prev = 1 if f["prev"] == r.blue_team else -1 if f["prev"] == r.red_team else 0
        wd = f["wins"].get(r.blue_team, 0) - f["wins"].get(r.red_team, 0)
        rows.append((b[0], d[0], b[1], d[1], f["game_no"], wd, prev, f["fearless"] and f["game_no"] >= 2))
    a = pd.DataFrame(rows, columns=["b5", "r5", "bs", "rs", "gno", "wd", "prev", "affected"], index=mdf.index)
    mdf = mdf.copy()
    mdf["b_fl_top5"], mdf["r_fl_top5"], mdf["diff_fl_top5"] = a.b5, a.r5, a.b5 - a.r5
    mdf["b_fl_share"], mdf["r_fl_share"], mdf["diff_fl_share"] = a.bs, a.rs, a.bs - a.rs
    mdf["ctx_game_no"], mdf["ctx_wins_diff"], mdf["ctx_prev"] = a.gno, a.wd, a.prev
    mdf["_affected"] = a.affected
    return mdf


def fit_predict(Xtr, ytr, Xca, yca, Xte, seed):
    m = XGBClassifier(**{**XGB, "random_state": seed})
    m.fit(Xtr, ytr, verbose=False)
    cal = LogisticRegression(C=1e6).fit(m.predict_proba(Xca)[:, 1].reshape(-1, 1), yca)
    return cal.predict_proba(m.predict_proba(Xte)[:, 1].reshape(-1, 1))[:, 1]


def metrics(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"brier": -brier_score_loss(y, p), "logloss": -log_loss(y, p, labels=[0, 1]),
            "acc": float(((p > 0.5).astype(int) == y).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()

    print("构建训练矩阵 (和 train.py 同一份) …")
    mdf = build_training_matrix(DATA, verbose=False)
    mdf = mdf.sort_values("date", kind="stable").reset_index(drop=True)
    base = select_features(mdf, "pre_draft")
    print(f"  {len(mdf)} 场, 赛前特征 {len(base)} 个")
    print("拼系列赛, 算英雄池损耗 …")
    pl, tm = load_rows()
    mdf = add_features(mdf, series_features(pl, tm))

    aff = mdf["_affected"].values
    fear = (mdf["date"] >= FEARLESS_FROM).values
    print(f"  全局 BP 时代 {fear.sum()} 场, 其中受影响 (第 2 局及以后) {aff.sum()} 场")
    sub = mdf[aff]
    lower = np.sign(-sub["diff_fl_top5"])            # +1: 蓝方英雄池损失更少
    known = lower != 0
    print(f"  受影响的局里, 英雄池损失更少的一方胜率: "
          f"{((lower[known] > 0) == (sub['y'][known] == 1)).mean():.1%}  (n={int(known.sum())})")

    variants = {"基线": base, "+英雄池损耗": base + FL, "+系列赛上下文": base + CTX, "+两者": base + FL + CTX}
    y = mdf["y"].astype(int).values
    cal_n = int(len(mdf) * CAL_FRAC)
    start = int(np.argmax(fear))
    blocks = np.array_split(np.arange(start, len(mdf)), a.folds)
    res = {v: {"all": defaultdict(list), "aff": defaultdict(list)} for v in variants}
    for k, blk in enumerate(blocks):
        s, e = blk[0], blk[-1] + 1
        tr, ca, te = slice(0, s - cal_n), slice(s - cal_n, s), slice(s, e)
        line = []
        for name, feats in variants.items():
            X = mdf[feats].astype(float).fillna(-999)
            p = np.mean([fit_predict(X.iloc[tr], y[tr], X.iloc[ca], y[ca], X.iloc[te], sd)
                         for sd in range(a.seeds)], axis=0)
            for kk, vv in metrics(y[te], p).items():
                res[name]["all"][kk].append(vv)
            m_aff = aff[te]
            for kk, vv in metrics(y[te][m_aff], p[m_aff]).items():
                res[name]["aff"][kk].append(vv)
            line.append(f"{name} {-res[name]['all']['brier'][-1]:.4f}")
        print(f"  段 {k + 1}: 训练 {s - cal_n} / 校准 {cal_n} / 测试 {e - s} (受影响 {int(aff[te].sum())})  "
              f"{mdf['date'].iloc[s].date()} → {mdf['date'].iloc[e - 1].date()}   " + "  ".join(line))

    for scope, label in (("all", "整个测试段"), ("aff", "只看受影响的局")):
        print(f"\n── {label} ──")
        print(f"  {'':<14}{'Brier':>9}{'对数损失':>10}{'准确率':>9}")
        for name in variants:
            r = res[name][scope]
            print(f"  {name:<14}{-np.mean(r['brier']):>9.4f}{-np.mean(r['logloss']):>10.4f}{np.mean(r['acc']):>9.1%}")
        for name in list(variants)[1:]:
            bits = []
            for metric in ("brier", "logloss", "acc"):
                d, _sd, t, _ = paired_t(np.array(res["基线"][scope][metric]), np.array(res[name][scope][metric]))
                bits.append(f"{metric} Δ{d:+.4f} t={t:+.2f}")
            verdict = ("过闸" if all(paired_t(np.array(res["基线"][scope][m]), np.array(res[name][scope][m]))[2] > T_ACCEPT
                                 for m in ("brier", "logloss")) else "不过闸")
            print(f"  {name} vs 基线:  " + "   ".join(bits) + f"   -> {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
