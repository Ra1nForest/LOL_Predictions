"""
闸门 G3 的次要检查: 逐分钟回填上的同一个"经济差 ±200 邻域平均"去抖 (Python, live 变体)
=====================================================================================
主闸门是 research/gate_dechatter.mjs (逐秒走势, 浏览器那份实现, 160 局)。这里换一份实现、换一批更多的局,
在逐分钟回填快照上复核同一个候选, 看结论是不是只是那 160 局的运气。**协议同主闸门, 看结果之前定死:**

  候选  每一行把经济差换成 [gd − 200, gd + 200] 里等距的 9 个值, 由经济差派生的列 (golddiff_norm,
        x_gold_scaling, x_goldnorm_scaling, lead_by_early_comp) 用 merge_backfill.derive 重算 (公式照抄
        ingame_model.build, 和 make_row 一致; 总经济 gold_total 是独立输入, 不动), 9 个**校准后**的 live
        概率取平均。切片取 nearest_slice (线上口径)。然后原样走看板的 3–15 分钟渐变 (api._blend_prior)。
  基线  同一行原样的校准概率, 同样渐变。两边都 round 到 4 位 (看板吐出来的精度)。
  考卷  backfill/snapshots.jsonl 按 oe_gameid 接上 ingame_model.build() 的产出 (借赛前特征、阵容强势期、
        is_*、标签 —— merge_backfill.borrow_cols, 同 gate_anchor / gate_objectives), 只留 Stage 4 测试段
        (split_by_game 的最后 20%) 里日期 ≥ 2026-01-15 的局。选边和 OE 对不上的局 (帧上第 10/15 分钟的经济差
        和 OE 的 golddiffat10/15 反号, |OE| > 300) 丢掉 —— 标签是 OE 蓝方的, 帧蓝方不是同一队时整局标签是反的。
  渐变锚  四大赛区: BP 后 (Stage 2) 概率, 取 feature_store.build_training_matrix 测试段里这一局的那一行
        (同 gate_anchor; 看板对国内比赛用的就是 BP 后)。国际赛和次级联赛: 回填里拿不到看板的锚
        (同母赛区要 BP 名单, 跨赛区要模型 C 的赛事内重放), 不渐变。两边永远同一个锚。
        另报一行"全部不渐变" (= /predict/ingame 的数) 作参照。
  判据  每局所有逐分钟点的平均 Brier, 逐局配对 t 和系列赛配对 t (系列赛 = 同赛区、同一对队伍、
        相邻两局开赛间隔 ≤ 12 小时连成一串, 同主闸门)。t > 0 = 候选更好。子集: 看板服务的赛区 (四大 + 国际赛,
        和主闸门同一个范围) / 全部赛区; 分钟段 5–15 / 15–25 / 25+。另报每局无事件 (人头/塔/龙/大龙/水晶
        都没变) 的 ≥4pp / ≥2pp 跳变 (相邻两分钟, 间隔 < 1.5 分钟)。

    python research/gate_dechatter_backfill.py

只读 artifacts/ (线上 live 模型和 Stage 2), 不训练任何东西。

**结果 (2026-10-17): 同向, 不显著变差 (而且是变好的方向) —— 主闸门的结论不是那 160 局的运气。**
1663 局 / 46234 行 (选边对不上丢 7 局); 看板赛区 756 局, 其中四大 601 局全都有 BP 后锚。

    看板口径 (四大从 BP 后渐变)      Δ(基线−候选, 逐局)   逐局 t   系列赛 t
      看板赛区 756 局 / 298 系列     +0.00011            +2.36    +2.51
      四大     601 / 232             +0.00009            +1.96    +1.51
      国际赛   155 / 66              +0.00019            +1.33    +2.09
      全部赛区 1663 / 683            +0.00009            +2.51    +1.74
    全部不渐变 (参照, = /predict/ingame 的数)
      看板赛区                       +0.00017            +2.98    +3.19
      全部赛区                       +0.00012            +3.01    +2.25

好处集中在开局 5-15 分钟 (看板赛区、不渐变: 系列赛 t +3.01; 渐变把它压到 +2.21), 25+ 分钟持平
(全部赛区 −0.91, 所有子集和分钟段里最差的一格)。相邻两分钟的无事件 >=4pp 跳变每局 2.11 → 1.79 (看板赛区)
—— 逐分钟的步子本来就以真实局势变化为主, 去抖的效果要在逐秒走势上看 (主闸门: 25.9 → 7.6)。
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

import ingame_model as IM                               # noqa: E402
from harness import T_ACCEPT, paired_t                  # noqa: E402
from ingame_service import IngameModel                  # noqa: E402
from merge_backfill import borrow_cols, derive          # noqa: E402
from multiyear_gate import data_path                    # noqa: E402

SNAP = ROOT / "backfill" / "snapshots.jsonl"
YEARS = (2022, 2023, 2024, 2025, 2026)                  # = ingame_train.YEARS = api.DATA
SINCE = "2026-01-15"
K, NPTS = 200, 9
SHIFTS = np.linspace(-K, K, NPTS)                       # −200, −150, …, +200
MAJORS = {"LPL", "LCK", "LEC", "LCS"}
INTL = {"WORLDS", "MSI", "FIRST STAND", "DCGI", "ESPORTS WORLD CUP"}   # = esports_feed.LEAGUE_IDS 的国际赛
BLEND_FROM, BLEND_TO = 3.0, 15.0                        # = api.BLEND_FROM / BLEND_TO
KEEP = ["game_id", "oe_gameid", "league", "date", "blue", "red", "minute", "golddiff", "csdiff",
        "blue_kills", "red_kills", "gold_total", "blue_towers", "red_towers", "blue_inhibitors",
        "red_inhibitors", "blue_barons", "red_barons", "blue_dragons", "red_dragons"]
EVENTS = ["blue_kills", "red_kills", "blue_towers", "red_towers", "blue_dragons", "red_dragons",
          "blue_barons", "red_barons", "blue_inhibitors", "red_inhibitors"]
SEGS = [("5-15", 0, 15), ("15-25", 15, 25), ("25+", 25, 999), ("全部", 0, 999)]


def blend_prior(prior, ingame, minute):
    """api._blend_prior, 逐元素。"""
    w = min(1.0, max(0.0, (float(minute) - BLEND_FROM) / (BLEND_TO - BLEND_FROM)))
    if w >= 1.0:
        return ingame
    cl = lambda q: min(max(q, 1e-6), 1 - 1e-6)
    lg = lambda q: math.log(q / (1 - q))
    z = (1 - w) * lg(cl(prior)) + w * lg(cl(ingame))
    return 1 / (1 + math.exp(-z))


def series_ids(g: pd.DataFrame) -> pd.Series:
    """同赛区 + 同一对队伍, 按开赛时间排, 相邻两局间隔 ≤ 12 小时连成一串 (同 gate_dechatter.mjs)。"""
    g = g.copy()
    g["t"] = pd.to_datetime(g["date"])
    g["pair"] = g.league.str.upper() + "|" + g[["blue", "red"]].apply(lambda r: "|".join(sorted(r)), axis=1)
    out = {}
    for pair, s in g.sort_values("t").groupby("pair"):
        sid, last = 0, None
        for gid, t in zip(s.game_id, s.t):
            if last is None or (t - last).total_seconds() > 12 * 3600:
                sid += 1
            last = t
            out[gid] = f"{pair}#{sid}"
    return pd.Series(out)


def main():
    t0 = time.time()
    print("[1/4] OE 快照表 (ingame_model.build)")
    df = IM.load([data_path(y) for y in YEARS])
    mdf, _ = IM.build(df, verbose=False)
    _tr, _cal, te = IM.split_by_game(mdf)
    const = mdf.drop_duplicates("gameid").set_index("gameid")[borrow_cols(mdf)]
    oe = mdf.pivot_table(index="gameid", columns="T", values="golddiff")
    print(f"      {time.time() - t0:.0f}s")

    print("[2/4] 回填考卷")
    bf = pd.DataFrame([json.loads(l) for l in open(SNAP, encoding="utf-8") if l.strip()])
    bf = bf[bf.oe_gameid.notna()].drop_duplicates(["game_id", "minute"])
    n0 = bf.game_id.nunique()
    bf = bf[bf.oe_gameid.isin(set(te.gameid)) & (bf.date >= SINCE)]
    n1 = bf.game_id.nunique()
    flip = []
    for gid, g in bf.groupby("game_id"):
        o_id = g.oe_gameid.iloc[0]
        bad = chk = 0
        for T in (10, 15):
            s = g[np.floor(g.minute) == T]
            if s.empty or o_id not in oe.index or pd.isna(oe.loc[o_id, T]):
                continue
            o, v = float(oe.loc[o_id, T]), float(s.golddiff.iloc[0])
            if abs(o) > 300:
                chk += 1
                bad += abs(v - o) > abs(v + o)
        if chk and bad == chk:
            flip.append(gid)
    bf = bf[~bf.game_id.isin(flip)]
    c = const.drop(columns=["league"])
    X = pd.DataFrame([derive(r) for r in bf[KEEP].merge(c, left_on="oe_gameid", right_index=True).to_dict("records")])
    X["y"] = X["y"].astype(int)
    X["board"] = X.league.str.upper().isin(MAJORS | INTL)
    X["major"] = X.league.str.upper().isin(MAJORS)
    X = X.sort_values(["game_id", "minute"]).reset_index(drop=True)
    print(f"      回填 {n0} 局, 在 Stage 4 测试段且 ≥ {SINCE} 的 {n1} 局, 选边对不上丢 {len(flip)} 局 → "
          f"{X.game_id.nunique()} 局 / {len(X)} 行 (看板赛区 {X[X.board].game_id.nunique()} 局, 四大 {X[X.major].game_id.nunique()})")

    print("[3/4] 渐变锚 (四大: Stage 2 测试段那一行) + 两种概率")
    import api                                          # 只为 Stage 和 DATA, 不起服务
    from feature_store import build_training_matrix
    mdf2 = build_training_matrix(api.DATA, verbose=False)
    test2 = mdf2.iloc[int(len(mdf2) * 0.80):]
    s2 = api.Stage("post_draft")
    rows2 = {r["gameid"]: r for r in test2.to_dict("records")}
    anchor = {}
    for gid, o_id, mj in X.drop_duplicates("game_id")[["game_id", "oe_gameid", "major"]].itertuples(index=False):
        if mj and o_id in rows2:
            anchor[gid] = s2.predict(rows2[o_id])[1]
    nmaj = X[X.major].game_id.nunique()
    print(f"      四大 {nmaj} 局里有 BP 后锚的 {sum(1 for g in X[X.major].game_id.unique() if g in anchor)} 局")

    live = IngameModel("_live")
    X["T"] = X.minute.map(live.nearest_slice)

    def cal(frame: pd.DataFrame) -> np.ndarray:
        x = np.nan_to_num(frame[live.features].astype(float).values, nan=-999.0)
        raw = live.model.predict_proba(x)[:, 1]
        return np.clip(np.interp(raw, live.iso_x, live.iso_y), live.clip, 1 - live.clip)

    base_cols = [c for c in X.columns]
    recs = X[base_cols].to_dict("records")
    p0 = cal(X)
    ps = []
    for d in SHIFTS:
        sh = pd.DataFrame([derive({**r, "golddiff": r["golddiff"] + d}) for r in recs])
        ps.append(cal(sh))
    assert np.array_equal(ps[NPTS // 2], p0), "挪动 0 的那一份和原样不等 —— 派生列重算出了偏差"
    pc = np.mean(ps, axis=0)
    for lab, use_anchor in (("blend", True), ("raw", False)):
        b, cc = np.empty(len(X)), np.empty(len(X))
        for i, (gid, m) in enumerate(zip(X.game_id, X.minute)):
            a = anchor.get(gid) if use_anchor else None
            b[i] = round(blend_prior(a, p0[i], m) if a is not None else p0[i], 4)
            cc[i] = round(blend_prior(a, pc[i], m) if a is not None else pc[i], 4)
        X[f"pb_{lab}"], X[f"pc_{lab}"] = b, cc
        X[f"bb_{lab}"], X[f"bc_{lab}"] = (b - X.y) ** 2, (cc - X.y) ** 2

    print(f"[4/4] 结果  (Brier 越小越好; t > 0 = 候选更好; 采纳规则同主闸门: 系列赛 t > −{T_ACCEPT})")
    sid = series_ids(X.drop_duplicates("game_id")[["game_id", "league", "date", "blue", "red"]])
    X["series"] = X.game_id.map(sid)
    for lab, name in (("blend", "看板口径 (四大从 BP 后渐变)"), ("raw", "全部不渐变 (参照)")):
        print(f"\n  == {name} ==")
        for sub_lab, sub in (("看板赛区 (四大 + 国际赛)", X[X.board]), ("四大", X[X.major]),
                             ("国际赛", X[X.board & ~X.major]), ("全部赛区", X)):
            print(f"   [{sub_lab}]")
            for seg, lo, hi in SEGS:
                s = sub[(sub.minute >= lo) & (sub.minute < hi)]
                if s.game_id.nunique() < 3:
                    continue
                gg = s.groupby("game_id")[[f"bb_{lab}", f"bc_{lab}"]].mean()
                ss = s.groupby("series")[[f"bb_{lab}", f"bc_{lab}"]].mean()
                dg, _, tg, _ = paired_t(-gg[f"bb_{lab}"].values, -gg[f"bc_{lab}"].values)
                ds, _, ts, _ = paired_t(-ss[f"bb_{lab}"].values, -ss[f"bc_{lab}"].values)
                print(f"     分钟 {seg:<6}{len(gg):>5} 局 /{len(ss):>5} 系列  基线 {gg[f'bb_{lab}'].mean():.5f}"
                      f"  候选 {gg[f'bc_{lab}'].mean():.5f}  Δ 逐局 {dg:+.5f} t {tg:+.2f}   系列赛 {ds:+.5f} t {ts:+.2f}")

    # 跳变: 相邻两分钟 (间隔 < 1.5), 无事件 = 人头/塔/龙/大龙/水晶都没变; 看板口径那一份
    g = X.groupby("game_id")
    dm = g.minute.diff()
    ev = g[EVENTS].diff().abs().sum(axis=1) > 0
    ok = dm.notna() & (dm < 1.5)
    noev = ok & ~ev
    print("\n  跳变 (看板口径, 相邻两分钟; 每局平均次数)")
    for sub_lab, msk in (("看板赛区", X.board), ("全部赛区", pd.Series(True, index=X.index))):
        ng = X[msk].game_id.nunique()
        line = f"   [{sub_lab}, {ng} 局]"
        for k in ("pb_blend", "pc_blend"):
            dp = g[k].diff().abs()
            j4 = ((dp >= 0.04 - 1e-9) & noev & msk).sum() / ng
            j2 = ((dp >= 0.02 - 1e-9) & noev & msk).sum() / ng
            line += f"  {'基线' if k.startswith('pb') else '候选'}: 无事件 >=4pp {j4:.2f}, >=2pp {j2:.2f}"
        print(line)
    print(f"\n  (相邻两分钟里无事件的占 {noev.sum() / ok.sum():.1%}; 用时 {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
