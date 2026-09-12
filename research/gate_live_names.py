"""
看板 BP 后预测: 名字对齐前后, 在模型没见过的比赛上谁更准
===========================================================
2026-09-12 查出看板把 lolesports 的名字原样查 OE 训练表 —— 选手名带战队前缀 (IGTheShy),
英雄是 Data Dragon id (LeeSin), 见 esports_feed.oe_player_name / feature_store.champion_key。
修复是恢复训练/线上一致; 这里量它在**测试集**上的实际效果: train.py 按日期留出的最后 20%,
当前 artifacts/ 的模型训练和校准都没见过。

每一局按看板的方式算两次, 其余输入完全相同:
  修复前  选手名带战队前缀 (查不到), 英雄用 Data Dragon id 且原样查表
  修复后  选手名是 OE 名字, 英雄用 Data Dragon id, 经 champion_key 对齐
特征都按比赛日期截断 (as_of), 和训练一样不看未来。修复后那一路的 BP 特征还要和训练矩阵
逐列相等 —— 这是"线上和训练一致"的直接证据。

判据: 逐局配对 (同一局两种输入); 另把测试集按时间切成 5 段按段配对, 和 harness.gate
同一套 |t| > 2.5。分数取负损失, t 为正表示修复后更好。

    python research/gate_live_names.py        (需要联网取 Data Dragon 的英雄 id 表)
"""
import json
import math
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

import api                                                    # noqa: E402  Stage 和 DATA
from feature_store import (FeatureStore, ROLES, LEAGUES,      # noqa: E402
                           build_training_matrix, select_features, _is_draft_feature)
from harness import T_ACCEPT, paired_t                        # noqa: E402

TRAIN_FRAC, CAL_FRAC = 0.60, 0.20      # 和 train.py 一致 —— 测试集就是模型没见过的那一段
N_BLOCKS = 5


def ddragon_ids():
    """OE 显示名 → Data Dragon id。看板收到的是后者。"""
    get = lambda u: json.load(urllib.request.urlopen(u, timeout=30))
    v = get("https://ddragon.leagueoflegends.com/api/versions.json")[0]
    data = get(f"https://ddragon.leagueoflegends.com/cdn/{v}/data/en_US/champion.json")["data"]
    return v, {c["name"]: c["id"] for c in data.values()}


def same(a, b, tol=0.0) -> bool:
    """tol 给 *_avg_champ_wr 那一族用: 训练矩阵里一局十个人的先后顺序被 df.sort_values("date")
    (默认不稳定排序) 打乱过, 五个胜率相加的顺序和线上 (按 top..sup) 不同, 末位会差一个 ulp
    (实测最大 3.3e-16)。那是训练端本来就有的, 和名字对不对齐无关; 胜率/场数本身必须逐位相等。"""
    fa, fb = float(a), float(b)
    return (math.isnan(fa) and math.isnan(fb)) or abs(fa - fb) <= tol


def main():
    ver, to_id = ddragon_ids()
    print(f"Data Dragon {ver}: {len(to_id)} 个英雄, 其中显示名和 id 不同的 "
          f"{sum(k != v for k, v in to_id.items())} 个")

    print("构建训练矩阵和特征库 (五年 CSV) …")
    mdf = build_training_matrix(api.DATA, verbose=False)
    n = len(mdf)
    test = mdf.iloc[int(n * (TRAIN_FRAC + CAL_FRAC)):].reset_index(drop=True)
    store = FeatureStore.from_csv(api.DATA, verbose=False)
    s1, s2 = api.Stage("pre_draft"), api.Stage("post_draft")
    draft_feats = [f for f in select_features(mdf, "post_draft") if _is_draft_feature(f)]

    raw = pd.concat([pd.read_csv(f, low_memory=False,
                                 usecols=["gameid", "league", "position", "side", "playername", "champion"])
                     for f in api.DATA if Path(f).exists()], ignore_index=True)
    raw["league"] = raw["league"].replace({"LTA N": "LCS", "LTA S": "CBLOL"})
    raw = raw[raw["league"].isin(LEAGUES) & raw["position"].isin(ROLES) & raw["gameid"].isin(set(test["gameid"]))]
    players = {g: d for g, d in raw.groupby("gameid")}
    unmapped = sorted(set(raw["champion"].dropna()) - set(to_id))
    if unmapped:
        print(f"  ⚠ Data Dragon 里没有的 OE 英雄名 (原样使用): {unmapped}")

    ys, pf, pb, p1, dates = [], [], [], [], []
    parity_bad, skipped = 0, 0
    found = {"fixed": [0, 0], "broken": [0, 0]}          # [查到的选手-英雄槽位, 总槽位]
    for r in test.to_dict("records"):
        gp = players.get(r["gameid"])
        if gp is None or len(gp) != 10:
            skipped += 1
            continue
        fixed, broken = {"blue": {}, "red": {}}, {"blue": {}, "red": {}}
        for p in gp.itertuples(index=False):
            side = "blue" if p.side == "Blue" else "red"
            cid = to_id.get(p.champion, p.champion)
            fixed[side][p.position] = (str(p.playername), cid)
            broken[side][p.position] = ("XX" + str(p.playername), cid)   # 带前缀: 看板修复前的样子
        kw = dict(playoffs=int(r["playoffs"]), as_of=r["date"])
        try:
            store.__dict__.pop("oe_champion", None)
            rf, _ = store.make_row(r["blue_team"], r["red_team"], r["league"], draft=fixed, **kw)
            store.oe_champion = lambda c: c                  # 修复前: 英雄名原样查表
            rb, _ = store.make_row(r["blue_team"], r["red_team"], r["league"], draft=broken, **kw)
        except ValueError:
            skipped += 1
            continue
        finally:
            store.__dict__.pop("oe_champion", None)
        if any(not same(rf.get(f, np.nan), r.get(f, np.nan), 1e-12 if "avg_champ_wr" in f else 0.0)
               for f in draft_feats):
            parity_bad += 1
        for tag, row in (("fixed", rf), ("broken", rb)):
            g = [row.get(f"{s}_{ro}_champ_games", 0) for s in "br" for ro in ROLES]
            found[tag][0] += sum(x > 0 for x in g)
            found[tag][1] += len(g)
        ys.append(int(r["y"]))
        pf.append(s2.predict(rf)[1])
        pb.append(s2.predict(rb)[1])
        p1.append(s1.predict(rf)[1])
        dates.append(r["date"])

    y = np.array(ys)
    P = {"修复前 (看板原样)": np.array(pb), "修复后 (名字对齐)": np.array(pf), "赛前模型 (参照)": np.array(p1)}
    eps = 1e-12
    brier = {k: (p - y) ** 2 for k, p in P.items()}
    ll = {k: -(y * np.log(np.clip(p, eps, 1)) + (1 - y) * np.log(np.clip(1 - p, eps, 1))) for k, p in P.items()}
    acc = {k: ((p > 0.5).astype(int) == y).astype(float) for k, p in P.items()}

    print(f"\n测试集 {len(test)} 局, 用上 {len(y)} 局 (跳过 {skipped}: 选手不齐或队伍历史不足)")
    print(f"选手-英雄有历史的槽位: 修复前 {found['broken'][0]}/{found['broken'][1]}   "
          f"修复后 {found['fixed'][0]}/{found['fixed'][1]}")
    print(f"修复后的 BP 特征和训练矩阵一致 (avg_champ_wr 族容许 1e-12, 其余逐位): "
          f"{len(y) - parity_bad}/{len(y)} 局")
    print(f"\n{'':<18}{'准确率':>8}{'Brier':>9}{'对数损失':>10}")
    for k in P:
        print(f"{k:<18}{acc[k].mean():>8.1%}{brier[k].mean():>9.4f}{ll[k].mean():>10.4f}")

    a, b = "修复前 (看板原样)", "修复后 (名字对齐)"
    order = np.argsort(np.array(dates, dtype="datetime64[ns]"), kind="stable")
    blocks = np.array_split(order, N_BLOCKS)
    print(f"\n配对比较 (修复后 − 修复前, 分数 = 负损失, t > {T_ACCEPT} 才算稳定变好):")
    for name, loss in (("Brier", brier), ("对数损失", ll), ("准确率", None)):
        sa = acc[a] if loss is None else -loss[a]
        sb = acc[b] if loss is None else -loss[b]
        d_g, _sd, t_g, _ = paired_t(sa, sb)
        ba = [sa[i].mean() for i in blocks]
        bb = [sb[i].mean() for i in blocks]
        d_b, _sd, t_b, _ = paired_t(ba, bb)
        per_block = "  ".join(f"{x - w:+.4f}" for w, x in zip(ba, bb))
        print(f"  {name:<6} 逐局 Δ {d_g:+.4f} t = {t_g:+.2f}   按 {N_BLOCKS} 段 t = {t_b:+.2f}  [{per_block}]")


if __name__ == "__main__":
    main()
