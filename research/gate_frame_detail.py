"""
闸门: 亚分钟级的帧细节值不值得进 Stage 4?
==========================================

问题
----
实时帧是 **0.2 秒一帧** (实测一个 window 响应 40 帧 / 9 秒), 而我们真正存下来
用的只有:
  · collected/     两分钟一条 (采集器轮询间隔)
  · backfill/      T=25/30/35/40 几个切片
  · Oracle's Elixir  只有 at10/15/20/25 四列

差三个数量级。问题是**这些细节到底有没有信息**, 还是只是更密的同一条曲线。

测什么
------
挑三类**只有帧才有、聚合数据结构上给不了**的东西:

1. **血量状态** —— 帧里每个人有 currentHealth/maxHealth。OE 一个字段都没有。
   这是"此刻是不是正在打团/有没有人残血"的直接读数, 而金币差要等人头
   落地才动。current_health=0 更是唯一直说"他倒了"的信号。
2. **9 秒内的斜率** —— 一个 window 响应本身就跨 9 秒, 首末帧相减即可, **零额外请求**。
3. **60 秒动量** —— 多取一个 T-60 的窗口, 得到一分钟内经济/人头怎么变的。

为什么不测"更密的采样点"
------------------------
把同一局按 0.2 秒切成上万条样本再训, 涨的是**样本数不是信息量** —— 同一局
相邻两帧几乎完全相关, 等于把一局的权重放大一万倍, 而验证集里它们同样相关,
分数会好看得毫无意义。所以这里固定用回填已有的那些 T, 只在**特征**上加东西。

为什么只能在回填子集里测
------------------------
和 gate_objectives 同一个理由: 这些列只有帧有。混进 OE 主表去训,
"有值 / NaN" 会变成"来源标签", 模型学到的是数据从哪来 —— 本项目最怕的
那类静默污染。

判据
----
配对 t 检验, |t| > 2.5 (research/harness.py 的口径)。按**局**分折。
同时报归一化 lift 和 Brier。

用法:
    python research/gate_frame_detail.py --games 400        # 先攒数据再判
    python research/gate_frame_detail.py --report           # 只用已攒的缓存
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, brier_score_loss

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "research"))
from harness import paired_t, T_ACCEPT          # noqa: E402
import esports_feed as EF                        # noqa: E402

SNAPS = _ROOT / "backfill" / "snapshots.jsonl"
CACHE = _ROOT / "research" / "frame_detail_cache.jsonl"

XGB_P = dict(n_estimators=400, max_depth=5, learning_rate=0.04,
             subsample=0.8, colsample_bytree=0.7,
             eval_metric="logloss", verbosity=0)

# 基线: 回填快照里 Stage 4 今天就能拿到的那几个。回填没有 XP, 所以整个
# 对比在 live 变体的口径上做 (同 gate_objectives 的 DROP 处理)。
BASE_F = ["golddiff", "csdiff", "killdiff", "gold_total", "T"]

NEW_F = [
    # 血量 —— 聚合数据结构上没有
    "hp_frac_diff", "blue_hp_frac", "red_hp_frac",
    "blue_low", "red_low", "blue_dead", "red_dead",
    # 9 秒斜率 (窗口内首末帧相减, 零额外请求)
    "gold_slope_9s", "kill_slope_9s",
    # 60 秒动量
    "gold_mom_60s", "kill_mom_60s",
]

LOW_HP = 0.35          # "残血"的阈值


def _hp(team: dict) -> tuple[float, int, int]:
    """(队伍总血量比例, 残血人数, 阵亡人数)。"""
    ps = team.get("participants") or []
    cur = sum(p.get("currentHealth") or 0 for p in ps)
    mx = sum(p.get("maxHealth") or 0 for p in ps)
    low = sum(1 for p in ps
              if (p.get("maxHealth") or 0) > 0
              and (p.get("currentHealth") or 0) / p["maxHealth"] < LOW_HP)
    dead = sum(1 for p in ps if (p.get("currentHealth") or 0) == 0)
    return (cur / mx if mx else float("nan")), low, dead


def _gk(frame: dict) -> tuple[int, int]:
    b, r = frame.get("blueTeam") or {}, frame.get("redTeam") or {}
    return (b.get("totalGold", 0) - r.get("totalGold", 0),
            b.get("totalKills", 0) - r.get("totalKills", 0))


def extract(feed: EF.EsportsFeed, game_id: str, frame_t: str) -> dict | None:
    """取 T 和 T-60 两个窗口, 算出新特征。任一步拿不到就整条丢掉 ——
    半条样本会让两个特征集的样本不一致, 配对 t 就不再配对了。"""
    try:
        at = datetime.strptime(frame_t[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    except Exception:
        return None

    def win(t):
        p = feed._window(game_id, EF.EsportsFeed._lagged(0, t), ttl=3600)
        return (p or {}).get("frames") or []

    now_fr = win(at)
    if not now_fr:
        return None
    f = now_fr[-1]
    b, r = f.get("blueTeam") or {}, f.get("redTeam") or {}
    bhp, blow, bdead = _hp(b)
    rhp, rlow, rdead = _hp(r)
    g_now, k_now = _gk(f)

    # 9 秒斜率: 同一个响应里的首末帧, 不多花请求
    g0, k0 = _gk(now_fr[0])
    span = max(1.0, (feed._ts(f) - feed._ts(now_fr[0])).total_seconds())
    out = {
        "blue_hp_frac": bhp, "red_hp_frac": rhp, "hp_frac_diff": bhp - rhp,
        "blue_low": blow, "red_low": rlow, "blue_dead": bdead, "red_dead": rdead,
        "gold_slope_9s": (g_now - g0) / span * 60,   # 折算成每分钟, 便于读
        "kill_slope_9s": (k_now - k0) / span * 60,
    }

    past = win(at - timedelta(seconds=60))
    if past:
        gp, kp = _gk(past[-1])
        out["gold_mom_60s"] = g_now - gp
        out["kill_mom_60s"] = k_now - kp
    else:
        out["gold_mom_60s"] = np.nan
        out["kill_mom_60s"] = np.nan
    return out


def collect(df: pd.DataFrame, workers: int = 8) -> int:
    """补齐缓存里还没有的样本。缓存按 (game_id, frame_t) 去重, 可反复跑。"""
    have = set()
    if CACHE.exists():
        for ln in CACHE.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                try:
                    o = json.loads(ln)
                    have.add((o["game_id"], o["frame_t"]))
                except Exception:
                    pass
    todo = [(row.game_id, row.frame_t) for row in df.itertuples()
            if (row.game_id, row.frame_t) not in have]
    if not todo:
        print(f"  缓存已覆盖全部 {len(df)} 条样本")
        return 0
    print(f"  需要补 {len(todo)} 条 (已有 {len(have)}), "
          f"每条最多 2 个上游请求…")

    feed = EF.EsportsFeed()
    n_ok = 0
    with CACHE.open("a", encoding="utf-8") as fh:
        def one(pair):
            gid, ft = pair
            try:
                d = extract(feed, gid, ft)
            except Exception:
                d = None
            return gid, ft, d

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for i, (gid, ft, d) in enumerate(ex.map(one, todo), 1):
                if d:
                    d.update({"game_id": gid, "frame_t": ft})
                    fh.write(json.dumps(d, ensure_ascii=False) + "\n")
                    n_ok += 1
                if i % 200 == 0:
                    print(f"    {i}/{len(todo)}  成功 {n_ok}", flush=True)
                    fh.flush()
    print(f"  补齐 {n_ok}/{len(todo)} 条")
    return n_ok


def folds_by_game(df: pd.DataFrame, n_folds: int):
    games = (df[["game_id", "date"]].drop_duplicates("game_id")
             .sort_values("date")["game_id"].values)
    return np.array_split(games, n_folds)


def score(tr, te, feats, seeds):
    X, y = tr[feats].astype(float).fillna(-999), tr["y"].astype(int)
    Xe = te[feats].astype(float).fillna(-999)
    ye = te["y"].astype(int).values
    accs, brs = [], []
    for s in seeds:
        m = XGBClassifier(random_state=s, **XGB_P)
        m.fit(X, y, verbose=False)
        p = m.predict_proba(Xe)[:, 1]
        base = max(ye.mean(), 1 - ye.mean())
        accs.append((accuracy_score(ye, (p >= .5).astype(int)) - base) / (1 - base))
        brs.append(-brier_score_loss(ye, p))
    return float(np.mean(accs)), float(np.mean(brs))


def run(df, base_f, cand_f, folds, seeds, label):
    a_acc, b_acc, a_br, b_br = [], [], [], []
    for i, hold in enumerate(folds):
        te = df[df.game_id.isin(hold)]
        tr = df[~df.game_id.isin(hold)]
        if len(te) < 50 or len(tr) < 200:
            continue
        aa, ab = score(tr, te, base_f, seeds)
        ba, bb = score(tr, te, cand_f, seeds)
        a_acc.append(aa); b_acc.append(ba)
        a_br.append(ab); b_br.append(bb)
        print(f"    fold {i+1}: {len(tr):>6} 训 / {len(te):>5} 测   "
              f"lift {aa:+.4f} -> {ba:+.4f}   Brier {-ab:.4f} -> {-bb:.4f}")
    if len(a_acc) < 3:
        print("    折数不足, 无法判定")
        return
    for name, A, B in (("归一化 lift", a_acc, b_acc), ("Brier(负)", a_br, b_br)):
        d, sd, t, _ratio = paired_t(np.array(A), np.array(B))
        verdict = ("采纳" if t > T_ACCEPT else
                   "拒绝 (稳定变差)" if t < -T_ACCEPT else "噪声内, 不采纳")
        print(f"  {label} · {name}: Δ={d:+.4f}  配对SD={sd:.4f}  "
              f"t={t:+.2f}  (判据 {T_ACCEPT}) -> {verdict}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=400, help="取样多少局")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--report", action="store_true", help="不抓新数据, 只用缓存")
    a = ap.parse_args()
    seeds = tuple(range(a.seeds))

    if not SNAPS.exists():
        print("没有 backfill/snapshots.jsonl, 先跑 research/backfill_late.py")
        return 1
    rows = [json.loads(l) for l in SNAPS.read_text(encoding="utf-8").splitlines() if l.strip()]
    snap = pd.DataFrame(rows)
    snap = snap.dropna(subset=["y", "game_id", "frame_t"])
    snap["killdiff"] = snap["blue_kills"] - snap["red_kills"]
    snap["date"] = pd.to_datetime(snap["date"], errors="coerce")

    # 按局取样, 不按行 —— 分折是按局做的, 按行抽会让同一局散进多个折
    games = snap[["game_id", "date"]].drop_duplicates("game_id").sort_values("date")
    if not a.report and a.games < len(games):
        # 均匀跨时间取样, 不是取最近的 N 局 —— 否则折与折之间没有时间跨度
        idx = np.linspace(0, len(games) - 1, a.games).astype(int)
        games = games.iloc[idx]
    sub = snap[snap.game_id.isin(set(games.game_id))]
    print(f"回填快照 {len(snap)} 条 / {snap.game_id.nunique()} 局; "
          f"本次取样 {sub.game_id.nunique()} 局 / {len(sub)} 条\n")

    if not a.report:
        collect(sub, a.workers)

    if not CACHE.exists():
        print("缓存是空的 —— 先不带 --report 跑一次")
        return 1
    extra = pd.DataFrame([json.loads(l) for l in
                          CACHE.read_text(encoding="utf-8").splitlines() if l.strip()])
    extra = extra.drop_duplicates(subset=["game_id", "frame_t"], keep="last")
    df = sub.merge(extra, on=["game_id", "frame_t"], how="inner")
    print(f"\n可用于判定的样本: {len(df)} 条 / {df.game_id.nunique()} 局")
    if len(df) < 300:
        print("样本太少, 判不了。加大 --games 再跑一次。")
        return 1

    print("\n  新特征的分布 (全常数或全缺失的话这个实验没有意义):")
    for c in NEW_F:
        v = pd.to_numeric(df[c], errors="coerce")
        print(f"    {c:<16} 缺失 {100*v.isna().mean():>5.1f}%   "
              f"非零 {100*(v.fillna(0) != 0).mean():>5.1f}%   "
              f"范围 [{v.min():.3g}, {v.max():.3g}]")

    folds = folds_by_game(df, a.folds)
    print("\n  [A] 全部快照")
    run(df, BASE_F, BASE_F + NEW_F, folds, seeds, "全部")

    # 分组看: 血量/动量最该起作用的是"局势胶着"的时候 —— 大顺风局
    # 谁赢已经写在金币上了, 加什么都没用, 混在一起会稀释掉效应。
    close = df[df.golddiff.abs() < 3000]
    if close.game_id.nunique() > 40:
        print(f"\n  [B] 只看经济差 <3k 的胶着局面 ({len(close)} 条)")
        run(close, BASE_F, BASE_F + NEW_F, folds_by_game(close, a.folds), seeds, "胶着")

    print("\n  [C] 只加血量 (最贵也最独有的那组)")
    hp_only = [c for c in NEW_F if "hp" in c or c.endswith(("_low", "_dead"))]
    run(df, BASE_F, BASE_F + hp_only, folds, seeds, "仅血量")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
