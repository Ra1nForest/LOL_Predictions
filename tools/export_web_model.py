"""
导出浏览器版模型 + 黄金用例
============================
GitHub Pages 只能放静态文件, 跑不了 Python, 所以赛前和局内的胜率都要搬进浏览器里算。
这个脚本把浏览器需要的东西从 Python 这边**原样**导出, 并顺手生成黄金用例 ——
同一批输入, 线上真函数算出的结果 —— 给 JS 版逐项对账。

为什么非要黄金用例, 而不是"看着差不多就行":
这个项目几乎每一个 bug 都是"一个看起来合理的错数字", 从不报错。训练在
Python、推理在 JS, 等于特征构造有了两份实现 —— 正是 CLAUDE.md 第一条
("训练和推理必须读同一份逻辑") 要防的形状。两份实现只能靠逐位对账来证明
它们是同一份。

期望值一律走线上的真函数, 不在这里另写一份:
  · 局内特征 / 概率 / 警告   ingame_service.IngameModel (make_row / predict)
  · 局内整条响应             api._ingame_core (看板头条和走势图每个点都用它)
  · 赛前整条响应             api.predict (POST /predict)
  · 赛前 9 项                api._pre_context
  · 数据范围                 api.DATA (五年列表, 和服务同一份 —— 见 CLAUDE.md)
另写一份的话, 两边会一起错, 对账照样通过, 等于没测。

文案表 (explain.py 的 STAT / TONE …) 也直接导出成数据, JS 版不手抄 ——
explain.py 改一句话, 重新导出就跟上了。

产物:
  frontend/public/web/ingame_live.json     局内模型 (树) + 保序表 + 阵容强势期表
  frontend/public/web/stage1_pre.json      赛前模型 (树) + Platt 系数
  frontend/public/web/teams.json           每队一行的滚动统计 + 各赛区已知队伍 + 队名别名
  frontend/public/web/explain.json         explain.py 的文案表
  frontend/scripts/golden/ingame_cases.json    黄金用例 (不发布)
  frontend/scripts/golden/stage1_cases.json

    python tools/export_web_model.py
    npm --prefix frontend run test:golden
"""
from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ART = ROOT / "artifacts"
OUT_WEB = ROOT / "frontend" / "public" / "web"
OUT_GOLD = ROOT / "frontend" / "scripts" / "golden"
COLLECTED = ROOT / "collected"
MAJORS = ("LPL", "LCK", "LEC", "LCS")
# 和 src/web/xgb.ts 里的 FORMAT 必须一致; 改导出格式就改这个号
FORMAT = "xgb-binary-logistic-v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def f32(v: float) -> float:
    """按 float32 取值再转回 Python float。

    XGBoost 内部的切分阈值和叶子值都是 float32。先落成精确的 float32, JSON
    往返后 JS 端的 Float32Array 就能拿回同一个比特。直接写 float64 的话,
    阈值附近的输入会在两边走进不同的分支 —— 差一个 ulp 就是另一片叶子。
    """
    return float(np.float32(v))


def _jsonable(o):
    """线上响应 → 和浏览器拿到的完全同形的 JSON。allow_nan=False: NaN 混进来就在这里炸。"""
    return json.loads(json.dumps(o, ensure_ascii=False, allow_nan=False))


# ══════════════════════════════════════════════════════════
#  模型
# ══════════════════════════════════════════════════════════

def _trees(path: Path, features: list[str]) -> tuple[float, list[dict]]:
    """XGBoost 模型文件 → (base_score, 紧凑的树)。

    每一条检查守的都是"JS 版没实现、但不会报错"的情形 —— 真遇上了, JS 会安静地
    给出另一个数。宁可在导出时就拒绝。
    """
    L = json.loads(path.read_text(encoding="utf-8"))["learner"]
    obj = L["objective"]["name"]
    if obj != "binary:logistic":
        raise SystemExit(f"{path.name}: 目标函数是 {obj}, JS 版只实现了 binary:logistic")
    if (L.get("feature_names") or []) != list(features):
        raise SystemExit(f"{path.name}: 模型文件里的特征顺序和 calib 不一致 —— JS 按 calib "
                         f"的顺序组向量, 顺序一错每个特征都喂进错的位置")
    gb = L["gradient_booster"]
    if gb["name"] != "gbtree":
        raise SystemExit(f"{path.name}: booster 是 {gb['name']}, JS 版只实现了 gbtree")
    mdl = gb["model"]
    if int(mdl["gbtree_model_param"]["num_parallel_tree"]) != 1 or any(mdl["tree_info"]):
        raise SystemExit(f"{path.name}: 有并行树或多分类输出, JS 版没实现")

    trees = []
    for t in mdl["trees"]:
        if any(int(s) != 0 for s in t["split_type"]):
            raise SystemExit(f"{path.name}: 出现类别型切分, JS 版只实现了数值切分")
        trees.append({
            "l": [int(v) for v in t["left_children"]],
            "r": [int(v) for v in t["right_children"]],
            "f": [int(v) for v in t["split_indices"]],
            # 叶子节点 (left == -1) 的 split_conditions 存的就是叶子值
            "c": [f32(v) for v in t["split_conditions"]],
            "d": [int(v) for v in t["default_left"]],
        })
    # XGBoost 3.x 把 base_score 存成 "[5.270288E-1]" (向量形式, 概率空间)
    base = float(str(L["learner_model_param"]["base_score"]).strip("[]"))
    return f32(base), trees


def export_ingame(ig) -> dict:
    from ingame_service import GOLD_TOTAL_MEDIAN

    base, trees = _trees(ART / "model_ingame_live.json", ig.features)
    m = ig.metrics
    return {
        "format": FORMAT,
        "variant": "live",
        "features": list(ig.features),
        "base_score": base,
        "trees": trees,
        "importance": {k: float(v) for k, v in ig.importance.items()},
        "calibrator": ig.calibrator,
        "iso_x": [float(v) for v in ig.iso_x],
        "iso_y": [float(v) for v in ig.iso_y],
        "clip": float(ig.clip),
        "platt": {"a": float(ig.a), "b": float(ig.b)},
        "p_min": m.get("p_min"),
        "p_max": m.get("p_max"),
        "slices": [int(s) for s in ig.slices],
        "gold_total_median": {str(k): int(v) for k, v in GOLD_TOTAL_MEDIAN.items()},
        "scaling_index": {k: float(v) for k, v in ig.scaling.items()},
        "has_xpdiff": bool(ig.has_xpdiff),
        "metrics": {"accuracy": m.get("accuracy"), "ece": m.get("ece"),
                    "per_T": m.get("per_T")},
        "trained_through": ig.trained_through,
        "exported_at": _now(),
    }


def export_stage1(s1, model_keys) -> dict:
    base, trees = _trees(ART / "model_pre_draft.json", s1.features)
    return {
        "format": FORMAT,
        "features": list(s1.features),
        "base_score": base,
        "trees": trees,
        "importance": {k: float(v) for k, v in s1.importance.items()},
        "calibrator": "platt",
        "platt": {"a": float(s1.a), "b": float(s1.b)},
        "metrics": {k: s1.metrics[k] for k in model_keys if k in s1.metrics},
        "exported_at": _now(),
    }


def export_teams(store) -> dict:
    """每队一行: 看板在"现在"这一刻会拿到的那份滚动统计。

    浏览器里赛前特征就是两队那一行逐项相减/相加, 和 store.make_row 里的
    `bv - rv` / `bv + rv` 是同一个运算。NaN 写成 null (严格 JSON 里没有 NaN)。
    """
    from esports_feed import TEAM_ALIASES
    from feature_store import LEAGUES, SUM_FEATS

    cols = list(store.roll_cols)
    teams = {}
    for name in sorted(store.team_history):
        s = store.team_snapshot(name)
        if s is None:
            continue
        vals = []
        for c in cols:
            v = float(s.get(c, float("nan")))
            vals.append(None if math.isnan(v) else v)
        teams[name] = {"v": vals, "n": int(s["_n_games"]),
                       "last": str(pd.Timestamp(s["_last_date"]).date())}
    return {
        "format": "teams-v1",
        "roll_cols": cols,
        "sum_feats": list(SUM_FEATS),
        "teams": teams,
        "known": {lg: store.known_teams(lg) for lg in LEAGUES},
        "league_last": {lg: str(pd.Timestamp(d).date()) for lg, d in store.league_last.items()},
        "aliases": dict(TEAM_ALIASES),
        "data_through": str(store.last_date.date()),
        "generated_at": _now(),
    }


def export_explain() -> dict:
    import explain as EX
    return {
        "format": "explain-v1",
        "stat": {k: list(v) for k, v in EX.STAT.items()},
        "alias": dict(EX.ALIAS),
        "live": {k: list(v) for k, v in EX.LIVE.items()},
        "meta": dict(EX.META),
        "interaction": dict(EX.INTERACTION),
        # 有序: _player_champ 按这个顺序试
        "role_cn": [[k, v] for k, v in EX.ROLE_CN.items()],
        "skip_in_reasons": sorted(EX.SKIP_IN_REASONS),
        "tone": [[th, t] for th, t in EX.TONE],
    }


# ══════════════════════════════════════════════════════════
#  黄金用例
# ══════════════════════════════════════════════════════════

def _import_api():
    """导入 api.py 只为借用它的函数和 DATA —— 不启动服务 (lifespan 只在 uvicorn 里跑)。"""
    try:
        import api
    except Exception as e:
        raise SystemExit(f"导入 api.py 失败: {type(e).__name__}: {e}")
    return api


def _pre_row(api, cache, blue, red, league):
    key = (blue, red, league)
    if key not in cache:
        if not blue or not red:
            cache[key] = None
        else:
            try:
                cache[key], _p, _w = api._pre_context(blue, red, league)
            except Exception:
                # 看板里 _pre_context 失败 (比如队伍不认识) 就是不带赛前特征,
                # 这里照同样的路走
                cache[key] = None
    return cache[key]


def _expect(ig, api, cache, inp) -> dict:
    pre = _pre_row(api, cache, inp["blue"], inp["red"], inp["league"])
    m, warns = ig.make_row(
        minute=inp["minute"], golddiff=inp["golddiff"], xpdiff=None,
        csdiff=inp["csdiff"], blue_kills=inp["blue_kills"], red_kills=inp["red_kills"],
        gold_total=inp["gold_total"], blue_champs=inp["blue_champs"],
        red_champs=inp["red_champs"], pre_row=pre, league=inp["league"])
    # 和 IngameModel.predict() 里组向量的两行一字不差
    x = np.array([[float(m.get(f, np.nan)) for f in ig.features]])
    x = np.nan_to_num(x, nan=-999.0)
    raw, cal = ig.predict(m)
    return {"T": int(m["T"]), "x": [float(v) for v in x[0]],
            "raw": float(raw), "cal": float(cal), "warnings": list(warns),
            "has_pre": pre is not None}


def _ingame_resp(api, inp) -> dict:
    """api._ingame_core 的真实输出 —— 看板头条调它的方式 (include_pregame=True, 不带 pre_ctx)。"""
    st = api.IngameState(minute=inp["minute"], golddiff=inp["golddiff"], xpdiff=None,
                         csdiff=inp["csdiff"], blue_kills=inp["blue_kills"],
                         red_kills=inp["red_kills"], gold_total=inp["gold_total"])
    out, _row = api._ingame_core(inp["blue"], inp["red"], inp["league"], st,
                                 inp["blue_champs"], inp["red_champs"], True)
    return _jsonable(out)


def real_inputs(store) -> list[tuple[str, dict]]:
    """collected/ 里的真实直播帧 —— 看板实际会喂给模型的那种输入。

    参数的取法照 api.esports_board 的走势图那段: minute 夹到 [3, 60],
    gold_total 用蓝方总经济, 阵容要凑满 5+5 才传。
    """
    from esports_feed import map_team

    out = []
    for mp in sorted(COLLECTED.glob("*.meta.json")):
        meta = json.loads(mp.read_text(encoding="utf-8"))
        lg = meta.get("league")
        if lg not in MAJORS:
            continue
        gid = mp.name[: -len(".meta.json")]
        jl = mp.with_name(f"{gid}.jsonl")
        if not jl.exists():
            continue
        ps = list((meta.get("players") or {}).values())
        bch = [p["champion"] for p in ps if p.get("side") == "blue" and p.get("champion")]
        rch = [p["champion"] for p in ps if p.get("side") == "red" and p.get("champion")]
        if len(bch) != 5 or len(rch) != 5:
            bch = rch = None
        known = store.known_teams(lg)          # 看板里映射用的就是本赛区的已知队伍
        blue = map_team(meta.get("blue") or "", known)
        red = map_team(meta.get("red") or "", known)
        for line in jl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            mn = row.get("minute")
            if mn is None or mn < 3:
                continue
            out.append((f"{gid}@{mn}", {
                "minute": min(60, max(3, mn)),
                "golddiff": row["golddiff"], "csdiff": row.get("csdiff", 0),
                "blue_kills": row["blue_kills"], "red_kills": row["red_kills"],
                "gold_total": row.get("blue_gold"),
                "blue_champs": bch, "red_champs": rch,
                "blue": blue, "red": red, "league": lg,
            }))
    return out


def synthetic_inputs(base: dict) -> list[tuple[str, dict]]:
    """在一场真实比赛上每次只改一个维度, 专门打边界。

    真实帧覆盖的是"常见情况"; 这里覆盖的是两份实现最容易分叉的地方 ——
    切片平局、.5 分钟的格式化、除零、阵容不全、队伍不认识、正负号。
    """
    def v(label, **kw):
        d = dict(base)
        d.update(kw)
        return (f"syn:{label}", d)

    k = base["blue_champs"]
    out = []
    # 切片边界: 7.5 / 12.5 / 17.5 / 22.5 是平局, 27.5 是"偏离超过 2.5"的临界
    for mn in (3, 5, 7.49, 7.5, 7.51, 10, 12.4, 12.5, 12.6, 15, 17.5, 20,
               22.5, 25, 27.5, 27.51, 30, 45, 60):
        out.append(v(f"minute={mn}", minute=mn))
    for gd in (-20000, -1, 0, 1, 20000):
        out.append(v(f"golddiff={gd}", golddiff=gd, minute=20))
    for mn in (10, 15, 20, 25, 30):
        out.append(v(f"gold_total=None@{mn}", gold_total=None, minute=mn))
    out.append(v("gold_total=0", gold_total=0))
    out.append(v("champs=None", blue_champs=None, red_champs=None))
    out.append(v("blue_champs=None", blue_champs=None))
    out.append(v("red_champs=[]", red_champs=[]))
    out.append(v("champs=fake", blue_champs=["NotAChampion"] * 5))
    out.append(v("champs=3known+2fake", blue_champs=k[:3] + ["Fake1", "Fake2"]))
    out.append(v("champs=2known+3fake", blue_champs=k[:2] + ["Fake1", "Fake2", "Fake3"]))
    out.append(v("champs=dup", blue_champs=[k[0]] * 5))
    out.append(v("blue=unknown", blue="Definitely Not A Team"))
    out.append(v("red=None", red=None))
    out.append(v("teams=None", blue=None, red=None))
    for lg in MAJORS:
        out.append(v(f"league={lg}", league=lg))
    out.append(v("kills=0/0", blue_kills=0, red_kills=0))
    out.append(v("kills=30/2", blue_kills=30, red_kills=2))
    out.append(v("csdiff=-300", csdiff=-300))
    out.append(v("csdiff=300", csdiff=300))
    out.append(v("swap", blue=base["red"], red=base["blue"],
                 blue_champs=base["red_champs"], red_champs=base["blue_champs"],
                 golddiff=-base["golddiff"]))
    return out


def stage1_cases(api, store) -> list[dict]:
    """POST /predict 的真实响应 —— 各赛区已知队伍两两对阵, 外加跨赛区、季后赛、不认识的队。"""
    from fastapi import HTTPException
    from feature_store import LEAGUES

    def run(b, r, lg, po=False):
        inp = {"blue_team": b, "red_team": r, "league": lg, "playoffs": po}
        try:
            return {"in": inp, "resp": _jsonable(api.predict(api.PredictRequest(**inp)))}
        except HTTPException as e:
            return {"in": inp, "status": e.status_code, "error": e.detail}

    k = {lg: store.known_teams(lg) for lg in LEAGUES}
    out = []
    for lg in LEAGUES:
        ks = k[lg][:12]
        for b in ks:
            for r in ks:
                if b != r:
                    out.append(run(b, r, lg))
    out.append(run(k["LCK"][0], k["LPL"][0], "LCK"))       # 国际赛: 跨赛区
    out.append(run(k["LEC"][0], k["LCS"][0], "LEC"))
    out.append(run(k["LPL"][0], k["LPL"][1], "LPL", True))  # 季后赛
    out.append(run(k["LCK"][1], k["LCK"][0], "LCK", True))
    out.append(run("Definitely Not A Team", k["LCK"][0], "LCK"))
    out.append(run(k["LCK"][0], "Definitely Not A Team", "LCK"))
    # 在队伍史里、但不在任何赛区已知队伍表里的 (场次太少或换了赛区)
    extra = [t for t in sorted(store.team_history) if all(t not in v for v in k.values())][:3]
    for t in extra:
        out.append(run(t, k["LPL"][0], "LPL"))
    return out


# ══════════════════════════════════════════════════════════

def _write(path: Path, obj) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    # allow_nan=False: 严格 JSON。NaN 混进来就在这里炸, 不要留到浏览器里
    txt = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    path.write_text(txt, encoding="utf-8")
    return len(txt.encode("utf-8"))


def main():
    t0 = time.time()
    from feature_store import FeatureStore
    from ingame_service import IngameModel

    print("加载模型 …")
    api = _import_api()
    ig = IngameModel("_live")
    s1 = api.Stage("pre_draft")

    print("加载特征库 (五年 CSV, 和服务同一份 api.DATA) …")
    store = FeatureStore.from_csv(api.DATA)
    # 线上函数要的全局状态 —— 和 lifespan 里装的是同一批东西
    api.STATE["store"] = store
    api.STATE["stages"] = {"pre_draft": s1}
    api.STATE["ingame_live"] = ig
    today = str(pd.Timestamp.now().normalize().date())

    for name, obj in (("ingame_live.json", export_ingame(ig)),
                      ("stage1_pre.json", export_stage1(s1, api.MODEL_KEYS)),
                      ("teams.json", export_teams(store)),
                      ("explain.json", export_explain())):
        n = _write(OUT_WEB / name, obj)
        print(f"  {name:18s} {n / 1024:>6,.0f} KB")
    stale = OUT_WEB / "teams_pre.json"          # 旧格式, 已并进 teams.json
    if stale.exists():
        stale.unlink()

    print("生成黄金用例 …")
    real = real_inputs(store)
    base = next((inp for _, inp in real
                 if inp["blue_champs"] and inp["blue"] in store.team_history
                 and inp["red"] in store.team_history), None)
    if base is None:
        raise SystemExit("collected/ 里找不到一场阵容齐全、两队都认识的比赛当合成用例的底子")
    syn = synthetic_inputs(base)

    cache: dict = {}
    cases, n_resp = [], 0
    for i, (src, inp) in enumerate(real + syn):
        c = {"src": src, "in": inp, **_expect(ig, api, cache, inp)}
        # 整条响应只抽一部分 (每条两三 KB): 全部合成用例 + 每 5 条真实用例取 1。
        # 两队都得有名字 —— 看板只对 predictable 的比赛调局内模型
        if inp["blue"] and inp["red"] and (src.startswith("syn:") or i % 5 == 0):
            c["resp"] = _ingame_resp(api, inp)
            n_resp += 1
        cases.append(c)
    n = _write(OUT_GOLD / "ingame_cases.json", {
        "generated_at": _now(), "today": today, "features": list(ig.features),
        "n_real": len(real), "n_synthetic": len(syn), "cases": cases})
    games = {c["src"].split("@")[0] for c in cases if not c["src"].startswith("syn:")}
    print(f"  ingame_cases.json  {n / 1024:>6,.0f} KB   真实 {len(real)} 条 ({len(games)} 局)  "
          f"合成 {len(syn)} 条  其中带整条响应 {n_resp} 条")

    s1c = stage1_cases(api, store)
    n = _write(OUT_GOLD / "stage1_cases.json", {"generated_at": _now(), "today": today, "cases": s1c})
    print(f"  stage1_cases.json  {n / 1024:>6,.0f} KB   {len(s1c)} 组对阵 "
          f"(其中 {sum('error' in c for c in s1c)} 组预期报错)")
    print(f"完成, {time.time() - t0:.0f} 秒。下一步: npm --prefix frontend run test:golden")


if __name__ == "__main__":
    main()
