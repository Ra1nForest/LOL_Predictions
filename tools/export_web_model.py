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
  frontend/public/web/stage2_post.json     BP 后模型 (树) + Platt + 英雄/选手-英雄的胜场场数表
  frontend/public/web/teams.json           每队一行的滚动统计 (含母赛区) + 各赛区已知队伍及其并集 + 队名别名
  frontend/public/web/explain.json         explain.py 的文案表
  frontend/scripts/golden/ingame_cases.json    黄金用例 (不发布)
  frontend/scripts/golden/stage1_cases.json
  frontend/scripts/golden/stage2_cases.json
  frontend/scripts/golden/intl_cases.json      国际赛: pregame_league / _known_for / map_team,
                                               以及看板的 _pregame_reason / _early_prediction (开局
                                               头 3 分钟, test:diff 碰不到) / _board_warnings

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
    from feature_store import CHAMP_ID_ALIAS
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
        # 看板传来的英雄名是 Data Dragon id, 表里是 OE 显示名 —— 按 champion_key 对齐, 见 names.ts
        "champion_alias": dict(CHAMP_ID_ALIAS),
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


def export_stage2(s2, store, model_keys) -> dict:
    """BP 后 (Stage 2): 树 + Platt, 外加两张查表 —— 每个 (英雄, 赛区) 和每个 (选手, 英雄)
    的 [胜场, 场数]。

    存整数不存胜率: FeatureStore 里是 np.mean(0/1 列表), 即精确求和再除以 n, JS 做同一次
    除法就逐位相同。英雄按 champion_key 存 (看板传来的 Data Dragon id 直接能查); 归一化后
    撞名就拒绝导出 —— 两个英雄共用一行胜率不会报错, 只会给出一个看着合理的错数字。
    """
    from feature_store import CHAMP_ID_ALIAS, champion_key

    base, trees = _trees(ART / "model_post_draft.json", s2.features)
    seen: dict = {}
    for c, _lg in store.champ_records:
        if isinstance(c, str) and seen.setdefault(champion_key(c), c) != c:
            raise SystemExit(f"英雄名归一化后撞名: {seen[champion_key(c)]} / {c} —— champion_key 要加区分")
    champ: dict = {}
    for (c, lg), recs in store.champ_records.items():
        if isinstance(c, str):
            champ.setdefault(lg, {})[champion_key(c)] = [int(sum(r for _d, r in recs)), len(recs)]
    player: dict = {}
    for (p, c), recs in store.player_champ.items():
        if isinstance(p, str) and isinstance(c, str):
            player.setdefault(p, {})[champion_key(c)] = [int(sum(r for _d, r in recs)), len(recs)]
    return {
        "format": FORMAT,
        "features": list(s2.features),
        "base_score": base,
        "trees": trees,
        "importance": {k: float(v) for k, v in s2.importance.items()},
        "calibrator": "platt",
        "platt": {"a": float(s2.a), "b": float(s2.b)},
        "metrics": {k: s2.metrics[k] for k in model_keys if k in s2.metrics},
        "champion_alias": dict(CHAMP_ID_ALIAS),
        "champ_league": champ,
        "player_champ": player,
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
                       "last": str(pd.Timestamp(s["_last_date"]).date()),
                       # 母赛区 (FeatureStore.home_league) —— 国际赛判断两队是不是同一个赛区出来的
                       "home": store.home_league(name)}
    return {
        "format": "teams-v1",
        "roll_cols": cols,
        "sum_feats": list(SUM_FEATS),
        "teams": teams,
        "known": {lg: store.known_teams(lg) for lg in LEAGUES},
        # 四大赛区的并集 = store.known_teams(None)。国际赛映射队名用它 (api._known_for);
        # 显式导出而不是让 JS 拼 "known" 的四张表, 免得两边对"并集"的理解不一样
        "known_all": store.known_teams(None),
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


def _pre_for(api, cache, inp):
    """这条输入的赛前特征行, 取法和 api._ingame_core 一字不差:
    include_pregame=False 就不带 (看板在跨赛区 / 队名不认识时这样调); 否则按 pre_league 算,
    没给 pre_league 就按 league (国际赛同母赛区时看板给的是两队的母赛区, league 仍是赛事本身)。
    这两个键只出现在国际赛的合成用例里, 老用例的输入一个字节都不变。"""
    if inp.get("include_pregame", True) is False:
        return None
    lg = inp.get("pre_league")
    return _pre_row(api, cache, inp["blue"], inp["red"], inp["league"] if lg is None else lg)


def _expect(ig, api, cache, inp) -> dict:
    pre = _pre_for(api, cache, inp)
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
    """api._ingame_core 的真实输出 —— 看板头条调它的方式 (不带 pre_ctx)。include_pregame 默认
    True; 国际赛用例另带 include_pregame=False (不带赛前特征) 或 pre_league (赛前特征按母赛区算)。"""
    st = api.IngameState(minute=inp["minute"], golddiff=inp["golddiff"], xpdiff=None,
                         csdiff=inp["csdiff"], blue_kills=inp["blue_kills"],
                         red_kills=inp["red_kills"], gold_total=inp["gold_total"])
    out, _row = api._ingame_core(inp["blue"], inp["red"], inp["league"], st,
                                 inp["blue_champs"], inp["red_champs"],
                                 inp.get("include_pregame", True),
                                 pre_league=inp.get("pre_league"))
    return _jsonable(out)


def real_inputs(store) -> list[tuple[str, dict]]:
    """collected/ 里的真实直播帧 —— 看板实际会喂给模型的那种输入。

    参数的取法照 api.esports_board 的走势图那段: minute 夹到 [3, 60],
    gold_total 用蓝方总经济, 阵容要凑满 5+5 才传。
    """
    from esports_feed import is_major, map_team

    out = []
    for mp in sorted(COLLECTED.glob("*.meta.json")):
        meta = json.loads(mp.read_text(encoding="utf-8"))
        lg = meta.get("league")
        # 真实用例只取四大赛区 —— 国际赛的走法 (不带赛前特征 / 按母赛区) 由 intl_inputs 的合成用例覆盖
        if not is_major(lg):
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


def intl_inputs(base: dict, store) -> list[tuple[str, dict]]:
    """国际赛看板喂给局内模型的三种输入 (见 api.esports_board / api.pregame_league):

      · 跨赛区或队名不认识: include_pregame=False —— 不带赛前特征; league 是赛事本身, is_* 全 0
      · 队名不认识: 蓝方用 lolesports 原名, 只当标签 (摘要、理由), 不进任何查表
      · 同母赛区: league 是赛事本身, 赛前特征按 pre_league = 两队母赛区算

    src 用 "syn:intl:" 打头, 和其余合成用例一样带整条响应。
    """
    out = []
    for lg in ("DCGI", "Worlds"):
        for mn in (3, 10, 20, 25, 34.5):
            out.append((f"syn:intl:{lg}@{mn}:无赛前",
                        {**base, "league": lg, "minute": mn, "include_pregame": False}))
        for gd in (-6000, 6000):
            out.append((f"syn:intl:{lg}:golddiff={gd}:无赛前",
                        {**base, "league": lg, "minute": 22, "golddiff": gd, "include_pregame": False}))
    out.append(("syn:intl:DCGI:不认识的队当标签",
                {**base, "league": "DCGI", "blue": "GAM Esports", "include_pregame": False}))
    out.append(("syn:intl:Worlds:无赛前:无阵容",
                {**base, "league": "Worlds", "include_pregame": False,
                 "blue_champs": None, "red_champs": None}))
    hb, hr = store.home_league(base["blue"]), store.home_league(base["red"])
    if hb is None or hb != hr:
        raise SystemExit(f"合成用例的底子 {base['blue']} / {base['red']} 不是同一个母赛区, 造不出同母赛区的国际赛用例")
    for lg in ("Worlds", "Esports World Cup"):
        out.append((f"syn:intl:{lg}:同母赛区按{hb}",
                    {**base, "league": lg, "pre_league": hb}))
    return out


def intl_cases(api, store) -> dict:
    """国际赛的几张小表 (golden/intl_cases.json), 给浏览器版逐条对账:

      pregame_league  api.pregame_league —— 赛前/BP 后按哪个赛区算, null = 不给
      known_for       api._known_for —— 映射队名用哪份已知队伍表 (四大按本赛区, 其余按并集)
      map_team        esports_feed.map_team —— known 的三种形状: null (不校验)、[] (空表, 必须返回
                      null)、"*" (并集) 或赛区名 (该赛区的表)
      pregame_reason  api._pregame_reason —— 看板上"为什么没有赛前/BP 后"那几句 (四大、同母赛区、
                      跨赛区、一队/两队不认识)
      early           api._early_prediction —— 开局头 3 分钟的 prediction (note 结尾三选一、source
                      可能为 null、同母赛区时的 warnings)。test:diff 只取终局帧, 碰不到这一段
      board_warnings  api._board_warnings —— 局内阶段提醒的顺序, 以及故意不带赛前特征时去掉那句
                      "没有可用的赛前队伍统计"
    """
    from esports_feed import TEAM_ALIASES, map_team
    from feature_store import LEAGUES

    k = {lg: store.known_teams(lg) for lg in LEAGUES}
    few = next((t for t in sorted(store.team_history) if t not in store.known_teams(None)), None)
    pairs = [(k["LCK"][0], k["LCK"][1]), (k["LPL"][1], k["LPL"][0]), (k["LEC"][0], k["LEC"][1]),
             (k["LCS"][0], k["LCS"][1]),
             (k["LCK"][0], k["LPL"][0]), (k["LEC"][0], k["LCS"][0]),       # 跨赛区
             (k["LCK"][0], None), (None, None), ("GAM Esports", k["LCK"][0]),
             ("GAM Esports", "RED Kalunga")]
    if few:                                    # 在队伍史里、但场次不够进已知队伍表的
        pairs.append((few, few))
    leagues = ["LCK", "lck", "LPL", "LEC", "LCS", "Worlds", "WORLDS", "MSI", "First Stand",
               "Esports World Cup", "DCGI", "LCK Challengers", "", None]
    pg = [{"in": [b, r, lg], "out": api.pregame_league(store, b, r, lg)}
          for b, r in pairs for lg in leagues]

    kf = [{"in": lg, "out": api._known_for(lg)} for lg in leagues]

    names = list(dict.fromkeys(
        sorted(TEAM_ALIASES)[:4] + ["Cloud9 Kia", "Team Liquid Alienware", k["LCK"][0],
                                    k["LCK"][0].upper(), k["LPL"][0].lower(), "GAM Esports",
                                    "RED Kalunga", "TBD", "", None]))
    specs = [None, [], "*", "LCK", "LPL", "LCS"]

    def known_of(spec):
        if spec is None or isinstance(spec, list):
            return spec
        return store.known_teams(None if spec == "*" else spec)

    mt = [{"in": [n, spec], "out": map_team(n, known_of(spec))} for n in names for spec in specs]

    # 看板上"为什么没有赛前/BP 后"那几句、开局头 3 分钟的 prediction、局内阶段的提醒顺序 ——
    # test:diff 只取终局帧, 开局那一段 (too_early) 不在它的范围里, 这三张表逐条对账。
    # 队伍形状: [lolesports 原名, 模型队名]; 模型队名为 None = 映射不出来
    import esports_feed as EF
    from ingame_service import NO_PRE_STATS
    hb = lambda lg: k[lg][0]                    # noqa: E731
    shapes = [
        ([("A", hb("LCK")), ("B", k["LCK"][1])], "LCK"),
        ([("A", hb("LCK")), ("B", k["LCK"][1])], "Worlds"),             # 同母赛区
        ([("A", hb("LPL")), ("B", k["LPL"][1])], "Esports World Cup"),
        ([("A", hb("LCK")), ("B", hb("LPL"))], "MSI"),                  # 跨赛区
        ([("A", hb("LEC")), ("B", hb("LCS"))], "DCGI"),
        ([("GAM Esports", None), ("B", hb("LCK"))], "DCGI"),            # 一队不认识
        ([("RED Kalunga", None), ("NAVI", None)], "DCGI"),              # 两队都不认识
        ([("TBD", None), ("TBD", None)], "LCK"),                        # 国内也可能映射不出来
        ([("X", None), ("B", hb("LCK"))], "LCK"),
        ([("A", hb("LCK"))], "LCK"),                                    # 不是两队
    ]
    reasons = []
    for ts, lg in shapes:
        m = EF.Match(match_id="x", league=lg, start_time="", state="unstarted", best_of=1,
                     teams=[EF.TeamRef(api_name=a, code="", model_name=mn) for a, mn in ts])
        api._pregame_of(m)
        reasons.append({"in": [[{"api": a, "model": mn} for a, mn in ts], lg, m.pregame_league],
                        "out": list(api._pregame_reason(m))})

    kinds = list(dict.fromkeys((r["out"][0], r["out"][1]) for r in reasons))
    early = [{"in": [mn, p1, p2, wk, why], "out": api._early_prediction(mn, p1, p2, wk, why)}
             for wk, why in kinds for mn in (None, 0, 2)
             for p1, p2 in ((None, None), (0.6123456789, None), (None, 0.4321), (0.6123456789, 0.4321))]
    warns0 = [["本次没有经验差数据, 改用不含经验差的局内模型。", NO_PRE_STATS, "概率已经触到…"],
              [NO_PRE_STATS], [], ["别的提醒"]]
    bw = [{"in": [w, wk, why], "out": api._board_warnings(w, wk, why)}
          for wk, why in kinds for w in warns0]
    return {"pregame_league": pg, "known_for": kf, "map_team": mt,
            "pregame_reason": reasons, "early": early, "board_warnings": bw}


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
    out.append(run(k["LCK"][0], k["LPL"][0], "LCK"))       # 跨赛区的两队 (看板上国际赛跨赛区不给赛前)
    out.append(run(k["LEC"][0], k["LCS"][0], "LEC"))
    # 国际赛同母赛区: 看板按 pregame_league (= 两队母赛区) 去问赛前, 即 Stage 1 拿到的 league
    # 是母赛区, 不是赛事名。src 只是标注, 对账照常按 in / resp 比
    for ev, hl in (("Worlds", "LCK"), ("MSI", "LPL"), ("Esports World Cup", "LEC")):
        b, r = k[hl][2], k[hl][3]
        lg = api.pregame_league(store, b, r, ev)
        if lg != hl:
            raise SystemExit(f"{b} / {r} 在 {ev} 上的 pregame_league 是 {lg}, 不是 {hl}")
        out.append({**run(b, r, lg), "src": f"intl:{ev}→{lg}"})
    out.append(run(k["LPL"][0], k["LPL"][1], "LPL", True))  # 季后赛
    out.append(run(k["LCK"][1], k["LCK"][0], "LCK", True))
    out.append(run("Definitely Not A Team", k["LCK"][0], "LCK"))
    out.append(run(k["LCK"][0], "Definitely Not A Team", "LCK"))
    # 在队伍史里、但不在任何赛区已知队伍表里的 (场次太少或换了赛区)
    extra = [t for t in sorted(store.team_history) if all(t not in v for v in k.values())][:3]
    for t in extra:
        out.append(run(t, k["LPL"][0], "LPL"))
    return out


def stage2_cases(api, store) -> dict:
    """看板那条"BP 后"参考线: 元数据里的十个人 → api._board_draft 组 BP → api._predict_core。

    两步都核对: 名字对齐 (去战队前缀、英雄 id 归一化) 发生在第一步, 概率在第二步; 另存一份
    特征向量 —— 名字对不齐时概率可能只差一点点, 向量能直接指出是哪一列。
    输入取 collected/ 的真实元数据; 战队简称用同一方五个名字的公共前缀代替 (元数据里没存
    简称, 而这里只要两边拿到同样的输入)。另有一组合成用例专打别名英雄和认不出来的名字。
    """
    import os
    from esports_feed import is_major, map_team, oe_player_name

    s2 = api.STATE["stages"]["post_draft"]

    class _Feed:
        def __init__(self, md):
            self.md = md

        def game_metadata(self, _gid):
            return self.md

    class _Team:
        def __init__(self, code):
            self.code = code

    class _Minfo:
        def __init__(self, codes):
            self.teams = [_Team(c) for c in codes]

    def run(src, players, codes, blue, red, league):
        c = {"src": src, "in": {"players": players, "codes": codes, "blue": blue, "red": red,
                                "league": league}}
        draft = api._board_draft(_Feed({i + 1: p for i, p in enumerate(players)}), "g", _Minfo(codes))
        c["draft"] = draft
        if draft:
            try:
                c["p2"] = api._predict_core(blue, red, league, False, draft)
                dd = {s: {r: (v["player"], v["champion"]) for r, v in draft[s].items()}
                      for s in ("blue", "red")}
                row, _w = store.make_row(blue, red, league, draft=dd, playoffs=0)
                # 和 Stage.predict 里组向量的两行一字不差
                x = np.nan_to_num(np.array([[float(row.get(f, np.nan)) for f in s2.features]]),
                                  nan=-999.0)
                c["x"] = [float(v) for v in x[0]]
            except Exception as e:
                c["error"] = str(e)
        return c

    cases = []
    for mp in sorted(COLLECTED.glob("*.meta.json")):
        meta = json.loads(mp.read_text(encoding="utf-8"))
        lg = meta.get("league")
        if not is_major(lg):
            continue
        players = [{k: p.get(k) for k in ("summoner_name", "champion", "role", "side")}
                   for p in (meta.get("players") or {}).values()]
        codes = []
        for side in ("blue", "red"):
            names = [p["summoner_name"] or "" for p in players if p["side"] == side]
            codes.append(os.path.commonprefix(names).strip() if len(names) == 5 else "")
        known = store.known_teams(lg)
        cases.append(run(mp.name[: -len(".meta.json")], players, codes,
                         map_team(meta.get("blue") or "", known),
                         map_team(meta.get("red") or "", known), lg))

    base = next((c for c in cases if "p2" in c), None)
    if base is None:
        raise SystemExit("collected/ 里找不到一场能算出 BP 后概率的比赛当合成用例的底子")
    bi = base["in"]

    def syn(label, **kw):
        d = {"players": [dict(p) for p in bi["players"]], "codes": list(bi["codes"]),
             "blue": bi["blue"], "red": bi["red"], "league": bi["league"]}
        edit = kw.pop("edit", None)
        d.update(kw)
        if edit:
            edit(d["players"])
        return run(f"syn:{label}", d["players"], d["codes"], d["blue"], d["red"], d["league"])

    def champs(names):
        def f(ps):
            for p, n in zip([p for p in ps if p["side"] == "blue"], names):
                p["champion"] = n
        return f

    cases += [
        syn("别名英雄", edit=champs(["MonkeyKing", "LeeSin", "Renata", "Nunu", "KSante"])),
        syn("认不出的英雄", edit=champs(["NotAChampion"] * 5)),
        syn("同一英雄五个", edit=champs(["Azir"] * 5)),
        syn("简称对不上", codes=["ZZ", "YY"]),
        syn("缺一个人", edit=lambda ps: ps.pop()),
        syn("位置缺失", edit=lambda ps: ps[0].update(role=None)),
        syn("选手名为空", edit=lambda ps: ps[1].update(summoner_name="")),
        syn("队伍不认识", blue="Definitely Not A Team"),
        syn("换边", blue=bi["red"], red=bi["blue"]),
    ]
    for lg in MAJORS:
        cases.append(syn(f"赛区={lg}", league=lg))
    # 国际赛同母赛区: 看板的 BP 后按 pregame_league (= 两队母赛区) 算, 英雄胜率查母赛区的表。
    # in.league 就是传给 _predict_core 的那个 (母赛区); in.match_league 只是标注
    for ev, hl in (("Worlds", "LCK"), ("Esports World Cup", "LPL")):
        kk = store.known_teams(hl)
        lg = api.pregame_league(store, kk[0], kk[1], ev)
        if lg != hl:
            raise SystemExit(f"{kk[0]} / {kk[1]} 在 {ev} 上的 pregame_league 是 {lg}, 不是 {hl}")
        c = syn(f"国际赛同母赛区 {ev}→{lg}", blue=kk[0], red=kk[1], league=lg)
        c["in"]["match_league"] = ev
        cases.append(c)

    names = [["IGTheShy", ["IG", "AL"]], ["TH Hype", ["TH", "G2"]], ["T1Faker", ["T", "T1"]],
             ["Faker", ["IG", "AL"]], ["IG", ["IG"]], [None, ["IG"]], ["  IGRookie ", ["IG", None]],
             ["", ["IG"]], ["iGRookie", ["IG"]]]
    return {"cases": cases,
            "names": [{"in": [s, c], "out": oe_player_name(s, c)} for s, c in names]}


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
    s2 = api.Stage("post_draft")

    print("加载特征库 (五年 CSV, 和服务同一份 api.DATA) …")
    store = FeatureStore.from_csv(api.DATA)
    # 线上函数要的全局状态 —— 和 lifespan 里装的是同一批东西
    api.STATE["store"] = store
    api.STATE["stages"] = {"pre_draft": s1, "post_draft": s2}
    api.STATE["ingame_live"] = ig
    today = str(pd.Timestamp.now().normalize().date())

    for name, obj in (("ingame_live.json", export_ingame(ig)),
                      ("stage1_pre.json", export_stage1(s1, api.MODEL_KEYS)),
                      ("stage2_post.json", export_stage2(s2, store, api.MODEL_KEYS)),
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
    syn = synthetic_inputs(base) + intl_inputs(base, store)

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

    g2 = stage2_cases(api, store)
    n = _write(OUT_GOLD / "stage2_cases.json", {"generated_at": _now(), "today": today,
                                                "features": list(s2.features), **g2})
    c2 = g2["cases"]
    print(f"  stage2_cases.json  {n / 1024:>6,.0f} KB   {len(c2)} 局阵容, 其中 "
          f"{sum('p2' in c for c in c2)} 局算出 BP 后概率、{sum(not c['draft'] for c in c2)} 局凑不齐十人、"
          f"{sum('error' in c for c in c2)} 局预期报错")

    gi = intl_cases(api, store)
    n = _write(OUT_GOLD / "intl_cases.json", {"generated_at": _now(), "today": today, **gi})
    print(f"  intl_cases.json    {n / 1024:>6,.0f} KB   pregame_league {len(gi['pregame_league'])} 条 "
          f"(其中给赛前 {sum(c['out'] is not None for c in gi['pregame_league'])})  "
          f"known_for {len(gi['known_for'])} 条  map_team {len(gi['map_team'])} 条  "
          f"pregame_reason {len(gi['pregame_reason'])} 条  开局 {len(gi['early'])} 条  "
          f"看板提醒 {len(gi['board_warnings'])} 条")
    print(f"完成, {time.time() - t0:.0f} 秒。下一步: npm --prefix frontend run test:golden")


if __name__ == "__main__":
    main()
