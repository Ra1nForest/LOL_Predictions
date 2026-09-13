"""
分段式预测服务
================
三段递进, 每段独立可用:

  Stage 1  pre_draft    仅队伍历史 — BP 之前
  Stage 2  post_draft   + 英雄选择 — BP 之后
  Stage 3  agent_review + LLM 复核 — 可选, 需 NVIDIA_API_KEY

启动:  uvicorn api:app --reload --port 8000
文档:  http://localhost:8000/docs

前置: python train.py  (生成 artifacts/)
"""
from __future__ import annotations
import json, math, os, numpy as np, pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Literal
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Header
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from xgboost import XGBClassifier

from feature_store import FeatureStore, ROLES
from ingame_service import IngameModel, SessionStore
import websearch as WS
import explain as EX
import esports_feed as EF
from prediction_log import log_prediction

_ROOT = Path(__file__).parent
DATA = [str(_ROOT / "data" / f"{y}_LoL_esports_match_data_from_OraclesElixir.csv")
        for y in (2022, 2023, 2024, 2025, 2026)]
ART = _ROOT / "artifacts"

STATE: dict = {}


# ══════════════════════════════════════════════════════════
#  模型封装
# ══════════════════════════════════════════════════════════

class Stage:
    def __init__(self, name: str):
        self.name = name
        self.model = XGBClassifier()
        self.model.load_model(str(ART / f"model_{name}.json"))
        cfg = json.loads((ART / f"calib_{name}.json").read_text())
        self.a, self.b = cfg["coef"], cfg["intercept"]
        self.features = cfg["features"]
        self.metrics = cfg["metrics"]
        self.importance = dict(zip(self.features, self.model.feature_importances_))

    def predict(self, row: dict) -> tuple[float, float]:
        x = np.array([[float(row.get(f, np.nan)) for f in self.features]])
        x = np.nan_to_num(x, nan=-999.0)
        raw = float(self.model.predict_proba(x)[0, 1])
        cal = 1.0 / (1.0 + math.exp(-(self.a * raw + self.b)))
        return raw, cal

    def top_evidence(self, row: dict, k: int = 8):
        items = []
        for f in self.features:
            v = row.get(f)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            items.append((f, float(v), float(self.importance.get(f, 0.0))))
        items.sort(key=lambda t: -t[2])
        return items[:k]


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("  loading feature store...")
    STATE["store"] = FeatureStore.from_csv(DATA)
    print("  loading models...")
    STATE["stages"] = {n: Stage(n) for n in ("pre_draft", "post_draft")}
    STATE["sessions"] = SessionStore()
    STATE["feed"] = EF.EsportsFeed()
    sb = WS.available()
    print(f"  search backend: {sb or '未配置 (agent 将无外部信息源)'}")
    try:
        full, live = IngameModel.load_pair()
        STATE["ingame"], STATE["ingame_live"] = full, live
        mt = full.metrics
        print(f"  ingame model: acc {mt['accuracy']:.1%}, "
              f"range [{mt['p_min']:.2f}, {mt['p_max']:.2f}]")
        if live:
            lm = live.metrics
            print(f"  ingame live 变体: {len(live.features)} 特征 (无经验差), "
                  f"acc {lm['accuracy']:.1%}")
        else:
            print("  ingame live 变体缺失 —— 实时数据源的请求会被拒绝而不是"
                  "用 0 顶替 xpdiff。重跑 ingame_train.py 可生成。")
    except Exception as e:
        STATE["ingame"] = STATE["ingame_live"] = None
        print(f"  ingame model 未加载 ({type(e).__name__}) — 先跑 ingame_train.py")
    print("  ready")
    yield
    STATE.clear()


app = FastAPI(title="LoL Match Prediction", version="1.0", lifespan=lifespan)


# ══════════════════════════════════════════════════════════
#  Schemas
# ══════════════════════════════════════════════════════════

class Pick(BaseModel):
    player: str
    champion: str


class DraftSide(BaseModel):
    top: Optional[Pick] = None
    jng: Optional[Pick] = None
    mid: Optional[Pick] = None
    bot: Optional[Pick] = None
    sup: Optional[Pick] = None


class Draft(BaseModel):
    blue: DraftSide
    red: DraftSide


class PredictRequest(BaseModel):
    blue_team: str = Field(..., examples=["JD Gaming"])
    red_team: str  = Field(..., examples=["Bilibili Gaming"])
    league: Literal["LPL", "LCK", "LEC", "LCS"] = "LPL"
    playoffs: bool = False
    draft: Optional[Draft] = None
    agent_review: bool = False
    raw_features: bool = Field(
        False, description="调试用。True 时额外返回英文特征名和原始数值。")



# ══════════════════════════════════════════════════════════
#  对外呈现层
#
#  模型内部字段名 (diff_avg_towers) 不进对外响应 —— 除非显式
#  raw_features=true。前端只拿翻译好的 reasons。
# ══════════════════════════════════════════════════════════

MODEL_KEYS = ("accuracy", "baseline", "lift", "ece_cal", "p_min", "p_max")


def _model_card(stage) -> dict:
    d = {k: stage.metrics[k] for k in MODEL_KEYS if k in stage.metrics}
    d["note"] = EX.confidence_note(d)
    d["range_note"] = EX.range_note(d.get("p_min"), d.get("p_max"))
    return d


def _present(block: dict, raw_ev, blue: str, red: str, raw: bool) -> dict:
    """给一个 stage 块装上 reasons; raw=False 时抹掉英文字段。"""
    block["reasons"] = EX.explain_evidence(raw_ev, blue, red)
    if raw:
        block["evidence"] = [
            {"feature": f, "value": round(v, 3), "importance": round(i, 4)}
            for f, v, i in raw_ev] if raw_ev and isinstance(raw_ev[0], tuple) \
            else raw_ev
    else:
        block.pop("evidence", None)
    return block


# ══════════════════════════════════════════════════════════
#  Endpoints
# ══════════════════════════════════════════════════════════

@app.get("/health")
def health():
    st = STATE.get("stages", {})
    store = STATE.get("store")
    fresh = {}
    if store:
        for lg in ("LPL", "LCK", "LEC", "LCS"):
            s = store.staleness(lg)
            fresh[lg] = {"last_date": s.get("last_date"), "days_old": s.get("days"),
                         "level": s["level"]}
    # 日更流程本身的死活。**和 data_freshness 是两回事**: 后者答"数据有多旧",
    # 这个答"更新流程还活着吗"。两者会分叉 —— 赛区休赛时数据本来就旧而流程
    # 健康 (实测 LEC/LCS 停在 08-30 是真没打); 反过来流程死了好几天, 只要赛区
    # 恰好也没比赛, data_freshness 依然报 fresh。
    # 整个 import 包在 try 里: 这一段是来报告故障的, 不该自己变成故障源。
    try:
        import update_status
        upd = update_status.summary()
    except Exception as e:
        upd = {"level": "unknown", "message": f"读取日更留档失败: {type(e).__name__}: {e}"}

    return {"status": "ok" if st else "loading",
            "stages": {k: v.metrics for k, v in st.items()},
            "data_through": str(store.last_date.date()) if store else None,
            "data_freshness": fresh,
            "update": upd}


@app.get("/teams")
def teams(league: Optional[str] = None):
    store = STATE.get("store")
    if not store:
        raise HTTPException(503, "still loading")
    return {"league": league, "teams": store.known_teams(league)}


# ══════════════════════════════════════════════════════════
#  赛事数据 (lolesports 非官方端点, 见 esports_feed.py)
# ══════════════════════════════════════════════════════════

def _warn(where: str, e: Exception) -> str:
    """静默吞异常是这个项目里几个 bug 能藏住的直接原因 —— 曲线和头条差
    十几个百分点那次就是被 except: pass 盖住的。凡是失败会**改变输出数字**
    (而不只是少一块数据) 的地方, 都要留一条日志, 并把说明带进响应。"""
    msg = f"{where}: {type(e).__name__}: {e}"
    print(f"  ⚠ {msg}", flush=True)
    return msg


def _feed():
    f = STATE.get("feed")
    if f is None:
        raise HTTPException(503, "赛事数据源未初始化")
    return f


def _predict_core(blue, red, league, playoffs, draft):
    """只要一个 Stage 2 概率, 不要整个响应。draft 是
    {side: {role: {player, champion}}}。算不出来返回 None, 不抛。"""
    store, stages = STATE.get("store"), STATE.get("stages")
    if not store or not stages:
        return None
    dd = {side: {r: (v["player"], v["champion"]) for r, v in (draft.get(side) or {}).items()}
          for side in ("blue", "red")}
    row, _w = store.make_row(blue, red, league, draft=dd, playoffs=int(playoffs))
    _raw, cal = stages["post_draft"].predict(row)
    return float(cal)


_ROLE_OE = {"jungle": "jng", "bottom": "bot", "support": "sup"}


def _board_draft(feed, game_id, minfo):
    """看板上本局的 BP, 给 Stage 2 用: {side: {role: {player, champion}}}; 凑不齐十个人返回 None。

    选手名剥掉战队简称前缀 (esports_feed.oe_player_name) —— 早先原样传 "IGTheShy", 训练里
    按 OE 名字 "TheShy" 存的熟练度一个都查不到, BP 后预测的熟练度特征恒为 0, 不报错。
    英雄名不在这里转: FeatureStore 查表时自己按 champion_key 对齐。
    浏览器版 frontend/src/web/board.ts 的 boardDraft 是同一套逻辑。
    """
    from esports_feed import oe_player_name
    codes = [t.code for t in minfo.teams]
    draft = {"blue": {}, "red": {}}
    for mm2 in feed.game_metadata(game_id).values():
        role = _ROLE_OE.get(mm2.get("role"), mm2.get("role"))
        nm = oe_player_name(mm2.get("summoner_name"), codes)
        ch = mm2.get("champion")
        if role and ch and nm and mm2.get("side") in draft:
            draft[mm2["side"]][role] = {"player": nm, "champion": ch}
    return draft if sum(len(v) for v in draft.values()) == 10 else None


def _clinched(m) -> bool:
    """这场 BO 是否已经分出胜负。

    不能用「赢过至少一局」来判断结束 —— BO3 打到 1-0、第二局正在进行是
    最常见的直播状态。实测就因为这个条件, JDG vs LGD 赢下首局后整场比赛
    从直播列表里消失了。按赛制算需要几胜。
    """
    need = ((m.best_of or 1) // 2) + 1
    return any((t.game_wins or 0) >= need for t in m.teams)


def _refresh_wins(m, feed) -> bool:
    """把系列赛比分换成 getEventDetails 那一份 (20 秒缓存)。改动了就返回 True。

    m.teams[].game_wins 是从 getSchedule 来的, 而那个响应缓存 300 秒 ——
    一局刚打完时列表里会挂着五分钟前的旧比分 (实测: BO3 已经 2-1 结束,
    外面还写 1-1)。同一个字段在 getEventDetails 里也有, 缓存只有 20 秒。

    只对**要返回给前端的那几场**做, 不是对整张赛程做: 每场一个请求, 而
    直播中的比赛通常只有个位数。看板那边更省 —— 它已经调过 feed.games(),
    是同一个 URL, 走缓存不多花请求。

    取不到就原样不动: 拿旧比分显示, 好过把已知的比分抹成 None。
    """
    try:
        wins = feed.match_wins(m.match_id)
    except Exception:
        return False
    if not wins:
        return False
    changed = False
    for t in m.teams:
        w = wins.get(EF._norm(t.api_name))
        if w is None:
            w = wins.get(EF._norm(t.code or ""))
        if w is not None and w != t.game_wins:
            t.game_wins = w
            changed = True
    return changed


def _match_json(m, feed):
    """比赛 -> JSON。队名映射不出来时明确标出, 不静默放过 ——
    队名对不上不会报错, 只会让那一侧的特征变成缺失值。"""
    return {
        "match_id": m.match_id,
        "league": m.league,
        "start_time": m.start_time,
        "state": m.state,
        "best_of": m.best_of,
        "predictable": m.predictable,
        "teams": [{
            "name": t.api_name,
            "code": t.code,
            "model_name": t.model_name,
            "known": t.usable,
            "game_wins": t.game_wins,
            "image": t.image,
        } for t in m.teams],
        "unknown_teams": [t.api_name for t in m.teams if not t.usable],
    }


@app.get("/esports/schedule")
def esports_schedule(league: str, state: Optional[str] = None):
    """赛程。state 可选 unstarted / inProgress / completed。"""
    store, feed = STATE.get("store"), _feed()
    known = store.known_teams(league) if store else None
    try:
        ms = feed.schedule(league, known=known)
    except Exception as e:
        raise HTTPException(502, f"赛程拉取失败: {e}")
    if state:
        ms = [m for m in ms if m.state == state]
    return {
        "league": league.upper(),
        "count": len(ms),
        "matches": [_match_json(m, feed) for m in ms],
    }


@app.get("/esports/ddragon")
def esports_ddragon():
    """Data Dragon 的当前版本。图标路径按版本走, 前端自己拼 URL 时要用。"""
    return {"version": _feed().ddragon_version(), "base": EF.DDRAGON}


@app.get("/esports/upcoming")
def esports_upcoming(limit: int = 24, past_hours: int = 5):
    """四大赛区接下来的比赛, 按开赛时间合并排序。

    **不按 state 过滤。** 实测 2026-08-16: LPL 当天三场 (含两场尚未到
    开赛时间的) 在 getSchedule 里全部标成 state="completed", 而
    getEventDetails 说三局都是 unstarted、比分 0-0。按 state 过滤会让
    当天的比赛在界面上整个消失。

    改用两个可信的信号:
      · start_time —— 未来的, 或刚开始不久的 (past_hours 内)
      · 实际战绩 —— 任一方赢过局数 > 0 就是真打过了, 不算"接下来"
    """
    store, feed = STATE.get("store"), _feed()
    now = datetime.now(timezone.utc)
    out, errs = [], []
    for lg in EF.LEAGUE_IDS:
        try:
            known = store.known_teams(lg) if store else None
            for m in feed.schedule(lg, known=known):
                if not m.start_time:
                    continue
                try:
                    ts = datetime.strptime(m.start_time[:19], "%Y-%m-%dT%H:%M:%S") \
                                 .replace(tzinfo=timezone.utc)
                except Exception:
                    continue
                if ts < now - timedelta(hours=past_hours):
                    continue
                if any((t.game_wins or 0) > 0 for t in m.teams):
                    continue
                out.append(_match_json(m, feed))
        except Exception as e:
            errs.append(f"{lg}: {e}")
    out.sort(key=lambda x: x["start_time"] or "")
    return {"count": len(out), "matches": out[:limit], "errors": errs}


@app.get("/esports/live")
def esports_live():
    """当前进行中的比赛。

    getLive 不可靠 —— 实测 2026-08-16 LPL 已经打到第 6 分钟, 它仍返回 0 场。
    所以再拿今天开赛不久的比赛逐个查帧补上。帧是唯一权威。
    """
    store, feed = STATE.get("store"), _feed()
    found, seen = [], set()
    try:
        for m in feed.live():
            found.append(m); seen.add(m.match_id)
    except Exception as e:
        print(f"  getLive 失败: {e}")

    # 补漏: 最近开赛的比赛里, 谁的帧显示真的在打
    try:
        cands = []
        for lg in EF.LEAGUE_IDS:
            known = store.known_teams(lg) if store else None
            for m in feed.schedule(lg, known=known):
                if m.match_id in seen or _clinched(m):
                    continue
                cands.append(m)
        for m in feed.live_by_frames(cands):
            if m.match_id not in seen:
                found.append(m); seen.add(m.match_id)

        # 局间休息: 已经赢过局、还没分出胜负, 但此刻没有任何一局有帧
        # (上一局刚结束, 下一局还在 BP)。这时比赛仍在进行, 不该从列表里
        # 消失 —— 之前它既不算"直播"也不算"即将开赛", 直接不见了。
        now = datetime.now(timezone.utc)
        for m in cands:
            if m.match_id in seen:
                continue
            if not any((t.game_wins or 0) > 0 for t in m.teams):
                continue
            try:
                ts = datetime.strptime(m.start_time[:19], "%Y-%m-%dT%H:%M:%S") \
                             .replace(tzinfo=timezone.utc)
            except Exception:
                continue
            if now - timedelta(hours=6) <= ts <= now:
                m.between_games = True
                found.append(m); seen.add(m.match_id)
    except Exception as e:
        print(f"  live_by_frames 失败: {e}")

    out = []
    for m in found:
        known = store.known_teams(m.league) if store else None
        for t in m.teams:
            t.model_name = EF.map_team(t.api_name, known)

        # 比分改用 20 秒缓存的那一份, 否则这里挂的是最多五分钟前的旧值
        _refresh_wins(m, feed)

        # 局间休息那一支是拿"赢过局数 > 0"筛进来的, 用的是同一份旧比分,
        # 于是**已经打完的系列赛会赖在直播列表里**: 决胜局结束后比分还没
        # 刷新, _clinched 判不出来 (它在上面用 cands 判过一次, 也是旧值)。
        # 比分刷新之后重判一次。只对这一支做 —— 有帧的那几场说明真的在打,
        # 不能因为比分先一步跳到 2-1 就把用户正在看的直播摘掉。
        if getattr(m, "between_games", False) and _clinched(m):
            continue

        j = _match_json(m, feed)
        j["between_games"] = bool(getattr(m, "between_games", False))
        out.append(j)
    return {"count": len(out), "matches": out}


@app.get("/esports/match/{match_id}")
def esports_match(match_id: str):
    """这场 BO 各局的状态 + 真正在进行的那一局的实时读数。

    games[].state 来自 persisted API, 实测会滞后 (曾见到它报 inProgress
    而帧里 gameState 已是 finished)。live_game 以帧为准。
    """
    feed = _feed()
    try:
        games = feed.games(match_id)
        cur = feed.current_game(match_id)
    except Exception as e:
        raise HTTPException(502, f"对局详情拉取失败: {e}")

    live = None
    if cur:
        g, st = cur
        ver = feed.ddragon_version()
        players = []
        try:
            for p in feed.players(st.game_id):
                players.append({
                    "participant_id": p.participant_id,
                    "side": p.side, "role": p.role,
                    "summoner_name": p.summoner_name,
                    "champion": p.champion,
                    "champion_icon": (f"{EF.DDRAGON}/cdn/{ver}/img/champion/{p.champion}.png"
                                      if p.champion else None),
                    "level": p.level,
                    "kills": p.kills, "deaths": p.deaths, "assists": p.assists,
                    "cs": p.cs, "gold": p.gold,
                    "kill_participation": p.kill_participation,
                    "damage_share": p.damage_share,
                    "wards_placed": p.wards_placed,
                    "wards_destroyed": p.wards_destroyed,
                    "current_health": p.current_health,
                    "max_health": p.max_health,
                    "items": [{"id": i,
                               "icon": f"{EF.DDRAGON}/cdn/{ver}/img/item/{i}.png"}
                              for i in p.items],
                    "stats": p.stats,
                })
        except Exception as e:
            # 选手面板拿不到不该让整个接口失败 —— 概率读数不依赖它
            players = []
            print(f"  players() 失败: {e}")
        live = {
            "game_id": st.game_id,
            "game_number": g.get("number"),
            "game_state": st.game_state,
            "minute": st.minute,
            "frame_time": st.frame_time,
            "state": st.to_state(),      # 可直接 POST 给 /predict/ingame
            "players": players,
            "ddragon_version": ver,
        }
    return {
        "match_id": match_id,
        "games": [{"number": g.get("number"), "game_id": g.get("id"),
                   "persisted_state": g.get("state")} for g in games],
        "live_game": live,
        "note": ("live_game 为 null 表示没有任何一局真的在进行中 —— "
                 "即使上面的 persisted_state 显示 inProgress。以帧为准。"),
    }


def _players_json(feed, game_id, ver):
    out = []
    for p in feed.players(game_id):
        out.append({
            "participant_id": p.participant_id, "side": p.side, "role": p.role,
            "summoner_name": p.summoner_name, "champion": p.champion,
            "champion_icon": (f"{EF.DDRAGON}/cdn/{ver}/img/champion/{p.champion}.png"
                              if p.champion else None),
            "level": p.level, "kills": p.kills, "deaths": p.deaths,
            "assists": p.assists, "cs": p.cs, "gold": p.gold,
            "kill_participation": p.kill_participation,
            "damage_share": p.damage_share,
            "wards_placed": p.wards_placed, "wards_destroyed": p.wards_destroyed,
            "current_health": p.current_health, "max_health": p.max_health,
            "items": [{"id": i, "icon": f"{EF.DDRAGON}/cdn/{ver}/img/item/{i}.png"}
                      for i in p.items],
            # 饰品单独一格 —— 帧里它混在 items 数组中且位置不固定, 见 trinket_ids()
            "trinket": ({"id": p.trinket,
                         "icon": f"{EF.DDRAGON}/cdn/{ver}/img/item/{p.trinket}.png"}
                        if p.trinket else None),
            "stats": p.stats,
            "perks": p.perks,
            "abilities": p.abilities,
        })
    return out


def _playable_games(games: list[dict], framed: dict) -> list[dict]:
    """哪些局"打过或正在打" —— 有帧即算数, 不只信 games[].state。

    坑: games[].state 两个方向都会错。实测 2026-09-06 LPL WE vs IG, 第 1 局
    已经打完 (帧 finished, 第 23 分钟, 18-7), 五局的 state 仍全写着
    unstarted。只信它的话这里返回空表 —— 于是选局没有回退可选、标签页一个
    不出, 整个看板空白显示"尚未开始", 而系列赛其实已经 1-0。

    反过来也有: state 说 completed 而比赛没开打 (2026-08-16 LPL 三场), 所以
    两个来源取并集, 谁也不单独当权威。
    """
    return [g for g in games
            if g.get("state") in ("inProgress", "completed")
            or g.get("id") in framed]


@app.get("/esports/board/{match_id}")
def esports_board(match_id: str, game_id: Optional[str] = None,
                  curves: bool = True, points: int = 60):
    """观赛板要的一切, 一个请求拿全。

    前端原本要打 /esports/match 再打 /predict/ingame, 每轮两次往返, 而且
    上游的 getEventDetails / window / details 各自重复取。这里在服务端聚合
    并复用同一份缓存, 轮询一次只剩一个请求。

    game_id 可指定看第几局 (Game 1/2/3 切换); 不给就自动选正在打的那局,
    没有在打的就取最后一局已完成的。
    """
    store, feed = STATE.get("store"), _feed()
    ig_full, ig_live = STATE.get("ingame"), STATE.get("ingame_live")

    try:
        games = feed.games(match_id)
    except Exception as e:
        raise HTTPException(502, f"对局详情拉取失败: {e}")

    # 这场比赛的基本信息 —— 从直播列表或赛程里找
    minfo = None
    try:
        for m in feed.live():
            if m.match_id == match_id:
                minfo = m; break
        if minfo is None:
            for lg in EF.LEAGUE_IDS:
                for m in feed.schedule(lg, known=(store.known_teams(lg) if store else None)):
                    if m.match_id == match_id:
                        minfo = m; break
                if minfo:
                    break
    except Exception as e:
        _warn("查找比赛信息失败", e)
    if minfo is not None and store:
        known = store.known_teams(minfo.league)
        for t in minfo.teams:
            t.model_name = EF.map_team(t.api_name, known)
    if minfo is not None:
        # 上面 feed.games() 已经打过 getEventDetails, 同一个 URL 走缓存,
        # 所以这里刷新比分不多花请求。不刷的话页面顶上的系列赛比分和
        # 列表一样会停在五分钟前。
        _refresh_wins(minfo, feed)

    # 每一局的选边。games[].teams[] 里有 {id, side} —— 不用靠选手名前缀去猜。
    # 这不只是配色: 蓝色方在模型里是一个特征, 把队伍摆错边预测就是错的。
    def sides_of(g):
        out = {}
        for t in (g.get("teams") or []):
            if t.get("side") in ("blue", "red"):
                out[t["side"]] = t.get("id")
        return out

    # 哪些局"打过或正在打"。**不能只看 games[].state** —— 那个字段两个方向
    # 都会错 (见 esports_feed 模块顶部)。实测 2026-09-06 LPL WE vs IG: 第 1
    # 局都打完了 (帧 finished, 第 23 分钟, 18-7), 五局的 state 仍全是
    # unstarted。只信它的话 playable 为空, 于是回退没得选、标签页一个不出,
    # 整个看板空白显示"尚未开始"。
    #
    # 有帧就说明这一局真的开过。帧有缓存, probe_games 和下面的 current_game
    # 共用同一批结果, 不会多打上游。
    framed = {g.get("id"): st for g, st in feed.probe_games(match_id) if st}
    playable = _playable_games(games, framed)
    chosen, st = None, None
    if game_id:
        chosen = next((g for g in games if g.get("id") == game_id), None)
        if chosen:
            st = feed.window(chosen["id"])
    else:
        cur = feed.current_game(match_id)
        if cur:
            chosen, st = cur
        elif playable:
            # 没有正在打的局 —— 局间休息, 或者整场打完了。停在最后一局,
            # 那是页面该对着的地方。
            chosen = playable[-1]
            st = framed.get(chosen["id"]) or feed.window(chosen["id"])

    # 选定之后再取一次, 这回要精确分钟数。
    # 选局阶段可能要把一场 BO 的每一局都探一遍, 不值得为每局都扫一遍暂停;
    # 但真正要显示、要喂给模型的这一局必须把暂停扣掉 —— 帧的时间戳是真实
    # 世界时刻, 暂停期间照走, 不扣就会报出比游戏内时钟大好几分钟的数,
    # 还会连带选错 Stage 4 的切片。窗口本身已经缓存, 这次只多付扫描。
    if chosen and st is not None:
        st = feed.window(chosen["id"], precise_minute=True) or st

    # 按选中这一局的选边给队伍排序: [0] 永远是蓝色方。
    # 界面上左蓝右红是固定的, 换的是哪支队站哪边。
    side_note = None
    side_warn = None
    if minfo and chosen and st is not None:
        # 以**帧**的选边为准, 不是 persisted 的 games[].teams[].side。
        # 两个源约 8% 的比赛会打架 (实测 2026-08-20 BLG vs LGD 第 2 局)。
        # 之所以选帧, 不是因为它更权威, 而是因为页面上每一个数字 ——
        # 金币、人头、逐路对位、喂给模型的 golddiff、P(blue) —— 都是按帧的
        # blueTeamMetadata 分组算出来的。队名跟另一个源排, 就会把 A 队的
        # 数字写在 B 队名下, 而且全程不报错。
        by_id = {}
        try:
            ev = (feed._get(f"{EF.PERSISTED}/getEventDetails?hl=en-US&id={match_id}",
                            feed.live_ttl) or {})
            for t in ((ev.get("data") or {}).get("event") or {}) \
                     .get("match", {}).get("teams", []):
                if t.get("id"):
                    by_id[str(t["id"])] = t.get("name")
        except Exception as e:
            side_warn = _warn("取队名失败, 无法确认本局选边", e)

        fs = {}
        try:
            fs = feed.frame_sides(st.game_id)
        except Exception as e:
            side_warn = side_warn or _warn("取帧选边失败", e)

        bname = by_id.get(fs.get("blue", ""))
        if bname and len(minfo.teams) == 2:
            # 归一化比对。**不能用 ==**: getEventDetails 返回全大写
            # ("BILIBILI GAMING"), schedule 返回的是 "Bilibili Gaming"。
            # 原来那版直接比字符串, 这场能对上纯属大小写碰巧一致, 换一场
            # 就静默地不重排了 —— 而不重排恰恰不会报错, 只是左右反了。
            want = EF._norm(bname)
            cur = [EF._norm(t.api_name) for t in minfo.teams]
            if cur[0] != want and cur[1] == want:
                minfo.teams.reverse()
                side_note = "按本局选边调整了左右"
            elif cur[0] != want and cur[1] != want:
                side_warn = side_warn or (
                    f"本局蓝方是 {bname}, 但和赛程里的队名都对不上 —— "
                    f"左右未调整, 可能是反的")

            # persisted 怎么说也看一眼, 不一致就摆出来。不是拿它纠正帧,
            # 而是不让"两个官方源打架"这件事悄悄过去。
            sd = sides_of(chosen)
            if sd.get("blue") and fs.get("blue") and str(sd["blue"]) != fs["blue"]:
                pname = by_id.get(str(sd["blue"])) or sd["blue"]
                side_warn = side_warn or (
                    f"两个官方源对本局选边说法不一: 实时帧说蓝方是 {bname}, "
                    f"赛程接口说是 {pname}。页面上的数字全部按实时帧分组, "
                    f"因此以帧为准; 若转播画面显示的是另一边, 以转播为准。")
        elif fs.get("blue") and not bname:
            side_warn = side_warn or "拿不到本局的队伍对照, 左右可能不准"
        elif not fs.get("blue"):
            side_warn = side_warn or "这一局的帧里没有选边信息, 左右可能不准"
    elif minfo and chosen:
        # 还没有帧 (这一局没开打)。此时无从判断选边, 队序就是赛程顺序 ——
        # 不能顶着"蓝色方/红色方"的标签当真, 否则等于拿系列赛顺序冒充本局选边。
        side_warn = "本局还没有实时帧, 无法确认选边 —— 左右按赛程顺序排, 不代表蓝红"

    ver = feed.ddragon_version()
    out = {
        "match_id": match_id,
        "match": _match_json(minfo, feed) if minfo else None,
        "side_note": side_note,
        "side_warning": side_warn,
        # playable 决定界面上出不出这一局的标签页, 用上面那个"有帧即算数"
        # 的口径, 和选局用的是同一份判断 —— 否则会出现"看板有数据、上面
        # 却一个局号都没有"。
        "games": [{"number": g.get("number"), "game_id": g.get("id"),
                   "persisted_state": g.get("state"),
                   "playable": g in playable,
                   "sides": sides_of(g)}
                  for g in games],
        "selected_game_id": chosen.get("id") if chosen else None,
        "ddragon_version": ver,
        "live": None,
        "prediction": None,
        "timeline": [],
        "note": ("以帧里的 gameState 为准, persisted_state 会滞后。"),
    }
    if not chosen or not st:
        return out

    out["live"] = {
        "game_id": st.game_id, "game_number": chosen.get("number"),
        "game_state": st.game_state, "minute": st.minute,
        "frame_time": st.frame_time, "state": st.to_state(),
        "in_progress": st.live,
        "teams": st.teams,          # 大龙/小龙/塔/水晶/经济/人头/补刀
        "stale_seconds": st.stale_seconds,
        # minute 已经扣掉暂停; 墙钟和暂停量一起给出去, 否则界面上的分钟数
        # 和转播里的比赛时钟对不上时没人说得清是哪边错了。
        "paused_seconds": round(st.paused_seconds),
        # 帧不再推进。gameState 还写着 in_game, 但那是帧流断在了这里,
        # 不是比赛还在打 —— 这一局多半已经结束却从没推送过 finished 帧。
        "stalled": st.stalled,
    }
    if st.stalled:
        out["live"]["note"] = (
            f"这一局的数据流已经 {st.stale_seconds // 60} 分钟没有更新了 —— "
            f"帧里还写着 {st.game_state}, 但它停在 {st.frame_time}, "
            f"不是实时战况。"
            + ("赛程也已经把这一局标为结束。"
               if chosen.get("state") == "completed" else ""))
    try:
        ps = _players_json(feed, st.game_id, ver)
        out["live"]["players"] = ps
        # 各路对位经济差 (蓝减红), 界面中间那一栏用
        alias = {"jng": "jungle", "bot": "bottom", "sup": "support"}
        lanes = []
        for lane in ("top", "jungle", "mid", "bottom", "support"):
            b = next((p for p in ps if p["side"] == "blue"
                      and alias.get(p["role"], p["role"]) == lane), None)
            r = next((p for p in ps if p["side"] == "red"
                      and alias.get(p["role"], p["role"]) == lane), None)
            if b and r:
                lanes.append({"lane": lane, "gold_diff": b["gold"] - r["gold"],
                              "blue_gold": b["gold"], "red_gold": r["gold"]})
        out["live"]["lanes"] = lanes
    except Exception as e:
        out["live"]["players"] = []
        out["live"]["lanes"] = []
        print(f"  players() 失败: {e}")

    # 概率: 服务端直接算, 省一次往返。
    # 开局前几分钟不算 —— 模型最早的切片是 T=10, IngameState 也要求
    # minute>=5。硬塞会抛校验错误 (实测: 第 2 分钟时整个响应变成一条
    # pydantic 报错文本)。这时候实时战况照常给, 只是没有概率。
    # minute 为 None 也走这条路: 开局头一两分钟 _game_start 还拿不到最初那几帧
    # (window 的默认调用返回空), 分钟数就是未知。之前的条件只挡了 minute<5,
    # None 直接漏进下面的 IngameState(minute=None) -> pydantic 校验错误,
    # 整个响应变成一段报错文本。不知道第几分钟 = 选不了切片, 和太早是一回事。
    board_p2 = None          # BP 后概率; 曲线开局那段的渐变也要用 (_blend_prior)
    MIN_MINUTE = 3
    if st.minute is None or st.minute < MIN_MINUTE:
        # 局内模型给不了不等于没有概率可给 —— 赛前和 BP 后这两段一直有效,
        # 而且开局几分钟局势本来就接近它们。之前这里直接不给数, 界面上就成了
        # "前十分钟不预测", 那是把能给的也藏了。
        p1 = p2 = None
        if minfo and minfo.predictable:
            b, r = minfo.teams[0].model_name, minfo.teams[1].model_name
            try:
                _row, p1, _w = _pre_context(b, r, minfo.league)
            except Exception:
                pass
            try:
                draft = _board_draft(feed, st.game_id, minfo)
                if draft:
                    p2 = _predict_core(b, r, minfo.league, False, draft)
            except Exception:
                pass
        board_p2 = p2
        # 留档: 赛前/BP 后的概率整局不变, minute=None 让它每局只记一次
        log_prediction(match_id=match_id, game_id=st.game_id,
                       league=minfo.league if minfo else None,
                       blue=minfo.teams[0].model_name if minfo else None,
                       red=minfo.teams[1].model_name if minfo else None,
                       probability_blue=(p2 if p2 is not None else p1),
                       source="post_draft" if p2 is not None else "pre_draft",
                       minute=None, game_number=chosen.get("number"))
        out["prediction"] = {
            "too_early": True, "minute": st.minute,
            "pregame_probability_blue": p1,
            "postdraft_probability_blue": p2,
            "probability_blue": p2 if p2 is not None else p1,
            "source": "post_draft" if p2 is not None else "pre_draft",
            "note": (f"开局第 {st.minute} 分钟, " if st.minute is not None
                     else "刚开局 (局内时间还没同步出来), ")
                    + "局内模型最早在第 10 分钟的切片上训练过。这里显示的是"
                    + ("BP 后" if p2 is not None else "赛前")
                    + "的概率 —— 它不看局内数据, 但一直有效。",
        }
    elif minfo and minfo.predictable and (ig_live or ig_full):
        try:
            b, r = minfo.teams[0].model_name, minfo.teams[1].model_name
            sd = IngameState(**st.to_state())
            # _ingame_core 返回 (响应, 特征行) —— 第二个元素里有 NaN,
            # 直接塞进 JSON 会让整个响应 500。只要第一个。
            # 阵容: 帧的 gameMetadata 里就有本局英雄, 之前没传下去, 于是
            # 每次都报"没填双方阵容, 无法判断谁的阵容更耐拖" —— 数据在手边
            # 却没用。局内模型靠它算阵容强势期指数。
            bch = rch = None
            try:
                meta0 = feed.game_metadata(st.game_id)
                bl = [v["champion"] for v in meta0.values()
                      if v.get("side") == "blue" and v.get("champion")]
                rd = [v["champion"] for v in meta0.values()
                      if v.get("side") == "red" and v.get("champion")]
                if len(bl) == 5 and len(rd) == 5:
                    bch, rch = bl, rd
            except Exception as e:
                _warn("头条取阵容失败, 本次预测不含阵容强势期", e)
            # BP 后 (第二段) 的概率 —— **先算**: 头条开局那段要从它渐变过渡到局内模型
            # (_blend_prior)。帧的 gameMetadata 里有本局英雄和选手, 名字怎么对齐见
            # _board_draft。凑不齐十个人就不给这条线, 不猜 (渐变就退回赛前概率)。
            p2 = None
            try:
                draft = _board_draft(feed, st.game_id, minfo)
                if draft:
                    p2 = _predict_core(minfo.teams[0].model_name,
                                       minfo.teams[1].model_name,
                                       minfo.league, False, draft)
            except Exception as e:
                print(f"  Stage2 参考线跳过: {type(e).__name__}: {e}")
            board_p2 = p2

            pred, _row = _ingame_core(b, r, minfo.league, sd, bch, rch, True,
                                      blend=True, blend_with=p2)
            log_prediction(match_id=match_id, game_id=st.game_id,
                           league=minfo.league, blue=b, red=r,
                           probability_blue=pred.get("probability_blue"),
                           source="ingame", minute=st.minute,
                           slice_used=pred.get("slice_used"),
                           game_number=chosen.get("number"),
                           golddiff=st.golddiff,
                           blue_kills=st.blue_kills, red_kills=st.red_kills)
            if p2 is not None:
                pred["postdraft_probability_blue"] = p2

            out["prediction"] = pred
        except HTTPException as e:
            out["prediction"] = {"error": e.detail}
        except Exception as e:
            out["prediction"] = {"error": str(e)}

    # 曲线
    if curves and minfo and minfo.predictable:
        try:
            # 上界用**这一局最后一帧**的时刻, 不是"现在" —— 见 gold_timeline
            upto = None
            if st.frame_time:
                try:
                    upto = datetime.strptime(st.frame_time[:19], "%Y-%m-%dT%H:%M:%S") \
                                   .replace(tzinfo=timezone.utc)
                except Exception:
                    upto = None
            rows = feed.downsample(feed.gold_timeline(st.game_id, upto=upto), points)
            b, r = minfo.teams[0].model_name, minfo.teams[1].model_name
            # 赛前特征只和队伍有关, 整条曲线算一次就够
            try:
                pre_ctx = _pre_context(b, r, minfo.league)
            except Exception as e:
                # 曲线丢掉赛前特征 -> 整条线和头条对不上, 但不会报错
                pre_ctx = None
                _warn("曲线取赛前特征失败, 曲线会偏离头条", e)
            # 阵容也要传给曲线。之前这里写死 None —— 头条带阵容、曲线不带,
            # 于是同一时刻两个数能差十几个百分点 (实测末端 75% 对头条 91.5%)。
            # 局内模型靠阵容算强势期指数, 缺了它整条曲线都偏。
            c_bch = c_rch = None
            try:
                _meta = feed.game_metadata(st.game_id)
                _bl = [v["champion"] for v in _meta.values()
                       if v.get("side") == "blue" and v.get("champion")]
                _rd = [v["champion"] for v in _meta.values()
                       if v.get("side") == "red" and v.get("champion")]
                if len(_bl) == 5 and len(_rd) == 5:
                    c_bch, c_rch = _bl, _rd
            except Exception as e:
                _warn("曲线取阵容失败, 曲线会偏离头条", e)

            series, last_p = [], None
            for row in rows:
                pt = {"minute": row["minute"], "golddiff": row["golddiff"],
                      "blue_kills": row["blue_kills"], "red_kills": row["red_kills"]}
                # 胜率只在 >=5 分钟后算 —— 模型的切片从 T=10 起, 更早没有意义
                if row["minute"] >= 3 and (ig_live or ig_full):
                    try:
                        sd = IngameState(
                            minute=min(60, max(3, row["minute"])),
                            golddiff=row["golddiff"], xpdiff=None,
                            csdiff=row.get("csdiff", 0),
                            blue_kills=row["blue_kills"], red_kills=row["red_kills"],
                            gold_total=row["blue_gold"])
                        # include_pregame 必须和头条数字一致 (True)。
                        # 传 False 会让模型丢掉全部赛前特征, 实测同一时刻
                        # 曲线末端 81.7% 对头条 87.9% —— 差 6pp, 看着就像 bug。
                        # blend 同头条: 曲线开局那段也从 BP 后渐变过渡, 两者同一个数
                        res, _row = _ingame_core(b, r, minfo.league, sd,
                                                 c_bch, c_rch, True,
                                                 pre_ctx=pre_ctx,
                                                 blend=True, blend_with=board_p2)
                        last_p = res.get("probability_blue")
                    except Exception as e:
                        # 这里以前是静默 pass, 于是元组解包失败也看不出来,
                        # 曲线只是"恰好"空的。至少要留一句。
                        print(f"  曲线点 {row['minute']:.1f}min 失败: "
                              f"{type(e).__name__}: {e}")
                pt["probability_blue"] = last_p
                series.append(pt)
            out["timeline"] = series
        except Exception as e:
            print(f"  timeline 失败: {e}")
    return out


@app.post("/predict")
def predict(req: PredictRequest):
    store, stages = STATE.get("store"), STATE.get("stages")
    if not store:
        raise HTTPException(503, "still loading")

    draft_dict = None
    if req.draft:
        draft_dict = {}
        for side in ("blue", "red"):
            ds = getattr(req.draft, side)
            draft_dict[side] = {r: (p.player, p.champion)
                                for r in ROLES if (p := getattr(ds, r))}

    try:
        row, warns = store.make_row(req.blue_team, req.red_team, req.league,
                                    draft=draft_dict, playoffs=int(req.playoffs))
    except ValueError as e:
        raise HTTPException(400, str(e))

    st = store.staleness(req.league)
    out = {"blue_team": req.blue_team, "red_team": req.red_team,
           "league": req.league, "playoffs": req.playoffs,
           "data_quality": {"league_data_through": st.get("last_date"),
                            "days_old": st.get("days"),
                            "level": st["level"],
                            "reliable": st["level"] == "fresh"},
           "warnings": warns, "stages": {}}

    # Stage 1
    s1 = stages["pre_draft"]
    raw1, cal1 = s1.predict(row)
    out["stages"]["1_pre_draft"] = _present({
        "probability_blue": round(cal1, 4),
        "probability_raw": round(raw1, 4),
        "summary": EX.summarize(cal1, req.blue_team, req.red_team, "1_pre_draft"),
        "model": _model_card(s1),
    }, s1.top_evidence(row), req.blue_team, req.red_team, req.raw_features)
    final_p, final_stage = cal1, "1_pre_draft"

    # Stage 2
    if draft_dict:
        s2 = stages["post_draft"]
        raw2, cal2 = s2.predict(row)
        out["stages"]["2_post_draft"] = _present({
            "probability_blue": round(cal2, 4),
            "probability_raw": round(raw2, 4),
            "delta_from_stage1": round(cal2 - cal1, 4),
            "summary": EX.summarize(cal2, req.blue_team, req.red_team, "2_post_draft"),
            "model": _model_card(s2),
        }, s2.top_evidence(row), req.blue_team, req.red_team, req.raw_features)
        final_p, final_stage = cal2, "2_post_draft"

    # Stage 3 —— advisory only, 不参与 final 概率
    if req.agent_review:
        try:
            import agents
            stg = stages["post_draft" if draft_dict else "pre_draft"]
            pred_ctx = {"blue_team": req.blue_team, "red_team": req.red_team,
                        "league": req.league, "stage": final_stage,
                        "probability": final_p, "warnings": warns,
                        "model_accuracy": stg.metrics["accuracy"],
                        "model_baseline": stg.metrics["baseline"],
                        "model_ece": stg.metrics["ece_cal"],
                        "p_range": [stg.metrics["p_min"], stg.metrics["p_max"]]}
            rv = agents.review(pred_ctx, stg.top_evidence(row))
            out["stages"]["3_agent_review"] = rv
        except Exception as e:
            out["stages"]["3_agent_review"] = {"advisory_only": True, "error": str(e)}
        # final_p / final_stage 故意不变 —— 见 agents.py 顶部注释

    out["final"] = {"probability_source": final_stage,
                    "summary": EX.summarize(final_p, req.blue_team,
                                            req.red_team, final_stage),
                    "note": "Stage 3 为 advisory-only, 不影响此概率",
                    "probability_blue": round(final_p, 4),
                    "probability_red": round(1 - final_p, 4),
                    "implied_fair_odds_blue": round(1 / final_p, 3),
                    "implied_fair_odds_red": round(1 / (1 - final_p), 3)}
    disc = ["概率来自留出测试集校准的统计模型, 非投注建议。",
            "模型概率输出范围有限, 对一边倒的对局判别力不足。",
            "walk-forward 估计的期望 lift 约 +0.13 ± 0.10, 单场预测不确定性大。"]
    if draft_dict:
        disc.append("数据源已公告: 2026 年 BP 规则变更导致部分比赛的英雄选择记录"
                    "可能不正确, draft 相关特征存在已知误差。")
    if st["level"] == "very_stale":
        disc.insert(0, f"⚠ 本次预测基于 {st['days']} 天前的数据, 结果不可靠。")
    elif st["level"] == "stale":
        disc.insert(0, f"⚠ 数据滞后 {st['days']} 天, 未反映最近比赛。")
    out["disclaimer"] = " ".join(disc)
    return out


# ══════════════════════════════════════════════════════════
#  Stage 4 — 局内
# ══════════════════════════════════════════════════════════

class IngameState(BaseModel):
    minute: float = Field(..., ge=3, le=60, examples=[20])
    golddiff: float = Field(..., description="蓝方减红方, 正数=蓝方领先")
    # null 表示「拿不到」, 不是「等于 0」。实时数据源 (lolesports) 的帧里
    # 没有 XP 字段, 传 null 会自动改用不含经验差的 live 变体模型。
    xpdiff: Optional[float] = Field(0, description="留 null = 无此数据, 改用 live 模型")
    csdiff: float = 0
    blue_kills: int = 0
    red_kills: int = 0
    gold_total: Optional[float] = Field(None, description="蓝方该时点总经济, 不填则估算")


class IngameRequest(BaseModel):
    blue_team: str
    red_team: str
    league: Literal["LPL", "LCK", "LEC", "LCS"] = "LPL"
    state: IngameState
    blue_champions: Optional[list[str]] = None
    red_champions: Optional[list[str]] = None
    include_pregame: bool = True
    raw_features: bool = False


def _pre_context(req_blue, req_red, league):
    """赛前特征 + 赛前概率。整条曲线共用同一份 —— 它只取决于队伍, 和局内
    时刻无关。之前每个曲线点都重算一遍, 60 个点就是 60 次全量队伍统计,
    切换一局要 16 秒。返回 (pre_row, pre_p, warnings)。"""
    store = STATE.get("store")
    if not store:
        return None, None, []
    r, w = store.make_row(req_blue, req_red, league)
    pre_row = {}
    for k, v in r.items():
        if k.startswith("diff_avg_"):
            pre_row["diff_pre_" + k[len("diff_avg_"):]] = v
    if "diff_rolling_wr" in r:
        pre_row["diff_pre_wr"] = r["diff_rolling_wr"]
    _, pre_p = STATE["stages"]["pre_draft"].predict(r)
    return pre_row, pre_p, w


def _ingame_disclaimer(ig) -> str:
    """局内模型的免责声明 —— 数字和校准方法**全部从 metrics 读, 不写死**。

    原先这里硬编码 "准确率约 73%, 概率经 Platt 校准 (ECE ≈ 0.03)"。换成保序
    回归之后真实是 77% / ECE 0.016, 那句话就成了假的 —— 而它会被界面原样
    印出来、被 Stage 3 的 agent 当事实引用。同一个坑今天在 clip_warning()
    上刚踩过一次: 陈旧的断言比没有断言更糟。
    """
    m = ig.metrics
    acc, ece = m.get("accuracy"), m.get("ece")
    t25 = ((m.get("per_T") or {}).get("25") or {}).get("accuracy")
    how = "保序回归" if getattr(ig, "calibrator", "platt") == "isotonic" else "Platt"
    bits = []
    if acc is not None:
        s = f"局内模型在留出测试集上准确率约 {acc:.0%}"
        if t25 is not None:
            s += f", 25 分钟时约 {t25:.0%}"
        bits.append(s + "。")
    if ece is not None:
        bits.append(f"概率经{how}校准 (ECE ≈ {ece:.3f})。")
    bits.append("非投注建议。")
    return "".join(bits)


# 看板开局那段的渐变 (research/gate_anchor.py 的 C2)。第 BLEND_FROM 分钟以前完全是 BP 后 (没有就
# 赛前) 的概率, 到第 BLEND_TO 分钟完全交给局内模型, 中间在对数几率上线性过渡。
#
# 为什么: 局内模型自带一套"赛前判断" (队伍滚动统计 + 阵容强势期), 看不到 BP 后模型用的英雄胜率和
# 选手熟练度; 12 分钟以前又一律按 T=10 评估, 第 3 分钟的小经济差被放大。原来第 3 分钟硬切,
# 2026-09-13 实测第 2→3 分钟平均跳 14.3 个百分点, 39% 的局跳超过 15 (BP 后说 MKOI 64%, 下一分钟
# 局内给 VIT 83%)。渐变后跳 0.1, 0% 的局超过 15。
# 准确度: 532 局两份模型都没见过的比赛, 全局 Brier t=+0.83, 10-19 分钟 t=+1.85, 各段都不比硬切差
# —— 没过 2.5 的闸, 采纳理由是"不变差 + 连续", 用户 2026-09-13 定的。3/15 在看结果之前定下。
BLEND_FROM, BLEND_TO = 3.0, 15.0


def _blend_prior(prior, ingame, minute):
    """(渐变后的概率, 局内模型的权重)。权重到 1 就原样返回局内模型的数。"""
    w = min(1.0, max(0.0, (float(minute) - BLEND_FROM) / (BLEND_TO - BLEND_FROM)))
    if w >= 1.0:
        return ingame, w
    cl = lambda q: min(max(q, 1e-6), 1 - 1e-6)
    lg = lambda q: math.log(q / (1 - q))
    z = (1 - w) * lg(cl(prior)) + w * lg(cl(ingame))
    return 1 / (1 + math.exp(-z)), w


def _ingame_core(req_blue, req_red, league, st, bch, rch, include_pre, raw=False,
                 pre_ctx=None, blend=False, blend_with=None):
    # blend=True (看板用): 开局那段从 blend_with (BP 后概率; None 则用赛前概率) 渐变过渡到局内模型,
    # 见 _blend_prior。/predict/ingame 不开 —— 它回答的是"局内模型怎么看", 不掺别的模型。
    # xpdiff 为 null = 数据源给不了 (实时接入)。换用训练时就不含经验差的
    # 变体, 而不是拿 0 顶替 —— 后者会让 evidence() 生成「双方经验持平」,
    # 把缺失讲成实测结论。
    if st.xpdiff is None:
        ig = STATE.get("ingame_live")
        if ig is None:
            raise HTTPException(
                503, "本次请求没有经验差数据, 需要 live 变体模型, 但它未加载。"
                     "跑 ingame_train.py 生成 model_ingame_live.json 后重启。")
    else:
        ig = STATE.get("ingame")
    if ig is None:
        raise HTTPException(503, "局内模型未加载, 先跑 ingame_train.py 再重启")
    store = STATE.get("store")

    pre_row, pre_p, warns = None, None, []
    if include_pre and pre_ctx is not None:
        pre_row, pre_p, w = pre_ctx          # 调用方已经算好, 直接用
        warns += list(w)
    elif include_pre and store:
        try:
            pre_row, pre_p, w = _pre_context(req_blue, req_red, league)
            warns += w
        except Exception as e:
            warns.append(f"赛前特征不可用: {e}")

    row, w2 = ig.make_row(
        minute=st.minute, golddiff=st.golddiff, xpdiff=st.xpdiff,
        csdiff=st.csdiff, blue_kills=st.blue_kills, red_kills=st.red_kills,
        gold_total=st.gold_total, blue_champs=bch, red_champs=rch,
        pre_row=pre_row, league=league)
    warns += w2
    raw_p, cal = ig.predict(row)
    ig_cal, w_blend = cal, None
    if blend:
        anchor = blend_with if blend_with is not None else pre_p
        if anchor is not None:
            cal, w_blend = _blend_prior(anchor, cal, st.minute)
    # 概率被上下限夹住的警告要看**实际输出**, 不能按经济差猜 —— 见 clip_warning
    # (渐变之后的数才是实际输出)
    cw = ig.clip_warning(cal)
    if cw:
        warns.append(cw)

    T = row["T"]
    perT = ig.metrics.get("per_T", {}).get(str(T), {})
    out = {
        "blue_team": req_blue, "red_team": req_red, "league": league,
        "minute": st.minute, "slice_used": T,
        "probability_blue": round(cal, 4),
        "probability_raw": round(raw_p, 4),
        "summary": EX.summarize(cal, req_blue, req_red, minute=st.minute),
        "evidence": ig.evidence(row),
        "model": {"accuracy_at_this_slice": perT.get("accuracy"),
                  "high_conf_share_at_this_slice": perT.get("high_conf_share"),
                  "overall_accuracy": ig.metrics["accuracy"],
                  "ece": ig.metrics["ece"],
                  "p_min": ig.metrics.get("p_min"),
                  "p_max": ig.metrics.get("p_max"),
                  "trained_through": ig.trained_through},
        "warnings": warns,
        "disclaimer": _ingame_disclaimer(ig),
    }
    if pre_p is not None:
        out["pregame_probability_blue"] = round(pre_p, 4)
        out["shift_from_pregame"] = round(cal - pre_p, 4)
        d = cal - pre_p
        if abs(d) >= 0.05:
            side = req_blue if d > 0 else req_red
            out["shift_note"] = (f"相比赛前, 局势已向 {side} 移动 "
                                 f"{abs(d) * 100:.0f} 个百分点")
    if w_blend is not None:
        # 头条是渐变后的数; 局内模型自己的读数和它占的权重也给出来, 核对用
        out["ingame_probability_blue"] = round(ig_cal, 4)
        out["blend_weight"] = round(w_blend, 4)
    out["model"]["note"] = EX.confidence_note(out["model"])
    out["model"]["range_note"] = EX.range_note(
        ig.metrics.get("p_min"), ig.metrics.get("p_max"))
    _present(out, out.get("evidence"), req_blue, req_red, raw)
    return out, row


@app.post("/predict/ingame")
def predict_ingame(req: IngameRequest):
    out, _ = _ingame_core(req.blue_team, req.red_team, req.league, req.state,
                          req.blue_champions, req.red_champions,
                          req.include_pregame, req.raw_features)
    # disclaimer 由 _ingame_core 按模型实际 metrics 生成, 这里不再覆盖
    return out


class SessionStart(BaseModel):
    blue_team: str
    red_team: str
    league: Literal["LPL", "LCK", "LEC", "LCS"] = "LPL"
    blue_champions: Optional[list[str]] = None
    red_champions: Optional[list[str]] = None
    note: Optional[str] = None


@app.post("/session/start")
def session_start(req: SessionStart):
    if STATE.get("ingame") is None:
        raise HTTPException(503, "局内模型未加载")
    sid = STATE["sessions"].start(req.model_dump())
    return {"session_id": sid,
            "next": f"POST /session/{sid}/tick 每隔几分钟报一次局势"}


@app.post("/session/{sid}/tick")
def session_tick(sid: str, st: IngameState):
    sess = STATE["sessions"].get(sid)
    if sess is None:
        raise HTTPException(404, "会话不存在")
    m = sess["meta"]
    out, _ = _ingame_core(m["blue_team"], m["red_team"], m["league"], st,
                          m.get("blue_champions"), m.get("red_champions"), True)
    entry = {"minute": st.minute, "golddiff": st.golddiff,
             "probability_blue": out["probability_blue"],
             "slice_used": out["slice_used"]}
    STATE["sessions"].tick(sid, entry)
    traj = STATE["sessions"].trajectory(sid)
    out["session"] = {"id": sid, "n_ticks": traj["session"]["n_ticks"],
                      "summary": traj["summary"]}
    return out


@app.get("/session/{sid}")
def session_get(sid: str):
    traj = STATE["sessions"].trajectory(sid)
    if traj is None:
        raise HTTPException(404, "会话不存在")
    return traj


# ══════════════════════════════════════════════════════════
#  多轮辩论 (可选 · 联网)
# ══════════════════════════════════════════════════════════

class DebateRequest(BaseModel):
    blue_team: str
    red_team: str
    league: Literal["LPL", "LCK", "LEC", "LCS"] = "LPL"
    playoffs: bool = False
    draft: Optional[Draft] = None
    search: bool = True
    ingame: Optional[IngameState] = None
    blue_champions: Optional[list[str]] = None
    red_champions: Optional[list[str]] = None
    # 十个人的逐个读数。**只进简报, 不进模型。**
    #
    # 给平均值是没用的: "等级 蓝 14.6 / 红 13.2" 抹掉了唯一值得判断的东西 ——
    # 是哪一路崩了、谁在吃线、谁被针对。agent 要能说"打野落后 1.8k 且死了 3 次,
    # 蓝方野区大概率被压", 就必须看到逐个人的数。
    players: Optional[list[dict]] = None
    lanes: Optional[list[dict]] = None



def _pre_row_from(row):
    """把 Stage 2 的特征行映射成局内模型认识的 diff_pre_* 命名。
    /predict/ingame 一直是这么做的, /debate 之前漏了 —— 导致同一场比赛
    走两个端点会得到不同的模型输入。"""
    out = {}
    for k, v in row.items():
        if k.startswith("diff_avg_"):
            out["diff_pre_" + k[len("diff_avg_"):]] = v
    if "diff_rolling_wr" in row:
        out["diff_pre_wr"] = row["diff_rolling_wr"]
    return out


ROLE_CN = {"top": "上单", "jungle": "打野", "jng": "打野", "mid": "中单",
           "middle": "中单", "bottom": "ADC", "bot": "ADC",
           "support": "辅助", "sup": "辅助"}
LANE_ORDER = ["top", "jungle", "mid", "bottom", "support"]


def _lane_table(req: DebateRequest) -> list[str]:
    """逐路对位。**不做平均** —— 平均值抹掉的正是唯一值得判断的东西:
    哪一路崩了、谁在吃线、谁被针对。同样的总经济差, 分布在上单和分布在
    打野身上, 含义完全不同。"""
    ps = req.players or []
    if not ps:
        return []

    def norm(r):
        r = (r or "").lower()
        return {"jng": "jungle", "bot": "bottom", "sup": "support",
                "middle": "mid"}.get(r, r)

    gap = {}
    for l in (req.lanes or []):
        gap[norm(l.get("lane"))] = l.get("gold_diff")

    out = ["", "逐路对位 (蓝方 vs 红方):"]
    used = set()
    for lane in LANE_ORDER:
        b = next((x for x in ps if x.get("side") == "blue"
                  and norm(x.get("role")) == lane and id(x) not in used), None)
        r = next((x for x in ps if x.get("side") == "red"
                  and norm(x.get("role")) == lane and id(x) not in used), None)
        if not (b and r):
            continue
        used.add(id(b)); used.add(id(r))

        def one(x):
            k = f"{x.get('kills', 0)}/{x.get('deaths', 0)}/{x.get('assists', 0)}"
            g = x.get("gold") or 0
            gs = f"{g/1000:.1f}k" if g >= 1000 else str(g)
            dmg = x.get("damage_share")
            d = f" 伤害{dmg*100:.0f}%" if isinstance(dmg, (int, float)) else ""
            return (f"{(x.get('summoner_name') or '?')[:12]}"
                    f"({x.get('champion') or '?'}) "
                    f"lv{x.get('level', '?')} {k} {x.get('cs', 0)}刀 {gs}{d}")

        g = gap.get(lane)
        tag = ""
        if isinstance(g, (int, float)) and abs(g) >= 200:
            who = "蓝" if g > 0 else "红"
            tag = f"   [{who}方领先 {abs(g)/1000:.1f}k]" if abs(g) >= 1000                   else f"   [{who}方领先 {abs(g):.0f}]"
        out.append(f"  {ROLE_CN.get(lane, lane):<4}{one(b)}  vs  {one(r)}{tag}")
    return out if len(out) > 2 else []


def _brief_for(req: DebateRequest):
    """把模型输出组装成给 agent 看的简报。"""
    store, stages = STATE["store"], STATE["stages"]
    draft_dict = None
    if req.draft:
        draft_dict = {}
        for side in ("blue", "red"):
            ds = getattr(req.draft, side)
            draft_dict[side] = {r: (p.player, p.champion)
                                for r in ROLES if (p := getattr(ds, r))}
    row, warns = store.make_row(req.blue_team, req.red_team, req.league,
                                draft=draft_dict, playoffs=int(req.playoffs))
    stg = stages["post_draft" if draft_dict else "pre_draft"]
    _, p = stg.predict(row)
    st = store.staleness(req.league)

    L = [f"比赛: {req.blue_team} (蓝方) vs {req.red_team} (红方)  [{req.league}]",
         f"模型预测: 蓝方胜率 {p:.1%}"]

    ig_block = None
    if req.ingame is not None:
        # xpdiff 为 None = 实时数据源给不了 (帧里没有 XP 字段)。必须换成训练时
        # 就不含经验差的 live 变体 —— 拿完整模型硬吃 None 会让特征行里出现
        # 训练时没见过的缺口。这和 _ingame_core 的选择逻辑保持一致。
        ig = STATE.get("ingame_live") if req.ingame.xpdiff is None              else STATE.get("ingame")
    if req.ingame is not None and ig:
        irow, iw = ig.make_row(
            minute=req.ingame.minute, golddiff=req.ingame.golddiff,
            xpdiff=req.ingame.xpdiff, csdiff=req.ingame.csdiff,
            blue_kills=req.ingame.blue_kills, red_kills=req.ingame.red_kills,
            gold_total=req.ingame.gold_total,
            blue_champs=req.blue_champions, red_champs=req.red_champions,
            pre_row=_pre_row_from(row), league=req.league)
        _, ip = ig.predict(irow)
        warns += iw
        perT = ig.metrics.get("per_T", {}).get(str(irow["T"]), {})
        # 经验差可能是 None —— 直接塞进 {:+.0f} 会抛 TypeError。而且要**明说**
        # 它是缺失的: 写成 "经验差 +0" 等于告诉 agent "双方经验持平", 那是
        # 把"不知道"讲成了实测结论。
        #
        # 拿不到经验值, 但**等级是帧里有的**, 所以退而给等级 —— 它承载的是
        # 同一类信息 (谁发育更快), 只是粒度粗。注意这只影响简报文字, 模型那边
        # 仍然走不含经验差的 live 变体, 两者不能混。
        if req.ingame.xpdiff is not None:
            xp = f"经验差 {req.ingame.xpdiff:+.0f}"
        else:
            xp = "经验差 无数据 (实时帧不提供, 见下方逐路等级)"
        L += ["",
              f"局内状态 ({req.ingame.minute:.0f} 分钟, 用 T={irow['T']} 模型):",
              f"  经济差 {req.ingame.golddiff:+.0f}   {xp}   "
              f"人头 {req.ingame.blue_kills}-{req.ingame.red_kills}",
              f"  局内模型预测: 蓝方 {ip:.1%}   (赛前 {p:.1%}, 变动 {ip-p:+.1%})",
              f"  该时间点历史准确率 {perT.get('accuracy', 0):.1%}"]
        L += _lane_table(req)
        ig_block = {"probability_blue": round(ip, 4), "slice": irow["T"],
                    "pregame": round(p, 4), "shift": round(ip - p, 4)}
        p = ip

    # 训练集构成必须写进来。之前只给准确率, agent 就在信息真空里推断"模型
    # 只训练了 LCK/LEC, 对 LPL 是外推" —— 听着合理, 但是错的。而且三个 agent
    # 的信息缺口相同, 交叉质证反而会一起放大这个错误前提。
    # 凡是它们会拿来当论据的事实, 简报里就该有。
    L += ["",
          "模型元信息:",
          f"  留出测试集准确率 {stg.metrics['accuracy']:.1%} (基线 {stg.metrics['baseline']:.1%})",
          f"  概率已 Platt 校准, ECE {stg.metrics['ece_cal']:.3f}",
          f"  概率输出范围受限于 [{stg.metrics['p_min']:.2f}, {stg.metrics['p_max']:.2f}]",
          f"  训练样本 {stg.metrics.get('n_train', '?')} 局 "
          f"(赛前/BP 段), 覆盖 LPL/LCK/LEC/LCS 四大赛区 2022 年起的比赛"]
    if ig_block is not None:
        igm = STATE["ingame_live" if req.ingame.xpdiff is None else "ingame"].metrics
        L += [f"  局内模型: 训练集含全部联赛 (不限四大赛区), 整体准确率 "
              f"{igm.get('accuracy', 0):.1%}, 本切片 "
              f"{igm.get('per_T', {}).get(str(ig_block['slice']), {}).get('accuracy', 0):.1%}",
              f"  局内概率范围 [{igm.get('p_min', 0):.2f}, {igm.get('p_max', 0):.2f}] "
              f"—— 这是校准的固有上限, 不是模型在低估领先方",
              "  (以上是事实, 不要据此推断训练集不含某赛区)"]
    L += ["", "权重最高的特征证据:"]
    for f, v, i in stg.top_evidence(row, 8):
        cn = EX.explain(f, v, req.blue_team, req.red_team)
        head = f"  · {cn}" if cn else f"  · {f}"
        L.append(f"{head}   [{f} = {v:+.3f}, 重要性 {i:.4f}]")
    if warns:
        L += ["", "数据警告:"] + [f"  · {w}" for w in warns]
    L += ["",
          f"数据新鲜度: {req.league} 截至 {st.get('last_date')} "
          f"({st.get('days')} 天前, {st['level']})"]
    return "\n".join(L), p, warns, ig_block


@app.post("/debate")
def debate(req: DebateRequest, x_debate_key: Optional[str] = Header(None)):
    # 服务挂在公网且无鉴权。/predict 只烧 CPU, 但 /debate 一次要打四次 LLM
    # 加若干次检索 —— 是唯一真花钱的端点, 所以单独护住。
    # 不设 DEBATE_KEY 就不校验, 维持原有行为。
    want = os.environ.get("DEBATE_KEY")
    if want and x_debate_key != want:
        raise HTTPException(401, "需要 X-Debate-Key 请求头")
    store = STATE.get("store")
    if not store:
        raise HTTPException(503, "still loading")
    try:
        brief, p, warns, ig_block = _brief_for(req)
    except ValueError as e:
        raise HTTPException(400, str(e))

    try:
        import debate as DB
        d = DB.Debate(WS.Search())
        log = d.run(brief, req.blue_team, req.red_team, req.league,
                    do_search=req.search)
    except Exception as e:
        raise HTTPException(500, f"辩论失败: {e}")

    return {
        "blue_team": req.blue_team, "red_team": req.red_team,
        "league": req.league,
        "model_probability_blue": round(p, 4),
        "ingame": ig_block,
        "warnings": warns,
        "debate": log,
        "final": {
            "probability_blue": round(p, 4),
            "probability_source": "model (Stage 2 或 Stage 4)",
            "note": "辩论层不修改概率。它输出经过交叉质证后仍然成立的主张。",
        },
        "disclaimer": (
            "概率来自留出测试集校准的模型。辩论层的作用是暴露风险点和外部事实, "
            "不参与定价。external_fact 必须带 URL, 无来源的一律降级。非投注建议。"),
    }


@app.get("/api")
def api_info():
    return {"service": "LoL Match Prediction",
            "stages": ["1_pre_draft", "2_post_draft", "3_agent_review", "4_ingame"],
            "endpoints": ["/health", "/teams", "/predict", "/predict/ingame",
                          "/esports/schedule", "/esports/live",
                          "/esports/match/{id}",
                          "/session/start", "/session/{id}/tick", "/session/{id}",
                          "/debate", "/docs"],
            "ingame_loaded": STATE.get("ingame") is not None,
            "ingame_live_loaded": STATE.get("ingame_live") is not None,
            "search_backend": WS.available()}


# ══════════════════════════════════════════════════════════
#  静态前端
#
#  Mount("/") 匹配所有路径, 所以它必须注册在全部 API 路由之后 ——
#  Starlette 按注册顺序匹配, 挂在前面会把 /predict 一起吃掉。
#  原来的 GET / 已改名为 GET /api。
# ══════════════════════════════════════════════════════════

_STATIC = Path(__file__).parent / "static"
if _STATIC.is_dir():
    app.mount("/", StaticFiles(directory=str(_STATIC), html=True), name="static")
else:
    print(f"  静态目录不存在, 前端未挂载: {_STATIC}")
