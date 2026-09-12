"""
lolesports 赛事数据接入 (非官方端点)
======================================
给服务端用: 拉赛程、拉进行中的比赛、把实时帧换算成 Stage 4 的输入。

为什么放服务端而不是前端直连:
  · 队名映射、gameState 校验、缓存、退避 这些只该有一份
  · 前端直连会把那个公共 key 明文暴露在浏览器里

实测记录 (2026-08-16) —— 每一条都是踩过的坑:

  1. startingTime 必须「对齐 10 秒后再减 60 秒」。实时流有延迟, 请求当前
     时刻必然返回 204 (空 body, 不是报错)。

  2. persisted API 的 state 会滞后。实测到一局 getLive 和 getEventDetails
     都报 inProgress, 而帧里 gameState 已经是 finished, 经济数字冻住不动。
     **权威信号是 frames[-1].gameState**, 信 state 会把冻住的概率当实时读数。

  3. 不带 startingTime 调用返回的是这一局**最初**的十来帧, 不是最新的。
     已结束的比赛也一样: 拿到的是开局帧, 经济 0/0, gameState 还是当时的
     in_game。所以它只能用来定位开局时刻, 绝不能当「当前战况」用。

  4. 赛程里的 startTime 是转播开始, 不是开局。实测差了 67 分钟, 不能用它
     算局内分钟数 —— 局内时间靠 _find_game_start 二分出来。

  5. 没有明显的频率限制 (实测 25 请求 / 2.5 req·s⁻¹ 全通), 但观测到过瞬时
     403。所以要退避重试, 但不必为躲限流设计复杂调度。

  6. 比赛结束之后, 任何更晚的 startingTime 都返回**同一批冻结帧** ——
     不是空。实测 2026-08-23 LCK 那局: 从终局后一分钟一直问到两小时,
     每次都是同样 10 帧, 末帧时间戳恒为 10:48:42.094, gameState=finished,
     经济定格。所以"窗口空了"不能当作比赛结束的信号 (真结束了反而不空);
     判结束看 gameState, 去重看时间戳。见 gold_timeline。
"""
from __future__ import annotations

import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

PERSISTED = "https://esports-api.lolesports.com/persisted/gw"
FEED = "https://feed.lolesports.com/livestats/v1"

# 公开的共享 key, lolesports.com 前端自己就用这个。不是机密。
API_KEY = "0TvQnueqKa5mxJntVWt0w4LpLfEkrV1Ta8rQBb9Z"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36")

LEAGUE_IDS = {
    "LCK": "98767991310872058",
    "LPL": "98767991314006698",
    "LEC": "98767991302996019",
    "LCS": "98767991299243165",
}

# ── 队名映射 ─────────────────────────────────────────────────────────
# 由 API 实际返回的队名和 /teams 比对生成 (40 个里 18 个需要映射)。
#
# 全部显式列出, 不做模糊匹配。理由是实测教训: 用 difflib 兜底时
# "Beijing JDG Esports" 被匹配成了 "LNG Esports" —— 看着合理, 是另一支队。
# 队名错了不会报错, 只会拿错队伍的历史算特征。宁可返回 None 让上层报错。
TEAM_ALIASES = {
    # LCK
    "Dplus KIA": "Dplus Kia",
    "Gen.G Esports": "Gen.G",
    "KIWOOM DRX": "Kiwoom DRX",
    "NONGSHIM RED FORCE": "Nongshim RedForce",
    "kt Rolster": "KT Rolster",
    # LPL
    "BILIBILI GAMING": "Bilibili Gaming",
    "Beijing JDG Esports": "JD Gaming",          # 缩写, 规则推不出来
    "EDWARD GAMING": "EDward Gaming",
    "LGD GAMING": "LGD Gaming",
    "Shenzhen NINJAS IN PYJAMAS": "Ninjas in Pyjamas",
    "Suzhou LNG Esports": "LNG Esports",
    "THUNDER TALK GAMING": "ThunderTalk Gaming",
    "TOP ESPORTS": "Top Esports",
    "WeiboGaming": "Weibo Gaming",
    "Xi'an Team WE": "Team WE",
    # LEC
    "GIANTX": "GiantX",
    # LCS
    "Cloud9 Kia": "Cloud9",
    "Team Liquid Alienware": "Team Liquid",
}

_norm = lambda s: re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _https(url: Optional[str]) -> Optional[str]:
    """队标 URL 升到 https。

    上游返回的是 http://static.lolesports.com/…。页面跑在 HTTPS 上时,
    浏览器会按「混合内容」把这些图片全部拦掉 —— 而且不报错, 只是不显示。
    实测该主机 http/https 都返回同一张图, 所以直接升级。
    """
    if url and url.startswith("http://"):
        return "https://" + url[len("http://"):]
    return url


def map_team(api_name: str, known: Optional[list[str]] = None) -> Optional[str]:
    """API 队名 -> 模型队名。映射不出来返回 None, 不猜。"""
    if not api_name or api_name == "TBD":
        return None
    if api_name in TEAM_ALIASES:
        return TEAM_ALIASES[api_name]
    if known:
        if api_name in known:
            return api_name
        by_norm = {_norm(k): k for k in known}
        hit = by_norm.get(_norm(api_name))
        if hit:
            return hit
        return None
    return api_name


# ── 数据结构 ─────────────────────────────────────────────────────────
@dataclass
class TeamRef:
    api_name: str
    code: str
    model_name: Optional[str] = None
    game_wins: Optional[int] = None
    image: Optional[str] = None          # static.lolesports.com 上的队标

    @property
    def usable(self) -> bool:
        return self.model_name is not None


@dataclass
class Match:
    match_id: str
    league: str
    start_time: str          # 转播开始, 不是开局
    state: str               # unstarted | inProgress | completed
    best_of: Optional[int]
    teams: list[TeamRef] = field(default_factory=list)
    between_games: bool = False       # 系列赛进行中, 但此刻没有一局有帧

    @property
    def predictable(self) -> bool:
        return len(self.teams) == 2 and all(t.usable for t in self.teams)


# Data Dragon —— 英雄头像和装备图标。
# 用 Riot 官方源。参考项目 (Aureom/live-lol-esports) 用的
# ddragon.bangingheads.net 镜像实测已经挂了 (HTTP 522), 别跟着用。
DDRAGON = "https://ddragon.leagueoflegends.com"
DDRAGON_FALLBACK_VER = "16.16.1"     # 版本接口拿不到时的兜底, 图标路径不会因此 404


@dataclass
class Player:
    participant_id: int
    side: str                 # blue | red
    role: Optional[str]
    summoner_name: Optional[str]
    champion: Optional[str]
    level: int = 0
    kills: int = 0
    deaths: int = 0
    assists: int = 0
    cs: int = 0
    gold: int = 0
    kill_participation: Optional[float] = None
    damage_share: Optional[float] = None
    wards_placed: Optional[int] = None
    wards_destroyed: Optional[int] = None
    # 血量只有 window 端点有 (details 的 participant 里没有这两个字段),
    # 所以 players() 要额外取一次同时刻的 window 再按 participantId 并起来。
    current_health: Optional[int] = None
    max_health: Optional[int] = None
    # 六格普通装备。**已经把饰品摘出去了** —— 帧给的是混装数组, 见 trinket_ids()
    items: list[int] = field(default_factory=list)
    trinket: Optional[int] = None
    stats: dict = field(default_factory=dict)
    # 符文。details 端点的 perkMetadata: {styleId, subStyleId, perks: [id, ...]}
    # 这里存**解析过**的形状 (名字 + 图标路径), 解析放服务端是为了让前端
    # 不必知道 runesReforged 的结构 —— UI 不放领域逻辑。
    perks: dict = field(default_factory=dict)
    # 技能加点顺序, 形如 ["Q","E","Q","W","Q","R",...]。整局只增不改。
    abilities: list[str] = field(default_factory=list)


@dataclass
class LiveState:
    """一帧换算成 Stage 4 的输入。"""
    game_id: str
    game_state: str          # in_game | finished | paused
    minute: Optional[int]
    golddiff: int
    csdiff: int
    blue_kills: int
    red_kills: int
    gold_total: int
    frame_time: str
    live: bool               # gameState == in_game 且这一帧还新鲜
    # 团队级战况。帧里本来就有, 之前只取了金币和人头。
    teams: dict = field(default_factory=dict)
    stale_seconds: int = 0   # 这一帧比"现在"旧多少秒 (暂停/断流时会变大)
    # 这一局到 frame_time 为止累计暂停了多久。minute 已经扣掉了它;
    # 单独留一份是为了让上层能说清"48 分钟的墙钟里有 7 分钟是暂停"。
    paused_seconds: float = 0.0
    # gameState 还写着 in_game, 但帧已经不推进了 —— 帧流中断或这一局其实
    # 已经打完却从没推送过 finished 帧。见 window() 的说明。
    stalled: bool = False

    def to_state(self) -> dict:
        """喂给 /predict/ingame 的 state。

        xpdiff 传 None 而不是 0 —— 帧里没有 XP 字段, 0 是编造的。服务端见到
        None 会改用不含经验差的 live 变体模型 (代价: T=20 约 1pp, 见
        research/gate_ingame_features.py 配置 F)。传 0 的话模型会吃到一个
        训练时没见过的常量, 而且 evidence() 会输出「双方经验持平」——
        把「不知道」讲成了「实测相等」。
        """
        return {
            "minute": self.minute,
            "golddiff": self.golddiff,
            "xpdiff": None,
            "csdiff": self.csdiff,
            "blue_kills": self.blue_kills,
            "red_kills": self.red_kills,
            "gold_total": self.gold_total,
        }


class FeedError(RuntimeError):
    pass


# ── 客户端 ───────────────────────────────────────────────────────────
class EsportsFeed:
    # window_ttl 决定实时读数最快能有多新。真正的上限是 10 秒 —— startingTime
    # 必须对齐到 10 秒边界, URL 每 10 秒才换一次。比 10 秒更密地重取同一个 URL
    # 仍有意义 (窗口后半段返回的帧更全), 但收益递减。
    # 3 秒: 每个 URL 每 3 秒最多取一次 = 0.33 req/s, 远在实测通过的 2.5 req/s 之内。
    # 升档用的细阶梯。window() 的 fallback_lags 是**入口**档 (给没见过的局用,
    # 保守地从 60 起步), 而这个是**落空之后往回退**用的 —— 低端必须密, 否则
    # 从 30 落空会一步跳回 60, 形成锯齿。见 window() 里的实测轨迹。
    LAG_STEPS = (20, 30, 40, 50, 60, 90, 120, 150, 210, 300)

    # min_lag / lag_probe_sec: 见 window() 里"自适应 lag"那一段。
    # 20 秒是**试探的下界**, 不是默认值 —— 默认仍从 fallback_lags[0] 起步,
    # 只有真的取到帧才会降下来。低于 20 秒没试过, 留着下界免得空转。
    def __init__(self, schedule_ttl=300, live_ttl=20, window_ttl=3,
                 timeout=15, retries=3, no_frames_ttl=30,
                 min_lag=20, lag_probe_sec=20):
        self.schedule_ttl = schedule_ttl
        self.live_ttl = live_ttl
        self.window_ttl = window_ttl
        self.min_lag = min_lag
        self.lag_probe_sec = lag_probe_sec
        self._lag: dict[str, int] = {}          # gameId -> 当前采用的 lag
        self._lag_probe: dict[str, float] = {}  # gameId -> 下次可试探更小 lag 的时刻
        # "这一局压根没有帧" 记多久 —— 见 window() 里的说明
        self.no_frames_ttl = no_frames_ttl
        self.timeout = timeout
        self.retries = retries
        self._cache: dict[str, tuple[float, object]] = {}
        self._starts: dict[str, datetime] = {}      # gameId -> 开局时刻
        self._start_retry: dict[str, float] = {}    # gameId -> 校准失败后何时再试 (见 _game_start)
        # gameId -> {第几分钟 -> (窗口时刻, 首帧时刻, 末帧时刻, 是否冻结)}
        # 缓存**逐窗观测**而不是算完的暂停区间: 观测一旦定型就不会再变, 于是
        # 每次只需补扫新增的那几分钟, 区间由观测现算 (纯 CPU, 几十项而已)。
        # 旧实现缓存的是区间, 键里含"到第几分钟", 比赛每走一分钟键就换一个,
        # 于是整局从头重扫 —— 直播中每次轮询都要付一遍。
        self._obs: dict[str, dict[int, tuple]] = {}
        self._meta: dict[str, dict] = {}            # gameId -> 十个人的名字/英雄
        self._no_frames: dict[str, float] = {}      # gameId -> 这之前别再探了
        self._lock = threading.Lock()
        self._sess = requests.Session()

    # -- 底层 --
    def _get(self, url: str, ttl: float, key_hdr: bool = True,
             empty_ttl: Optional[float] = None):
        """empty_ttl: 上游没给数据 (204 / 4xx) 时改用这个更短的 ttl。

        为什么要分开: 空结果和真数据的"有效期"根本不是一回事。真数据里的
        历史帧永远不变, 缓存一小时是对的; 而空**不代表这个时刻永远没有数据**
        —— 实测同一时刻 lag=60 返回 404 而 lag=150 返回 200 (见 window() 的
        注释), 这个 404/204 的交替看不出规律。把一次瞬时 404 按一小时缓存,
        等于把偶发抖动焊死成一小时的确定性缺口, 期间每次轮询都重现同一个洞。
        """
        now = time.time()
        with self._lock:
            hit = self._cache.get(url)
            if hit and hit[0] > now:
                return hit[1]

        headers = {"User-Agent": UA, "Accept": "application/json"}
        if key_hdr:
            headers["x-api-key"] = API_KEY

        last = None
        for attempt in range(self.retries):
            try:
                r = self._sess.get(url, headers=headers, timeout=self.timeout)
                if r.status_code == 204:            # 该时刻没有数据, 不是错误
                    payload = None
                elif r.status_code == 200:
                    payload = r.json()
                elif 400 <= r.status_code < 500 and r.status_code not in (403, 429):
                    # 客户端错误重试没有意义 —— 比如请求了这一局根本不存在的
                    # 时刻会返回 400。当作"没有数据", 别退避三次浪费十几秒。
                    payload = None
                else:
                    # 观测到过瞬时 403, 退避后重试
                    last = f"HTTP {r.status_code}"
                    time.sleep(1.5 * (attempt + 1))
                    continue
                keep = ttl if payload is not None else (
                    ttl if empty_ttl is None else empty_ttl)
                with self._lock:
                    self._cache[url] = (now + keep, payload)
                return payload
            except requests.RequestException as e:
                last = str(e)
                time.sleep(1.5 * (attempt + 1))
        raise FeedError(f"{url} 取数失败: {last}")

    @staticmethod
    def _lagged(lag_seconds: int = 60, at: Optional[datetime] = None) -> str:
        """对齐 10 秒后再减 lag。见模块顶部第 1 条。"""
        t = at or datetime.now(timezone.utc)
        t = t.replace(second=(t.second // 10) * 10, microsecond=0)
        return (t - timedelta(seconds=lag_seconds)).strftime("%Y-%m-%dT%H:%M:%SZ")

    # -- 赛程 / 直播 --
    def _matches(self, payload, known: Optional[list[str]], league: str) -> list[Match]:
        out = []
        for e in (payload or {}).get("data", {}).get("schedule", {}).get("events", []):
            m = e.get("match") or {}
            if not m.get("id"):
                continue
            lg = (e.get("league") or {}).get("name") or league
            teams = []
            for t in m.get("teams", []):
                nm = t.get("name")
                teams.append(TeamRef(
                    api_name=nm,
                    code=t.get("code", ""),
                    model_name=map_team(nm, known),
                    game_wins=(t.get("result") or {}).get("gameWins"),
                    image=_https(t.get("image")),
                ))
            out.append(Match(
                match_id=m["id"], league=lg,
                start_time=e.get("startTime", ""),
                state=e.get("state", ""),
                best_of=(m.get("strategy") or {}).get("count"),
                teams=teams,
            ))
        return out

    def schedule(self, league: str, known: Optional[list[str]] = None,
                 league_id: Optional[str] = None) -> list[Match]:
        """league_id 显式给出时不查 LEAGUE_IDS —— 回填要扫二十多个次级联赛,
        而 LEAGUE_IDS 只有四大赛区 (它同时决定直播检测的范围, 不该为回填加宽)。"""
        lid = league_id or LEAGUE_IDS.get(league.upper())
        if not lid:
            raise FeedError(f"未知赛区 {league}")
        p = self._get(f"{PERSISTED}/getSchedule?hl=en-US&leagueId={lid}",
                      self.schedule_ttl)
        return self._matches(p, known, league.upper())

    def live(self, known: Optional[list[str]] = None) -> list[Match]:
        p = self._get(f"{PERSISTED}/getLive?hl=en-US", self.live_ttl)
        return [m for m in self._matches(p, known, "")
                if m.league.upper() in LEAGUE_IDS]

    def match_teams(self, match_id: str) -> list[dict]:
        """这场比赛的队伍, 带 id。games[].teams 里只有 id 和选边, 要靠它换名字。

        和 games() 打同一个 URL, 所以走缓存, 不多花请求。
        """
        p = self._get(f"{PERSISTED}/getEventDetails?hl=en-US&id={match_id}",
                      self.live_ttl)
        ev = (p or {}).get("data", {}).get("event") or {}
        return (ev.get("match") or {}).get("teams", []) or []

    def games(self, match_id: str) -> list[dict]:
        p = self._get(f"{PERSISTED}/getEventDetails?hl=en-US&id={match_id}",
                      self.live_ttl)
        ev = (p or {}).get("data", {}).get("event") or {}
        return (ev.get("match") or {}).get("games", [])

    def match_wins(self, match_id: str) -> dict[str, int]:
        """系列赛比分 {队名归一化 -> 赢下的局数}。

        为什么不直接用 schedule 里的 game_wins: 两处**是同一个字段**, 但
        getSchedule 走 schedule_ttl(300 秒) 而这里走 live_ttl(20 秒)。一局刚打完
        时, 前者会把旧比分焊死五分钟 —— 实测表现就是 BO3 已经 2-1 结束了,
        列表里还挂着 1-1。上游对这个字段本身没有分歧 (已定型的比赛两个端点
        完全一致), 差的纯粹是我们自己的缓存。

        逐局胜者 API 里没有 —— games[] 只有 number/id/state/side, 谁赢了只能
        从这个系列赛总分看出来 (collect_live.reconcile 的注释讲了同一件事)。

        和 games() / match_teams() 打同一个 URL, 所以已经调过其中之一时
        **不多花任何一个请求**。
        """
        p = self._get(f"{PERSISTED}/getEventDetails?hl=en-US&id={match_id}",
                      self.live_ttl)
        ev = (p or {}).get("data", {}).get("event") or {}
        out = {}
        for t in (ev.get("match") or {}).get("teams", []) or []:
            w = (t.get("result") or {}).get("gameWins")
            if w is None:
                continue
            # 名字和缩写都记一份: 调用方手里是 api_name, 但两边偶尔只有一个对得上
            for key in (t.get("name"), t.get("code")):
                if key:
                    out[_norm(key)] = int(w)
        return out

    # -- 实时帧 --
    def _window(self, game_id: str, starting: Optional[str], ttl: float,
                empty_ttl: Optional[float] = None):
        url = f"{FEED}/window/{game_id}"
        if starting:
            url += f"?startingTime={starting}"
        return self._get(url, ttl, key_hdr=False, empty_ttl=empty_ttl)

    @staticmethod
    def _ts(frame) -> datetime:
        return datetime.strptime(frame["rfc460Timestamp"][:19],
                                 "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)

    @staticmethod
    def _objectives(t: dict) -> dict:
        """一方的目标资源。

        **这是帧独有的信息**: Oracle's Elixir 的 60 个分时刻列 (at10/15/20/25)
        全是经济/经验/补刀/KDA, 塔龙大龙只有整局总数 —— 想知道"第 20 分钟时
        谁拿了大龙、谁攒了几条龙", 除了帧没有别的来源。
        而这些字段就在同一批帧里, 收它们不额外花任何一个请求。

        dragons 可能是 list (每条龙的属性, 能数出龙魂) 也可能是计数, 两种都见过。
        """
        d = t.get("dragons")
        types = [x for x in d if isinstance(x, str)] if isinstance(d, list) else []
        return {
            "towers": t.get("towers") or 0,
            "inhibitors": t.get("inhibitors") or 0,
            "barons": t.get("barons") or 0,
            "dragons": len(d) if isinstance(d, list) else (d or 0),
            "dragon_types": types,
        }

    def _game_start(self, game_id: str) -> Optional[datetime]:
        """开局时刻。赛程的 startTime 是转播时间, 实测差 67 分钟, 不能用。

        不带 startingTime 调用 window, 返回的是这一局**最初**的十来帧
        (实测: 已结束的比赛也一样, 拿到的是 20:08 开局帧、经济 0/0、
        gameState 还停在当时的 in_game)。所以取 frames[0] 即开局时刻,
        一个请求就够 —— 早先那版在这里做了七次二分, 属于白费。

        **校准失败不能当成结果长缓存。** 服务第一次看到一局通常在开打后一两分钟,
        校准要用的那批窗口 (开局后 0~CALIB_SPAN_SEC 秒) 那时多半还没发布完, 校准
        失败、退回 frames[0]。早先这个退回值和成功的结果一样进了 _starts, 整局就
        一直按 frames[0] 算: 分钟数偏 8~20 秒, collect_live 采到的快照分钟标签跟着
        偏 (2026-09-12 差分测试查出: 在跑的服务说 GX vs NAVI 第 3 局第 36 分钟,
        全新进程说第 35 分钟)。
        现在失败时先用 frames[0] 顶着但不缓存, 每 START_RETRY_SEC 秒再试一次;
        那批窗口全部定型之后还失败, 才认定这一局确实校不了, 缓存 frames[0]。
        """
        if game_id in self._starts:
            return self._starts[game_id]
        p = self._window(game_id, None, ttl=3600)     # 开局时刻不会变, 长缓存
        frames = (p or {}).get("frames") or []
        if not frames:
            return None
        raw = self._ts(frames[0])
        now = time.time()
        if now < self._start_retry.get(game_id, 0):
            return raw                     # 刚试过没成, 先用 frames[0] 顶着
        start = self._calibrate_start(game_id, raw)
        if start is None:
            if now - raw.timestamp() <= self.CALIB_SPAN_SEC + self.OBS_SETTLE_SEC:
                self._start_retry[game_id] = now + self.START_RETRY_SEC
                return raw
            start = raw                    # 窗口都定型了还算不出来: 这一局确实校不了
        self._starts[game_id] = start
        if self._start_retry.pop(game_id, None) is not None:
            # 此前一直按 frames[0] 算, 暂停观测是按旧零点分的桶, 作废重扫
            with self._lock:
                self._obs.pop(game_id, None)
        return start

    # 队伍累计金币: 开局 5×500 = 2500, 之后第一次超过 2500 的时刻记作
    # FIRST_INCOME_SEC。
    #
    # 这个常数**不是**按游戏机制推的, 是拿 Oracle's Elixir 的精确局长当真值
    # 标出来的 —— 早先按"小兵出生 1:05"填了 65, 结果分钟数系统性偏大 32 秒
    # (n=20, 中位 +30, SD 15; 方法: 精确定位 in_game -> finished 的转折帧,
    #  和 OE 的 gamelength 比)。改成 33 后偏差归零, 残余 SD 约 15 秒 ——
    # 那是 10 秒窗口粒度加真实抖动, 压不下去了。
    #
    # 校准本身仍然只用金币, 因为直播时拿不到局长; 局长只用来定这个常数。
    START_GOLD = 2500
    FIRST_INCOME_SEC = 33
    # 校准扫开局后 0~CALIB_SPAN_SEC 秒的窗口。这批窗口全部定型 (落后现在超过
    # OBS_SETTLE_SEC) 之前, 校准失败只说明"还没发布", 不说明"校不了"。
    CALIB_SPAN_SEC = 180
    START_RETRY_SEC = 30       # 还没定型时, 失败后隔多久再试

    def _calibrate_start(self, game_id: str,
                         raw: datetime) -> Optional[datetime]:
        """把零点校到真正的游戏内 0:00。

        frames[0] 不是开局时刻, 而是**数据流开始推送**的时刻, 实测落在游戏内
        0:18~0:21 —— 直接拿它当零点, 报出来的分钟数会一直偏小二十秒, 切片
        选择在边界上就会选错一档。

        校准靠金币: 队伍累计金币开局恒为 2500, 直到小兵出生 (1:05) 才涨。
        找出第一个超过 2500 的帧, 它就是 1:05, 往回退 65 秒就是零点。
        窗口本身只能按 10 秒对齐请求, 但**一个窗口里有三四十帧、间隔 0.2 秒**,
        所以命中的时刻是亚秒级精度, 不受 10 秒粒度限制。

        算成功一次就进 _starts 长缓存, 并行发出。算不出来就返回 None, 由调用方
        决定是稍后重试还是退回 frames[0] (见 _game_start) —— 宁可沿用旧口径,
        不要瞎调。
        """
        times = [raw + timedelta(seconds=s) for s in range(0, self.CALIB_SPAN_SEC, 10)]
        now = datetime.now(timezone.utc)

        def grab(t):
            # 还没定型的窗口只短缓存: 看得太早时它是空的或帧不全, 按一小时缓存的话,
            # _game_start 之后每次重试拿回的都是同一批空结果, 重试形同虚设
            settled = (now - t).total_seconds() > self.OBS_SETTLE_SEC
            return (self._window(game_id, self._lagged(0, t),
                                 ttl=3600 if settled else self.window_ttl,
                                 empty_ttl=self.window_ttl) or {}).get("frames") or []

        try:
            with ThreadPoolExecutor(max_workers=min(8, len(times))) as ex:
                batches = list(ex.map(grab, times))
        except Exception:
            return None

        seen, first = set(), None
        for fr in sorted((f for b in batches for f in b),
                         key=lambda f: f.get("rfc460Timestamp") or ""):
            ts = fr.get("rfc460Timestamp")
            if not ts or ts in seen:
                continue
            seen.add(ts)
            bg = (fr.get("blueTeam") or {}).get("totalGold", 0)
            rg = (fr.get("redTeam") or {}).get("totalGold", 0)
            if not bg and not rg:
                continue                       # 空帧
            if bg > self.START_GOLD or rg > self.START_GOLD:
                first = self._ts(fr)
                break
        if first is None:
            return None

        t0 = first - timedelta(seconds=self.FIRST_INCOME_SEC)
        # 合理性闸: 零点应当略早于 frames[0] (实测早 18~21 秒)。差太多说明
        # 这一局有反常情况 (比如一血发生在小兵出生前), 宁可不调。
        if not (raw - timedelta(seconds=120) <= t0 <= raw + timedelta(seconds=10)):
            return None
        return t0

    # ── 暂停 ─────────────────────────────────────────────────────────
    # 帧的 rfc460Timestamp 是**真实世界时刻**, 不是游戏内时钟。暂停期间上游
    # 干脆不推新帧, 窗口返回的是冻结的最后一帧 —— 于是"最后一帧 - 开局"这个
    # 墙钟差会把暂停的时间一起算进去。
    #
    # 实测 2026-08-23 LEC GX vs TH 第 1 局: 第 3 分钟暂停到第 8 分钟
    # (帧冻在 15:10:25, 到 15:16:19 才恢复), 第 38 分钟又停了 105 秒。
    # 墙钟 48.2 分钟, 游戏内 41:05, 差的 7.6 分钟全是暂停。
    #
    # 这不只是显示难看: minute 要拿去选 Stage 4 的切片 (T=10/15/20/25),
    # 报大 5 分钟就可能拿 T=15 的局面去套 T=20 的模型 —— 又一个不报错的
    # 训练-服务偏移。
    #
    # 判据是"窗口的末帧早于窗口起点": 正常窗口的帧必然落在 [tt, tt+10] 内,
    # 冻结窗口给的是过去某一刻的帧。帧里的 gameState=paused 也能看出来,
    # 但冻结判据同时覆盖了"断流"这种 gameState 来不及改的情况。
    FREEZE_SLACK = 5           # 秒, 容忍窗口对齐和时钟抖动
    # 冻结段到底是"暂停"还是"断流", 靠**跨过这一段游戏内数据有没有前进**分辨。
    # 真暂停: 比赛没在走, 经济一分不涨 (实测冻结帧之间只有 ±450 的读数抖动)。
    # 断流:   比赛照打, 恢复的那一刻数字跳一大截 —— 实测 2026-09-05 LPL
    #         NIP vs JDG 第 2 局断了 16.3 分钟, 恢复时双方总经济从 25113
    #         跳到 85636 (+60523, 约 3700/分钟), 补刀 +1214。
    # 阈值 = 基数 + 每分钟增量: 基数高过抖动, 斜率远低于真实进账速度。
    PLAY_GOLD_BASE = 1000
    PLAY_GOLD_PER_MIN = 800
    LIVE_STALE_SEC = 600       # 帧超过这么久没动就不算"实时", 见 window()
    # 逐窗观测多久之后算定型 (见 pause_spans)。feed 的发布延迟实测约 70 秒
    # (research/probe_lag.py), 取 180 秒留 2.5 倍余量。
    OBS_SETTLE_SEC = 180

    def pause_seconds(self, game_id: str, upto: datetime,
                      step_sec: int = 60) -> float:
        """开局到 upto 之间累计暂停了多少秒。见 pause_spans。"""
        return sum(
            max(0.0, (min(end, upto) - begin).total_seconds())
            for begin, end in self.pause_spans(game_id, upto, step_sec)
            if begin < upto)

    def pause_spans(self, game_id: str, upto: datetime,
                    step_sec: int = 60) -> list:
        """暂停区间 [(冻结在哪一刻, 恢复于哪一刻), ...]。

        返回区间而不是一个总数, 是因为曲线上每个点要扣的是**截至那一点**
        的暂停 —— 第 2 分钟的点不该替第 20 分钟的暂停买单。

        步长和 gold_timeline 一致 (60 秒, 同样的 _lagged(0, tt) URL), 所以
        两者共用 _get 的缓存 —— 谁先跑谁付请求, 另一个几乎全部命中。

        精度就是步长: 只能判定"这一分钟游戏有没有在走", 暂停的起止时刻则
        取冻结帧和恢复后首帧的实际时间戳, 所以残余误差在半分钟量级 ——
        和开局零点校准的残差 (约 15 秒) 同一个数量级, 不再细分。
        """
        start = self._game_start(game_id)
        if not start or upto <= start:
            return []

        times, t = [], start
        while t <= upto:
            times.append(t)
            t += timedelta(seconds=step_sec)
        bucket = {tt: int((tt - start).total_seconds()) // step_sec for tt in times}

        now = datetime.now(timezone.utc)
        with self._lock:
            cached = dict(self._obs.get(game_id) or {})

        # 只补扫**没有观测**或**观测还可能变**的窗口。
        #
        # 一个窗口一旦落后现在超过 OBS_SETTLE_SEC 就不会再变: 帧是历史事实,
        # 而 feed 的发布延迟实测约 70 秒 (research/probe_lag.py), 180 秒有
        # 2.5 倍余量。比这更近的窗口帧可能还没到齐, 末帧偏早会被误判成冻结 ——
        # 所以近端每次重取, 远端一次定终身。
        need = [tt for tt in times
                if bucket[tt] not in cached
                or (now - tt).total_seconds() <= self.OBS_SETTLE_SEC]

        def grab(tt):
            settled = (now - tt).total_seconds() > 600
            p = self._window(game_id, self._lagged(0, tt),
                             ttl=3600 if settled else self.window_ttl,
                             empty_ttl=self.window_ttl)
            return (p or {}).get("frames") or []

        if need:
            with ThreadPoolExecutor(max_workers=min(8, len(need))) as ex:
                framesets = list(ex.map(grab, need))
            fresh = {}
            for tt, frames in zip(need, framesets):
                fr = [f for f in frames if f.get("rfc460Timestamp")]
                # 空窗是"不知道", 不能算暂停, 也**不写进缓存** —— 上游的 404
                # 没有规律 (见 _get 的注释), 把抖动当暂停会把分钟数越扣越小,
                # 而缓存下来就再也没机会自愈。
                if not fr:
                    continue
                first, last = self._ts(fr[0]), self._ts(fr[-1])
                # 末帧的双方总经济。用来区分"暂停"和"断流" —— 见类顶部
                # PLAY_GOLD_BASE 的说明。存观测里而不是事后再取, 因为事后
                # 那些窗口早就被长缓存定型了, 再问一遍也是同一份。
                lf = fr[-1]
                gold = ((lf.get("blueTeam") or {}).get("totalGold", 0)
                        + (lf.get("redTeam") or {}).get("totalGold", 0))
                fresh[bucket[tt]] = (tt, first, last,
                                     last < tt - timedelta(seconds=self.FREEZE_SLACK),
                                     gold)
            cached.update(fresh)
            with self._lock:
                self._obs.setdefault(game_id, {}).update(fresh)

        seq = [cached[bucket[tt]] for tt in times if bucket[tt] in cached]

        spans, i = [], 0
        while i < len(seq):
            if not seq[i][3]:
                i += 1
                continue
            j = i
            while j < len(seq) and seq[j][3]:
                j += 1
            # 冻结段: 从冻住的那一帧, 到恢复后第一帧。走到序列末尾还没恢复
            # 说明这不是暂停而是这一局到此为止了 —— 不计。
            if j < len(seq) and seq[j][1] > seq[i][2]:
                begin, end = seq[i][2], seq[j][1]
                # **断流不是暂停。** 两者在帧上长得一样 (都是没有新帧然后恢复),
                # 唯一分得开的是跨过这一段游戏内数据有没有前进: 暂停时比赛没在
                # 走, 经济不动; 断流时比赛照打, 恢复瞬间数字跳一大截。
                # 分不清的代价很实在: 把 16 分钟断流当暂停扣掉, 界面报第 12
                # 分钟而比赛其实打到第 28 分钟 —— 分钟数还要拿去选 Stage 4 的
                # 切片, 于是用一个时间点的局面去套另一个时间点的模型。
                gap_min = (end - begin).total_seconds() / 60
                grew = (seq[j][4] or 0) - (seq[i][4] or 0)
                if grew > self.PLAY_GOLD_BASE + self.PLAY_GOLD_PER_MIN * gap_min:
                    i = j
                    continue
                spans.append((begin, end))
            i = j

        return spans

    def window(self, game_id: str, at: Optional[datetime] = None,
               fallback_lags=(60, 90, 120, 150, 210, 300),
               precise_minute: bool = False) -> Optional[LiveState]:
        """当前帧。取不到就返回 None —— 绝不回退到默认调用。

        回退是危险的: 默认调用给的是开局帧, 会把第 0 分钟的 0/0 当成当前
        战况报出去, 而且 gameState 还是开局时的 in_game。宁可没有读数。
        """
        # 比赛暂停 / 转播中断时, 退 60 秒那个窗口会是空的。往回多退几档再试
        # —— 实测 IG vs WBG 暂停期间界面整个变成"无数据", 其实两分钟前的
        # 战况仍然取得到。拿到的是稍旧的一帧, 总比什么都没有强。
        #
        # 档位在早段要密, 因为**相邻档位的成败并不连续**。实测 LEC NAVI vs
        # MKOI 开局约两分钟时, 同一时刻扫下来:
        #     lag=60 -> HTTP 404    lag=150 -> HTTP 200, 34 帧    lag=300 -> 404
        #     lag=600 -> 204        lag=1200 -> 404               lag=2400 -> 204
        # 能取到的只是中间窄窄一条, 两边都是 404。原来的三档 (60,150,300) 只有
        # 一档落在带子里, 稍微错开一点就三档全空, 界面就成了"等待开局"。
        # 90/120/210 是用来提高命中率的, 不是因为知道带子在哪 —— 这个 404/204
        # 的交替看不出规律, 只能多试几档。正常直播时 lag=60 一发命中, 不多花请求。
        # 没开打的局要花掉整条链 (六个注定 404 的请求), 而失败结果只按
        # window_ttl(3 秒) 缓存 —— 于是每轮轮询都要把 BO 里没打的那几局
        # 重探一遍。current_game() 会遍历每一局, 一场 BO3 打第一局就是每轮
        # 12 个白发的请求, 实测把 board 的热路径从 1.3 秒拖到 11 秒。
        #
        # 所以"这一局整条链都没有帧"这个结论单独记一段时间。数据本身就落后
        # 70 秒, 晚半分钟发现某局开打完全无所谓。at 不为 None 时是在查历史
        # 时刻, 不走这层。
        now = time.time()
        if at is None:
            until = self._no_frames.get(game_id)
            if until and until > now:
                return None

        # 两段式。到 feed.lolesports.com 的往返实测约 1 秒, 串行跑完六档就是
        # 六秒 —— current_game() 还要对每一局都跑一遍, 实测热路径被拖到 14 秒。
        # 正常直播时第一档 (60 秒) 一发命中, 所以先单发它; 只有落空了才把剩下
        # 几档并发打出去, 一次往返拿回全部结果。
        # 于是: 常见情况 1 个请求 / 1 次往返, 最坏情况 6 个请求 / 2 次往返。
        def fetch(lag):
            return (self._window(game_id, self._lagged(lag, at),
                                 self.window_ttl) or {}).get("frames") or []

        # 自适应 lag。60 秒是**当年测出来能稳定命中的值, 不是上游的真实发布
        # 延迟** —— 而窗口实测是以 T 为中心的 (帧覆盖 T-6s..T+3s, 步长 0.2 秒),
        # 所以 lag 有多大, 画面就落后多少, 一秒不少。上游到底提前多久发布,
        # 各赛区/各转播不一定一样, 写死一个数只能取最保守的那个。
        #
        # 于是不猜: 每隔 lag_probe_sec 顺手多问一个"更近 10 秒"的窗口, 取到帧
        # 就把这一局的 lag 降下来, 取不到就维持原样。降是一次一档, 升由下面
        # 那条 fallback 链兜底 (转播中断、暂停时延迟会变大)。
        #
        # 代价: 试探那一轮多**一个**请求, 而且只对已经在推帧的那一局发生
        # (没帧的局被 _no_frames 挡在外面)。收敛后回到每轮一个请求。
        # 历史查询 (at 不为 None) 不参与 —— 那是在问过去的某个确定时刻,
        # 没有"发布延迟"可言, 自适应只会把状态搞脏。
        base = fallback_lags[0]
        probe = None
        if at is None:
            base = self._lag.get(game_id, fallback_lags[0])
            if (base > self.min_lag
                    and now >= self._lag_probe.get(game_id, 0.0)):
                probe = base - 10
                self._lag_probe[game_id] = now + self.lag_probe_sec

        if probe is not None:
            with ThreadPoolExecutor(max_workers=2) as ex:
                got = dict(zip((probe, base), ex.map(fetch, (probe, base))))
            # 更近的那一档只要有帧就采纳 —— 这是整个自适应的目的
            if got[probe]:
                frames, used_lag = got[probe], probe
                self._lag[game_id] = probe
            else:
                frames, used_lag = got[base], base
        else:
            frames, used_lag = fetch(base), base

        # 兜底链: 只试比当前档更远的, 更近的刚才已经试过了。
        #
        # **低端必须密**。fallback_lags 最小的一档是 60, 直接拿它当升档阶梯的话:
        # 自适应降到 30 之后, 只要 30 偶尔落空 (它就在发布边界上, 实测约七八成
        # 命中), 就会一步跳回 60, 再花一分钟一档一档爬回来。实测轨迹是
        #   50 → 40 → 30 → 30 → 30 → 30 → 30 → 60 → 50 → 40 → 30 → …
        # 平均落后被这个锯齿从 ~30 秒拉到 ~45 秒, 等于自适应白做了一半。
        # 补上 20/30/40/50 之后, 30 落空只退到 40, 在 30↔40 之间小幅摆动。
        ladder = sorted(set(self.LAG_STEPS) | set(fallback_lags))
        rest = [l for l in ladder if l > used_lag] if not frames else []
        if rest:
            with ThreadPoolExecutor(max_workers=len(rest)) as ex:
                got = list(ex.map(fetch, rest))
            # 取**最新**的那一档 (lag 最小), 不是随便哪个命中的
            for lag, fr in zip(rest, got):
                if fr:
                    frames, used_lag = fr, lag
                    break
            # 升过档就先别急着往回降: 刚证明了更近的那档此刻给不了数据,
            # 二十秒后再试只会大概率再落空一次, 白发请求还把画面又拖旧一轮。
            if frames and at is None:
                self._lag_probe[game_id] = now + self.lag_probe_sec * 2
        if not frames:
            if at is None:
                self._no_frames[game_id] = now + self.no_frames_ttl
                # 整条链都空: 别把 lag 停在一个已经取不到帧的档上, 退回默认,
                # 否则这一局会一直卡在那个档上重试。
                self._lag.pop(game_id, None)
            return None
        if at is None:
            # 采用的档位可能比原来的远 (转播中断时会), 记下来免得每轮重爬
            self._lag[game_id] = used_lag
        self._no_frames.pop(game_id, None)

        f = frames[-1]
        b, r = f["blueTeam"], f["redTeam"]
        bcs = sum(x.get("creepScore", 0) for x in b.get("participants", []))
        rcs = sum(x.get("creepScore", 0) for x in r.get("participants", []))

        minute = None
        paused = 0.0
        start = self._game_start(game_id)
        if start:
            elapsed = (self._ts(f) - start).total_seconds()
            # 默认不扣暂停: window() 是热路径, current_game() 会对一场 BO 的
            # 每一局都调一次, 而那里只需要判断哪局在打, 不需要准确分钟数。
            # 要拿去显示或喂模型的那一局才值得付这次扫描 (且多半命中缓存)。
            if precise_minute:
                paused = self.pause_seconds(game_id, self._ts(f))
            minute = max(0, int((elapsed - paused) // 60))

        stale = max(0, int((datetime.now(timezone.utc)
                            - self._ts(f)).total_seconds()))
        # gameState 说 in_game, 但这一帧已经很久没动了。
        #
        # 模块顶部第 2 条记的是反向情形 —— 帧已经 finished 而 persisted 还说
        # inProgress, 那时帧是权威。这里是帧自己卡住了: 实测 2026-08-23 LEC
        # GX vs TH 第 1 局, persisted 说 completed, 而帧停在 in_game、
        # 时间戳 44 分钟没变 —— 帧流中断了, 从没推送过 finished。
        # 卡住的帧不是"实时", 信它会把一局早就打完的比赛标成正在直播,
        # 而 current_game() 正是靠 live 挑当前局, 局间休息时会一直挑中它。
        #
        # 阈值给到 10 分钟而不是压到一两分钟: 正常直播 stale 约 70-100 秒
        # (数据源本身就落后 70 秒), 而暂停期间它会一路涨到几分钟 —— 暂停的
        # 比赛仍然在进行, 不该被判成没在打。真正卡死的局 stale 是几十分钟,
        # 两者差着一个数量级, 不需要卡得很紧。
        stalled = (f.get("gameState") == "in_game"
                   and stale > self.LIVE_STALE_SEC)

        return LiveState(
            game_id=game_id,
            game_state=f.get("gameState", ""),
            minute=minute,
            golddiff=b.get("totalGold", 0) - r.get("totalGold", 0),
            csdiff=bcs - rcs,
            blue_kills=b.get("totalKills", 0),
            red_kills=r.get("totalKills", 0),
            gold_total=b.get("totalGold", 0),
            frame_time=f.get("rfc460Timestamp", ""),
            # 开局那几帧经济确实是 0/0, 那是真的在打, 不该被当成「无效数据」
            # —— 分钟数太小自然进不了 Stage 4 的切片。所以只看 gameState,
            # 外加一道"这帧还在动吗"(stalled), 理由见上面。
            live=(f.get("gameState") == "in_game" and not stalled),
            stale_seconds=stale,
            paused_seconds=paused,
            stalled=stalled,
            teams={
                side: {
                    "gold": t.get("totalGold", 0),
                    "kills": t.get("totalKills", 0),
                    "towers": t.get("towers", 0),
                    "inhibitors": t.get("inhibitors", 0),
                    "barons": t.get("barons", 0),
                    "dragons": len(t.get("dragons") or [])
                              if isinstance(t.get("dragons"), list)
                              else (t.get("dragons") or 0),
                    "dragon_types": (t.get("dragons") or [])
                                    if isinstance(t.get("dragons"), list) else [],
                    "cs": sum(x.get("creepScore", 0)
                              for x in t.get("participants", [])),
                }
                for side, t in (("blue", b), ("red", r))
            },
        )

    # -- 选手面板 --
    def ddragon_version(self) -> str:
        """Data Dragon 的最新版本号。每两周变一次, 缓存 6 小时。

        拿不到就退回一个固定版本 —— 图标路径按版本走, 用旧版本顶多是
        新英雄/新装备缺图, 比整个面板没图标好。
        """
        key = "__ddragon_ver__"
        now = time.time()
        with self._lock:
            hit = self._cache.get(key)
            if hit and hit[0] > now:
                return hit[1]
        try:
            r = self._sess.get(f"{DDRAGON}/api/versions.json",
                               headers={"User-Agent": UA}, timeout=self.timeout)
            v = r.json()[0]
        except Exception:
            v = DDRAGON_FALLBACK_VER
        with self._lock:
            self._cache[key] = (now + 6 * 3600, v)
        return v

    def rune_index(self) -> dict:
        """符文 id -> {name, icon, style}。整份表按 ddragon 版本缓存 6 小时。

        帧里给的只有一串数字 (`perkMetadata.perks` = [8010, 9111, ...]),
        光有 id 在界面上没有意义。ddragon 的 runesReforged.json 是**唯一**
        把 id 换成名字和图标的地方。

        放服务端解析而不是丢给前端: UI 不放领域逻辑 (见 CLAUDE.md), 而且
        这份表十个选手共用一次, 在前端要么每人拉一遍要么自己做缓存。

        拿不到就返回空字典 —— 调用方会退回显示原始 id, 比整个面板消失强。
        """
        key = "__runes__"
        now = time.time()
        with self._lock:
            hit = self._cache.get(key)
            if hit and hit[0] > now:
                return hit[1]
        idx: dict[int, dict] = {}
        try:
            ver = self.ddragon_version()
            r = self._sess.get(f"{DDRAGON}/cdn/{ver}/data/zh_CN/runesReforged.json",
                               headers={"User-Agent": UA}, timeout=self.timeout)
            for style in r.json():
                sid, sname = style.get("id"), style.get("name")
                # 系本身也要能查 (perkMetadata 的 styleId/subStyleId 指向它)
                idx[sid] = {"name": sname, "icon": style.get("icon"),
                            "style": sname}
                for slot in style.get("slots") or []:
                    for p in slot.get("runes") or []:
                        idx[p["id"]] = {"name": p.get("name"),
                                        "icon": p.get("icon"), "style": sname}
        except Exception as e:
            print(f"  符文表拉取失败, 退回显示原始 id: {type(e).__name__}: {e}")
            idx = {}
        with self._lock:
            self._cache[key] = (now + 6 * 3600, idx)
        return idx

    # 小符文碎片。**不在 runesReforged.json 里** —— 那份表只有六个大符文,
    # 于是 5008/5011 这些解析出来是 None。id 段固定在 5000-5099, 一共十条,
    # 名字取自 Community Dragon 的 perks.json (2026-09-05 实拉核对)。
    # 写死而不是再加一个网络依赖: 十条、id 稳定, 不值得为它多一个会挂的上游。
    STAT_SHARDS = {
        5001: "成长生命值", 5002: "护甲", 5003: "魔法抗性",
        5005: "攻击速度", 5007: "技能急速", 5008: "适应之力",
        5010: "移动速度", 5011: "生命值", 5012: "抗性成长",
        5013: "韧性和减速抗性",
    }

    def trinket_ids(self) -> set:
        """饰品的 id 集合, 取自 ddragon 的 item.json (tags 含 "Trinket")。

        为什么需要: 帧的 items 是**一个混装数组** —— 饰品、真眼、合剂和正经
        装备混在一起, 而且**位置不固定** (实测同一局里有人饰品在下标 0、
        有人在 1)。界面上饰品该有自己的格子, 就必须先认出它是哪一件。

        为什么不写死 3340/3363/3364: 那是当前版本的三件, 改版加一件就漏。
        按标签查是跟着版本走的。拿不到就返回空集 —— 调用方退回"不拆分",
        顶多是饰品混在普通格子里, 不会开天窗。
        """
        key = "__trinkets__"
        now = time.time()
        with self._lock:
            hit = self._cache.get(key)
            if hit and hit[0] > now:
                return hit[1]
        out = set()
        try:
            ver = self.ddragon_version()
            r = self._sess.get(f"{DDRAGON}/cdn/{ver}/data/en_US/item.json",
                               headers={"User-Agent": UA}, timeout=self.timeout)
            for iid, d in (r.json().get("data") or {}).items():
                if "Trinket" in (d.get("tags") or []):
                    out.add(int(iid))
        except Exception as e:
            print(f"  装备表拉取失败, 饰品不单独拆分: {type(e).__name__}: {e}")
        with self._lock:
            self._cache[key] = (now + 6 * 3600, out)
        return out

    def _perks(self, meta: Optional[dict]) -> dict:
        """把 perkMetadata 解析成能直接渲染的形状。"""
        if not meta:
            return {}
        idx = self.rune_index()
        ver = self.ddragon_version()

        def one(pid):
            info = idx.get(pid)
            if not info:
                # 大符文表里没有 = 多半是小符文碎片, 它没有独立图标
                return {"id": pid, "name": self.STAT_SHARDS.get(pid),
                        "icon": None, "shard": pid in self.STAT_SHARDS}
            # runesReforged 里的 icon 是相对路径 (perk-images/...), 而且
            # **不带版本号** —— 拼 URL 时不能套 /cdn/{ver}/
            return {"id": pid, "name": info["name"],
                    "icon": f"{DDRAGON}/cdn/img/{info['icon']}" if info.get("icon") else None}

        return {
            "style": one(meta.get("styleId")),
            "sub_style": one(meta.get("subStyleId")),
            "perks": [one(p) for p in (meta.get("perks") or [])],
            "ddragon_version": ver,
        }

    def game_metadata(self, game_id: str) -> dict:
        """{participantId: {summonerName, championId, role, side}}。

        这份信息整局不变, 拿到就长缓存。

        **不能只依赖不带 startingTime 的那个调用**: 开局头一两分钟它会返回空
        (实测 LPL 一局刚开打时 frames=[]), 于是选手名和英雄全成了 "?"。而
        gameMetadata 其实挂在**每一个** window 响应上 (已验证: 默认调用和
        中段某时刻的调用返回的是同一份), 所以默认调用空了就退回到当前窗口。
        """
        if game_id in self._meta:
            return self._meta[game_id]
        md = {}
        for at in (None, self._lagged(60), self._lagged(150)):
            p = self._window(game_id, at, ttl=3600 if at is None else self.window_ttl)
            md = (p or {}).get("gameMetadata") or {}
            if md:
                break
        out = {}
        for side, key in (("blue", "blueTeamMetadata"), ("red", "redTeamMetadata")):
            for m in (md.get(key) or {}).get("participantMetadata", []):
                out[m.get("participantId")] = {
                    "summoner_name": m.get("summonerName"),
                    "champion": m.get("championId"),
                    "role": m.get("role"),
                    "side": side,
                }
        # 只缓存完整的十个人 —— 缺人的半成品缓存下来会一整局都是残的
        if len(out) == 10:
            self._meta[game_id] = out
        return out

    def frame_sides(self, game_id: str) -> dict:
        """本局蓝红方的 esportsTeamId, 取自帧自己的 gameMetadata。

        为什么不用 getEventDetails 里的 games[].teams[].side: 两个源会打架。
        实测 2026-08-20 BLG vs LGD 第 2 局, persisted 说蓝方是 BLG, 而帧的
        blueTeamMetadata 说是 LGD (36 场里抽查 13 场系列赛, 12 场两源一致,
        1 场不一致 —— 约 8% 会对不上)。

        **这里必须以帧为准**, 理由不是"帧更权威", 而是内部一致性:
        participantId 1-5 的金币、人头、装备、逐路对位、以及喂给模型的
        golddiff, 全都是按 blueTeamMetadata 这个分组算出来的。队名要是跟着
        另一个源排, 页面就会把 A 队的数字写在 B 队名下 —— 而且不报错。
        """
        md = {}
        for at in (None, self._lagged(60), self._lagged(150)):
            p = self._window(game_id, at, ttl=3600 if at is None else self.window_ttl)
            md = (p or {}).get("gameMetadata") or {}
            if md:
                break
        out = {}
        for side, key in (("blue", "blueTeamMetadata"), ("red", "redTeamMetadata")):
            tid = (md.get(key) or {}).get("esportsTeamId")
            if tid:
                out[side] = str(tid)
        return out

    def players(self, game_id: str, at: Optional[datetime] = None) -> list[Player]:
        """十个人的等级/装备/KDA/属性。取不到返回空表。"""
        url = f"{FEED}/details/{game_id}?startingTime={self._lagged(60, at)}"
        p = self._get(url, self.window_ttl, key_hdr=False)
        frames = (p or {}).get("frames") or []
        if not frames:
            return []
        meta = self.game_metadata(game_id)

        # 血量: details 里没有, window 里有 (currentHealth / maxHealth)。
        # 同一个滞后时刻再取一次 window, 按 participantId 并进来。这次请求
        # 通常已经在缓存里 —— window() 刚用同样的参数取过。
        hp: dict[int, dict] = {}
        try:
            w = self._window(game_id, self._lagged(60, at), ttl=self.window_ttl)
            wf = (w or {}).get("frames") or []
            if wf:
                for side_key in ("blueTeam", "redTeam"):
                    for x in (wf[-1].get(side_key) or {}).get("participants", []):
                        hp[x.get("participantId")] = x
        except Exception:
            pass                       # 拿不到就是没血量, 不影响其余字段

        trinkets = self.trinket_ids()
        out = []
        for x in frames[-1].get("participants", []):
            pid = x.get("participantId")
            m = meta.get(pid, {})
            h = hp.get(pid, {})
            # 饰品单独拿出来。剩下的截到六格 —— 数组里还混着控制守卫和合剂,
            # 实测一个人能有八项, 不截的话界面那一排会越来越长。
            raw_items = [i for i in (x.get("items") or []) if i]
            trinket = next((i for i in raw_items if i in trinkets), None)
            plain = [i for i in raw_items if i != trinket][:6]
            out.append(Player(
                participant_id=pid,
                # metadata 缺失时按 participantId 兜底: 1-5 蓝, 6-10 红
                side=m.get("side") or ("blue" if (pid or 0) <= 5 else "red"),
                role=m.get("role"),
                summoner_name=m.get("summoner_name"),
                champion=m.get("champion"),
                level=x.get("level", 0),
                kills=x.get("kills", 0),
                deaths=x.get("deaths", 0),
                assists=x.get("assists", 0),
                cs=x.get("creepScore", 0),
                gold=x.get("totalGoldEarned", 0),
                kill_participation=x.get("killParticipation"),
                damage_share=x.get("championDamageShare"),
                wards_placed=x.get("wardsPlaced"),
                wards_destroyed=x.get("wardsDestroyed"),
                current_health=h.get("currentHealth"),
                max_health=h.get("maxHealth"),
                items=plain,
                trinket=trinket,
                stats={
                    "attack_damage": x.get("attackDamage"),
                    "ability_power": x.get("abilityPower"),
                    "armor": x.get("armor"),
                    "magic_resist": x.get("magicResistance"),
                    "attack_speed": x.get("attackSpeed"),
                    "crit": x.get("criticalChance"),
                    "life_steal": x.get("lifeSteal"),
                    "tenacity": x.get("tenacity"),
                },
                perks=self._perks(x.get("perkMetadata")),
                abilities=[a for a in (x.get("abilities") or []) if a],
            ))
        return out

    # -- 时间线 (画曲线用) --
    def gold_timeline(self, game_id: str, upto: Optional[datetime] = None,
                      step_sec: int = 60) -> list[dict]:
        """整局的队伍级时间线, 用来画经济曲线。

        **一次 window 请求只覆盖约 10 秒** (实测: 28~48 帧, 帧间隔 0.2-0.3 秒,
        跨度 9.3~9.9 秒)。早先按 5 分钟步进是基于"一次覆盖 5 分钟"的错误
        假设, 结果每 5 分钟只采到 10 秒, 中间全靠直线插值 —— 第 12 分钟打的
        团在图上完全看不出来。

        现在按 step_sec 步进, 每个窗口只取最后一帧, 得到每分钟一个点。

        **已经过去的帧不会再变**, 所以按时刻缓存 1 小时; 只有最后一段
        (还在推进的那几分钟) 用短缓存。直播时只有新增的那一两个窗口要真取。
        """
        start = self._game_start(game_id)
        if not start:
            return []
        now = datetime.now(timezone.utc)
        # upto 必须是**这一局的最后一帧时刻**, 不是"现在"。
        # 之前默认用 now, 对一场昨晚打完的比赛就会从开局一路步进到此刻 ——
        # 十小时 / 5 分钟 = 一百多个请求, 而且越过比赛结束后全是 HTTP 400。
        end = min(upto or now, now)
        if end <= start:
            return []
        # 再加一道硬上限: 一局 LoL 不会超过两小时
        end = min(end, start + timedelta(minutes=120))

        # 要拉的所有时刻。之前是串行 while 循环, 一局 35 分钟 = 35 个顺序
        # 请求, 实测 25 秒 —— 第一次打开某一局时前端就卡这么久。
        # 实测并发对比 (35 个未缓存的时刻):
        #     串行 25.2s | 并发4 7.7s | 并发8 6.0s | 并发12 4.8s
        # 全程零 403。取 8: 12 只再快 1.2 秒, 但把速率推到 7.3 req/s,
        # 离已验证的范围更远, 不值得。
        times, t = [], start
        while t <= end + timedelta(seconds=step_sec):
            times.append(t)
            t += timedelta(seconds=step_sec)

        def grab(tt):
            settled = (now - tt).total_seconds() > 600  # 定型的窗口长缓存
            p = self._window(game_id, self._lagged(0, tt),
                             ttl=3600 if settled else self.window_ttl,
                             # 空不进长缓存 —— 否则一次瞬时 404 会把这一分钟
                             # 焊死一小时, 曲线在整整一小时里反复缺同一个点。
                             empty_ttl=self.window_ttl)
            return (p or {}).get("frames") or []

        # 一次全发。请求数和串行版一样, 只是多打了那几个注定为空的窗口
        # (end 已经被最后一帧时刻和 120 分钟双重夹住, 多不出几个)。
        with ThreadPoolExecutor(max_workers=min(8, len(times) or 1)) as ex:
            framesets = list(ex.map(grab, times))

        # **不要在这里提前 break。** 曾经有一版是"取到过数据之后连续三窗空
        # 就停, 后面的都不要", 那是串行版为了避开赛后一片 400 留下来的,
        # 改成并发一次全发之后它不再省任何请求 —— framesets 已经在手上,
        # 提前停只是**把已经取回来的真实数据丢掉**。
        #
        # 代价是实打实的: 比赛中途暂停 / 转播断流会连着几分钟取不到帧
        # (window() 那段注释里记的就是这个), 一旦断够三窗, 后面整局的曲线
        # 全被丢弃, 页面上就是"走势卡在第几分钟不动了"。而空窗又曾按一小时
        # 缓存 (见 grab), 于是这一小时里每次刷新都卡在同一个点, 看着像
        # 曲线坏了, 其实数据一直在。
        #
        # 赛后那一片重复窗口不需要靠 break 挡: 实测已结束的比赛, 之后每个
        # 窗口返回的都是**同一批帧**(末帧时间戳恒定, gameState=finished),
        # 按 ts 去重天然只留下终局那一个点。
        # 曲线的横轴也是局内时间, 同样要扣暂停 —— 而且每个点扣的是**截至
        # 该点**的累计量。这里顺带把 backfill 也管住了:
        # research/backfill_late.py 是拿这里的 minute 当快照的 T 存进训练集的,
        # 不扣的话训练数据里就会有一批"标着 T=25 其实是 T=18"的样本。
        # 窗口 URL 和上面 grab 的完全一致, 所以这次扫描全部命中缓存。
        spans = self.pause_spans(game_id, end)

        def ingame_secs(ts: datetime) -> float:
            paused = sum(max(0.0, (min(e, ts) - b).total_seconds())
                         for b, e in spans if b < ts)
            return (ts - start).total_seconds() - paused

        rows: dict[str, dict] = {}
        for frames in framesets:
            # 一个窗口只取最后一帧 —— 窗口本身只有 10 秒, 全收进来只是
            # 在同一个时间点上堆几十个几乎相同的样本。
            for f in reversed(frames):
                b, r = f.get("blueTeam") or {}, f.get("redTeam") or {}
                bg, rg = b.get("totalGold", 0), r.get("totalGold", 0)
                if not bg and not rg:
                    continue                       # 开局前 / 赛后的空帧
                ts = f.get("rfc460Timestamp")
                if not ts or ts in rows:
                    break
                secs = ingame_secs(self._ts(f))
                bo, ro = self._objectives(b), self._objectives(r)
                rows[ts] = {
                    "t": ts,
                    "minute": round(secs / 60, 2),
                    # 目标资源 —— 见 _objectives。塔/龙/大龙/水晶是金币表达不了
                    # 的信息 (大龙是时限 buff, 龙魂是永久跃迁, 破水晶是持续
                    # 兵线压力), 而 OE 分时刻列里一个都没有。
                    "blue_towers": bo["towers"], "red_towers": ro["towers"],
                    "blue_inhibitors": bo["inhibitors"],
                    "red_inhibitors": ro["inhibitors"],
                    "blue_barons": bo["barons"], "red_barons": ro["barons"],
                    "blue_dragons": bo["dragons"], "red_dragons": ro["dragons"],
                    "blue_dragon_types": bo["dragon_types"],
                    "red_dragon_types": ro["dragon_types"],
                    "golddiff": bg - rg,
                    "blue_gold": bg, "red_gold": rg,
                    "blue_kills": b.get("totalKills", 0),
                    "red_kills": r.get("totalKills", 0),
                    # 补刀差: 帧里本来就有 (每人的 creepScore), 之前没收。
                    # 曲线因此只能传 csdiff=0, 和头条用的真实值对不上。
                    "csdiff": (sum(x.get("creepScore", 0)
                                   for x in b.get("participants", []))
                               - sum(x.get("creepScore", 0)
                                     for x in r.get("participants", []))),
                    "state": f.get("gameState", ""),
                }
                break

        return sorted(rows.values(), key=lambda x: x["t"])

    @staticmethod
    def downsample(rows: list[dict], n: int = 60) -> list[dict]:
        """等间隔抽稀。曲线上 60 个点足够, 全量传给前端是浪费。"""
        if len(rows) <= n:
            return rows
        step = len(rows) / n
        picked = [rows[min(len(rows) - 1, int(i * step))] for i in range(n)]
        if picked[-1] is not rows[-1]:
            picked.append(rows[-1])                # 最新一点必须保留
        return picked

    def probe_games(self, match_id: str) -> list[tuple[dict, Optional[LiveState]]]:
        """把这场 BO 的每一局都探一遍帧, 按局号顺序返回 (局, 帧状态)。

        单独抽出来是因为**有两个调用方需要同一批结果**: current_game() 要
        挑当前局, esports_board 要判断哪些局该出标签页。以前后者用的是
        games[].state, 而那个字段不可信 (见下), 于是"看板有数据、上面
        一个局号都没有"和"比赛打完了、页面说尚未开始"两种情形反复出现。

        窗口本身有缓存, 所以两处各调一次并不会多打上游。
        """
        gs = [g for g in self.games(match_id) if g.get("id")]
        if not gs:
            return []

        # 各局并发探。串行的话一场 BO3 里没开打的那两局要各花一次完整回退,
        # 加起来好几秒; 并发之后总耗时就是最慢的那一局。
        def probe(g):
            try:
                return self.window(g["id"])
            except FeedError:
                return None

        with ThreadPoolExecutor(max_workers=len(gs)) as ex:
            states = list(ex.map(probe, gs))
        return list(zip(gs, states))

    def current_game(self, match_id: str) -> Optional[tuple[dict, LiveState]]:
        """找这场 BO 里真正在打的那一局。

        **完全不看 games[].state。** 实测 2026-08-16 LPL JDG vs LGD:
        游戏已经打到第 6 分钟 (帧里 gameState=in_game, 经济 10688/9760),
        而 getEventDetails 说三局都是 unstarted、getLive 返回 0 场。
        之前这里先按 state 筛过一遍, 于是正在打的那局被直接跳过。

        帧是唯一权威。从最后一局往前试, 命中即止。
        """
        pairs = self.probe_games(match_id)
        if not pairs:
            return None
        gs = [g for g, _ in pairs]
        states = [st for _, st in pairs]

        # 从最后一局往前找 —— 命中的是当前正在打的那局
        for g, st in zip(reversed(gs), reversed(states)):
            if st and st.live:
                return g, st

        # 没有一局是 live 时**不要直接返回 None**。
        #
        # 实测 2026-09-06 LPL WE vs IG 第 1 局: 上游按真实时间的 15-62%
        # 发布帧 (连续采样 606s → 635s → 654s → 698s), 延迟一路累积到 11
        # 分钟。stale 越过 LIVE_STALE_SEC 之后 stalled=True → live=False,
        # 这里就一无所获; 而 esports_board 的回退用的是 games[].state ——
        # 那场比赛五局全写着 unstarted (正是模块顶部记的那个不可信字段),
        # 于是 chosen=None, 整个看板连同走势图、计分板、预测一起变成空白。
        #
        # 比赛明明在打, 页面却说"尚未开始", 这比显示一份旧数据糟得多。
        # 卡住的帧仍然是我们**唯一**的战况来源, 交出去、由上层标明它有多旧,
        # 远好过什么都不给。只回退到 in_game 的那一局: 已经 finished 的局
        # 不是"当前局", 那种情形走 esports_board 原有的 playable 回退。
        for g, st in zip(reversed(gs), reversed(states)):
            if st and st.game_state == "in_game":
                return g, st
        return None

    def live_by_frames(self, matches: list[Match],
                       within_hours: int = 5) -> list[Match]:
        """从候选比赛里挑出**真的在打**的 —— 靠帧判断, 不靠 state。

        getLive 会漏 (实测 LPL 开打六分钟仍返回 0 场), 所以对开赛时间落在
        最近 within_hours 内的比赛逐个查帧。查一场要 1 次 getEventDetails
        + 最多 N 次 window, 都有缓存; 候选一般只有两三场。
        """
        now = datetime.now(timezone.utc)
        out = []
        for m in matches:
            if not m.start_time:
                continue
            try:
                ts = datetime.strptime(m.start_time[:19], "%Y-%m-%dT%H:%M:%S") \
                             .replace(tzinfo=timezone.utc)
            except Exception:
                continue
            if not (now - timedelta(hours=within_hours) <= ts <= now + timedelta(minutes=10)):
                continue
            try:
                if self.current_game(m.match_id):
                    out.append(m)
            except Exception:
                continue
        return out
