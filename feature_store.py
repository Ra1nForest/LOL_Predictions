"""
Feature Store — 特征仓库
==========================
把训练时的特征构建逻辑抽出来, 支持两种用法:

  1. build_training_matrix()  离线: 从 CSV 构建完整训练矩阵
  2. FeatureStore.make_row()  在线: 给定两队 (+可选 draft), 构造一行特征

关键: 在线路径必须和离线路径产出完全一致的特征, 否则模型输入分布错位。
      本模块是唯一的特征定义处, 训练和推理共用。

赛区内 Elo (闸门 G1, 2026-10-17)
--------------------------------
10 局滚动窗口看不到长期实力; 赛区内 Elo 是线上残差里最强的信号。research/gate_pregame_package.py
预先登记的组合包里, 全部收益几乎都来自它 (B0+{d} 单独 pre t +6.43 / post t +4.42; 上线配方 {a, d}
= 浅树 + Elo 进校准层: pre t +6.55, post t +4.67)。这里是它**唯一**的定义 (elo_games / elo_replay /
elo_rating), build_training_matrix 和 make_row 都走这一份:

  · 每个联赛一个池 (特征库只装四大, LTA N → LCS 已在读 CSV 时改名), 初始 1500, K = 20, 胜负 0/1,
    不带比分差; 每跨过一个日历年 (UTC) 交界, 池内所有队向池内均值回归 50% (跨 n 个交界就是 0.5^n)。
  · 训练行: 一局的 Elo 只用**更早 UTC 日期**的比赛 —— 同一天的结果不进 (对齐 OE 晚一天登记)。
  · 在线: 库里每一场之后的评分; 请求的年份晚于这个**池**最后一个比赛日的年份, 就补上那几次跨年回归。
    回归是整池一起做的 (闸门就是这样): 一支队去年没打、池子今年打过, 它已经被今年的交界回归过一次,
    不能按"这支队最后一场的年份"再回归一遍。
  · 存成 elo_gap (蓝减红)。这个名字不以 diff_/b_/r_/sum_/is_ 开头, select_features 不会把它喂给树
    —— Elo 只进校准层 (train.py 的 logit_elo; 闸门里 Elo 当树特征只拿到一半收益)。
"""
from __future__ import annotations
import math
import re
import pandas as pd, numpy as np
from pathlib import Path
from dataclasses import dataclass, field

import fdlibm

LEAGUES     = ["LPL", "LCK", "LEC", "LCS"]
ROLL_WINDOW = 10
# 滚动统计至少要几场才出数。**离线和在线必须用同一个值** —— 离线是
# rolling(..., min_periods=) 的参数, 在线是 team_snapshot 里的下限。
# 两边不一致不会报错, 只会让新队伍在线上拿到训练分布之外的极端值
# (只打过 1 场时 rolling_wr 就是 0.0 或 1.0)。
MIN_ROLL_PERIODS = 3
ROLES       = ["top", "jng", "mid", "bot", "sup"]

ROLL_STATS = ["kills","deaths","assists","totalgold","damagetochampions",
              "wardsplaced","wardskilled","dragons","barons","towers",
              "cspm","golddiffat10","golddiffat15",
              "firstblood","firstdragon","firsttower","ckpm","team kpm"]

SUM_FEATS = ["avg_kills","avg_deaths","avg_gamelen","avg_game_kills",
             "avg_ckpm","avg_team_kpm","avg_towers","avg_dragons"]

# 已知污染列 — 结构性剔除
EXCLUDED = {"blue_firstpick", "atakhans", "turretplates"}

DRAFT_PREFIXES = ("champ_wr", "champ_games", "comfort")


def _is_draft_feature(name: str) -> bool:
    return any(k in name for k in DRAFT_PREFIXES)


# ── 英雄名 ──────────────────────────────────────────────────
# lolesports 的帧给的是 Data Dragon 的英雄 id (LeeSin, KSante, MissFortune, MonkeyKing),
# Oracle's Elixir 用的是显示名 (Lee Sin, K'Sante, Miss Fortune, Wukong)。训练只见过后者,
# 看板拿前者直接查表, 名字带空格/撇号的英雄全都查不到 —— collected/ 里 2540 个选手槽位
# 有 337 个 (13%) 这样丢掉 (2026-09-12), BP 后的英雄胜率和局内的阵容强势期都受影响, 不报错。
# 统一成"小写、只留字母数字"再比; id 和显示名差得更远的三个单独列出。五年 OE 的 168 个
# 英雄归一化后没有撞名。这不是模糊匹配: 对不上就是对不上, 查表照旧查不到。
CHAMP_ID_ALIAS = {"monkeyking": "wukong", "renata": "renataglasc", "nunu": "nunuwillump"}


def champion_key(name) -> str:
    k = re.sub(r"[^a-z0-9]", "", str(name or "").lower())
    return CHAMP_ID_ALIAS.get(k, k)


# ── 赛区内 Elo (见模块文档; 常数是闸门冻结的, 改了要重新过闸) ────────────
ELO_K, ELO_INIT, ELO_REG = 20.0, 1500.0, 0.5
# 训练矩阵 / make_row 里的列名。不许带 diff_/b_/r_/sum_/is_ 前缀: 那样 select_features 会把它喂给树
ELO_COL = "elo_gap"
# 校准层的两个输入: logit(clip(树的原始概率)) 和 elo_gap · ln(10)/400 (Elo 的对数几率单位)。
# 两个常数都写进 calib 文件, api.Stage 和浏览器版从文件读, 不各自重算 (Math.log(10) 和 math.log(10)
# 不保证同一个比特)
LOGIT_CLIP = 1e-7
ELO_SCALE = math.log(10) / 400


def elo_games(teams: pd.DataFrame) -> pd.DataFrame:
    """队伍行 → 每局一行 (gameid, league, date, b, r, y, day), 按 (date, gameid) 排好。

    每局恰好两行队伍行的才算, participantid 小的是蓝方 —— 和 build_training_matrix 同一个口径。
    排序和 research/gate_pregame_package.elo_inputs 一字不差: 同一天里按 (时间, gameid) 依次更新,
    池内评分的插入顺序 (决定 np.mean 的加法顺序) 也就一样。
    """
    tt = teams.sort_values(["gameid", "participantid"], kind="stable")
    tt = tt[tt.groupby("gameid")["gameid"].transform("size") == 2]
    blue = tt.groupby("gameid").nth(0).set_index("gameid")
    red = tt.groupby("gameid").nth(1).set_index("gameid").loc[blue.index]
    G = pd.DataFrame({"gameid": blue.index.values, "league": blue["league"].values,
                      "date": blue["date"].values, "b": blue["teamname"].values,
                      "r": red["teamname"].values, "y": blue["result"].astype(int).values})
    G = G.sort_values(["date", "gameid"], kind="stable").reset_index(drop=True)
    G["day"] = G["date"].dt.normalize()
    return G


def elo_replay(G: pd.DataFrame, before=None) -> tuple[dict, dict]:
    """按日重放 elo_games 的表, 返回 (pregame, pools)。

      pregame  gameid → 蓝方 Elo − 红方 Elo, 只用**更早 UTC 日期**的比赛 (训练行用)
      pools    联赛 → {"ratings": {队: 评分}, "year": 最后一个比赛日的年份, "mean": 池内均值,
                       "last": {队: 在这个池里最后一场的日期}}  (在线用, 见 elo_rating)
      before   只走 day < before 的比赛 —— make_row(as_of=...) 用, 和训练行同一个口径

    逐行照抄闸门的 elo_inputs (连 y 是 numpy int64、评分从第一次更新起变成 np.float64 都一样),
    不然 10 ** x 走的是另一条 pow, 末位就可能不同。新队从 1500 进、更新零和, 池内均值理论上恒为
    1500 —— 照字面算均值, 不假设它。
    """
    if before is not None:
        G = G[G["day"] < pd.Timestamp(before).normalize()]
    pre, pools = {}, {}
    for lg, g in G.groupby("league", sort=False):
        R: dict = {}
        last: dict = {}
        cur_year = None
        for day, d in g.groupby("day", sort=True):
            yr = day.year
            if cur_year is not None and yr > cur_year and R:
                mean = float(np.mean(list(R.values())))
                f = (1 - ELO_REG) ** (yr - cur_year)
                for k in R:
                    R[k] = mean + f * (R[k] - mean)
            cur_year = yr
            rows = list(zip(d["gameid"].values, d["b"].values, d["r"].values, d["y"].values))
            for gid, bt, rt, _ in rows:
                pre[gid] = R.get(bt, ELO_INIT) - R.get(rt, ELO_INIT)
            for gid, bt, rt, yy in rows:
                rb, rr = R.get(bt, ELO_INIT), R.get(rt, ELO_INIT)
                e = 1.0 / (1.0 + 10 ** ((rr - rb) / 400.0))
                R[bt], R[rt] = rb + ELO_K * (yy - e), rr - ELO_K * (yy - e)
                last[bt] = last[rt] = day
        pools[lg] = {"ratings": R, "year": cur_year, "last": last,
                     "mean": float(np.mean(list(R.values()))) if R else ELO_INIT}
    return pre, pools


def elo_rating(pool: dict | None, team, year: int) -> float:
    """某队在某池里的评分; 请求年份晚于池子最后一个比赛日的年份, 就补上跨年回归。

    不在池里的队 (新队、别的赛区的队) 是初始 1500, 不回归 —— 闸门里 R.get(队, 1500) 就是这样。
    浏览器版是 web/stage1.ts 的 eloRating, 数据来自 teams.json 的 elo (同一份 ratings / year / mean)。
    """
    if not pool or team not in pool["ratings"]:
        return ELO_INIT
    r = pool["ratings"][team]
    if year > pool["year"]:
        f = (1 - ELO_REG) ** (year - pool["year"])
        r = pool["mean"] + f * (r - pool["mean"])
    return float(r)


def calib_logit(p, clip: float = LOGIT_CLIP):
    """校准层的第一个输入: logit(clip(p))。标量 (线上) 走 fdlibm.log, 数组 (train.py 拟合 / 评估) 走 numpy。

    线上用可移植的 log: 浏览器版 predictStage1 算同一个数, 平台的 log 两边会差一位 (见 fdlibm.py)。
    训练那一路仍是 numpy —— 系数是在它上面拟的, 两种 log 至多差一位, 远在模型的意义之下。
    """
    if np.ndim(p) == 0:
        q = min(max(float(p), clip), 1 - clip)
        return fdlibm.log(q / (1 - q))
    q = np.clip(np.asarray(p, dtype=np.float64), clip, 1 - clip)
    return np.log(q / (1 - q))


def logit_elo_inputs(p_raw, elo_gap, clip: float = LOGIT_CLIP, scale: float = ELO_SCALE) -> np.ndarray:
    """train.py 拟合 / 评估 logit_elo 校准层的设计矩阵: [logit(clip(p_raw)), elo_gap · scale]。"""
    return np.column_stack([calib_logit(p_raw, clip), np.asarray(elo_gap, dtype=np.float64) * scale])


def logit_elo_prob(p_raw: float, elo_gap: float, coef, intercept: float,
                   clip: float, scale: float) -> float:
    """线上 logit_elo 校准: σ(c₁·logit(clip(p_raw)) + c₂·(elo_gap·scale) + b)。

    运算顺序和 sklearn 的 decision_function 一样 ((x₁c₁ + x₂c₂) + b); 浏览器版 web/stage1.ts 的
    predictStage1 照抄这一行。exp / log 用 fdlibm 的移植, 不用 math: 看板上 BP 后 / 赛前的概率不取整就输出,
    平台的 exp / log 让约 4.5% 的输入两边差一位, test:diff 用 === 比 (2026-10-17 实测)。
    """
    z = calib_logit(p_raw, clip) * coef[0] + (elo_gap * scale) * coef[1] + intercept
    return 1.0 / (1.0 + fdlibm.exp(-z))


@dataclass
class FeatureStore:
    """持有全部历史, 支持按 (队伍, 日期) 查询滚动统计。"""
    team_history: dict = field(default_factory=dict)
    champ_records: dict = field(default_factory=dict)
    player_champ: dict = field(default_factory=dict)
    roll_cols: list = field(default_factory=list)
    last_date: pd.Timestamp = None
    league_last: dict = field(default_factory=dict)   # 每个赛区最后一场的日期
    # 赛区内 Elo (见模块文档): 每局一行的表 (make_row(as_of) 截断重放用)、库里全部比赛之后的各池状态、
    # 每局"当天开始前"的 elo_gap (build_training_matrix 用)。直接构造 (测试) 时都是空的: elo_gap = 0
    elo_table: pd.DataFrame = None
    elo_pools: dict = field(default_factory=dict)
    elo_pre: dict = field(default_factory=dict)

    # ── 构建 ────────────────────────────────────────────
    @classmethod
    def from_csv(cls, files: list[str], verbose: bool = True) -> "FeatureStore":
        dfs = []
        for f in files:
            if Path(f).exists():
                if verbose: print(f"    loading {f}")
                dfs.append(pd.read_csv(f, low_memory=False))
        if not dfs:
            raise FileNotFoundError("no CSV found")
        raw = pd.concat(dfs, ignore_index=True)
        raw["league"] = raw["league"].replace({"LTA N": "LCS", "LTA S": "CBLOL"})
        df = raw[raw["league"].isin(LEAGUES)].copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)
        return cls.from_frame(df, verbose=verbose)

    @classmethod
    def from_frame(cls, df: pd.DataFrame, verbose: bool = True) -> "FeatureStore":
        teams = df[df["position"] == "team"].copy()
        teams["result"] = teams["result"].astype(int)
        gl = pd.to_numeric(teams["gamelength"], errors="coerce")
        teams["gamelength_min"] = gl / 60 if gl.max() > 200 else gl
        teams["game_total_kills"] = (pd.to_numeric(teams["teamkills"], errors="coerce")
                                     + pd.to_numeric(teams["teamdeaths"], errors="coerce"))
        players = df[df["position"].isin(ROLES)].copy()
        players["result"] = players["result"].astype(int)

        avail = [c for c in ROLL_STATS if c in teams.columns]
        hist = {}
        for tm in teams["teamname"].unique():
            t = teams[teams["teamname"] == tm].sort_values("date").copy()
            for c in avail:
                t[f"avg_{c.replace(' ', '_')}"] = (
                    pd.to_numeric(t[c], errors="coerce")
                    .rolling(ROLL_WINDOW, min_periods=MIN_ROLL_PERIODS).mean().shift(1))
            t["rolling_wr"] = t["result"].rolling(ROLL_WINDOW, min_periods=MIN_ROLL_PERIODS).mean().shift(1)
            s, st = 0, []
            for r in t["result"]:
                st.append(s)
                s = (s + 1 if s > 0 else 1) if r == 1 else (s - 1 if s < 0 else -1)
            t["streak"] = st
            t["avg_gamelen"] = t["gamelength_min"].rolling(ROLL_WINDOW, min_periods=MIN_ROLL_PERIODS).mean().shift(1)
            t["avg_game_kills"] = t["game_total_kills"].rolling(ROLL_WINDOW, min_periods=MIN_ROLL_PERIODS).mean().shift(1)
            hist[tm] = t

        champ, pchamp = {}, {}
        for r in players.itertuples(index=False):
            champ.setdefault((r.champion, r.league), []).append((r.date, r.result))
            pchamp.setdefault((r.playername, r.champion), []).append((r.date, r.result))
        for k in champ:  champ[k].sort(key=lambda x: x[0])
        for k in pchamp: pchamp[k].sort(key=lambda x: x[0])

        roll_cols = [c for c in list(hist.values())[0].columns
                     if c.startswith("avg_") or c in ("rolling_wr", "streak")]

        ll = teams.groupby("league")["date"].max().to_dict()
        eg = elo_games(teams)
        e_pre, e_pools = elo_replay(eg)
        if verbose:
            print(f"    {len(hist)} teams, {len(champ)} champion-league pairs, "
                  f"{len(pchamp)} player-champion pairs")
            for k in sorted(ll):
                print(f"      {k:<6} 数据截至 {ll[k].date()}")
        return cls(hist, champ, pchamp, roll_cols, teams["date"].max(), ll,
                   eg, e_pools, e_pre)

    # ── 查询 ────────────────────────────────────────────
    def team_snapshot(self, team: str, before: pd.Timestamp | None = None) -> dict | None:
        """取某队在某日期之前的最新滚动统计。"""
        if team not in self.team_history:
            return None
        t = self.team_history[team]
        if before is not None:
            t = t[t["date"] < before]
        if t.empty:
            return None
        # 用最后一行的 shift 值 —— 但最后一行的 avg_* 是「进入该场时」的状态。
        # 要预测未来, 需要把最后一场也纳入, 所以重算一次末尾窗口。
        last = t.iloc[-1]
        snap = {}
        for c in self.roll_cols:
            if c == "streak":
                s = last["streak"]
                snap[c] = (s + 1 if s > 0 else 1) if last["result"] == 1 else (s - 1 if s < 0 else -1)
            elif c == "rolling_wr":
                s = t["result"].tail(ROLL_WINDOW)
                # min_periods=3 要和离线的 rolling(...) 对齐。少了这道下限,
                # 一支只打过 1 场的队伍在线上会拿到 rolling_wr = 0.0 或 1.0,
                # 而模型训练时**从没见过**这种值 —— 离线保证至少 3 场才出数。
                # 不报错, 只是给一个训练分布之外的输入。
                snap[c] = s.mean() if len(s) >= MIN_ROLL_PERIODS else np.nan
            else:
                src = c[4:]                      # avg_xxx -> xxx
                col = {"gamelen": "gamelength_min",
                       "game_kills": "game_total_kills"}.get(src, src.replace("_", " "))
                if col not in t.columns:
                    col = src
                if col in t.columns:
                    v = pd.to_numeric(t[col], errors="coerce").tail(ROLL_WINDOW)
                    snap[c] = (v.mean() if v.notna().sum() >= MIN_ROLL_PERIODS
                               else np.nan)
                else:
                    snap[c] = np.nan
        snap["_n_games"] = len(t)
        snap["_last_date"] = last["date"]
        return snap

    def oe_champion(self, champion: str) -> str:
        """任意写法的英雄名 → 本特征库里的 OE 显示名; 认不出来就原样返回 (查表自然查不到)。

        表本身仍按 OE 名字存 —— 训练传进来的就是 OE 名字, 换回来还是它自己, 结果不变;
        research/ 里也有脚本直接按 OE 名字读这两张表。见 champion_key。
        """
        idx = self.__dict__.get("_champ_idx")
        if idx is None:
            idx = {champion_key(c): c for c, _lg in self.champ_records}
            self._champ_idx = idx
        return idx.get(champion_key(champion), champion)

    def champ_winrate(self, champion: str, league: str,
                      before: pd.Timestamp | None = None, min_g: int = 5) -> float:
        recs = self.champ_records.get((self.oe_champion(champion), league), [])
        p = [r for d, r in recs if before is None or d < before]
        return float(np.mean(p)) if len(p) >= min_g else np.nan

    def player_champ_stats(self, player: str, champion: str,
                           before: pd.Timestamp | None = None) -> tuple[float, int]:
        recs = self.player_champ.get((player, self.oe_champion(champion)), [])
        p = [r for d, r in recs if before is None or d < before]
        wr = float(np.mean(p)) if len(p) >= 3 else np.nan
        return wr, len(p)

    def elo_pools_asof(self, as_of: pd.Timestamp | None = None) -> dict:
        """各池的 Elo 状态。as_of=None: 库里全部比赛之后; 否则只算 as_of 那天之前 (更早的 UTC 日期) 的比赛
        —— 和训练行同一个口径, 所以 make_row(as_of=某局的时间) 拿到的 elo_gap 就是训练矩阵里那一格。
        按天缓存: research/ 的脚本按比赛逐局调 make_row(as_of), 同一天的不必重放第二遍。"""
        if as_of is None or self.elo_table is None:
            return self.elo_pools
        day = pd.Timestamp(as_of).normalize()
        cache = self.__dict__.setdefault("_elo_asof_cache", {})
        if day not in cache:
            if len(cache) > 256:
                cache.clear()
            cache[day] = elo_replay(self.elo_table, before=day)[1]
        return cache[day]

    def elo_gap(self, blue: str, red: str, league: str,
                as_of: pd.Timestamp | None = None, year: int | None = None) -> float:
        """蓝方减红方的赛区内 Elo, 用 league 这个池 (国际赛同母赛区时调用方传母赛区, 见 api.pregame_league)。

        year = 请求的年份, 决定要不要补跨年回归: 默认取 as_of 的年份, 没有 as_of 就是此刻的本地年份 ——
        和 staleness 的"今天"同一个口径, 浏览器版用的也是本地的 today。只在元旦前后几小时和 UTC 不同。
        """
        if year is None:
            year = (pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now()).year
        pool = self.elo_pools_asof(as_of).get(league)
        return elo_rating(pool, blue, year) - elo_rating(pool, red, year)

    def home_league(self, team: str) -> str | None:
        """这支队的母赛区 = 它在特征库里最后一场比赛所在的赛区; 不认识返回 None。

        特征库只装四大赛区 (from_csv 按 LEAGUES 过滤), 所以结果只可能是四大之一 —— 国际赛的
        场次根本不在库里, 不会把"最后一场"带偏。known_teams(league) 用的是同一个口径。
        国际赛靠它判断两队是不是同一个赛区出来的, 见 api.pregame_league。
        """
        t = self.team_history.get(team)
        if t is None or t.empty:
            return None
        return str(t["league"].iloc[-1])

    def known_teams(self, league: str | None = None) -> list[str]:
        """场次够 (>= 3) 的队伍。league=None 是四大赛区的并集 —— 国际赛映射队名时用它
        (按赛事名去查恒为空表, 见 esports_feed.map_team)。"""
        out = []
        for tm, t in self.team_history.items():
            if league and t["league"].iloc[-1] != league:
                continue
            if len(t) >= 3:
                out.append(tm)
        return sorted(out)

    # ── 数据新鲜度 ──────────────────────────────────────
    def staleness(self, league: str) -> dict:
        """
        返回该赛区数据的新鲜度。

        阈值依据: LEC/LCS 这类赛区**只在周末打** —— 上周日打完到本周六是 6 天, OE 还要晚
        一两天登记, 所以正常情况下最多 8 天没有新数据。原来的 3 天在它们身上几乎每周都报,
        报的是赛程本身, 不是数据出了问题 (2026-09-13 用户指出, 当时 LCS 停在 7 天前)。
          <= 8 天   正常 (覆盖一个完整的周末间隔)
          9-13 天   错过了一个比赛周末, 滚动窗口缺最近比赛, 预测仍有参考价值
          >= 14 天  连续两个周末没数据, 可能整段赛程缺失 (含转会/换人), 不可靠
        """
        last = self.league_last.get(league)
        if last is None:
            return dict(league=league, ok=False, days=None,
                        level="unknown", message=f"{league} 无数据")
        days = (pd.Timestamp.now().normalize() - pd.Timestamp(last).normalize()).days
        if days <= 8:
            lvl, msg = "fresh", None
        elif days < 14:
            lvl = "stale"
            msg = (f"{league} 数据截至 {pd.Timestamp(last).date()} ({days} 天前) — "
                   f"滚动窗口缺少最近比赛, 预测偏向旧状态")
        else:
            lvl = "very_stale"
            msg = (f"{league} 数据截至 {pd.Timestamp(last).date()} ({days} 天前) — "
                   f"可能整段赛程缺失, 若期间有转会或换人, 队伍统计描述的是"
                   f"一支已不存在的阵容, 预测不可靠")
        return dict(league=league, ok=(lvl == "fresh"), days=days,
                    level=lvl, last_date=str(pd.Timestamp(last).date()), message=msg)

    # ── 在线构造一行特征 ────────────────────────────────
    def make_row(self, blue: str, red: str, league: str,
                 draft: dict | None = None,
                 playoffs: int = 0,
                 as_of: pd.Timestamp | None = None) -> tuple[dict, list[str]]:
        """
        draft 格式 (可选):
          {"blue": {"top": ("PlayerName","Champion"), ...},
           "red":  {...}}
        返回 (特征字典, 警告列表)
        """
        warn = []
        st = self.staleness(league)
        if st.get("message"):
            warn.append(st["message"])
        bs = self.team_snapshot(blue, as_of)
        rs = self.team_snapshot(red, as_of)
        if bs is None: raise ValueError(f"unknown or insufficient history: {blue}")
        if rs is None: raise ValueError(f"unknown or insufficient history: {red}")
        for nm, s in ((blue, bs), (red, rs)):
            if s["_n_games"] < 5:
                warn.append(f"{nm} 仅有 {s['_n_games']} 场历史, 统计不稳定")
            gap = (pd.Timestamp.now().normalize()
                   - pd.Timestamp(s["_last_date"]).normalize()).days
            if gap >= 14:
                warn.append(f"{nm} 最后一场是 {pd.Timestamp(s['_last_date']).date()} "
                            f"({gap} 天前), 近期状态未知")

        m = {"is_lpl": int(league == "LPL"), "is_lck": int(league == "LCK"),
             "is_lec": int(league == "LEC"), "is_lcs": int(league == "LCS"),
             "playoffs": int(playoffs),
             # 不进树, 只进校准层 (api.Stage 的 logit_elo); 见模块文档
             ELO_COL: self.elo_gap(blue, red, league, as_of)}
        for c in self.roll_cols:
            bv, rv = float(bs.get(c, np.nan)), float(rs.get(c, np.nan))
            m[f"diff_{c}"], m[f"b_{c}"], m[f"r_{c}"] = bv - rv, bv, rv
            if c in SUM_FEATS:
                m[f"sum_{c}"] = bv + rv

        if draft:
            bw, rw, bc, rc = [], [], [], []
            for side, key, accw, accc in (("b", "blue", bw, bc), ("r", "red", rw, rc)):
                picks = draft.get(key, {})
                for role in ROLES:
                    ent = picks.get(role)
                    if not ent:
                        warn.append(f"{key} {role} 缺少 draft 信息")
                        continue
                    player, champion = ent
                    cw = self.champ_winrate(champion, league, as_of)
                    if not np.isnan(cw): accw.append(cw)
                    pw, ng = self.player_champ_stats(player, champion, as_of)
                    accc.append(ng)
                    m[f"{side}_{role}_champ_wr"] = pw
                    m[f"{side}_{role}_champ_games"] = ng
                    if ng == 0:
                        warn.append(f"{player} 无 {champion} 的历史记录")
            m["b_avg_champ_wr"] = float(np.mean(bw)) if bw else np.nan
            m["r_avg_champ_wr"] = float(np.mean(rw)) if rw else np.nan
            if bw and rw:
                m["diff_avg_champ_wr"] = m["b_avg_champ_wr"] - m["r_avg_champ_wr"]
                m["sum_avg_champ_wr"] = m["b_avg_champ_wr"] + m["r_avg_champ_wr"]
            m["b_comfort"], m["r_comfort"] = sum(bc), sum(rc)
            m["diff_comfort"], m["sum_comfort"] = sum(bc) - sum(rc), sum(bc) + sum(rc)
            for role in ROLES:
                m[f"diff_{role}_comfort"] = ((m.get(f"b_{role}_champ_games") or 0)
                                             - (m.get(f"r_{role}_champ_games") or 0))
        return m, warn


# ══════════════════════════════════════════════════════════
#  离线: 构建训练矩阵
# ══════════════════════════════════════════════════════════

def build_training_matrix(files: list[str], verbose: bool = True) -> pd.DataFrame:
    try:
        from data_freshness import check
        import pandas as _pd
        _d = _pd.read_csv(files[-1], usecols=["date"], low_memory=False)
        check(_pd.to_datetime(_d["date"], errors="coerce").max(),
              label="    训练数据")
    except Exception:
        pass
    dfs = []
    for f in files:
        if Path(f).exists():
            if verbose: print(f"    loading {f}")
            dfs.append(pd.read_csv(f, low_memory=False))
    if not dfs:
        raise FileNotFoundError("no CSV found")
    raw = pd.concat(dfs, ignore_index=True)
    raw["league"] = raw["league"].replace({"LTA N": "LCS", "LTA S": "CBLOL"})
    df = raw[raw["league"].isin(LEAGUES)].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    store = FeatureStore.from_frame(df, verbose=False)
    teams = df[df["position"] == "team"].copy()
    teams["result"] = teams["result"].astype(int)
    players = df[df["position"].isin(ROLES)]
    pbg = {g: d for g, d in players.groupby("gameid")}

    rows = []
    for gid in teams["gameid"].unique():
        grp = teams[teams["gameid"] == gid]
        if len(grp) != 2: continue
        r = grp.sort_values("participantid")
        bT, rT = r.iloc[0], r.iloc[1]
        bt, rt, lg, gd = bT["teamname"], rT["teamname"], bT["league"], bT["date"]
        if bt not in store.team_history or rt not in store.team_history: continue
        bh = store.team_history[bt]; bh = bh[bh["gameid"] == gid]
        rh = store.team_history[rt]; rh = rh[rh["gameid"] == gid]
        if bh.empty or rh.empty: continue
        bh, rh = bh.iloc[0], rh.iloc[0]

        m = {"gameid": gid, "date": gd, "league": lg,
             "blue_team": bt, "red_team": rt, "y": bT["result"],
             # 当天开始前的 Elo 差 (同一天的结果不进); 不进树, 只给 train.py 的校准层
             ELO_COL: store.elo_pre.get(gid, np.nan),
             "is_lpl": int(lg=="LPL"), "is_lck": int(lg=="LCK"),
             "is_lec": int(lg=="LEC"), "is_lcs": int(lg=="LCS"),
             "playoffs": int(pd.to_numeric(bT.get("playoffs",0), errors="coerce") or 0)}
        for c in store.roll_cols:
            try:
                bv, rv = float(bh.get(c, np.nan)), float(rh.get(c, np.nan))
                m[f"diff_{c}"], m[f"b_{c}"], m[f"r_{c}"] = bv-rv, bv, rv
                if c in SUM_FEATS: m[f"sum_{c}"] = bv + rv
            except (TypeError, ValueError): pass

        gp = pbg.get(gid)
        if gp is not None:
            bp, rp = gp[gp["side"]=="Blue"], gp[gp["side"]=="Red"]
            if len(bp) >= 5 and len(rp) >= 5:
                bw, rw, bc, rc = [], [], [], []
                for side, sub, accw, accc in (("b",bp,bw,bc), ("r",rp,rw,rc)):
                    for x in sub.itertuples(index=False):
                        cw = store.champ_winrate(x.champion, lg, gd)
                        if not np.isnan(cw): accw.append(cw)
                        pw, ng = store.player_champ_stats(x.playername, x.champion, gd)
                        accc.append(ng)
                        m[f"{side}_{x.position}_champ_wr"] = pw
                        m[f"{side}_{x.position}_champ_games"] = ng
                m["b_avg_champ_wr"] = float(np.mean(bw)) if bw else np.nan
                m["r_avg_champ_wr"] = float(np.mean(rw)) if rw else np.nan
                if bw and rw:
                    m["diff_avg_champ_wr"] = m["b_avg_champ_wr"] - m["r_avg_champ_wr"]
                    m["sum_avg_champ_wr"] = m["b_avg_champ_wr"] + m["r_avg_champ_wr"]
                m["b_comfort"], m["r_comfort"] = sum(bc), sum(rc)
                m["diff_comfort"], m["sum_comfort"] = sum(bc)-sum(rc), sum(bc)+sum(rc)
                for role in ROLES:
                    m[f"diff_{role}_comfort"] = ((m.get(f"b_{role}_champ_games") or 0)
                                                 - (m.get(f"r_{role}_champ_games") or 0))
        rows.append(m)

    out = pd.DataFrame(rows)
    core = [c for c in out.columns if c.startswith("diff_avg_")][:5]
    out = out.dropna(subset=core + ["y"]).reset_index(drop=True)
    if verbose: print(f"    {len(out)} games")
    return out


def select_features(mdf: pd.DataFrame, stage: str = "post_draft") -> list[str]:
    """stage: 'pre_draft' (仅队伍统计) | 'post_draft' (含 draft)"""
    feats = [c for c in mdf.columns
             if (c.startswith(("diff_","b_","r_","sum_","is_")) or c == "playoffs")
             and c not in ("blue_team","red_team")
             and c not in EXCLUDED
             and c != ELO_COL           # 名字本来就选不上; 显式再挡一道 —— Elo 只进校准层
             and mdf[c].dtype in ("float64","int64","float32","int32")]
    if stage == "pre_draft":
        feats = [c for c in feats if not _is_draft_feature(c)]
    return feats
