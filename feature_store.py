"""
Feature Store — 特征仓库
==========================
把训练时的特征构建逻辑抽出来, 支持两种用法:

  1. build_training_matrix()  离线: 从 CSV 构建完整训练矩阵
  2. FeatureStore.make_row()  在线: 给定两队 (+可选 draft), 构造一行特征

关键: 在线路径必须和离线路径产出完全一致的特征, 否则模型输入分布错位。
      本模块是唯一的特征定义处, 训练和推理共用。
"""
from __future__ import annotations
import re
import pandas as pd, numpy as np
from pathlib import Path
from dataclasses import dataclass, field

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


@dataclass
class FeatureStore:
    """持有全部历史, 支持按 (队伍, 日期) 查询滚动统计。"""
    team_history: dict = field(default_factory=dict)
    champ_records: dict = field(default_factory=dict)
    player_champ: dict = field(default_factory=dict)
    roll_cols: list = field(default_factory=list)
    last_date: pd.Timestamp = None
    league_last: dict = field(default_factory=dict)   # 每个赛区最后一场的日期

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
        if verbose:
            print(f"    {len(hist)} teams, {len(champ)} champion-league pairs, "
                  f"{len(pchamp)} player-champion pairs")
            for k in sorted(ll):
                print(f"      {k:<6} 数据截至 {ll[k].date()}")
        return cls(hist, champ, pchamp, roll_cols, teams["date"].max(), ll)

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

    def known_teams(self, league: str | None = None) -> list[str]:
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

        阈值依据: LoL 赛区通常每周 3-5 个比赛日, 10 场滚动窗口约 2-3 周。
          <= 3 天   正常
          4-9 天    滚动窗口缺了最近一两个比赛日, 预测仍有参考价值
          >= 10 天  可能整整错过一个赛段的开局 (含转会/换人), 不可靠
        """
        last = self.league_last.get(league)
        if last is None:
            return dict(league=league, ok=False, days=None,
                        level="unknown", message=f"{league} 无数据")
        days = (pd.Timestamp.now().normalize() - pd.Timestamp(last).normalize()).days
        if days <= 3:
            lvl, msg = "fresh", None
        elif days < 10:
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
             "playoffs": int(playoffs)}
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
             and mdf[c].dtype in ("float64","int64","float32","int32")]
    if stage == "pre_draft":
        feats = [c for c in feats if not _is_draft_feature(c)]
    return feats
