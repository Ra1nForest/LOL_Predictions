"""
跨赛区国际赛: 所有候选模型共用的局级数据表
============================================
用户 2026-10-03 问能不能给国际赛单独训一个模型 —— 现在看板上只要是跨赛区的对阵 (或有一队不在
四大赛区, 比如 GAM / RED Canids), 赛前和 BP 后就不给数。research/gate_international.py 已经量过:
线上 Stage 1/2 在 523 局跨赛区国际赛上准确率约 50%、校准斜率约 0, 它看不见赛区之间的差距。

要比较的候选不止一个 (只用赛区强度的 Bradley-Terry、跨赛区 Elo、Stage 1/2 重训 + 赛区项 ……),
它们必须在**同一张表、同一个"跨赛区"定义、同一个系列赛分组**上比, 否则 Brier 的差可能只是
各自数据口径的差。这个模块就是那张表, 本身不拟合任何东西, 也不碰线上模型文件。

冻结的协议 (结果出来之前定下, 不随结果改)
------------------------------------------
· 单位: OE 的国际赛局。蓝方 = 同一局两行 team 里 participantid 较小的那行, 标签 = 蓝方那行的
  result —— 和 feature_store.build_training_matrix、gate_international.event_rows 完全同一个口径
  (verify() 会拿训练矩阵逐局对账)。
· 母赛区 = 这支队在**这局之前** (时间戳严格更早) 最后一场"常规联赛"比赛所在的赛区, 范围是 OE 的
  全部赛区, 不只是四大。两队母赛区不同 = 跨赛区 (目标); 相同 = 同赛区 (多半是国际赛名义下的
  赛区内预选, 单独报, 不是目标); 有一方查不到 = unknown (国家队之类)。
· 时期: DEV = 2025-01-01 之前, HOLDOUT = 之后。候选的设计、特征、超参只在 DEV 上定。
· 赛事键 = 比赛日期的年份 × 赛事代码 ("2025 WLDs")。walk-forward 按赛事拟合: 预测一个赛事时
  只用该赛事第一局之前的数据, 见 walk_forward_events()。

哪些赛事代码算国际赛 —— 以及为什么
----------------------------------
协议给了 WLDs / MSI / FST / EWC, 要求把 OE 里其余"真正跨赛区"的赛事代码找出来, 并排除 DCup
(德玛西亚杯 2022/2023/2025 都是 LPL + LDL 的国内杯赛)。做法: 对 2022-2026 每个代码 × 年份, 看
参赛队来自哪些常规联赛、有多少局是两个不同母赛区的一级联赛队伍对打 (一级联赛 = 当年往
WLDs/MSI/FST/EWC 送过队的赛区)。实测 (数据截至 2026-10-02):

  列入 (跨赛区的一级联赛队伍对打):
    WLDs MSI FST EWC   协议给定。EWC 2026 的 232 局里大半是 4-5 月的赛区内预选 (同赛区, 不是目标)
    ASI  2025          LCK / LPL / LCP 一级队的邀请赛 (DK、NS、BFX、JDG、WBG、NIP、GAM、MVK), 62% 跨赛区
    LTA  2025          LTA 跨分区的季后赛和总决赛: LTA N (= LCS) 对 LTA S (= CBLOL), 31/40 局跨赛区。
                       这两个赛区平时互不交手, 只在这里和国际赛上碰面, 正是要学的那种差距
    KeSPA Cup 2025     LCK 杯赛邀请了 Cloud9 / Team Liquid 和日本、越南国家队: 10 局 LCK 对 LCS
    AC   2026          Americas Cup: Cloud9 / Sentinels (LCS) 对 FURIA / RED Canids (CBLOL), 15/19 局跨赛区

  不列入:
    DCup               协议排除 (国内杯赛)。2026 的全球邀请赛 DCGI 还没进 OE
    AC 2024            同一个代码, 但那届是 Americas Challengers: NACL / CBLOLA / LRN / LRS 的次级队伍
    KeSPA Cup 2026     只有 LCK 队伍, 没有跨赛区的局
    EM EUM EPL AM ASCI WSCI WLRQ   跨赛区, 但参赛的是次级联赛 / 青训队 (欧洲各国联赛、LCKC、LDL、
                       LJL/PCS/VCS 在 2025 合并进 LCP 之后的剩余联赛) —— 看板不覆盖, Stage 1/2 也
                       不认识这些队; 混进来会让跨赛区样本以欧洲次级联赛为主 (EM 一个代码就 1562 局),
                       HOLDOUT 的结论就不再是关于全球赛的了

  这一判断只用了参赛队伍是谁, 没用胜负, 所以用到 2025-26 的参赛名单不构成泄漏。同一条规则写成了
  _lookalikes(): 一个非国际赛的赛事 (年 × 代码) 里 ≥ 5 局且 ≥ 10% 是两个不同母赛区的一级联赛队伍
  对打, 就打警告。只列协议的四个代码时, 它在全部 OE 赛事里恰好标出 ASI 2025 / KeSPA Cup 2025 /
  LTA 2025 / AC 2026 这四个, 别的一个都没有 —— 上面的名单就是这条规则的结果。build 时每次都重扫,
  将来 DCGI 换个新代码进了 OE 也会被它报出来。

母赛区为什么不能停在"最近一场非国际赛"上
------------------------------------------
协议字面是"最近一场**非国际赛**比赛的赛区"。但 OE 里还有一批杯赛和次级跨区赛 (KeSPA Cup 2026、
DCup、CDF、EM ……), 它们既不是国际赛也不是谁的母赛区: 一支 LCK 队如果最近一场是 KeSPA Cup,
字面口径会把它的母赛区记成 "KeSPA Cup", 赛区强度模型就多出一个假赛区。所以母赛区只在**常规
联赛**里找 (NON_HOME 列出的杯赛 / 次级跨区赛和国际赛一起跳过)。build 时会把字面口径也算一遍,
打印两者在国际赛上不一致的局数。实测 23 局, 全是这类假母赛区: Team BDS 在 2023 全球总决赛前
打了法国杯 (CDF), 字面口径把它记成 "CDF" (19 局); 越南国家队记成 "KeSPA" (4 局, 这里是不明)。

队名标错时按阵容认队
--------------------
OE 偶尔把一支队记在另一个队名下: 2026 电竞世界杯里的 "Team Secret" 五个人 (Bie / Dire / Eddie /
Hizto / Pun) 就是 LCP 的 Team Secret Whales, 但 OE 用的是 2024 年就离开 VCS 的老 Team Secret 的
队名和 teamid —— 按队名查, 母赛区是 648 天前的 VCS。所以: 按队名查到的最后一场常规联赛比赛
**早于 365 天**时, 改用阵容认队 —— 这局五个人 (playerid) 各自之前最后一场常规联赛比赛的赛区,
至少三人相同就取它 (home_source = "roster"), 否则记为不明。365 天是看数据定的: 国际赛里其余
所有队的间隔最长 117 天 (2023 全球总决赛的 WBG)。这不是模糊匹配 —— 认的是同一批人, 不是长得像
的队名。按队名查不到任何历史的 (国家队) 不走这条路, 仍记为不明: 那不是标错名的俱乐部。
阵容口径对所有局都算了 (home_roster_*), 只作参照; 它和队名口径在国际赛上另外只有 KeSPA Cup 2025
(LCK 队派二队 / 新人上场) 和 Vivo Keyd Stars 在 EWC 2026 预选派了青训阵容这两处不同, 都保持队名口径。

系列赛分组
----------
OE 没有系列赛 id, 只有每局在系列赛里的序号 (game 列)。gate_international 用"日期 + 队伍对",
跨 UTC 零点的 BO5 (美洲赛区傍晚开打) 会被拆成两个系列赛, 同一天两队打两次 (小组赛加赛) 会被
并成一个。这里按 (赛事代码, 队伍对) 排时间, game 序号不增或相隔超过 12 小时就开新系列赛。
按系列赛聚合的配对 t 是协议的判据 (同一系列赛的几局共享同一对队伍, 不是独立样本), 见 group_t()。
国际赛 1438 局分成 648 个系列赛; "日期 + 队伍对"给 654 个, 拆开了 9 个、并错了 3 个。

按赛事 walk-forward 的两个副作用 (协议的结果, 只会更保守, 不会泄漏)
------------------------------------------------------------------
· LTA 2025 是一个赛事键, 但横跨 2 月到 9 月: 9 月的总决赛也只能用 2 月 15 日之前的数据拟合。
· EWC 2026 从 4 月 14 日的赛区预选算起: 7 月的正赛拿不到 6-7 月 MSI 2026 的结果。

计数 (2026-10-03, 数据截至 2026-10-02)
--------------------------------------
国际赛 1438 局: 跨赛区 979、同赛区 452、不明 7 (KeSPA Cup 2025 的日本 / 越南国家队)。
跨赛区 DEV 529 局 / 302 系列 / 7 赛事 (四大v四大 301, 含非四大 228); HOLDOUT 450 局 / 177 系列 /
11 赛事 (四大v四大 245, 含非四大 205)。BP 全部齐全。跨赛区里母赛区不是四大的队伍 29 支、9 个赛区
(CBLOL / LCP / PCS / VCS / LLA / LCO / LJL / TCL / LFL), 例: GAM Esports (2022-24 VCS, 2025+ LCP)、
RED Canids (CBLOL)。build_training_matrix 的 8844 局四大内战逐局对账: 蓝方、红方、标签 0 处不一致。

    from xregion_data import load_games, summarize, walk_forward_events, group_t
    python research/xregion_data.py            打印各项计数 (首次约 20 秒, 之后读缓存)
    python research/xregion_data.py --verify   额外和 build_training_matrix 逐局对账方向和标签 (约 3.5 分钟)
    缓存目录: --cache-dir, 或环境变量 LOL_XREGION_CACHE, 都没有就用系统临时目录
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "research"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from feature_store import LEAGUES, ROLES          # noqa: E402  四大赛区的名单只在那里写一次
from harness import paired_t                       # noqa: E402  配对 t 全项目只有一份
from multiyear_gate import data_path               # noqa: E402  CSV 的位置只在那里写一次

# 改了表的构造逻辑就加一, 旧缓存自动作废
VERSION = 2

# 和 train.py / api.py 的五年列表一致; _check_years() 在构建时核对, 不一致就报错 ——
# 少加载一年不会报错, 只会让"母赛区"和"之前的历史"悄悄变短。
YEARS = (2022, 2023, 2024, 2025, 2026)
HOLDOUT_START = pd.Timestamp("2025-01-01")

# ── 国际赛 (目标集合) ──────────────────────────────────────────
# 值 = 只在这些年份算国际赛; None = 所有年份。理由见模块文档。
INTL_EVENTS: dict[str, set[int] | None] = {
    "WLDs": None, "MSI": None, "FST": None, "EWC": None,      # 协议给定
    "ASI": None,                                             # 2025 亚洲邀请赛
    "LTA": None,                                             # 2025 LTA 跨分区 (LCS 对 CBLOL)
    "KeSPA Cup": {2025},                                     # 2025 届邀请了 LCS 队伍
    "AC": {2026},                                            # 2026 Americas Cup; 2024 是次级联赛
}
PROTOCOL_EXCLUDED = {"DCup"}

# ── 不是任何队的母赛区 ─────────────────────────────────────────
# 杯赛和次级跨区赛: 参赛队平时在别的联赛打。母赛区查找时和国际赛一起跳过。
# (国际赛代码本身也在跳过之列, 不管那一年算不算国际赛 —— AC 2024 不是国际赛, 但同样不是母赛区。)
NON_HOME = {
    "DCup", "KeSPA", "KeSPA Cup", "CDF", "CT", "USP", "TSC", "GLLPA", "EBLPA", "PRMP",
    "NLC Aurora Open", "HW", "CCWS", "IC",                    # 国内 / 区内杯赛
    "EM", "EUM", "EPL", "AM", "ASCI", "WSCI", "WLRQ",         # 次级联赛的跨区赛
} | set(INTL_EVENTS)

SERIES_GAP = pd.Timedelta(hours=12)
STALE_HOME_DAYS = 365          # 按队名查到的母赛区比这还旧, 改按阵容认队 (见模块文档)
ROSTER_MIN_AGREE = 3           # 五个人里至少几个人来自同一赛区才算数

_COLS = ["gameid", "datacompleteness", "league", "playoffs", "date", "game", "patch",
         "participantid", "side", "position", "playername", "playerid", "teamname", "teamid",
         "champion", "result"]


def is_international(code, year) -> bool:
    if code in PROTOCOL_EXCLUDED or code not in INTL_EVENTS:
        return False
    yrs = INTL_EVENTS[code]
    return yrs is None or int(year) in yrs


def default_cache_dir() -> Path:
    """缓存目录: 环境变量 LOL_XREGION_CACHE, 否则系统临时目录。

    不写死在源码里 —— 仓库是公开的, 本机路径不进源码 (CLAUDE.md 的约定)。
    """
    env = os.environ.get("LOL_XREGION_CACHE")
    return Path(env) if env else Path(tempfile.gettempdir()) / "lol_xregion"


# ══════════════════════════════════════════════════════════
#  构建
# ══════════════════════════════════════════════════════════

def _files() -> list[str]:
    return [data_path(y) for y in YEARS]


def _check_years():
    """五年列表必须和训练用的一致。在函数里 import train: 只有真要构建时才付 xgboost 的导入开销。"""
    import train
    ours = [Path(f).resolve() for f in _files()]
    theirs = [Path(f).resolve() for f in train.DATA]
    if ours != theirs:
        raise RuntimeError(f"xregion_data 的年份列表和 train.DATA 不一致:\n  {ours}\n  {theirs}")


def _fingerprint() -> dict:
    fp = {"version": VERSION, "files": []}
    for f in _files():
        p = Path(f)
        if not p.exists():
            raise FileNotFoundError(f"缺数据文件 {p} —— 少一年不会报错, 只会让母赛区和历史悄悄变短, 所以这里直接停")
        st = p.stat()
        fp["files"].append([p.name, st.st_size, st.st_mtime_ns])
    return fp


def _read_raw(verbose: bool) -> pd.DataFrame:
    dfs = []
    for f in _files():
        if verbose:
            print(f"    读 {Path(f).name}")
        dfs.append(pd.read_csv(f, usecols=_COLS, dtype={"patch": str}, low_memory=False))
    raw = pd.concat(dfs, ignore_index=True)
    # 和 feature_store / gate_international 同一行: 漏了它, 2025 一整年北美 / 巴西的比赛会消失
    raw["league"] = raw["league"].replace({"LTA N": "LCS", "LTA S": "CBLOL"})
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    bad = raw["date"].isna()
    if bad.any():
        if verbose:
            print(f"    ⚠ {int(bad.sum())} 行日期解析失败, 丢掉")
        raw = raw[~bad]
    return raw


def _home_league(games: pd.DataFrame, team_rows: pd.DataFrame, skip) -> dict:
    """每局两队的母赛区: 该队时间戳严格早于这局的最后一场"非 skip"比赛的赛区和日期。

    merge_asof(allow_exact_matches=False) = 严格更早; 同一时刻的比赛 (就是这局本身) 不算。
    """
    reg = team_rows[~skip(team_rows)][["teamname", "date", "league"]]
    reg = reg.rename(columns={"league": "home", "date": "home_date"})
    reg["date"] = reg["home_date"]
    reg = reg.sort_values("date")
    out = {}
    for side in ("blue", "red"):
        left = games[["gameid", "date", side]].rename(columns={side: "teamname"}).sort_values("date")
        m = pd.merge_asof(left, reg, on="date", by="teamname", direction="backward",
                          allow_exact_matches=False)
        out[side] = m.set_index("gameid")[["home", "home_date"]]
    return out


def _status(hb, hr) -> np.ndarray:
    """两方母赛区 → "cross" / "same" / "unknown" (有一方查不到)。"""
    hb = pd.Series(np.asarray(hb, dtype=object), dtype=object)
    hr = pd.Series(np.asarray(hr, dtype=object), dtype=object)
    known = (hb.notna() & hr.notna()).values
    return np.where(~known, "unknown", np.where(hb.values == hr.values, "same", "cross"))


def _roster_home(games: pd.DataFrame, player_rows: pd.DataFrame, skip) -> dict:
    """阵容口径的母赛区: 这一方五个人 (playerid) 各自严格早于这局的最后一场常规联赛比赛的赛区,
    至少 ROSTER_MIN_AGREE 人相同才取, 日期取这几个人里最近的那场。

    返回 {side: DataFrame[home, home_date]}, 行序和 games 一致。
    """
    reg = player_rows[~skip(player_rows)][["playerid", "date", "league"]].dropna(subset=["playerid"])
    reg = reg.rename(columns={"league": "home", "date": "home_date"})
    reg["date"] = reg["home_date"]
    reg = reg.sort_values("date")
    parts = []
    for side in ("blue", "red"):
        for role in ROLES:
            col = f"{side}_{role}_playerid"
            left = (games[["gameid", "date", col]].rename(columns={col: "playerid"})
                    .dropna(subset=["playerid"]).sort_values("date"))
            m = pd.merge_asof(left, reg, on="date", by="playerid", direction="backward",
                              allow_exact_matches=False)
            parts.append(m.assign(side=side)[["gameid", "side", "home", "home_date"]])
    L = pd.concat(parts, ignore_index=True).dropna(subset=["home"])
    c = (L.groupby(["gameid", "side", "home"]).agg(k=("home", "size"), home_date=("home_date", "max"))
         .reset_index())
    c = c[c["k"] >= ROSTER_MIN_AGREE]          # 五个人里三个以上同一赛区, 不会有并列
    out = {}
    for side in ("blue", "red"):
        s = c[c["side"] == side].drop_duplicates("gameid").set_index("gameid")[["home", "home_date"]]
        out[side] = s.reindex(games["gameid"].values)
    return out


def _series_keys(g: pd.DataFrame) -> pd.Series:
    """(赛事代码, 队伍对) 内按时间排, game 序号不增或间隔 > 12 小时就开新系列赛。"""
    d = g[["gameid", "league", "date", "game", "blue", "red"]].copy()
    d["pair"] = np.where(d["blue"].astype(str) < d["red"].astype(str),
                         d["blue"].astype(str) + " | " + d["red"].astype(str),
                         d["red"].astype(str) + " | " + d["blue"].astype(str))
    d = d.sort_values(["league", "pair", "date", "game"]).reset_index(drop=True)
    same = (d["league"] == d["league"].shift()) & (d["pair"] == d["pair"].shift())
    cont = same & (d["game"] > d["game"].shift()) & ((d["date"] - d["date"].shift()) <= SERIES_GAP)
    sid = (~cont).cumsum()
    start = d.groupby(sid)["date"].transform("min")
    key = d["league"].astype(str) + " | " + start.dt.strftime("%Y-%m-%d %H:%M") + " | " + d["pair"]
    return pd.Series(key.values, index=d["gameid"].values)


def build(verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """从五年 CSV 构建局级表 (全部赛区)。返回 (表, 构建时的诊断)。"""
    _check_years()
    raw = _read_raw(verbose)
    diag = {}

    # ── 两行 team → 一局, 方向和 build_training_matrix 一致 ─────────
    t = raw[raw["position"] == "team"].sort_values(["gameid", "participantid"]).copy()
    n_rows = t.groupby("gameid")["gameid"].transform("size")
    diag["dropped_not_two_rows"] = int(t.loc[n_rows != 2, "gameid"].nunique())
    t = t[n_rows == 2]
    t["slot"] = t.groupby("gameid").cumcount()
    b = t[t["slot"] == 0].set_index("gameid")
    r = t[t["slot"] == 1].set_index("gameid").loc[b.index]
    # gate_international 会跳过"participantid 小的那行不是 Blue"的局; 五年里一局都没有, 有了就报出来
    diag["side_mismatch"] = int(((b["side"] != "Blue") | (r["side"] != "Red")).sum())
    diag["result_not_one_winner"] = int(((b["result"] + r["result"]) != 1).sum())

    g = pd.DataFrame({
        "gameid": b.index.values,
        "date": b["date"].values,
        "league": b["league"].values,
        "blue": b["teamname"].values, "red": r["teamname"].values,
        "blue_teamid": b["teamid"].values, "red_teamid": r["teamid"].values,
        # 标签和训练矩阵一样直接取蓝方那行的 result; 三局 OE 没登记胜者 (两边都是 0, 都不是国际赛),
        # 训练矩阵把它们当蓝方负, 这里照样, 另给 result_ok 让候选自己决定要不要剔
        "blue_win": b["result"].astype(int).values,
        "result_ok": ((b["result"] + r["result"]) == 1).values,
        "game": pd.to_numeric(b["game"], errors="coerce").values,
        "playoffs": pd.to_numeric(b["playoffs"], errors="coerce").fillna(0).astype(int).values,
        "patch": b["patch"].astype(str).values,
        "datacompleteness": b["datacompleteness"].values,
    })
    g["year"] = g["date"].dt.year.astype(int)
    g["event"] = g["year"].astype(str) + " " + g["league"].astype(str)
    g["period"] = np.where(g["date"] < HOLDOUT_START, "DEV", "HOLDOUT")
    g["is_international"] = [is_international(c, y) for c, y in zip(g["league"], g["year"])]
    g["event_class"] = np.where(g["is_international"], "intl",
                                np.where(g["league"].isin(NON_HOME), "non_home", "league"))

    # ── BP: 十个人 (OE 选手名 / id) 和英雄 —— 母赛区的阵容口径要用 playerid, 所以先做 ──
    p = raw[raw["position"].isin(ROLES)].copy()
    p["slot"] = p["side"].str.lower() + "_" + p["position"]
    diag["draft_duplicate_slots"] = int(p.duplicated(["gameid", "slot"]).sum())
    p = p.drop_duplicates(["gameid", "slot"])
    for val, suf in (("playername", "player"), ("playerid", "playerid"), ("champion", "champ")):
        w = p.pivot(index="gameid", columns="slot", values=val)
        for side in ("blue", "red"):
            for role in ROLES:
                c = f"{side}_{role}"
                g[f"{c}_{suf}"] = w[c].reindex(g["gameid"]).values if c in w.columns else np.nan
    champ_cols = [f"{s}_{r}_champ" for s in ("blue", "red") for r in ROLES]
    player_cols = [f"{s}_{r}_player" for s in ("blue", "red") for r in ROLES]
    g["draft_complete"] = g[champ_cols].notna().all(axis=1) & g[player_cols].notna().all(axis=1)

    # ── 母赛区 ─────────────────────────────────────────────
    skip_ref = lambda d: d["league"].isin(NON_HOME)                         # noqa: E731
    skip_lit = lambda d: pd.Series([is_international(c, y) for c, y in      # noqa: E731
                                    zip(d["league"], d["date"].dt.year)], index=d.index)
    intl = g["is_international"].values
    home = _home_league(g, t, skip_ref)
    roster = _roster_home(g, p, skip_ref)
    stale_rows = np.zeros(len(g), bool)
    for side in ("blue", "red"):
        h = home[side].loc[g["gameid"]]
        rh = roster[side]
        name_home = pd.Series(h["home"].values, dtype=object)
        name_date = pd.Series(h["home_date"].values)
        gap = (g["date"] - name_date).dt.total_seconds() / 86400
        stale = (gap > STALE_HOME_DAYS).fillna(False).values
        stale_rows |= stale
        g[f"home_name_{side}"] = name_home.values
        g[f"home_roster_{side}"] = pd.Series(rh["home"].values, dtype=object).values
        final = name_home.where(~stale, pd.Series(rh["home"].values, dtype=object))
        fdate = name_date.where(~stale, pd.Series(rh["home_date"].values))
        g[f"home_{side}"] = final.values
        g[f"home_source_{side}"] = np.where(final.isna(), None, np.where(stale, "roster", "team"))
        g[f"home_gap_days_{side}"] = ((g["date"] - fdate).dt.total_seconds() / 86400).values
        g[f"home_major_{side}"] = g[f"home_{side}"].isin(LEAGUES).values
    g["region_status"] = _status(g["home_blue"], g["home_red"])
    known = g["region_status"] != "unknown"
    g["cross_region"] = g["region_status"] == "cross"
    g["pair_type"] = np.where(~known, "unknown",
                              np.where(g["home_major_blue"] & g["home_major_red"],
                                       "major_v_major", "involves_nonmajor"))
    diag["intl_stale_name_home_games"] = int((stale_rows & intl).sum())
    diag["intl_stale_name_home_examples"] = (
        g.loc[stale_rows & intl, ["event", "blue", "red", "home_name_blue", "home_name_red",
                                  "home_blue", "home_red"]].head(10).to_dict("records"))
    rdiff = np.zeros(len(g), bool)
    for side in ("blue", "red"):
        rdiff |= (g[f"home_roster_{side}"].astype(object).fillna("-").values
                  != g[f"home_{side}"].astype(object).fillna("-").values)
    diag["intl_roster_vs_final_home_diff_games"] = int((rdiff & intl).sum())
    diag["intl_roster_vs_final_home_diff_by_event"] = {
        k: int(v) for k, v in g.loc[rdiff & intl].groupby("event").size().items()}

    # 协议字面口径 (只跳过国际赛) 在国际赛上和这里差几局 —— 打印出来, 不静默
    lit = _home_league(g, t, skip_lit)
    lit_b = lit["blue"].loc[g["gameid"], "home"].astype(object).values
    lit_r = lit["red"].loc[g["gameid"], "home"].astype(object).values
    diff = np.zeros(len(g), bool)
    for side, lv in (("blue", lit_b), ("red", lit_r)):
        a = g[f"home_name_{side}"].astype(object).fillna("-").values
        diff |= a != pd.Series(lv, dtype=object).fillna("-").values
    diag["home_literal_vs_ref_diff_intl"] = int((diff & intl).sum())
    diag["home_literal_vs_ref_diff_all"] = int(diff.sum())
    m = diff & intl
    diag["home_literal_vs_ref_examples"] = (
        g.loc[m, ["event", "blue", "red", "home_name_blue", "home_name_red"]]
        .assign(literal_blue=lit_b[m], literal_red=lit_r[m])
        .drop_duplicates(["event", "home_name_blue", "home_name_red", "literal_blue", "literal_red"])
        .head(10).to_dict("records"))

    # 用 teamid 查母赛区作对照: 队名在 OE 里偶有改名 / 重名, 两种身份给出的跨赛区判断应一致。
    # 比的是队名口径 (换阵容口径之前), 这里只想知道"队名"和"teamid"这两种身份是否等价。
    t_id = t[t["teamid"].notna()].copy()
    t_id["teamname"] = t_id["teamid"].astype(str)
    gid = g[["gameid", "date"]].copy()
    gid["blue"], gid["red"] = g["blue_teamid"].astype(str), g["red_teamid"].astype(str)
    hid = _home_league(gid, t_id, skip_ref)
    st_id = _status(hid["blue"].loc[g["gameid"], "home"], hid["red"].loc[g["gameid"], "home"])
    st_name = _status(g["home_name_blue"], g["home_name_red"])
    has_id = g["blue_teamid"].notna().values & g["red_teamid"].notna().values
    dd = intl & has_id & (st_id != st_name)
    diag["region_status_teamid_vs_name_diff_intl"] = int(dd.sum())
    diag["region_status_teamid_vs_name_examples"] = (
        g.loc[dd, ["event", "blue", "red", "home_name_blue", "home_name_red"]].head(10).to_dict("records"))

    # ── 系列赛 ─────────────────────────────────────────────
    g["series"] = _series_keys(g).loc[g["gameid"]].values
    # 和 gate_international 的"赛事 | 日期 | 队伍对"口径比一下, 差多少就是那边被拆 / 被并的系列赛
    pair = np.where(g["blue"].astype(str) < g["red"].astype(str),
                    g["blue"].astype(str) + "|" + g["red"].astype(str),
                    g["red"].astype(str) + "|" + g["blue"].astype(str))
    dk = g["league"].astype(str) + "|" + g["date"].dt.date.astype(str) + "|" + pair
    sub = pd.DataFrame({"s": g["series"], "d": dk})[intl]
    diag["intl_series"] = int(sub["s"].nunique())
    diag["intl_series_by_datekey"] = int(sub["d"].nunique())
    diag["intl_series_split_by_datekey"] = int((sub.groupby("s")["d"].nunique() > 1).sum())
    diag["intl_datekeys_merging_series"] = int((sub.groupby("d")["s"].nunique() > 1).sum())

    g = g.sort_values(["date", "gameid"]).reset_index(drop=True)
    order = (["gameid", "date", "year", "league", "event", "period", "is_international", "event_class",
              "blue", "red", "blue_win", "result_ok",
              "home_blue", "home_red", "home_major_blue", "home_major_red",
              "home_gap_days_blue", "home_gap_days_red", "home_source_blue", "home_source_red",
              "home_name_blue", "home_name_red", "home_roster_blue", "home_roster_red",
              "region_status", "cross_region", "pair_type",
              "series", "game", "playoffs", "patch", "datacompleteness", "blue_teamid", "red_teamid",
              "draft_complete"]
             + [f"{s}_{r}_{k}" for s in ("blue", "red") for r in ROLES for k in ("player", "playerid", "champ")])
    g = g[order]
    diag["unlisted_lookalikes"] = _lookalikes(g)
    return g, diag


def _lookalikes(g: pd.DataFrame) -> list[dict]:
    """没列入国际赛、却有跨赛区一级联赛对打的赛事 —— 将来 OE 加了新代码 (比如 DCGI) 就会出现在这里。

    一级联赛 (按年) = 当年 WLDs/MSI/FST/EWC 参赛队的母赛区。只看参赛队是谁, 不看胜负。
    扫全部非国际赛的赛事 (包括常规联赛): 新出现的代码不在 NON_HOME 里, 只扫杯赛会漏掉它。
    常规联赛只有升降级队伍的头几场会被算成"跨赛区", 占比远低于阈值。
    """
    core = g[g["league"].isin(["WLDs", "MSI", "FST", "EWC"])]
    fd = {y: set(s[["home_blue", "home_red"]].stack().dropna()) for y, s in core.groupby("year")}
    out = []
    for (ev, y), s in g[~g["is_international"]].groupby(["event", "year"]):
        f = fd.get(y, set())
        m = s["cross_region"] & s["home_blue"].isin(f) & s["home_red"].isin(f)
        if m.sum() >= 5 and m.mean() >= 0.10:
            out.append({"event": ev, "n": len(s), "cross_first_division": int(m.sum())})
    return out


# ══════════════════════════════════════════════════════════
#  对外接口
# ══════════════════════════════════════════════════════════

def load_games(cache_dir=None, refresh: bool = False, verbose: bool = True) -> pd.DataFrame:
    """局级表 (全部赛区, 2022-2026)。CSV 没变就读缓存; CSV 或 VERSION 变了自动重建。

    cache_dir 不给就用 default_cache_dir()。构建时的诊断放在 df.attrs["diag"]。
    """
    cdir = Path(cache_dir) if cache_dir else default_cache_dir()
    cdir.mkdir(parents=True, exist_ok=True)
    pkl, meta = cdir / f"xregion_games_v{VERSION}.pkl", cdir / f"xregion_games_v{VERSION}.json"
    fp = _fingerprint()
    if not refresh and pkl.exists() and meta.exists():
        try:
            m = json.loads(meta.read_text(encoding="utf-8"))
            if m.get("fingerprint") == fp:
                df = pd.read_pickle(pkl)
                df.attrs["diag"] = m.get("diag", {})
                if verbose:
                    print(f"    读缓存 {pkl}")
                return df
        except Exception as e:            # 缓存坏了就重建, 不让它挡路
            if verbose:
                print(f"    缓存不可用 ({e}), 重建")
    df, diag = build(verbose=verbose)
    df.to_pickle(pkl)
    meta.write_text(json.dumps({"fingerprint": fp, "diag": diag}, ensure_ascii=False, indent=1,
                               default=str), encoding="utf-8")
    df.attrs["diag"] = diag
    if verbose:
        print(f"    已缓存到 {pkl}")
    return df


def target(df: pd.DataFrame) -> pd.Series:
    """协议的目标集合: 国际赛里的跨赛区局。"""
    return df["is_international"] & df["cross_region"]


def event_table(df: pd.DataFrame, mask=None) -> pd.DataFrame:
    """每个赛事 (年 × 代码) 的首末日期和局数, 按首局日期排。walk-forward 的切点就是 first。"""
    d = df if mask is None else df[mask]
    e = d.groupby("event").agg(first=("date", "min"), last=("date", "max"), n=("gameid", "size"),
                               n_cross=("cross_region", "sum"), period=("period", "first"))
    return e.sort_values("first")


def walk_forward_events(df: pd.DataFrame, eval_mask=None):
    """按赛事 walk-forward: 逐个产出 (赛事, 切点, 评估行的 index)。

    拟合只能用 df["date"] < 切点 的行 (切点 = 该赛事第一局的时间, 包括它的预选) —— 协议的
    时间纪律。注意 EWC 2026 从 4 月的赛区预选就开始了, 所以它 7 月的正赛也拿不到 6-7 月 MSI 的结果。
    """
    m = target(df) if eval_mask is None else eval_mask
    ev = event_table(df, df["is_international"])
    for name, row in ev.iterrows():
        idx = df.index[m & (df["event"] == name)]
        if len(idx):
            yield name, row["first"], idx


def brier(p, y) -> np.ndarray:
    return (np.asarray(p, float) - np.asarray(y, float)) ** 2


def logloss(p, y, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def group_t(loss_base, loss_cand, groups):
    """先按组 (系列赛 / 赛事) 求平均损失, 再在组之间做配对 t。正 = 候选更好。

    返回 (Δ = 基线损失 − 候选损失 的组均值, t, 组数, Δ > 0 的组数)。和 gate_international.group_t
    同一个算法, 这里单独放一份是为了候选脚本不必 import 那个会加载线上模型的文件。
    """
    d = pd.DataFrame({"a": -np.asarray(loss_base, float), "b": -np.asarray(loss_cand, float),
                      "g": np.asarray(groups)})
    s = d.groupby("g")[["a", "b"]].mean()
    if len(s) < 2:
        return np.nan, np.nan, len(s), int((s["b"] - s["a"] > 0).sum())
    delta, _sd, t, _ = paired_t(s["a"].values, s["b"].values)
    return float(delta), float(t), len(s), int((s["b"] - s["a"] > 0).sum())


# ══════════════════════════════════════════════════════════
#  计数
# ══════════════════════════════════════════════════════════

def summarize(df: pd.DataFrame, verbose: bool = True) -> dict:
    """协议要求报的计数。返回 dict, verbose 时同时打印。"""
    out = {}
    say = print if verbose else (lambda *a, **k: None)
    intl = df[df["is_international"]]
    diag = df.attrs.get("diag", {})

    say(f"\n  全表 {len(df)} 局 ({df['date'].min().date()} … {df['date'].max().date()}), "
        f"国际赛 {len(intl)} 局")
    for k in ("side_mismatch", "result_not_one_winner", "dropped_not_two_rows", "draft_duplicate_slots",
              "home_literal_vs_ref_diff_intl", "home_literal_vs_ref_diff_all",
              "region_status_teamid_vs_name_diff_intl", "intl_stale_name_home_games",
              "intl_roster_vs_final_home_diff_games", "intl_roster_vs_final_home_diff_by_event",
              "intl_series", "intl_series_by_datekey", "intl_series_split_by_datekey",
              "intl_datekeys_merging_series"):
        if k in diag:
            say(f"    {k:<42}{diag[k]}")
    for k in ("home_literal_vs_ref_examples", "region_status_teamid_vs_name_examples",
              "intl_stale_name_home_examples"):
        if diag.get(k):
            say(f"    {k}:")
            for row in diag[k]:
                say(f"      {row}")
    look = diag.get("unlisted_lookalikes", [])
    if look:
        say(f"    ⚠ 没列入但有跨赛区一级联赛对打的赛事: {look}")
    out["diag"] = {k: v for k, v in diag.items() if not k.endswith("examples")}

    # 1. 每个赛事代码 × 年份
    say("\n  [1] 国际赛 按代码 × 年份 (局数: 全部 / 跨赛区 / 同赛区 / 不明)")
    tab = (intl.groupby(["league", "year", "region_status"]).size().unstack(fill_value=0)
           .reindex(columns=["cross", "same", "unknown"], fill_value=0))
    tab.insert(0, "all", tab.sum(axis=1))
    for (lg, y), r in tab.iterrows():
        say(f"    {lg:<10}{y}  {r['all']:>4}  {r['cross']:>4}  {r['same']:>4}  {r['unknown']:>3}")
    say(f"    {'合计':<14}  {tab['all'].sum():>4}  {tab['cross'].sum():>4}  {tab['same'].sum():>4}  "
        f"{tab['unknown'].sum():>3}")
    out["by_event"] = {f"{y} {lg}": {k: int(v) for k, v in r.items()} for (lg, y), r in tab.iterrows()}

    # 2. 跨赛区 DEV / HOLDOUT × 对阵类型
    cr = intl[intl["cross_region"]]
    say("\n  [2] 跨赛区 (目标) 按时期 × 对阵类型 —— 局数 / 系列赛数 / 赛事数")
    out["cross"] = {}
    for per in ("DEV", "HOLDOUT"):
        s = cr[cr["period"] == per]
        row = {}
        for pt in ("major_v_major", "involves_nonmajor"):
            x = s[s["pair_type"] == pt]
            row[pt] = {"games": len(x), "series": int(x["series"].nunique())}
        row["all"] = {"games": len(s), "series": int(s["series"].nunique()), "events": int(s["event"].nunique()),
                      "draft_complete": int(s["draft_complete"].sum())}
        out["cross"][per] = row
        say(f"    {per:<8} 四大v四大 {row['major_v_major']['games']:>4} 局 / {row['major_v_major']['series']:>3} 系列   "
            f"含非四大 {row['involves_nonmajor']['games']:>4} 局 / {row['involves_nonmajor']['series']:>3} 系列   "
            f"合计 {row['all']['games']:>4} 局 / {row['all']['series']:>3} 系列 / {row['all']['events']:>2} 赛事  "
            f"(BP 齐全 {row['all']['draft_complete']})")
        say(f"             蓝方胜率 {s['blue_win'].mean():.3f}")

    # 3. 同赛区 / 不明
    say("\n  [3] 同赛区 (不是目标, 单独报) 和 母赛区不明")
    out["same"], out["unknown"] = {}, {}
    for per in ("DEV", "HOLDOUT"):
        s = intl[(intl["period"] == per) & (intl["region_status"] == "same")]
        u = intl[(intl["period"] == per) & (intl["region_status"] == "unknown")]
        out["same"][per] = {"games": len(s), "series": int(s["series"].nunique()),
                            "by_home": {k: int(v) for k, v in s.groupby("home_blue").size().items()}}
        out["unknown"][per] = {"games": len(u),
                               "teams": sorted(set(u.loc[u["home_blue"].isna(), "blue"])
                                               | set(u.loc[u["home_red"].isna(), "red"]))}
        say(f"    {per:<8} 同赛区 {len(s):>4} 局 / {s['series'].nunique():>3} 系列  "
            + ", ".join(f"{k} {v}" for k, v in out["same"][per]["by_home"].items()))
        say(f"    {per:<8} 不明   {len(u):>4} 局  {out['unknown'][per]['teams']}")

    # 4. 跨赛区国际赛里母赛区不是四大的队伍
    say("\n  [4] 跨赛区国际赛里母赛区是非四大的队伍 (队名 [母赛区]: DEV 局数 / HOLDOUT 局数)")
    rows = []
    for side in ("blue", "red"):
        x = cr[~cr[f"home_major_{side}"]][[side, f"home_{side}", "period"]]
        x.columns = ["team", "home", "period"]
        rows.append(x)
    nm = pd.concat(rows)
    piv = nm.groupby(["home", "team", "period"]).size().unstack(fill_value=0)
    piv = piv.reindex(columns=["DEV", "HOLDOUT"], fill_value=0)
    out["nonmajor_teams"] = []
    for home_lg, grp in piv.groupby(level=0):
        items = [f"{tm} {r['DEV']}/{r['HOLDOUT']}" for (_h, tm), r in grp.iterrows()]
        say(f"    {home_lg:<7}({len(grp):>2} 队) " + ", ".join(items))
        for (_h, tm), r in grp.iterrows():
            out["nonmajor_teams"].append({"team": tm, "home": home_lg,
                                          "dev_games": int(r["DEV"]), "holdout_games": int(r["HOLDOUT"])})
    say(f"    非四大母赛区: {nm['home'].nunique()} 个, 队伍 {nm['team'].nunique()} 支")

    # 5. 跨赛区的赛区对
    say("\n  [5] 跨赛区 按赛区对 (DEV / HOLDOUT 局数)")
    pr = cr.assign(pair=[" v ".join(sorted([a, b])) for a, b in zip(cr["home_blue"], cr["home_red"])])
    pp = pr.groupby(["pair", "period"]).size().unstack(fill_value=0).reindex(columns=["DEV", "HOLDOUT"],
                                                                             fill_value=0)
    pp = pp.assign(tot=pp.sum(axis=1)).sort_values("tot", ascending=False)
    say("    " + ";  ".join(f"{k} {r['DEV']}/{r['HOLDOUT']}" for k, r in pp.iterrows()))
    out["pairs"] = {k: {"DEV": int(r["DEV"]), "HOLDOUT": int(r["HOLDOUT"])} for k, r in pp.iterrows()}
    return out


# ══════════════════════════════════════════════════════════
#  对账
# ══════════════════════════════════════════════════════════

def verify(df: pd.DataFrame) -> dict:
    """方向和标签: 和 build_training_matrix (训练矩阵, 四大赛区内战) 逐局比蓝方、红方、标签。"""
    from feature_store import build_training_matrix
    mdf = build_training_matrix(_files(), verbose=False)
    j = mdf[["gameid", "blue_team", "red_team", "y"]].merge(
        df[["gameid", "blue", "red", "blue_win", "league"]], on="gameid", how="left")
    res = {
        "matrix_games": len(mdf),
        "missing_in_table": int(j["blue"].isna().sum()),
        "blue_mismatch": int((j["blue_team"] != j["blue"]).sum()),
        "red_mismatch": int((j["red_team"] != j["red"]).sum()),
        "label_mismatch": int((j["y"].astype(int) != j["blue_win"]).sum()),
        "league_not_major": int((~j["league"].isin(LEAGUES)).sum()),
    }
    print("\n  [对账] build_training_matrix vs 本表: " + ", ".join(f"{k} {v}" for k, v in res.items()))
    return res


def main():
    ap = argparse.ArgumentParser(description="跨赛区国际赛的共用局级表")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    print("=" * 96)
    print("  跨赛区国际赛: 共用局级表")
    print("=" * 96)
    df = load_games(a.cache_dir, refresh=a.refresh)
    summarize(df)
    if a.verify:
        verify(df)


if __name__ == "__main__":
    main()
