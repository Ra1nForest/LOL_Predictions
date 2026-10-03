"""
跨赛区模型 C (分层 Elo) —— 运行时的唯一定义
==========================================
看板上跨赛区的对阵 (或有一队不在四大赛区, 比如 GAM / RED Canids) 原来不给赛前数: 线上 Stage 1/2 的特征
全是"在自己赛区里"打出来的相对量, 看不见赛区之间的差距 (research/gate_international.py: 523 局历史跨赛区
国际赛上准确率约 50%、校准斜率约 0)。research/gate_cross_region.py 用冻结的 DEV / HOLDOUT 协议闸了一个
专门的模型 C, 用户 2026-10-04 批准上线。这个模块是它在运行时的**唯一定义**:

  · 全赛区局级表的构造和它要的规则 (哪些赛事算国际赛、母赛区 —— 跳过杯赛、按队名过期 365 天改按阵容认队 ——、
    系列赛分组)。研究脚本 research/xregion_data.py 从这里 import, 不再自己写一份。
  · 引擎 Engine: 按时间顺序的评分更新, 同一时间戳的几局先全部取赛前值再统一更新。research/gate_cross_region.py
    的 run_elo 就是本模块的 run_elo —— 闸门量的和看板用的是同一段代码。
  · 系数拟合 (岭 logistic)、o 的衰减、predict()。
  · 看板的赛事内重放 (board_predict 和它用的纯函数: 去重、终局帧判胜负、比分约束解码、排序), 见文件末尾那一段。
  · build_state(): 把五年 OE 走完之后的状态导出成 JSON (artifacts/xregion.json, 日更生成), 看板从它出发。
浏览器版 (frontend/src/web/) 是第二份实现, 要和这里逐位一致 (test:golden)。本模块不 import research/ 下的任何
东西 —— 那是研究代码, 部署时不一定在。

模型 (参数全部来自闸门, 在 DEV 上定、HOLDOUT 只评了一次, 一个数都不许改 —— 改了就得重新过闸)
------------------------------------------------------------------------------------------
  全球实力 = o[母赛区] + r[队], 差 S = 400/ln10 分 = logit 差 1。
  · r: 每个 OE 赛区一个 Elo 池, K=2, 由常规联赛和同池杯赛更新; 跨赛区国际赛另以 kint=24 更新两队的 r。
  · o: 只由跨赛区国际赛更新, eta=24, 零和。第一次出现时取先验: 四大 0, 其余 −600; 迁入还没有偏移的新赛区
    (2025 的 LCP) 取迁入队伍原赛区偏移的均值。o 以半衰期 365 天向先验衰减 (懒计算: 用到时才衰减)。
  · 输出 p = σ(a + b_o·Δo/S + b_r·Δr/S)。(a, b_o, b_r) 只在更早的跨赛区国际赛上拟合, 岭先验朝 (logit c, 1, 1),
    λ = 30, c = 切点前全部 OE 比赛的蓝方胜率。
  HOLDOUT (2025-01 起 450 局 / 177 系列赛): Brier 0.2231, 常数 0.2478, 对常数系列赛 t +2.87 (过 2.5 的线);
  对"只看母赛区身份"的赛区 BT 打平 (t +0.40)。偏自信: 给 0.8-0.9 的局实际只赢约 76%。

为什么必须在赛事内实时重放
--------------------------
闸门里**只有赛事内在线更新的版本能过**: 每一局打完, 评分就更新, 再预测下一局 —— 包括同一赛事、同一系列赛的
前几局。只用 OE 数据、结果隔天才生效的版本 (OE 晚约一天) 系列赛 t 只有 +2.14 ~ +2.48, 没过。所以看板不能只用
这里导出的状态: 必须把同一赛事里 OE 还没收录、lolesports 上已打完的局按 Engine.play() 重放进去; 重放失败
(取不到赛程) 就不显示 C。导出的 events[*].games 列出 OE 里已经有的局, 重放按它去重, 免得一局算两遍。

服务时和闸门的三处差别 (都只会让看板少给数, 不会给出另一个数)
-----------------------------------------------------------
  · 母赛区按队名查到的最后一场常规联赛早于 365 天: 闸门改按阵容认队, 这里直接返回 None (不给数)。
    看板拿不到 OE 的 playerid; 唯一实际出现过的情形 (2026 EWC 的 "Team Secret") 由 esports_feed 的显式别名解决。
    这样的队在导出时就被剪掉了, 它在以后任何时刻都是过期的。
  · 同母赛区的国际赛局闸门不评, 这里也不给 (predict 返回 None, 看板仍按母赛区用 Stage 1/2)。
  · 重放只更新跨赛区国际赛; OE 截止之后的国内比赛不重放 (K=2, 一局最多挪 2 分, 可以忽略)。
BP 后不另给数: 加英雄胜率的候选 D 在 HOLDOUT 上没过 (对常数 t +2.47, 对 C 反而 −1.64)。

系数按赛事 walk-forward —— 服务时怎么选切点
-----------------------------------------
闸门预测一个赛事时, 系数只在该赛事第一局 (含它的赛区预选) 之前的跨赛区国际赛上拟合, 评分则在赛事内照常更新。
服务时照搬: 对每个 lolesports 国际赛的键 (LE_EVENT_CODES), 若"当年"(OE 最后一局的年份) 的这个赛事已经在 OE 里,
切点 = 它在 OE 里的第一局; 否则 (赛事还没进 OE, 比如数据截至 10-02 时的 DCGI) 切点 = 全部数据之后。
第一种和闸门逐位相同 (2026 MSI / FST / EWC 的导出系数和闸门那一赛事的系数一个比特不差); 第二种只比闸门多
"同时在打的别的国际赛已进 OE 的局" —— 系数是近千局上的岭回归, 多几局几乎不动。赛事一进 OE 就切到第一种。

    python xregion.py --build [--asof 2026-10-01] [--out 文件]   构建状态 (默认写 $LOL_ARTIFACTS 或 artifacts/xregion.json)
    python xregion.py --check 文件                               只做健全性检查
    from xregion import build_state, predict, Engine
    from xregion import board_predict, replay_frames_needed   看板: 状态 + 本赛事重放 (取数在 api._xregion_board)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from feature_store import LEAGUES, ROLES          # 四大赛区的名单只在那里写一次

ROOT = Path(__file__).resolve().parent

# 改了表的构造逻辑就加一, research/xregion_data 的旧缓存自动作废
# 3: INTL_EVENTS 加了 DCGI; 诊断里的"没归类的新代码"拆成常规联赛 (拦) / 杯赛 (只提醒) 两张表
VERSION = 3

# 和 train.py / api.py 的五年列表一致; _check_years() 在构建时核对, 不一致就报错 ——
# 少加载一年不会报错, 只会让"母赛区"和"之前的历史"悄悄变短。
YEARS = (2022, 2023, 2024, 2025, 2026)
DATA = [str(ROOT / "data" / f"{y}_LoL_esports_match_data_from_OraclesElixir.csv") for y in YEARS]
HOLDOUT_START = pd.Timestamp("2025-01-01")        # 闸门协议的 DEV / HOLDOUT 分界 (表里的 period 列)

# ── 国际赛 (目标集合) ──────────────────────────────────────────
# 值 = 只在这些年份算国际赛; None = 所有年份。为什么是这几个见 research/xregion_data.py 的文档。
INTL_EVENTS: dict[str, set[int] | None] = {
    "WLDs": None, "MSI": None, "FST": None, "EWC": None,      # 协议给定
    "ASI": None,                                             # 2025 亚洲邀请赛
    "LTA": None,                                             # 2025 LTA 跨分区 (LCS 对 CBLOL)
    "KeSPA Cup": {2025},                                     # 2025 届邀请了 LCS 队伍
    "AC": {2026},                                            # 2026 Americas Cup; 2024 是次级联赛
    # 2026 德玛西亚杯全球邀请赛 (10-03 起, LCK / LPL / LEC / LCS / CBLOL / LCP 的一级队)。数据截至 10-02 时 OE
    # 还没收它, 代码未知 —— 先按 lolesports 的名字登记 (OE 给新赛事起的代码多半跟 lolesports 的简称走: FST、
    # EWC、ASI)。不登记的话它一进 OE 就被当成"常规联赛": 参赛队的 Elo 池和母赛区整个挪过去 (_lookalikes 会拦)。
    # 现在的数据里没有这个代码, 闸门的表一局都不变。
    # OE 若沿用 "DCup": DCup 仍然排除 (下一行), 这届的局就只是"不计入状态" —— 看板照样按 lolesports 重放
    # 整个赛事 (OE 里没有可去重的, 不会算两遍), 构建时打一条提醒, 由人决定要不要按日期把它单独归类。
    "DCGI": None,
}
# 德玛西亚杯 2022/2023/2025 都是 LPL + LDL 的国内杯赛: LDL 对 LPL 在母赛区上是"跨赛区", 当国际赛算会拿
# 国内杯赛的胜负去挪 LPL 的偏移 —— 协议排除它就是为此
PROTOCOL_EXCLUDED = {"DCup"}
# OE 里算国际赛、但看板上没有自己的 lolesports 键的代码 (它们的局只进状态, 不需要按赛事定系数切点)。
# 国际赛代码必须二选一: 在 LE_EVENT_CODES 的值里, 或者在这里 —— 否则加了国际赛代码却忘了 LE_EVENT_CODES,
# 那个赛区键的系数切点会落到"全部数据之后", 把这个赛事自己的 OE 局也拟合进去 (有测试, sanity_check 也查)。
NO_LE_KEY = {"ASI", "LTA", "KeSPA Cup", "AC"}

# ── 不是任何队的母赛区 ─────────────────────────────────────────
# 杯赛和次级跨区赛: 参赛队平时在别的联赛打。母赛区查找时和国际赛一起跳过。
# (国际赛代码本身也在跳过之列, 不管那一年算不算国际赛 —— AC 2024 不是国际赛, 但同样不是母赛区。)
# OE 将来给某个杯赛 / 邀请赛用了新代码而这里没收, 它会被当成一个"常规联赛": 参赛队的 Elo 池和母赛区整个
# 挪过去, 不报错。build() 的 _lookalikes() 会把这种赛事报出来, sanity_check() 见到就拒绝导出。
# 已经在这里的杯赛哪怕有跨赛区的一级队对打 (比如 OE 把 DCGI 记成 DCup) 也不会挪任何东西 —— 只是那些局
# 不计入状态, 所以只提醒、不拦 (见 _lookalikes)。
NON_HOME = {
    "DCup", "KeSPA", "KeSPA Cup", "CDF", "CT", "USP", "TSC", "GLLPA", "EBLPA", "PRMP",
    "NLC Aurora Open", "HW", "CCWS", "IC",                    # 国内 / 区内杯赛
    "EM", "EUM", "EPL", "AM", "ASCI", "WSCI", "WLRQ",         # 次级联赛的跨区赛
} | set(INTL_EVENTS)

SERIES_GAP = pd.Timedelta(hours=12)
STALE_HOME_DAYS = 365          # 按队名查到的母赛区比这还旧, 改按阵容认队 (服务时: 不给数)
ROSTER_MIN_AGREE = 3           # 五个人里至少几个人来自同一赛区才算数

# lolesports 国际赛的键 (esports_feed.LEAGUE_IDS, = league.name.upper()) → OE 的赛事代码。
# DCGI → "DCGI": 和 INTL_EVENTS 同一个猜测。猜对了, 它一进 OE 切点就按闸门的规则落在它的第一局; 猜错了
# (OE 起了别的名字) 切点仍是"全部数据之后", 和现在一样 —— 那个名字会被 _lookalikes 拦下来, 改 INTL_EVENTS
# 时这里要一起改 (NO_LE_KEY 那条检查会提醒)。绝不能猜成 "DCup": 那是 12 月的国内杯赛, 切点会落到不相干的
# 赛事上。esports_feed.LEAGUE_IDS 里每个非四大的键都必须在这里 (有测试)。
LE_EVENT_CODES = {"WORLDS": "WLDs", "MSI": "MSI", "FIRST STAND": "FST", "ESPORTS WORLD CUP": "EWC",
                  "DCGI": "DCGI"}

_COLS = ["gameid", "datacompleteness", "league", "playoffs", "date", "game", "patch",
         "participantid", "side", "position", "playername", "playerid", "teamname", "teamid",
         "champion", "result"]


def is_international(code, year) -> bool:
    if code in PROTOCOL_EXCLUDED or code not in INTL_EVENTS:
        return False
    yrs = INTL_EVENTS[code]
    return yrs is None or int(year) in yrs


# ══════════════════════════════════════════════════════════
#  局级表 (全部 OE 赛区)
# ══════════════════════════════════════════════════════════

def _check_years():
    """五年列表必须和训练用的一致。在函数里 import train: 只有真要构建时才付 xgboost 的导入开销。"""
    import train
    ours = [Path(f).resolve() for f in DATA]
    theirs = [Path(f).resolve() for f in train.DATA]
    if ours != theirs:
        raise RuntimeError(f"xregion 的年份列表和 train.DATA 不一致:\n  {ours}\n  {theirs}")


def _read_raw(files, verbose: bool) -> pd.DataFrame:
    dfs = []
    for f in files:
        p = Path(f)
        if not p.exists():
            raise FileNotFoundError(f"缺数据文件 {p} —— 少一年不会报错, 只会让母赛区和历史悄悄变短, 所以这里直接停")
        if verbose:
            print(f"    读 {p.name}")
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
    """(赛事代码, 队伍对) 内按时间排, game 序号不增或间隔 > 12 小时就开新系列赛。

    OE 没有系列赛 id。按"日期 + 队伍对"分, 跨 UTC 零点的 BO5 (美洲赛区傍晚开打) 会被拆成两个系列赛,
    同一天两队打两次 (小组赛加赛) 会被并成一个。
    """
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
    """从五年 CSV 构建局级表 (全部赛区)。返回 (表, 构建时的诊断)。约 15 秒。"""
    _check_years()
    raw = _read_raw(DATA, verbose)
    return build_table(raw)


def build_table(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """原始 OE 行 (已做赛区改名、日期解析, 见 _read_raw) → 局级表。拆出来是为了测试能喂合成数据。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _build_table(raw)


def _build_table(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
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
    look = _lookalikes(g)
    diag["unlisted_lookalikes"] = [x for x in look if x["class"] == "league"]
    diag["nonhome_crossregion"] = [x for x in look if x["class"] == "non_home"]
    return g, diag


def _lookalikes(g: pd.DataFrame) -> list[dict]:
    """没列入国际赛、却有跨赛区一级联赛对打的赛事 —— 将来 OE 加了新代码就会出现在这里。

    一级联赛 (按年) = 当年 WLDs/MSI/FST/EWC 参赛队的母赛区。只看参赛队是谁, 不看胜负。
    扫全部非国际赛的赛事 (包括常规联赛): 新出现的代码不在 NON_HOME 里, 只扫杯赛会漏掉它。
    常规联赛只有升降级队伍的头几场会被算成"跨赛区", 占比远低于阈值。

    每条带 class, 后果完全不同, 调用方分开处理:
      "league"    代码被当成常规联赛 —— 参赛队的 Elo 池和母赛区整个挪过去, 状态是错的。sanity_check 拒绝导出,
                  要先在 INTL_EVENTS (它的跨赛区局该计入 C) 或 NON_HOME (只是杯赛) 里归类。
      "non_home"  代码已在 NON_HOME (比如 OE 把 DCGI 记成了 DCup): 不挪池子、不当母赛区, 只是这些跨赛区局不计入
                  状态 (看板仍按 lolesports 重放正在打的赛事, OE 里没有可去重的, 不会算两遍)。只提醒 ——
                  拦下来的话日更要一直卡到有人改代码, 而状态本身没有错。
    """
    core = g[g["league"].isin(["WLDs", "MSI", "FST", "EWC"])]
    fd = {y: set(s[["home_blue", "home_red"]].stack().dropna()) for y, s in core.groupby("year")}
    out = []
    for (ev, y), s in g[~g["is_international"]].groupby(["event", "year"]):
        f = fd.get(y, set())
        m = s["cross_region"] & s["home_blue"].isin(f) & s["home_red"].isin(f)
        if m.sum() >= 5 and m.mean() >= 0.10:
            cls = "non_home" if s["event_class"].iloc[0] == "non_home" else "league"
            out.append({"event": ev, "n": len(s), "cross_first_division": int(m.sum()), "class": cls})
    return out


def target(df: pd.DataFrame) -> pd.Series:
    """协议的目标集合: 国际赛里的跨赛区局。系数只在这些局上拟合。"""
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


# ══════════════════════════════════════════════════════════
#  引擎: 分层 Elo
# ══════════════════════════════════════════════════════════

S = 400 / math.log(10)            # Elo 刻度: 差 S 分 = logit 差 1
EPOCH = pd.Timestamp("1970-01-01")
LN2 = math.log(2)

# 冻结的候选 C (闸门在 DEV 上选定; 一个数都不许改 —— research/gate_cross_region.py 的 DEV 复现会拿它
# 对账, 改了就复现不出来)。DEV 上还调到了 reg=0 (不跨年回归)、rebrand=否 (不按阵容继承改名队的评分)
# —— 这两项在这个取值下什么都不做, 所以实现里没有它们。h = 蓝方加成 (0), knew / nnew = 新队前 nnew 局 K 乘
# knew (1, 即不加速), lag = "none" 即赛事内在线更新 (只有这个版本过了闸)。
P_C = dict(K=2.0, knew=1.0, nnew=10, h=0.0, init=0.0, cup=True, intl_same=False,
           eta=24.0, kint=24.0, hl=365.0, dnm=600.0, lag="none")
LAM_C = 30.0                      # walk-forward logistic: Δo、Δr 分开, 岭先验朝 (logit c, 1, 1), λ = 30


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def to_t(ts) -> float:
    """时间戳 → 引擎的时间 (1970-01-01 起的天数, 浮点)。不带时区的按 UTC (OE 的 date 就是 UTC)。

    和 prepare() 对整列的算法逐位相同: 整秒的时间戳 → 秒数是精确整数, 再除以 86400。
    浏览器版: Date.parse(iso) / 1000 / 86400。
    """
    x = pd.Timestamp(ts)
    if x.tzinfo is not None:
        x = x.tz_convert("UTC").tz_localize(None)
    return (x - EPOCH).total_seconds() / 86400


# ── 可移植的 exp (fdlibm e_exp.c) ─────────────────────────────
# 为什么不用 math.exp: 看板的 C 由两份实现算 (这里和浏览器的 frontend/src/web/xregion.ts), test:diff 要求
# 逐位相同, 而**没有哪两个平台的 exp 逐位一致**: 本机 math.exp 是 MSVC 运行库的 (实测 20 万个输入里 1147 个
# 不是正确舍入), V8 的 Math.exp 是 fdlibm (和本机 math.exp 有 7% 的输入差一位), Safari 用系统 libm, Linux 上
# 的 Python 用 glibc。差一位本身无所谓, 但它会让"同一个看板两边算出不同的数"变成常态, 真正的移植错误就藏在
# 里面看不见。所以从导出的状态出发的那一段 (Engine.from_state 之后的重放、衰减、logistic) 两边都用这一份:
# 只有 IEEE 加减乘除和 2 的整数次幂缩放, 每一步在任何平台上都是同一个结果。浏览器版是同一段代码的逐行移植,
# 和 V8 的 Math.exp 在 20 万个随机输入上逐位相同 (V8 本来就是 fdlibm, 只对 x = 1 特判返回 Math.E; 这里不跟,
# 两份移植一致就够)。
# 闸门和每天导出状态时整表走一遍 (run_elo / build_state) 仍用 math.exp: 闸门的复现 (重构前后 max|Δp| = 0)
# 一个比特都不动, 状态本身是数据, 两份实现读的是同一个 JSON。两种 exp 至多差一位, 远在模型的意义之下。
_FD_HALF = (0.5, -0.5)
_FD_LN2HI = (6.93147180369123816490e-01, -6.93147180369123816490e-01)   # 0x3fe62e42 fee00000
_FD_LN2LO = (1.90821492927058770002e-10, -1.90821492927058770002e-10)   # 0x3dea39ef 35793c76
_FD_INVLN2 = 1.44269504088896338700e+00
_FD_P = (1.66666666666666019037e-01, -2.77777777770155933842e-03, 6.61375632143793436117e-05,
         -1.65339022054652515390e-06, 4.13813679705723846039e-08)
_FD_HUGE = 1.0e+300
_FD_TWOM1000 = 9.33263618503218878990e-302                          # 2**-1000
_FD_O_THRESHOLD = 7.09782712893383973096e+02
_FD_U_THRESHOLD = -7.45133219101941108420e+02


def portable_exp(x) -> float:
    """fdlibm 的 __ieee754_exp, 逐行移植 (理由见上)。浏览器版: xregion.ts 的 exp, 两边逐位相同 (test:golden)。"""
    x = float(x)
    bits = struct.unpack("<Q", struct.pack("<d", x))[0]
    hx, lx = bits >> 32, bits & 0xFFFFFFFF
    xsb = (hx >> 31) & 1
    hx &= 0x7FFFFFFF
    if hx >= 0x40862E42:                            # |x| >= 709.78...
        if hx >= 0x7FF00000:
            if ((hx & 0xFFFFF) | lx) != 0:
                return x + x                        # NaN
            return x if xsb == 0 else 0.0           # exp(±inf)
        if x > _FD_O_THRESHOLD:
            return math.inf
        if x < _FD_U_THRESHOLD:
            return 0.0
    k, hi, lo = 0, 0.0, 0.0
    if hx > 0x3FD62E42:                             # |x| > 0.5 ln2: 约简到 [-0.5 ln2, 0.5 ln2]
        if hx < 0x3FF0A2B2:                         # 且 |x| < 1.5 ln2
            hi, lo, k = x - _FD_LN2HI[xsb], _FD_LN2LO[xsb], 1 - xsb - xsb
        else:
            k = int(_FD_INVLN2 * x + _FD_HALF[xsb])     # 向零取整, 同 C 的 (int)
            t = float(k)
            hi = x - t * _FD_LN2HI[0]
            lo = t * _FD_LN2LO[0]
        x = hi - lo
    elif hx < 0x3E300000:                           # |x| < 2**-28
        if _FD_HUGE + x > 1.0:
            return 1.0 + x
    p1, p2, p3, p4, p5 = _FD_P
    t = x * x
    c = x - t * (p1 + t * (p2 + t * (p3 + t * (p4 + t * p5))))
    if k == 0:
        return 1.0 - ((x * c) / (c - 2.0) - x)
    y = 1.0 - ((lo - (x * c) / (2.0 - c)) - hi)
    # fdlibm 直接往 y 的指数位上加 k —— 乘 2 的整数次幂是精确的, 结果一样。2.0 ** 1024 在 Python 里溢出, 拆两步
    if k >= -1021:
        return y * 2.0 ** k if k < 1024 else y * 2.0 * 2.0 ** 1023
    return y * 2.0 ** (k + 1000) * _FD_TWOM1000


def _iso(x) -> str:
    """时间戳 / 引擎时间 → "2026-10-02T18:20:44Z" (UTC)。"""
    if isinstance(x, (int, float)):
        return datetime.fromtimestamp(round(x * 86400), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    x = pd.Timestamp(x)
    if x.tzinfo is not None:
        x = x.tz_convert("UTC").tz_localize(None)
    return x.strftime("%Y-%m-%dT%H:%M:%SZ")


def pairwise_mean(xs) -> float:
    """float(np.mean(xs)) 的逐位等价写法 —— 新赛区的先验 = 迁入队伍原赛区偏移的均值, 就用它。

    np.mean 不是顺序累加: 8 个以上的数用 8 路分块的成对求和 (numpy 的 pairwise_sum), 超过 128 个再对半
    递归。顺序累加在 8 个以上时会差最后一两位, 浏览器版照这个顺序加才能和这里逐位一致 (有测试对 np.mean)。
    """
    xs = [float(v) for v in xs]
    if not xs:
        raise ValueError("空列表没有均值")

    def psum(a, lo, n):
        if n < 8:
            s = 0.0
            for i in range(lo, lo + n):
                s += a[i]
            return s
        if n <= 128:
            r = a[lo:lo + 8]
            i = 8
            while i < n - (n % 8):
                for j in range(8):
                    r[j] += a[lo + i + j]
                i += 8
            s = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
            while i < n:
                s += a[lo + i]
                i += 1
            return s
        n2 = n // 2
        n2 -= n2 % 8
        return psum(a, lo, n2) + psum(a, lo + n2, n - n2)

    return psum(xs, 0, len(xs)) / len(xs)


def status_of(hb, hr) -> str:
    """两方母赛区 → "cross" / "same" / "unknown", 和 _status() 同一个口径。"""
    if hb is None or hr is None:
        return "unknown"
    return "same" if hb == hr else "cross"


class Engine:
    """分层 Elo 的状态机。闸门的 run_elo 和看板的重放 / 预测都只走这里的方法 —— 同一段算术。

    状态: r[队] / pool[队] (Elo 池 = 最后一场常规联赛的赛区) / ng[队] (常规局数) / last[队] (最后一场常规联赛的
    时间), o[赛区] / oprior[赛区] / olast[赛区] (偏移、先验、上次衰减到的时刻), migr[赛区] (迁入队伍的原赛区)。
    时间单位是 to_t() 的天数。

    exp: 引擎里所有 exp (衰减、赛前胜率) 用哪一个。整表走一遍 (闸门 / build_state) 用 math.exp —— 闸门的复现
    一个比特不动; 从导出状态出发 (from_state: 看板的重放和预测) 用 portable_exp —— 浏览器版逐位相同。见 portable_exp。
    """

    def __init__(self, P: dict | None = None, majors=None, exp=math.exp):
        P = dict(P_C if P is None else P)
        self.P = P
        self.exp = exp
        self.K, self.knew, self.nnew = P["K"], P["knew"], int(P["nnew"])
        self.h, self.init = P["h"], P["init"]
        self.eta, self.kint, self.hl, self.dnm = P["eta"], P["kint"], P["hl"], P["dnm"]
        self.cup, self.intl_same = P["cup"], P["intl_same"]
        self.majors = set(LEAGUES if majors is None else majors)
        self.r: dict[str, float] = {}
        self.pool: dict[str, str] = {}
        self.ng: dict[str, int] = {}
        self.last: dict[str, float] = {}
        self.o: dict[str, float] = {}
        self.oprior: dict[str, float] = {}
        self.olast: dict[str, float] = {}
        self.migr: dict[str, list] = defaultdict(list)     # 还没有偏移的赛区 → 迁入队伍的原赛区
        self.stats = {"intl_pool_mismatch": 0}

    # ── 基本操作 ──
    def enter(self, team, league):
        """常规联赛比赛: 确保队伍在这个池子里有 r (新队 / 换赛区)。"""
        r, ng, pool, o = self.r, self.ng, self.pool, self.o
        old = pool.get(team)
        if old == league:
            return
        if old is None:
            r[team], ng[team], pool[team] = self.init, 0, league
            return
        if old in o and league in o:                    # 两个赛区都有偏移: 保持全球实力不变
            r[team] = o[old] + r[team] - o[league]
        elif old in o:                                  # 迁入还没有偏移的新赛区: r 带过去
            self.migr[league].append(old)
        else:                                           # 原赛区没有可比的零点: 当新队
            r[team] = self.init
            ng[team] = 0
        pool[team] = league
        ng.setdefault(team, 0)

    def off(self, L, t):
        """赛区偏移: 懒初始化 + 向先验的指数衰减, 并把 olast 推到 t (会改状态)。

        衰减是分段乘的: 先衰减到 t1 再到 t2, 和一步衰减到 t2 差最后一位 —— 所以重放必须在和闸门同样的时刻调它
        (每一局跨赛区局开打时, 两个赛区各一次)。t 早于 olast 时不动 (不会往回"反衰减")。
        """
        o = self.o
        if L not in o:
            src = [o[x] for x in self.migr.get(L, []) if x in o]
            pri = pairwise_mean(src) if src else (0.0 if L in self.majors else -self.dnm)
            o[L] = self.oprior[L] = pri
            self.olast[L] = t
        elif self.hl != math.inf:
            dt = t - self.olast[L]
            if dt > 0:
                o[L] = self.oprior[L] + (o[L] - self.oprior[L]) * self.exp(-LN2 * dt / self.hl)
                self.olast[L] = t
        return o[L]

    def home(self, team, t):
        """服务时的母赛区: Elo 池 (= 最后一场常规联赛的赛区); 那场比赛早于 t 超过 365 天就是 None。

        闸门遇到这种队会按阵容认队, 看板没有 OE 的 playerid, 所以不给数 (见模块文档)。
        """
        L = self.pool.get(team)
        last = self.last.get(team)
        if L is None or last is None or t - last > STALE_HOME_DAYS:
            return None
        return L

    def pregame(self, ec, league, b, rd, hb, hr, status, t, ok, key):
        """一局开打前: 返回 (o_蓝, o_红, r_蓝, r_红, 待更新) —— 母赛区不明的国际赛返回 None (不出数也不更新)。

        ec: 0 = 常规联赛, 1 = 杯赛 / 次级跨区赛, 2 = 国际赛。待更新 (没有就是 None) 要等同一时间戳的整组都取完
        赛前值之后再交给 apply(): 同时开打的两局互相看不到。key 原样放进待更新里, 调用方用它找这局的胜负。
        """
        nan = math.nan
        r, pool, exp = self.r, self.pool, self.exp
        if ec == 0:                                     # 常规联赛
            self.enter(b, league)
            self.enter(rd, league)
            self.last[b] = t
            self.last[rd] = t
            u = ("dom", key, b, rd, 1 / (1 + exp(-(r[b] + self.h - r[rd]) / S))) if ok else None
            return nan, nan, r[b], r[rd], u
        if ec == 2:                                     # 国际赛
            if status == "unknown":
                return None
            # 母赛区 (表的口径, 可能按阵容认队) 和 Elo 池不一致的队, r 取 init 且不更新
            okb, okr = pool.get(b) == hb, pool.get(rd) == hr
            self.stats["intl_pool_mismatch"] += (not okb) + (not okr)
            vb = r[b] if okb else self.init
            vr = r[rd] if okr else self.init
            if status == "cross":
                ob, orr = self.off(hb, t), self.off(hr, t)
                pe = 1 / (1 + exp(-(ob + vb + self.h - orr - vr) / S))
                u = ("x", key, b, rd, pe, hb, hr, okb, okr) if ok else None
                return ob, orr, vb, vr, u
            oo = self.o.get(hb, nan)                    # 同赛区: 偏移相同, 只看 r
            pe = 1 / (1 + exp(-(vb + self.h - vr) / S))
            u = ("dom", key, b, rd, pe) if (self.intl_same and okb and okr and ok) else None
            return oo, oo, vb, vr, u
        # 杯赛 / 次级跨区赛: 同池两队才更新
        if self.cup and pool.get(b) is not None and pool.get(b) == pool.get(rd):
            pe = 1 / (1 + exp(-(r[b] + self.h - r[rd]) / S))
            return nan, nan, nan, nan, (("dom", key, b, rd, pe) if ok else None)
        return nan, nan, nan, nan, None

    def apply(self, u, y):
        """把一局的结果 y (蓝方胜 = 1.0) 按 pregame() 给的待更新记进状态。"""
        if u[0] == "dom":
            _, _key, b, rd, pe = u
            d = y - pe
            ng = self.ng
            kb = self.K * (self.knew if ng.get(b, 0) < self.nnew else 1.0)
            kr = self.K * (self.knew if ng.get(rd, 0) < self.nnew else 1.0)
            self.r[b] += kb * d
            self.r[rd] -= kr * d
            ng[b] = ng.get(b, 0) + 1
            ng[rd] = ng.get(rd, 0) + 1
        else:
            _, _key, b, rd, pe, hb, hr, okb, okr = u
            d = y - pe
            self.o[hb] += self.eta * d
            self.o[hr] -= self.eta * d
            if self.kint:
                if okb:
                    self.r[b] += self.kint * d
                if okr:
                    self.r[rd] -= self.kint * d

    # ── 看板用的两步: 重放、预测 ──
    def play(self, games) -> int:
        """按顺序重放一批已打完的国际赛局 (OE 截止之后的), 返回实际更新了评分的局数。

        games: [{"t": to_t(开打时间), "blue": OE 队名, "red": OE 队名, "blue_win": 0/1}, …], t 不得递减 (调用方
        先排好: 系列赛按开打时间, 局按局序 —— 递减直接抛 ValueError, 不悄悄重排)。t 相同的连续几局是一组:
        先全部取赛前值, 再统一更新, 和 run_elo 对同一时间戳的处理一样。组内按给定顺序更新。
        母赛区由本状态按 home() 查; 两方都查得到且不同 (跨赛区) 才更新, 同赛区和查不到的局什么都不做 ——
        和闸门对这两类局的处理一样。OE 截止之后的国内比赛不在这里 (K=2, 可以忽略)。
        """
        games = list(games)
        ts = [float(g["t"]) for g in games]
        if any(b_ < a_ for a_, b_ in zip(ts, ts[1:])):
            raise ValueError("重放的局必须按时间排好 (t 不递减)")
        n_upd = 0
        i = 0
        while i < len(games):
            j = i
            while j < len(games) and ts[j] == ts[i]:
                j += 1
            upd = []
            for k in range(i, j):
                g = games[k]
                b, rd, t = g["blue"], g["red"], ts[k]
                hb, hr = self.home(b, t), self.home(rd, t)
                pre = self.pregame(2, None, b, rd, hb, hr, status_of(hb, hr), t, True, k)
                if pre is not None and pre[4] is not None:
                    upd.append(pre[4])
            for u in upd:
                self.apply(u, float(games[u[1]]["blue_win"]))
                n_upd += u[0] == "x"
            i = j
        return n_upd

    def features(self, blue, red, t):
        """t 时刻的 (o_蓝, o_红, r_蓝, r_红) —— 两队母赛区都查得到且不同才有, 否则 None。会把 o 衰减到 t。"""
        hb, hr = self.home(blue, t), self.home(red, t)
        if status_of(hb, hr) != "cross":
            return None
        ob, orr = self.off(hb, t), self.off(hr, t)
        return ob, orr, self.r[blue], self.r[red]

    def predict(self, coef: dict, blue, red, t):
        f = self.features(blue, red, t)
        return None if f is None else logistic(coef, *f, exp=self.exp)

    # ── 状态 ↔ JSON ──
    @classmethod
    def from_state(cls, state: dict, exp=None) -> "Engine":
        """从 build_state() 的导出 (或它的 JSON) 重建引擎。只读 state, 不改它。

        exp 默认 portable_exp: 从这里出发的都是看板的计算 (重放、衰减、预测), 浏览器版要逐位复现。
        """
        P = {k: state["params"][k] for k in P_C}
        e = cls(P, majors=state["params"]["majors"], exp=portable_exp if exp is None else exp)
        for team, v in state["teams"].items():
            e.pool[team] = v["pool"]
            e.r[team] = float(v["r"])
            e.ng[team] = int(v["ng"])
            e.last[team] = float(v["last_regular_t"])
        for L, v in state["leagues"].items():
            if v["has_crossregion_history"]:
                e.o[L] = float(v["o"])
                e.oprior[L] = float(v["prior"])
                e.olast[L] = float(v["last_t"])
        for L, src in state.get("migr", {}).items():
            e.migr[L] = list(src)
        return e


def logistic(coef: dict, ob, orr, rb, rr, exp=None) -> float:
    """p = σ(a + b_o·Δo/S + b_r·Δr/S), 和闸门 predict_wf 的向量写法同样的运算顺序。

    只在看板 (服务) 这一侧用, exp 默认 portable_exp —— 浏览器版逐位复现 (见 portable_exp)。传 math.exp 时和
    闸门的 np.exp 写法逐位相同 (两者在本机 200 万个随机输入上一致)。"""
    xo = (ob - orr) / S
    xr = (rb - rr) / S
    z = coef["a"] + coef["b_o"] * xo + coef["b_r"] * xr
    return 1 / (1 + (portable_exp if exp is None else exp)(-z))


# ══════════════════════════════════════════════════════════
#  整表走一遍 (闸门和 build_state 共用)
# ══════════════════════════════════════════════════════════

def prepare(df: pd.DataFrame) -> dict:
    """局级表 → 扁平数组 (Elo 主循环吃 Python 列表, 比逐行 pandas 快两个数量级)。"""
    if not df["date"].is_monotonic_increasing:
        raise RuntimeError("局级表没按时间排序 —— Elo 和英雄表都假定按时间走")
    A = {"n": len(df)}
    # 天 (浮点)。不能用 astype("int64") / 86400e9: 缓存里的 date 是 datetime64[us] —— 候选 C 的第一版就这样
    # 差了 1000 倍, 半衰期整个失效, 没有任何报错
    A["t"] = ((df["date"] - EPOCH).dt.total_seconds() / 86400).tolist()
    ns = df["date"].astype("int64").tolist()                  # 只用来判断"同一时间戳", 单位无所谓
    A["year"] = df["year"].astype(int).tolist()
    A["ec"] = df["event_class"].map({"league": 0, "non_home": 1, "intl": 2}).astype(int).tolist()
    A["league"] = df["league"].astype(str).tolist()
    A["blue"] = df["blue"].astype(str).tolist()
    A["red"] = df["red"].astype(str).tolist()
    A["y"] = df["blue_win"].astype(float).tolist()
    A["ok"] = df["result_ok"].astype(bool).tolist()
    A["hb"] = [x if isinstance(x, str) else None for x in df["home_blue"].tolist()]
    A["hr"] = [x if isinstance(x, str) else None for x in df["home_red"].tolist()]
    A["status"] = df["region_status"].astype(str).tolist()
    # 同一时间戳的几局一组: 组内先全部取赛前值, 再统一更新 —— 同时开打的两局互相看不到
    starts = [0] + [i for i in range(1, len(ns)) if ns[i] != ns[i - 1]]
    A["groups"] = list(zip(starts, starts[1:] + [len(ns)]))
    day = [math.floor(x) for x in A["t"]]
    dstarts = [0] + [i for i in range(1, len(day)) if day[i] != day[i - 1]]
    A["day_groups"] = list(zip(dstarts, dstarts[1:] + [len(day)]))
    return A


def run_elo(P: dict, A: dict, exp=math.exp) -> dict:
    """按时间走一遍全部 OE 比赛, 给每局记下赛前的 o_蓝 / o_红 (只有国际赛才有) 和 r_蓝 / r_红。

    时间纪律: 预测只用时间戳严格更早的结果 (lag="none", 赛事内的更早局也算 —— 只有这个版本过了闸);
    lag="nextday" 是闸门的参照行: 结果从下一个 UTC 日起才生效 (OE 隔天才有数据)。
    返回的 "engine" 是走完之后的 Engine, build_state 从它导出状态。
    exp 默认 math.exp (闸门的口径); 测试拿 portable_exp 走一遍, 和"状态 + 重放"逐位对账。
    """
    eng = Engine(P, exp=exp)
    nextday = P["lag"] == "nextday"
    pending: list = []                              # nextday: (生效时刻, 更新), 按时间排队
    n = A["n"]
    ob, orr = np.full(n, np.nan), np.full(n, np.nan)
    rb, rr = np.full(n, np.nan), np.full(n, np.nan)
    tl, yl, ecl, lgl = A["t"], A["y"], A["ec"], A["league"]
    bl, rl, okl, hbl, hrl, stl = A["blue"], A["red"], A["ok"], A["hb"], A["hr"], A["status"]
    pi = 0
    for g0, g1 in A["groups"]:
        if nextday:                                 # 到期的结果先生效 (按原来的时间顺序)
            while pi < len(pending) and pending[pi][0] <= tl[g0]:
                u = pending[pi][1]
                eng.apply(u, yl[u[1]])
                pi += 1
        upd = []
        for i in range(g0, g1):
            pre = eng.pregame(ecl[i], lgl[i], bl[i], rl[i], hbl[i], hrl[i], stl[i], tl[i], okl[i], i)
            if pre is None:
                continue
            ob[i], orr[i], rb[i], rr[i], u = pre
            if u is not None:
                upd.append(u)
        for u in upd:
            if nextday:
                pending.append((math.floor(tl[u[1]]) + 1.0, u))
            else:
                eng.apply(u, yl[u[1]])
    return {"ob": ob, "or": orr, "rb": rb, "rr": rr, "o": dict(eng.o), "oprior": dict(eng.oprior),
            "olast": dict(eng.olast), "migr": {k: list(v) for k, v in eng.migr.items()},
            "stats": eng.stats, "r": dict(eng.r), "pool": dict(eng.pool), "engine": eng}


# ══════════════════════════════════════════════════════════
#  系数 (walk-forward 的岭 logistic)
# ══════════════════════════════════════════════════════════

def fit_ridge(X, y, prior, lam, pen=None, iters=60):
    """最小化 NLL + λ/2·Σ pen·(β − prior)²。没有训练行 (2022 MSI 之前) 就返回先验。"""
    prior = np.asarray(prior, float)
    if lam == math.inf or len(y) == 0:
        return prior.copy()
    pen = np.ones_like(prior) if pen is None else np.asarray(pen, float)
    beta = prior.copy()
    for _ in range(iters):
        p = sigmoid(X @ beta)
        g = X.T @ (y - p) - lam * pen * (beta - prior)
        H = X.T @ (X * (p * (1 - p))[:, None]) + lam * np.diag(pen) + 1e-9 * np.eye(len(beta))
        step = np.linalg.solve(H, g)
        beta = beta + step
        if np.abs(step).max() < 1e-10:
            break
    return beta


def const_rate(df, cut=None) -> float:
    """切点之前全部 OE 比赛的蓝方胜率 (闸门的基线 (a), 也是系数截距的先验)。cut=None = 全部数据。"""
    m = df["result_ok"] if cut is None else (df["date"] < cut) & df["result_ok"]
    return float(df.loc[m, "blue_win"].mean())


def fit_coef(x_o, x_r, y, tr, c, extra=(), lam=LAM_C):
    """在训练行 tr 上拟合 (a, b_o, b_r): 岭先验朝 (logit c, 1, 1), λ = LAM_C。

    x_o = Δo/S、x_r = Δr/S (整表的数组), tr = 布尔掩码。extra = [(整表的列, 先验, 罚权)]: 闸门的候选 D 在同一个
    logistic 里加 BP 项用 (没过闸, 线上不用)。返回 numpy 数组 [a, b_o, b_r, *extra]。
    """
    cols = [np.ones(tr.sum()), x_o[tr], x_r[tr]]
    prior, pen = [logit(c), 1.0, 1.0], [1.0, 1.0, 1.0]
    for col, pr, pn in extra:
        cols.append(col[tr])
        prior.append(pr)
        pen.append(pn)
    return fit_ridge(np.column_stack(cols), y[tr], prior, lam, pen)


def oe_event(league_key, year):
    """lolesports 的赛区键 + 年份 → OE 的赛事键 ("WORLDS", 2026 → "2026 WLDs"); 没有对应代码 → None。"""
    code = LE_EVENT_CODES.get(str(league_key).upper())
    return None if code is None else f"{int(year)} {code}"


def serving_coefficients(d: pd.DataFrame, R: dict, year: int) -> dict:
    """每个 lolesports 国际赛的键一组系数, 切点规则见模块文档。

    d 和 R 必须是同一张表 (run_elo(P_C, prepare(d)) 的结果), 已截断到 asof。
    """
    x_o = (R["ob"] - R["or"]) / S
    x_r = (R["rb"] - R["rr"]) / S
    y = d["blue_win"].values.astype(float)
    xmask = target(d).values
    dates = d["date"].values
    first = event_table(d, d["is_international"])["first"]
    out = {}
    for key in LE_EVENT_CODES:
        ev = oe_event(key, year)
        if ev is not None and ev in first.index:
            cut = first[ev]
            tr = xmask & (dates < np.datetime64(cut))
            c = const_rate(d, cut)
        else:
            cut = None
            tr = xmask.copy()
            c = const_rate(d)
        if not np.isfinite(x_o[tr]).all() or not np.isfinite(x_r[tr]).all():
            raise RuntimeError(f"{key}: 训练行里有缺失的评分 —— 跨赛区局应当都有 o / r")
        beta = fit_coef(x_o, x_r, y, tr, c)
        out[key] = {"a": float(beta[0]), "b_o": float(beta[1]), "b_r": float(beta[2]), "c": c,
                    "event": ev if cut is not None else None,
                    "cutoff": None if cut is None else _iso(cut), "n_train": int(tr.sum())}
    return out


# ══════════════════════════════════════════════════════════
#  导出的状态
# ══════════════════════════════════════════════════════════

SCHEMA = 1
MIN_TEAMS = 150                   # 2026-10 实测 400 支左右; 少到这个数说明少加载了数据
MIN_MAJOR_TEAMS = 6
GAME_FIELDS = ["blue", "red", "game", "date", "blue_win", "status"]


def build_state(df: pd.DataFrame | None = None, asof=None, verbose: bool = False, exp=math.exp) -> dict:
    """五年 OE 走完之后的状态 (JSON 可序列化)。df 不给就从 CSV 构建; asof 给了就只用严格早于它的局。

    字段:
      schema, built_at, asof, oe_asof (用到的最后一局 OE 的时间), oe_asof_t
      params        P_C 全部键 + lam / S / stale_home_days / majors
      teams         {OE 队名: {pool, home, r, ng, last_regular, last_regular_t}} —— 最后一场常规联赛早于 oe_asof
                    365 天以上的队不导出 (以后任何时刻都是过期的, 见模块文档); home = pool
      leagues       {赛区: {o, prior, last_t, last, has_crossregion_history, crossregion_games, major}} —— o 是
                    没衰减的原值, 上次衰减到 last_t; 没打过跨赛区国际赛的赛区 o / prior / last_t 为 null,
                    第一次出现时由 Engine.off() 取先验
      migr          {还没有偏移的赛区: [迁入队伍的原赛区, …]} (新赛区的先验 = 这些赛区当时偏移的均值)
      coefficients  {lolesports 键: {a, b_o, b_r, c, event, cutoff, n_train}}
      event_codes   LE_EVENT_CODES
      events        {OE 赛事键: {code, year, league_key, first, last, n, n_cross, games: [[blue, red, game, date,
                    blue_win, status], …]}} —— OE 里已有的国际赛局, 都已计入状态, 重放按它去重; 字段名在 game_fields
      counts, diag
    exp: 整表走一遍时用的 exp, 默认 math.exp (导出的系数和闸门那一赛事的系数逐位相同); 测试传 portable_exp,
    让"状态 + 重放"和整表在线走一遍在同一个 exp 下逐位对账。
    """
    if df is None:
        df, diag = build(verbose=verbose)
    else:
        diag = df.attrs.get("diag", {})
    d = df
    if asof is not None:
        cut = pd.Timestamp(asof)
        if cut.tzinfo is not None:
            cut = cut.tz_convert("UTC").tz_localize(None)
        d = df[df["date"] < cut]
    d = d.reset_index(drop=True)
    if not len(d):
        raise ValueError("asof 之前没有任何比赛")
    A = prepare(d)
    R = run_elo(P_C, A, exp=exp)
    eng: Engine = R["engine"]
    oe_last = pd.Timestamp(d["date"].iloc[-1])
    t_end = A["t"][-1]

    teams, pruned = {}, 0
    for team in sorted(eng.pool):
        last = eng.last[team]
        if t_end - last > STALE_HOME_DAYS:
            pruned += 1
            continue
        teams[team] = {"pool": eng.pool[team], "home": eng.pool[team], "r": eng.r[team],
                       "ng": int(eng.ng.get(team, 0)), "last_regular": _iso(last), "last_regular_t": last}

    tg = d[target(d)]
    xcount = pd.concat([tg["home_blue"], tg["home_red"]]).value_counts().to_dict()
    leagues = {}
    for L in sorted(set(eng.o) | {v["pool"] for v in teams.values()}):
        has = L in eng.o
        leagues[L] = {"o": eng.o[L] if has else None, "prior": eng.oprior[L] if has else None,
                      "last_t": eng.olast[L] if has else None, "last": _iso(eng.olast[L]) if has else None,
                      "has_crossregion_history": has, "crossregion_games": int(xcount.get(L, 0)),
                      "major": L in LEAGUES}
    # migr 只在赛区还没有偏移时被读 (Engine.off 的懒初始化); 已有偏移的赛区导出它没有意义
    migr = {L: list(v) for L, v in sorted(eng.migr.items()) if L not in eng.o and v}

    rev = {v: k for k, v in LE_EVENT_CODES.items() if v}
    events = {}
    intl = d[d["is_international"]]
    for ev, s in sorted(intl.groupby("event"), key=lambda kv: kv[1]["date"].min()):
        code = str(s["league"].iloc[0])
        games = [[str(b), str(r), None if g != g else int(g), _iso(dt), int(y), str(st)]
                 for b, r, g, dt, y, st in zip(s["blue"], s["red"], s["game"], s["date"], s["blue_win"],
                                               s["region_status"])]
        events[ev] = {"code": code, "year": int(s["year"].iloc[0]), "league_key": rev.get(code),
                      "first": _iso(s["date"].min()), "last": _iso(s["date"].max()), "n": len(s),
                      "n_cross": int(s["cross_region"].sum()), "games": games}

    return {
        "schema": SCHEMA,
        "model": "C 分层 Elo (research/gate_cross_region.py, 2026-10-04 上线)",
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "asof": None if asof is None else _iso(asof),
        "oe_asof": _iso(oe_last),
        "oe_asof_t": t_end,
        "params": {**P_C, "lam": LAM_C, "S": S, "stale_home_days": STALE_HOME_DAYS, "majors": list(LEAGUES)},
        "teams": teams,
        "leagues": leagues,
        "migr": migr,
        "coefficients": serving_coefficients(d, R, oe_last.year),
        "event_codes": dict(LE_EVENT_CODES),
        "game_fields": list(GAME_FIELDS),
        "events": events,
        "counts": {"games": len(d), "intl_games": int(d["is_international"].sum()),
                   "crossregion_intl_games": int(target(d).sum()), "teams_exported": len(teams),
                   "teams_pruned_stale": pruned},
        "diag": {"intl_pool_mismatch": int(R["stats"]["intl_pool_mismatch"]),
                 "unlisted_lookalikes": diag.get("unlisted_lookalikes"),
                 "nonhome_crossregion": diag.get("nonhome_crossregion")},
    }


def offset_at(state: dict, league: str, t: float):
    """展示用: 赛区偏移衰减到 t 时的值 (不改状态)。没有跨赛区记录的赛区返回 None。"""
    v = state["leagues"].get(league)
    if not v or not v["has_crossregion_history"]:
        return None
    dt = t - v["last_t"]
    if dt <= 0:
        return v["o"]
    return v["prior"] + (v["o"] - v["prior"]) * math.exp(-LN2 * dt / state["params"]["hl"])


def predict(state: dict, league_key, blue, red, t, games=()):
    """看板的一步到位: 导出的状态 + 本赛事已完成局的重放 (Engine.play 的格式) → t 时刻蓝方胜率。

    这个赛区键没有系数、任一队母赛区查不到、或者两队同母赛区 → None (看板保持原来的行为)。
    blue / red / games 里的队名都必须已经映射成 OE 队名。
    """
    coef = state["coefficients"].get(str(league_key).upper())
    if coef is None:
        return None
    eng = Engine.from_state(state)
    eng.play(games)
    return eng.predict(coef, blue, red, t)


def sanity_check(state: dict) -> list[str]:
    """导出前 / 换上线前的健全性检查。返回问题列表, 空 = 通过。"""
    bad = []
    coefs = state.get("coefficients") or {}
    for key in LE_EVENT_CODES:
        c = coefs.get(key)
        if c is None:
            bad.append(f"缺 {key} 的系数")
            continue
        for k in ("a", "b_o", "b_r", "c"):
            v = c.get(k)
            if not isinstance(v, (int, float)) or not math.isfinite(v):
                bad.append(f"{key} 的系数 {k} 不是有限数: {v}")
    teams = state.get("teams") or {}
    if len(teams) < MIN_TEAMS:
        bad.append(f"导出的队伍只有 {len(teams)} 支 (下限 {MIN_TEAMS}) —— 很可能少加载了数据")
    for team, v in teams.items():
        if not math.isfinite(v["r"]):
            bad.append(f"{team} 的 r 不是有限数")
            break
    leagues = state.get("leagues") or {}
    for L in LEAGUES:
        n = sum(1 for v in teams.values() if v["pool"] == L)
        lv = leagues.get(L)
        if n < MIN_MAJOR_TEAMS:
            bad.append(f"四大赛区 {L} 只有 {n} 支队 (下限 {MIN_MAJOR_TEAMS})")
        if not lv or not lv["has_crossregion_history"] or not math.isfinite(lv["o"]):
            bad.append(f"四大赛区 {L} 没有有效的赛区偏移")
    for L, v in leagues.items():
        if v["has_crossregion_history"] and not all(math.isfinite(v[k]) for k in ("o", "prior", "last_t")):
            bad.append(f"赛区 {L} 的偏移不是有限数")
    look = (state.get("diag") or {}).get("unlisted_lookalikes")
    if look:
        bad.append(f"有没列入国际赛、却有跨赛区一级联赛对打的赛事 {look} —— OE 给某个杯赛 / 邀请赛用了新代码, "
                   f"它被当成了常规联赛, 参赛队的 Elo 池和母赛区整个挪了过去。先归类: 它的跨赛区局该计入 C 就加进 "
                   f"INTL_EVENTS (有 lolesports 键的同时改 LE_EVENT_CODES, 没有的加进 NO_LE_KEY), 只是杯赛就加进 NON_HOME")
    # 国际赛代码必须二选一 (见 NO_LE_KEY): 漏了 LE_EVENT_CODES, 那个赛区键的切点会落到全部数据之后, 把赛事自己的
    # OE 局也拟合进系数。只查 OE 最后一年: 切点只对当年的赛事有意义
    year = str(state.get("oe_asof") or "")[:4]
    mapped = {c for c in LE_EVENT_CODES.values() if c} | NO_LE_KEY
    for ev, v in (state.get("events") or {}).items():
        if str(v.get("year")) == year and v.get("code") not in mapped:
            bad.append(f"国际赛 {ev} 的代码 {v.get('code')} 既不在 LE_EVENT_CODES 里也不在 NO_LE_KEY 里 —— "
                       f"看板那个赛区键的系数切点会把这个赛事自己的局也拟合进去")
    if not state.get("oe_asof"):
        bad.append("缺 oe_asof")
    return bad


def artifact_path() -> Path:
    """$LOL_ARTIFACTS (日更训到暂存目录时设) 或 artifacts/ 下的 xregion.json。"""
    return Path(os.environ.get("LOL_ARTIFACTS") or (ROOT / "artifacts")) / "xregion.json"


def write_state(state: dict, path) -> Path:
    """写 JSON (先写临时文件再原子替换: 服务和日更可能同时读它)。NaN 直接报错 —— 浏览器的 JSON.parse 不认。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
                   encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_state(path=None) -> dict:
    return json.loads(Path(path or artifact_path()).read_text(encoding="utf-8"))


# ══════════════════════════════════════════════════════════
#  看板: 赛事内重放 (浏览器版 frontend/src/web/xregion.ts 逐位一致, test:golden 的 xregion_cases 对账)
# ══════════════════════════════════════════════════════════
#
# 为什么要重放见模块文档: 只用 OE 的"隔天"版本没过闸 (系列赛 t +2.02), 只有赛事内每打完一局就更新的版本过了。
# OE 晚约一天, 所以同一赛事里 OE 还没收录、lolesports 上已经打完的局由看板自己补进去。这一段全是纯函数:
# 取数 (赛程、终局帧) 在 api._xregion_board / esports_feed, 这里只吃取好的数据, 黄金用例才能逐条对账。
#
# 输入的"系列赛" (一场 lolesports 比赛) 形状:
#   {"match_id": str, "start": 赛程 startTime (ISO, 必须带 Z), "best_of": int | None,
#    "teams": [{"name": lolesports 队名, "oe": OE 队名或 None, "wins": 赢的局数, "id": esportsTeamId 或 None}] × 2,
#    "frames": {"<局号>": 终局帧 | None}}            ← 只有 replay_frames_needed 列出的局才需要
#   终局帧: {"blue_id", "red_id", "towers": [蓝, 红], "inhibitors": [蓝, 红], "gold": [蓝, 红]}
#
# 规则 (2026-10-04 定, 可行性见 research/gate_cross_region.py 文档):
#   · 哪些局: 同一赛事 (同一个 lolesports tournament) 里开赛时间早于当前系列赛的系列赛的全部已完成局, 加上
#     当前系列赛里局号小于当前局的已完成局。已完成的局数 = 两队比分之和 (games[].state 不可信, 见
#     CLAUDE.md)。OE 里已经有的局 (oe_lookup) 不重放 —— 它们已经在导出的状态里, 再放一遍就算了两次。
#     近似: 和当前系列赛同时或更晚开赛、却在当前这局之前打完的系列赛 (多路并行的赛程) 不放 —— 要放就得知道每个
#     系列赛几点打完 (每个系列赛多取一次详情和终局帧), 而用"现在已完赛"代替会把当前这局开打之后才结束的结果
#     也算进来 (开局渐变的锚在局中途跟着变)。实测 (HOLDOUT 450 局, 和闸门的在线版本比): 受影响 17 局,
#     mean|Δp| 0.0008、max 0.053, Brier Δ −0.00012 (系列赛 t +0.86, 不显著); 改成"放全部已完赛"也只到
#     Δ −0.00022 (t +1.17)。
#   · 当前系列赛的比分落后于局号 (第 number 局已经开打, 比分之和却 < number − 1: 上游登记比分比帧晚约 6 分钟)
#     时不给数 (withheld = "score_lag"): 少放的那一局没有任何信号, 给出的是另一个看着合理的数, 而开局头 3 分钟
#     那一行赛前数每局只留档一次。比分跟上之后照常给。
#   · 顺序和时间: 系列赛按赛程 startTime (相同再按 match_id)、局按局号。引擎时间 t = startTime + (局号 − 1) 秒 (game_t): 只用来
#     排序和衰减 (半衰期 365 天, 差几小时可以忽略), 不冒充真实开局时刻; 两个系列赛 startTime 相同时, 同局号的
#     两局 t 相同, 按引擎的规矩一起取赛前值再一起更新 (同时开打的局互相看不到)。
#   · 胜负 (decode_series): OE 里已有的局按 OE 的胜负, 先从比分里扣掉 (一个系列赛一半在 OE 里时, 剩下的局只在
#     剩下的配额里解码 —— 不扣的话, 状态里的局加上重放的局可能连系列赛胜负都反了); BO1 和横扫直接由比分定;
#     分出胜负的系列赛最后一局归系列赛胜者; 其余的局看终局帧 —— 塔多的赢, 塔一样看水晶, 再一样看总经济 (蓝方
#     是哪队看帧的 gameMetadata.esportsTeamId) —— 再按比分约束解码 (把握最大的判断先定, 哪队的胜局数用完了,
#     剩下判给它的局改判给对方)。有一局没有终局帧, 这个系列赛剩下的局就整体退回交替序 (领先方先赢, 平分时
#     teams[0] 先)。可行性: 47 局终局帧约束解码 47/47; 局序近似对 Brier 的影响 ≤ 0.0005。OE 的胜负和比分对不上
#     (OE 里某队赢的局比比分还多) 就抛 ValueError, 看板不给 C —— 不是去重认错了局, 就是比分错了。
#   · 选边: 用了终局帧的局按帧的蓝红; 比分定的局和交替序的局按 teams 的顺序 (h = 0, 选边只影响最后几位舍入)。
#     要预测的那一局还没有帧 (没开打) 时, 哪队蓝方不知道 —— 赛程里的队序不是选边, 而系数的截距 a ≈ +0.14 是
#     蓝方优势, 按队序当蓝方会让数随一个任意的顺序挪 5-7 个百分点, 热门都可能换边。这时给两种选边的平均
#     (sides_known = False): (p(A 蓝) + 1 − p(B 蓝)) / 2。有帧之后按帧的蓝红算。
#   · OE 截止之后的国内比赛不重放 (K = 2, 一局最多挪 2 分)。
# 历史回看 (比赛打完之后再打开看板) 用的仍是"现在"的状态: OE 已经收录的局 (哪怕在被看的这一局之后) 都在状态里,
# 和 Stage 1/2 在历史看板上用"现在"的队伍统计是同一个做法。

REPLAY_GAP_S = 1                  # 同一系列赛相邻两局在引擎里差 1 秒
DEDUP_BEFORE_S = 6 * 3600         # 去重: OE 那局的开打时间最早可以比赛程 startTime 早 6 小时 (改期提前)
DEDUP_AFTER_S = 18 * 3600         #       最晚可以晚 18 小时 (BO5 打满 + 延误; 跨 UTC 零点也在这个窗里)


def epoch_s(iso) -> int:
    """ISO 时间 (必须带时区, 上游的都带 Z) → 1970 起的整秒, 向下取整。浏览器版: Math.floor(Date.parse(iso) / 1000)。

    不带时区的串在 JS 里按本地时间解析, 在这里按 UTC —— 两边会差几个小时, 所以直接拒绝。
    """
    x = pd.Timestamp(iso)
    if x.tzinfo is None:
        raise ValueError(f"时间必须带时区: {iso!r}")
    return int(x.value // 10**9)


def game_t(start_iso, number) -> float:
    """系列赛第 number 局的引擎时间: (startTime 的整秒 + (局号 − 1) × REPLAY_GAP_S) / 86400。

    第 1 局和 to_t(startTime) 逐位相同 (整秒除以 86400, 同一次舍入)。浏览器版同一个式子。
    """
    return (epoch_s(start_iso) + (int(number) - 1) * REPLAY_GAP_S) / 86400


def series_wins(s) -> list[int]:
    return [int(t.get("wins") or 0) for t in s["teams"]]


def series_clinched(s) -> bool:
    """按赛制算是否已分出胜负 (和 api._clinched 同一个口径); best_of 不知道就当没分出。"""
    bo = s.get("best_of")
    return bool(bo) and max(series_wins(s)) >= int(bo) // 2 + 1


def frames_needed(s, known=None) -> list[int]:
    """这个系列赛解码要终局帧的局号: 有一方 0 胜 (BO1、横扫) 不要; 分出胜负的系列赛最后一局归胜者, 也不要;
    OE 里已有的局 (known 的键) 按 OE 的胜负, 也不要。"""
    wins = series_wins(s)
    n = sum(wins)
    if min(wins) == 0:
        return []
    last = n if series_clinched(s) else None
    kn = {int(i) for i in (known or {})}
    return [i for i in range(1, n + 1) if i != last and i not in kn]


def frame_verdict(frame, ids):
    """终局帧 → (胜方下标, 蓝方下标, 把握度) 或 None。下标指 teams 的顺序。

    塔多的赢; 塔一样看水晶; 再一样看总经济; 三样都一样 → None。蓝方是哪队只看帧的 esportsTeamId, 两个 id
    必须正好是这两队 (对不上就 None, 不猜)。把握度 = (|塔差|, |水晶差|, |经济差|), 按字典序比。
    """
    if not frame or not ids[0] or not ids[1]:
        return None
    bid, rid = frame.get("blue_id"), frame.get("red_id")
    if (bid, rid) == (ids[0], ids[1]):
        blue = 0
    elif (bid, rid) == (ids[1], ids[0]):
        blue = 1
    else:
        return None
    if any(not isinstance(frame.get(k), list) or len(frame[k]) != 2 for k in ("towers", "inhibitors", "gold")):
        return None
    d = [int(frame[k][0]) - int(frame[k][1]) for k in ("towers", "inhibitors", "gold")]
    lead = next((x for x in d if x != 0), 0)
    if lead == 0:
        return None
    return (blue if lead > 0 else 1 - blue), blue, (abs(d[0]), abs(d[1]), abs(d[2]))


def decode_series(s, known=None) -> list[dict]:
    """系列赛已完成的 n 局 (n = 比分之和) 各自的胜方: [{"number", "winner", "blue", "how"}], 局号 1..n。

    winner / blue 是 teams 的下标。how: "oe" (OE 里已有, 按 OE 的胜负和蓝红) / "score" (比分直接定) /
    "frames" (终局帧 + 比分约束) / "alternate" (有局没有终局帧, 交替序)。
    known = {局号: (胜方, 蓝方)} —— oe_lookup 的结果。先从比分里扣掉, 剩下的局只在剩下的配额里解码;
    OE 的胜负和比分对不上 (某队在 OE 里赢的比比分还多、或者决胜局之前就已经赢满) 抛 ValueError。
    规则见本段开头; 解码保证每队的胜局数等于比分。
    """
    wins = series_wins(s)
    n = sum(wins)
    if n == 0:
        return []
    out = {}
    quota = list(wins)
    for i, (w, bl) in sorted((int(i), v) for i, v in (known or {}).items()):
        if 1 <= i <= n:
            out[i] = (int(w), int(bl), "oe")
            quota[int(w)] -= 1
    if min(quota) < 0:
        raise ValueError(f"系列赛 {s.get('match_id')}: OE 里已有的局和比分 {wins} 对不上")
    if min(wins) == 0:
        w = 0 if wins[0] else 1
        for i in range(1, n + 1):
            out.setdefault(i, (w, 0, "score"))
        return [{"number": i, "winner": out[i][0], "blue": out[i][1], "how": out[i][2]} for i in range(1, n + 1)]
    if series_clinched(s) and n not in out:
        w = 0 if wins[0] > wins[1] else 1
        if quota[w] == 0:
            raise ValueError(f"系列赛 {s.get('match_id')}: OE 里胜者在决胜局之前就赢满了 (比分 {wins})")
        out[n] = (w, 0, "score")
        quota[w] -= 1
    need = frames_needed(s, known)
    frames = s.get("frames") or {}
    ids = [t.get("id") for t in s["teams"]]
    ver = {i: frame_verdict(frames.get(str(i)), ids) for i in need}
    if all(v is not None for v in ver.values()):
        # 把握最大的先定; 把握一样按局号
        for i in sorted(need, key=lambda i: (-ver[i][2][0], -ver[i][2][1], -ver[i][2][2], i)):
            w, blue, _k = ver[i]
            if quota[w] == 0:
                w = 1 - w
            quota[w] -= 1
            out[i] = (w, blue, "frames")
    else:
        pref = 0 if wins[0] >= wins[1] else 1
        for i in need:
            w = pref if quota[pref] > 0 else 1 - pref
            quota[w] -= 1
            v = ver[i]
            out[i] = (w, v[1] if v is not None else 0, "alternate")
            pref = 1 - pref
    return [{"number": i, "winner": out[i][0], "blue": out[i][1], "how": out[i][2]} for i in range(1, n + 1)]


def oe_index(state, aliases=None) -> dict:
    """OE 国际赛局的去重索引: {"by": {(队名, 局号): [(对手队名, 开打时间整秒, 这一方赢了, 这一方是蓝方), …]},
    "teams": 状态里的现役队名, "aliases": {旧名: 现名}}。

    不分赛事 (DCGI 将来进 OE 时叫什么代码、LE_EVENT_CODES 有没有跟上, 都不影响去重)。每局两个队名各记一条。
    aliases: esports_feed.TEAM_ALIASES (浏览器版: teams.json 的 aliases) —— 只用来认 OE 用过期旧名记的局, 见 oe_lookup。
    浏览器版: by 用 "队名\\u0000局号" 当键, teams 用 Set。
    """
    by = defaultdict(list)
    for ev in state["events"].values():
        for b, r, g, date, y, _st in ev["games"]:
            if g is not None:
                x = epoch_s(date)
                bw = int(y) == 1
                by[(b, int(g))].append((r, x, bw, True))
                by[(r, int(g))].append((b, x, not bw, False))
    return {"by": by, "teams": set(state.get("teams") or ()), "aliases": dict(aliases or {})}


def oe_lookup(idx, a, b, number, start_iso):
    """这一局 OE 里有没有 (= 已经计入状态)。有就返回 OE 的 (胜方, 蓝方) —— 下标 0 = a, 1 = b; 没有返回 None。

    同时满足:
      · 同一局号, OE 的开打时间落在 [startTime − 6 小时, startTime + 18 小时] 里 —— 不按日历日期比, 跨 UTC
        零点的系列赛会被拆开;
      · 两队对得上 (不论蓝红): 至少一方队名相同, 另一方要么也相同, 要么 OE 那边是个过期旧名 (不是状态里的
        现役队) 而且别名表把它指向另一方 (aliases[旧名] == 另一方)。
    后一条是实测出来的: 2026 EWC 上 OE 把 Team Secret Whales 记成过期的旧名 "Team Secret" (状态里早被剪掉),
    看板按别名映射成 "Team Secret Whales", 只认完全相同的队名就配不上 —— 那 4 局已经在状态里, 又被重放一遍。
    最早的版本对"任何不是现役队的名字"都放行, 会把同一天同局号、对手是另一支 (OE 用旧名记的) 队的比赛错认成
    这一局, 这一局就再也不重放了; 所以只认别名表里写明的那一个。
    窗口里命中几条时取开打时间离 startTime 最近的 (一样近取先出现的)。
    """
    s = epoch_s(start_iso)
    lo, hi = s - DEDUP_BEFORE_S, s + DEDUP_AFTER_S
    teams, al = idx["teams"], idx["aliases"]
    best = None
    for k, (me, other) in enumerate(((a, b), (b, a))):
        for opp, x, won, blue in idx["by"].get((me, int(number)), ()):
            if lo <= x <= hi and (opp == other or (opp not in teams and al.get(opp) == other)):
                d = abs(x - s)
                if best is None or d < best[0]:
                    best = (d, k if won else 1 - k, k if blue else 1 - k)
    return None if best is None else (best[1], best[2])


def in_oe(idx, a, b, number, start_iso) -> bool:
    """oe_lookup 命中与否 (条件见那里)。"""
    return oe_lookup(idx, a, b, number, start_iso) is not None


def _replay_selection(state, series, current_id, number, aliases=None):
    """[(系列赛下标, [要重放的局号…], {局号: OE 的 (胜方, 蓝方)})] 和计数 —— 还没去取终局帧、还没解码胜负。

    OE 的胜负 (known) 查的是系列赛的全部 n 局, 不只是要重放的那几局: 回看一个打完的系列赛的前几局时, 后面已在
    OE 里的局一样要从比分里扣掉。计数里 score_lag = 当前系列赛的比分落后于局号 (见本段开头), 调用方不给 C。
    """
    cur = next(s for s in series if s["match_id"] == current_id)
    cs = epoch_s(cur["start"])
    idx = oe_index(state, aliases)
    sel, deduped, unmapped, lag = [], 0, 0, False
    for k, s in enumerate(series):
        if len(s["teams"]) != 2:
            continue
        n = sum(series_wins(s))
        if s["match_id"] == current_id:
            lag = lag or int(number) - 1 > n
            hi = min(n, int(number) - 1)
        elif epoch_s(s["start"]) < cs:
            hi = n
        else:
            continue
        if hi <= 0:
            continue
        a, b = s["teams"][0].get("oe"), s["teams"][1].get("oe")
        if not a or not b:
            unmapped += hi                      # 映射不出 OE 队名: 引擎里也没有母赛区, 本来就不会更新
            continue
        known = {}
        for i in range(1, n + 1):
            hit = oe_lookup(idx, a, b, i, s["start"])
            if hit is not None:
                known[i] = hit
        keep = [i for i in range(1, hi + 1) if i not in known]
        deduped += hi - len(keep)
        if keep:
            sel.append((k, keep, known))
    return sel, {"deduped": deduped, "unmapped": unmapped, "score_lag": lag}


def replay_frames_needed(state, series, current_id, number, aliases=None) -> list[tuple[str, int]]:
    """要去取终局帧的局: [(match_id, 局号)]。只看有局要重放的系列赛 —— 整个系列赛都在 OE 里就一个请求都不发。
    一个系列赛只要有一局要重放, 解码就要它全部的非定局 (比分约束是对整个系列赛的), OE 里已有的局除外。
    当前系列赛比分落后 (score_lag, 反正不给 C) 时一个都不取。"""
    sel, counts = _replay_selection(state, series, current_id, number, aliases)
    if counts["score_lag"]:
        return []
    return [(series[k]["match_id"], i) for k, _keep, known in sel for i in frames_needed(series[k], known)]


def board_replay(state, series, current_id, number, aliases=None) -> dict:
    """要重放的局 (已解码、已排序, Engine.play 的输入) 和计数。number = 当前局的局号。"""
    sel, counts = _replay_selection(state, series, current_id, number, aliases)
    games = []
    for k, keep, known in sel:
        s = series[k]
        dec = {d["number"]: d for d in decode_series(s, known)}
        names = [s["teams"][0]["oe"], s["teams"][1]["oe"]]
        start = epoch_s(s["start"])
        for i in keep:
            d = dec[i]
            games.append({"t": game_t(s["start"], i), "blue": names[d["blue"]], "red": names[1 - d["blue"]],
                          "blue_win": 1 if d["winner"] == d["blue"] else 0, "match_id": s["match_id"],
                          "number": i, "how": d["how"], "_key": (start, str(s["match_id"]))})
    # 排序键: t, 开赛时间, match_id, 局号 —— 不用输入列表里的位置: 同一时刻开打的两个系列赛, 组内的更新顺序
    # (浮点累加的顺序) 不能随上游返回的顺序变
    games.sort(key=lambda g: (g["t"], g["_key"][0], g["_key"][1], g["number"]))
    for g in games:
        del g["_key"]
    return {"games": games, **counts}


def board_predict(state, league_key, blue, red, series, current_id, number, aliases=None,
                  sides_known=True) -> dict:
    """看板的跨赛区赛前数: 导出的状态 + 本赛事重放 → 第 number 局开打时的蓝方胜率。

    blue / red 是 OE 队名。probability_blue 为 None = 不给 (没有这个赛区键的系数、母赛区查不到或过期、
    两队同母赛区, 或者 withheld 非空) —— 看板保持原来的行为。
      withheld      "score_lag" = 当前系列赛的比分还没跟上局号 (见本段开头); 否则 None
      side_neutral  sides_known = False (这一局还没有帧, 不知道谁蓝方) 时给两种选边的平均, 这里是 True
      no_history    两队母赛区里在导出的状态中没有跨赛区国际赛记录的 (先验占大头), 看板据此加一句说明
    aliases: 去重认 OE 旧名用的别名表 (见 oe_lookup)。
    """
    rep = board_replay(state, series, current_id, number, aliases)
    cur = next(s for s in series if s["match_id"] == current_id)
    t = game_t(cur["start"], number)
    eng = Engine.from_state(state)
    n_upd = eng.play(rep["games"])
    hb, hr = eng.home(blue, t), eng.home(red, t)
    coef = state["coefficients"].get(str(league_key).upper())
    withheld = "score_lag" if rep["score_lag"] else None
    p = None
    f = eng.features(blue, red, t) if coef is not None and withheld is None else None
    if f is not None:
        ob, orr, rb, rr = f
        p = logistic(coef, ob, orr, rb, rr)
        if not sides_known:
            p = (p + (1 - logistic(coef, orr, ob, rr, rb))) / 2
    lg = state["leagues"]
    no_hist = [L for L in (hb, hr) if L is not None and not (lg.get(L) or {}).get("has_crossregion_history")]
    return {"probability_blue": p, "t": t, "home_blue": hb, "home_red": hr, "no_history": no_hist,
            "withheld": withheld, "side_neutral": not sides_known,
            "replayed": len(rep["games"]), "updated": n_upd, "deduped": rep["deduped"],
            "unmapped": rep["unmapped"], "games": rep["games"]}


def _summary(state: dict) -> list[str]:
    t = state["oe_asof_t"]
    offs = {L: offset_at(state, L, t) for L, v in state["leagues"].items() if v["has_crossregion_history"]}
    lines = [f"OE 截至 {state['oe_asof']}  ({state['counts']['games']} 局, 国际赛 {state['counts']['intl_games']}, "
             f"跨赛区 {state['counts']['crossregion_intl_games']})",
             f"队伍 {state['counts']['teams_exported']} 支 (过期没导出 {state['counts']['teams_pruned_stale']} 支), "
             f"赛区 {len(state['leagues'])} 个, 有跨赛区记录 {len(offs)} 个",
             "赛区偏移 (衰减到 OE 截止, Elo 分): " + ", ".join(
                 f"{L} {v:+.0f}" for L, v in sorted(offs.items(), key=lambda kv: -kv[1]))]
    for k, c in state["coefficients"].items():
        lines.append(f"系数 {k:<18} a {c['a']:+.4f}  b_o {c['b_o']:.4f}  b_r {c['b_r']:.4f}  "
                     f"切点 {c['cutoff'] or '全部数据之后'} ({c['event'] or '-'}, 训练 {c['n_train']} 局)")
    for x in (state.get("diag") or {}).get("nonhome_crossregion") or []:
        lines.append(f"⚠ 杯赛 {x['event']} 里有 {x['cross_first_division']}/{x['n']} 局是跨赛区的一级队对打, 不计入状态 "
                     f"(不拦导出; 要计入, 得按年份把这个代码单独归成国际赛 —— 同一代码下的国内杯赛不能跟着进去)")
    return lines


def main(argv=None) -> int:
    for _s in (sys.stdout, sys.stderr):              # 计划任务里控制台是 GBK, 打 ✓ ✗ 会抛 UnicodeEncodeError
        if _s is not None and hasattr(_s, "reconfigure"):
            _s.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="跨赛区模型 C 的状态导出")
    ap.add_argument("--build", action="store_true", help="从五年 CSV 构建状态并写出")
    ap.add_argument("--asof", default=None, help="只用严格早于这个时间 (UTC) 的局")
    ap.add_argument("--out", default=None, help="输出文件 (默认 $LOL_ARTIFACTS 或 artifacts/ 下的 xregion.json)")
    ap.add_argument("--check", default=None, metavar="FILE", help="只检查一个已有的状态文件")
    a = ap.parse_args(argv)
    if a.check:
        state = load_state(a.check)
        bad = sanity_check(state)
        for line in _summary(state):
            print(f"  {line}")
        for b in bad:
            print(f"  ✗ {b}")
        print("  ✓ 健全性检查通过" if not bad else "  ✗ 健全性检查不通过")
        return 1 if bad else 0
    if not a.build:
        ap.print_help()
        return 2
    print("[xregion] 构建跨赛区模型 C 的状态")
    state = build_state(asof=a.asof, verbose=True)
    for line in _summary(state):
        print(f"  {line}")
    bad = sanity_check(state)
    if bad:
        for b in bad:
            print(f"  ✗ {b}")
        print("  ✗ 健全性检查不通过, 没有写文件")
        return 1
    out = write_state(state, a.out or artifact_path())
    print(f"  ✓ 写出 {out} ({out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
