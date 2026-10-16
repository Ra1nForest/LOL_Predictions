"""
闸门 G1: Stage 1/2 组合包 —— 浅树 + 次级联赛训练行 + 选手-英雄胜率收缩 + 赛区内 Elo 进校准层
==========================================================================================

[复核, 2026-10-17] 原先用 --jobs 6 跑, 复核改成默认不强制 n_jobs (= 和 train.py 一样不传), 判决以默认线程为准。
  原因: XGBoost 3.4.1 的 subsample / colsample 随机流按线程切分, 同一个 random_state 在 n_jobs=6 / 12 / 1
  下训出的树不同 (同一块同一种子, 原始概率最大差 0.35-0.40; 复核脚本 r9_threads.py)。原脚本的 --jobs 6 因此
  等于换了一套随机数, B0 不是 train.py 那条代码路径 (train.py 不传 n_jobs; 用闸门的 M 按 train.py 的切分在默认
  线程数下重训, coef / intercept / Brier 与 artifacts/calib_*.json 逐位相同)。协议没规定线程数, 所以两种跑法
  都"合规"; 但 B0 写的是"train.py 的参数, 原样", 默认线程才是线上那条路径。
  复核结论: 主判决 (P 过闸) 对线程数 / 种子不敏感; pre_draft 上 (a) 的逐项剔除 t 是随机数决定的 (同一协议
  三种随机流: +1.03 / −0.35 / +2.27), 简洁规则对它的取舍不可靠。默认线程重跑: pre P 块 t +6.47、post +4.86, 配方仍是 {a, d}。

为什么做成一个"包"
------------------
2026-10 的模型复查 (scratch 里的探针, 不是闸) 在 Stage 1/2 上分别看到四个方向的收益:

  (a) 树太深。约 5 千行四大训练集上 400 棵 depth-5 的树, 原始输出的校准斜率只有 ~0.5,
      不同种子之间特征重要性的 Spearman 只有 ~0.5 —— 典型的过拟合。
  (b) 次级联赛 3.6 万局没用上。Stage 4 加次级行已经过闸上线 (t = +5.19), Stage 1/2 还只看四大。
  (c) 选手-英雄胜率收缩 08 月就过了闸 (K=20 t+3.79, K=40 t+3.96, 见 champ_shrinkage_kscan.log),
      但 feature_store.player_champ_stats 仍是 len >= 3 的硬阈值, 一直没上线。
  (d) 10 局滚动窗口看不到长期实力: 赛区内 Elo 是线上残差里最强的信号 (系列赛聚类 z ≈ +7)。

四项分别过闸就是四次多重比较, 而且可能各自过闸、叠起来却不相加 (探针里 Elo 作 XGB 特征只拿到
单独用 Elo 时一半的收益)。所以预先登记成一个包 P 整体过闸, 再用逐项剔除 P−{x} 按一条事先写死的
"简洁规则" 决定哪一项可以从上线配方里省掉。

冻结的协议 (跑之前写死; 不在评估数据上调任何东西)
------------------------------------------------
基线 B0 = 现在线上的配方, 每块重训: train.py 的 XGB 参数和特征 (select_features), 选手-英雄胜率
  `len >= 3` 硬阈值, 只用四大的训练行, Platt 校准 (LogisticRegression(C=1e6) 吃原始概率)。
  B0 的四大矩阵和 feature_store.build_training_matrix 逐格相等 (verify_majors, 每次运行都核对)。

候选 P = B0 加上全部四项:
  (a) XGB 参数 max_depth=3, learning_rate=0.03, n_estimators=300, min_child_weight=10, 其余照 train.py;
  (b) 训练行加上非四大的**常规**联赛 (国际赛和杯赛不要: xregion.is_international / xregion.NON_HOME),
      每个联赛的特征只在它自己的池子里、用同一份代码算; 评估只在四大的比赛上。
      四大行的"池子"仍是线上那个四大合池 —— 改它就不是 B0 了;
  (c) 选手-英雄胜率收缩 wr = (胜 + K/2) / (场 + K), K = 40, 替掉硬阈值 (同样那 10 列, 别的不动;
      0 场时是 0.5 而不是 NaN);
  (d) 赛区内队伍 Elo: 每个联赛一个池 (次级联赛也有), K=20, 初始 1500, 每个日历年 (UTC) 交界时
      向池内均值回归 50%, 不带比分差项; 一局的 Elo 输入只用**更早 UTC 日期**的比赛 (同一天的结果
      不进, 对齐 OE 晚一天登记)。接法: 校准层把 Platt 的单输入换成 logistic 回归
      [logit(p_xgb_raw), diff_elo · ln(10)/400], 在校准段上拟合。Elo 不进树。

切块: 2024-07-01 到数据末尾的四大比赛按日期切成 10 个连续、局数相等的测试块 (扩展窗口)。每块:
  训练 = 块起点之前的全部行 (该配方的训练池), 按日期最早的 75% 训树, 最近的 25% 拟合校准层。
  种子 0, 1, 2; 每局对三个种子的校准后概率取平均。
  · "全部行"对带 (b) 的配方包括次级行, 所以它们的校准段里也有次级行 —— (d) 写明"次级联赛也有
    Elo 池", 只有校准段里有次级行时才用得上, 这是按字面读的。只用校准段里四大行拟合校准层的版本
    作为口径敏感性单独报告, 不参与判决。

判决 (两个阶段分开): P 过闸当且仅当 块级配对 t (Brier, B0 − P) > 2.5 且 块级 t (log loss) > 0。
  另报准确率、AUC、校准斜率, 按赛区、季后赛 / 常规赛、系列赛第一局拆开。

归因 (只报告, 只用于预先登记的简洁规则): 逐项剔除 P−{a}, P−{b}, P−{c}, P−{d}。
  规则: 某一项从上线配方里去掉, 当且仅当去掉它对 Brier 的块级 t < 1.0 (P 更好的方向为正), 即它没帮上忙。
  另报逐项加入 B0+{x}。
  (c) 只动 draft 列, pre_draft 看不到它 —— pre_draft 上 P−{c} 与 P、B0+{c} 与 B0 逐位相同, t = 0。

另报: 建次级联赛矩阵的耗时和内存。

结论 (2026-10-17, 数据截至 2026-10-13 的四大比赛; 4222 局分 10 块, 每块 422-423 局)
------------------------------------------------------------------------------
B0 矩阵与 build_training_matrix 逐位相等 (8869 局、两个阶段的特征列表和顺序、每一格; 现场重建核对)。
Brier 的 Δ 取 B0 − 候选, 正数表示候选更好; t 是 10 块的配对 t。

  pre_draft   P 过闸: Brier 0.2348 → 0.2200 (Δ +0.0148, t = +6.41, 10/10 块), log loss t = +6.03,
              准确率 60.5% → 64.8% (t +9.21), AUC 0.635 → 0.696, 校准斜率 0.94 → 0.87。
  post_draft  P 过闸: Brier 0.2302 → 0.2194 (Δ +0.0108, t = +4.90, 9/10), log loss t = +4.81,
              准确率 61.3% → 65.1% (t +4.60), AUC 0.656 → 0.699, 校准斜率 0.97 → 0.84。

逐项剔除 (P 相对 P−{x} 的 Brier 块 t) 和简洁规则 (t < 1.0 就去掉):
  pre_draft   a +1.03 (5/10, 保留, 刚过线)   b +0.11 去掉   c 0 (看不到, 去掉)   d +7.66 保留
  post_draft  a +1.61 (8/10, 保留)           b +0.15 去掉   c −0.14 去掉        d +4.46 保留
  → 两个阶段的上线配方都是 {a, d}: 浅树 + Elo 进校准层。次级联赛行和选手-英雄胜率收缩都不上线。
  次级行单独加在 B0 上是有用的 (pre t +2.95, post t +1.96), 但 Elo 进来以后就不剩什么了;
  收缩在新底座上没有作用 (加在 B0 上 post t +1.31, 从 P 里去掉 t −0.14)。
  全部收益几乎都来自 (d): B0+{d} 一项就有 pre t +6.43 / post t +4.42。

补充臂 B0+a+d (第一轮跑完后才加, 不参与判决; 它的树和 B0+a 相同, 加它不改变其他臂的任何数字):
  pre_draft   Brier 0.2201 (t +6.55, 10/10), 准确率 65.5%, 斜率 0.92 (它就是 P−{b})
  post_draft  Brier 0.2196 (t +4.67, 9/10), log loss t +4.66, 准确率 65.5%, 斜率 0.93
  相对 P: Brier −0.0001 (t −0.11 / −0.15), 校准斜率反而更接近 1 (P 的校准段里约 77% 是次级行,
  校准层的系数主要由次级行定, 对四大偏自信)。相对 B0+{d}: pre t +2.61, post t +1.32 —— (a) 在 Elo
  之上的增量很小。2026 年单独看收益约少三成 (配方 pre Δ +0.0108 t +3.28, post Δ +0.0069 t +2.19)。

次级联赛矩阵: 52 个联赛 38212 局 → 35240 行, 约 17-22 s; tracemalloc 峰值约 78 MB, 矩阵 42 MB,
  次级原始行 (只读需要的列) 266 MB; 进程峰值工作集 688 MB, 由读 CSV 决定, 不是建矩阵。

注意: 冻结的常数 (K=20、回归 0.5、浅树参数) 来自之前看过 2024-06 以后数据的探针, 这次是正式的
预先登记复核, 但不是对这些常数完全样本外的检验 —— 上线前仍建议按探针的建议做几周前向影子对比。

用法 (从项目根目录; 只用项目 venv)::

    .venv\\Scripts\\python.exe research/gate_pregame_package.py --out <输出目录> [--cache <缓存目录>]
        [--verify fresh|skip|<build_training_matrix 产出的 pickle>] [--jobs 6]

输出目录里: result.json (全部数字), report.md (表格), preds.pkl (每局每臂的概率), run.log。
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import math
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "research"))

import xregion as XR
from feature_store import (LEAGUES, MIN_ROLL_PERIODS, ROLES, ROLL_STATS, ROLL_WINDOW, SUM_FEATS,
                           build_training_matrix, select_features)
from harness import paired_t
from multiyear_gate import data_path
import train   # 只取 XGB 参数和 DATA。导入时 OUT.mkdir(exist_ok=True): artifacts/ 已存在, 是空操作

# ══════════════════════════════════════════════════════════
#  冻结的常数 (协议的一部分, 不许按结果改)
# ══════════════════════════════════════════════════════════
YEARS = (2022, 2023, 2024, 2025, 2026)
A_PARAMS = dict(max_depth=3, learning_rate=0.03, n_estimators=300, min_child_weight=10)
K_SHRINK = 40.0
ELO_K, ELO_INIT, ELO_REG = 20.0, 1500.0, 0.5
EVAL_START = pd.Timestamp("2024-07-01")
N_BLOCKS = 10
SEEDS = (0, 1, 2)
TREE_FRAC = 0.75
T_PASS = 2.5          # Brier 块级 t 必须 > 2.5
T_KEEP = 1.0          # 简洁规则: 剔除某项后 Brier 的 t < 1.0 就从上线配方里去掉
LN10_400 = math.log(10) / 400

COMPONENTS = ("a", "b", "c", "d")
ARMS = {"B0": "", "P": "abcd",
        "P-a": "bcd", "P-b": "acd", "P-c": "abd", "P-d": "abc",
        "B0+a": "a", "B0+b": "b", "B0+c": "c", "B0+d": "d"}
# 补充臂, 第一轮跑完之后才加的, **不参与任何判决**: 简洁规则在两个阶段都给出配方 {a, d}, 而这个组合不在
# 预先登记的臂里 (P−{b} 在 pre_draft 上恰好就是它, post_draft 上不是)。它的树和 B0+a 完全相同, 只是校准层
# 换成 (d), 一棵树都不多训 —— 加它不改变其他任何一臂的数字。
SUPPLEMENTARY = {"B0+a+d": "ad"}
ARMS = {**ARMS, **SUPPLEMENTARY}
RECIPE = "B0+a+d"
STAGES =("pre_draft", "post_draft")
WR_COLS = [f"{s}_{r}_champ_wr" for s in ("b", "r") for r in ROLES]     # (c) 只动这 10 列
SHR_COLS = [f"shr_{c}" for c in WR_COLS]                             # shr_ 前缀: select_features 不会选

RAW_COLS = (["gameid", "league", "date", "game", "playoffs", "participantid", "side", "position",
             "playername", "teamname", "champion", "result", "gamelength", "teamkills", "teamdeaths"]
            + list(ROLL_STATS))


# ══════════════════════════════════════════════════════════
#  小工具
# ══════════════════════════════════════════════════════════
class Log:
    def __init__(self, path: Path | None):
        self.f = path.open("a", encoding="utf-8") if path else None

    def __call__(self, msg=""):
        print(msg, flush=True)
        if self.f:
            self.f.write(str(msg) + "\n")
            self.f.flush()


def mem_mb() -> tuple[float, float]:
    """(当前工作集, 进程峰值工作集) MB。Windows 用 GetProcessMemoryInfo, 别处用 ru_maxrss。"""
    if sys.platform == "win32":
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        k32 = ctypes.WinDLL("kernel32")
        psapi = ctypes.WinDLL("psapi")
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
        return pmc.WorkingSetSize / 2 ** 20, pmc.PeakWorkingSetSize / 2 ** 20
    import resource
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return float("nan"), r


def data_files() -> list[str]:
    """CSV 路径只从 multiyear_gate.data_path 来, 并且必须和 train.DATA 是同一份五年列表。"""
    files = [data_path(y) for y in YEARS]
    if [Path(f).resolve() for f in files] != [Path(f).resolve() for f in train.DATA]:
        raise RuntimeError("闸门的年份列表和 train.DATA 不一致 —— 少加载一年不会报错, 只会悄悄改变结果")
    for f in files:
        if not Path(f).exists():
            raise FileNotFoundError(f)
    return files


def data_fingerprint(files) -> str:
    """缓存键 = 数据文件 (名字, 大小, 修改时间) + 本脚本的源码: 改了构建代码, 旧缓存自动作废。"""
    h = hashlib.sha1(Path(__file__).read_bytes())
    for f in files:
        st = Path(f).stat()
        h.update(f"{Path(f).name}:{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()[:16]


def load_raw(files) -> pd.DataFrame:
    """全部赛区五年, 只读需要的列 (省内存)。LTA 改名那一行和 feature_store 完全相同。"""
    dfs = []
    for f in files:
        hdr = pd.read_csv(f, nrows=0).columns
        dfs.append(pd.read_csv(f, usecols=[c for c in RAW_COLS if c in hdr], low_memory=False))
    raw = pd.concat(dfs, ignore_index=True)
    del dfs
    raw["league"] = raw["league"].replace({"LTA N": "LCS", "LTA S": "CBLOL"})
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    if raw["date"].isna().any():
        raise RuntimeError(f"{int(raw['date'].isna().sum())} 行日期解析失败 —— 线上构建器对 NaT 的处理没核对过, 先停")
    return raw


def minor_mask(raw: pd.DataFrame) -> np.ndarray:
    """(b) 的次级常规联赛: 非四大、非国际赛 (按年份)、非杯赛 / 次级跨区赛 (xregion.NON_HOME)。"""
    yrs = raw["date"].dt.year.values
    lg = raw["league"].values
    intl = np.fromiter((XR.is_international(c, y) for c, y in zip(lg, yrs)), bool, len(raw))
    return (~raw["league"].isin(LEAGUES).values) & (~intl) & (~raw["league"].isin(XR.NON_HOME).values)


# ══════════════════════════════════════════════════════════
#  训练矩阵 (build_training_matrix 的向量化版本)
# ══════════════════════════════════════════════════════════

def _prior(p: pd.DataFrame, keys: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """每个选手行: 同一 key 下日期**严格早于**本行日期的记录数和胜场数。
    等价于 FeatureStore.champ_winrate / player_champ_stats 里的 `d < before` (本局自己同日期, 不算)。
    计数和胜场都是整数, 所以 胜/场 与线上 np.mean(0/1 列表) 逐位相同。"""
    s = p.sort_values(keys + ["date", "_ord"], kind="stable")
    g1 = s.groupby(keys, sort=False, dropna=False)
    g2 = s.groupby(keys + ["date"], sort=False, dropna=False)
    r = s["result"].to_numpy(dtype=np.int64)
    w = (g1["result"].cumsum().to_numpy(dtype=np.int64) - r) - (g2["result"].cumsum().to_numpy(dtype=np.int64) - r)
    n = g1.cumcount().to_numpy() - g2.cumcount().to_numpy()
    on, ow = np.empty(len(p), np.int64), np.empty(len(p), np.int64)
    pos = s["_ord"].to_numpy()
    on[pos], ow[pos] = n, w
    return on, ow


def build_pool(df: pd.DataFrame, k_shrink: float = K_SHRINK) -> pd.DataFrame:
    """一个池的训练矩阵。四大合池上与 feature_store.build_training_matrix 逐格相等 (verify_majors 核对)。

    逐行逻辑照抄线上 (含边角: 每方 >= 5 个选手行才算 draft, 同一位置多行时后一行覆盖, 列顺序 =
    线上 DataFrame(rows) 的首次出现顺序), 只是把逐行的二分查找换成了分组累计。另外带出:
      game_no      系列赛内局号 (只作切片用; 不以 diff_/b_/r_/sum_/is_ 开头, select_features 不会选)
      shr_*        (c) 收缩版的 10 列选手-英雄胜率
    """
    # 排序必须和线上一样用默认的 quicksort (不稳定): 同一局 10 个选手行在 df 里的先后由它决定, 而这个先后
    # 决定 (1) draft 列在 DataFrame(rows) 里首次出现的顺序 = 特征顺序 (colsample_bytree 按列号抽列, 顺序
    # 不同树就不同), (2) 英雄胜率取平均时的加法顺序 (末位差 1e-16)。输入行序和线上相同 (五个 CSV 依次拼接),
    # 所以同一个排序给出同一个置换 —— verify_majors 逐位核对。
    df = df.sort_values("date").reset_index(drop=True)
    teams = df[df["position"] == "team"].copy()
    game_order = pd.unique(teams["gameid"].values)        # 线上逐局循环的顺序 (teams["gameid"].unique())
    teams["result"] = teams["result"].astype(int)
    gl = pd.to_numeric(teams["gamelength"], errors="coerce")
    teams["gamelength_min"] = gl / 60 if gl.max() > 200 else gl
    teams["game_total_kills"] = (pd.to_numeric(teams["teamkills"], errors="coerce")
                                 + pd.to_numeric(teams["teamdeaths"], errors="coerce"))
    teams = teams.sort_values(["teamname", "date"], kind="stable")
    g = teams.groupby("teamname", sort=False)

    def roll(s):
        return s.rolling(ROLL_WINDOW, min_periods=MIN_ROLL_PERIODS).mean().shift(1)

    for c in ROLL_STATS:
        if c in teams.columns:
            teams[f"avg_{c.replace(' ', '_')}"] = g[c].transform(lambda s: roll(pd.to_numeric(s, errors="coerce")))
    teams["rolling_wr"] = g["result"].transform(roll)
    st = np.zeros(len(teams))
    prev, s = None, 0
    for i, (tm, r) in enumerate(zip(teams["teamname"].values, teams["result"].values)):
        if tm != prev:
            s, prev = 0, tm
        st[i] = s
        s = (s + 1 if s > 0 else 1) if r == 1 else (s - 1 if s < 0 else -1)
    teams["streak"] = st
    teams["avg_gamelen"] = g["gamelength_min"].transform(roll)
    teams["avg_game_kills"] = g["game_total_kills"].transform(roll)
    roll_cols = [c for c in teams.columns if c.startswith("avg_") or c in ("rolling_wr", "streak")]

    tt = teams.sort_values(["gameid", "participantid"], kind="stable")
    tt = tt[tt.groupby("gameid")["gameid"].transform("size") == 2]
    b = tt.groupby("gameid").nth(0).set_index("gameid")
    b = b.loc[[gid for gid in game_order if gid in b.index]]
    r = tt.groupby("gameid").nth(1).set_index("gameid").loc[b.index]
    out = pd.DataFrame({"gameid": b.index.values, "date": b["date"].values, "league": b["league"].values,
                        "blue_team": b["teamname"].values, "red_team": r["teamname"].values,
                        "y": b["result"].values.astype(int)})
    for lg in ("lpl", "lck", "lec", "lcs"):
        out[f"is_{lg}"] = (out["league"] == lg.upper()).astype(int).values
    out["playoffs"] = pd.to_numeric(b["playoffs"], errors="coerce").fillna(0).astype(int).values
    for c in roll_cols:
        bv, rv = b[c].astype(float).values, r[c].astype(float).values
        out[f"diff_{c}"], out[f"b_{c}"], out[f"r_{c}"] = bv - rv, bv, rv
        if c in SUM_FEATS:
            out[f"sum_{c}"] = bv + rv
    game_no = pd.to_numeric(b["game"], errors="coerce").values

    # ── draft: 英雄 × 赛区胜率 (>= 5 场), 选手 × 英雄 (硬阈值 >= 3 / 收缩) ──
    players = df[df["position"].isin(ROLES)].copy()
    players["result"] = players["result"].astype(int)
    players["_ord"] = np.arange(len(players))
    n1, s1 = _prior(players, ["champion", "league"])
    n2, s2 = _prior(players, ["playername", "champion"])
    cw = np.where(n1 >= 5, s1 / np.maximum(n1, 1), np.nan)
    pw = np.where(n2 >= 3, s2 / np.maximum(n2, 1), np.nan)
    pws = (s2 + k_shrink * 0.5) / (n2 + k_shrink)
    side, pos = players["side"].values, players["position"].values
    gidx = players.groupby("gameid", sort=False).indices
    recs, shr = [], np.full((len(out), len(WR_COLS)), np.nan)
    col_i = {c: i for i, c in enumerate(WR_COLS)}
    for gi, gid in enumerate(out["gameid"].values):
        m = {}
        ii = gidx.get(gid)
        if ii is not None:
            bi, ri = ii[side[ii] == "Blue"], ii[side[ii] == "Red"]
            if len(bi) >= 5 and len(ri) >= 5:
                bw, rw, bc, rc = [], [], [], []
                for sd, sub, accw, accc in (("b", bi, bw, bc), ("r", ri, rw, rc)):
                    for j in sub:
                        if not np.isnan(cw[j]):
                            accw.append(cw[j])
                        accc.append(int(n2[j]))
                        m[f"{sd}_{pos[j]}_champ_wr"] = pw[j]
                        m[f"{sd}_{pos[j]}_champ_games"] = int(n2[j])
                        shr[gi, col_i[f"{sd}_{pos[j]}_champ_wr"]] = pws[j]
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
        recs.append(m)
    out = pd.concat([out, pd.DataFrame(recs, index=out.index)], axis=1)
    out["game_no"] = game_no
    for c, sc in zip(WR_COLS, SHR_COLS):
        out[sc] = shr[:, col_i[c]]
    core = [c for c in out.columns if c.startswith("diff_avg_")][:5]
    out = out.dropna(subset=core + ["y"]).reset_index(drop=True)
    return out


def verify_majors(M: pd.DataFrame, prod: pd.DataFrame, log) -> dict:
    """B0 = 线上配方的证据: 与 build_training_matrix 同一批局、两个阶段的特征列表和顺序相同、每一格相等。"""
    A, B = prod.set_index("gameid"), M.set_index("gameid")
    only_a, only_b = A.index.difference(B.index), B.index.difference(A.index)
    res = {"n_prod": len(prod), "n_fast": len(M), "only_prod": len(only_a), "only_fast": len(only_b)}
    for stg in STAGES:
        fa, fb = select_features(prod, stg), select_features(M, stg)
        res[f"features_equal_{stg}"] = fa == fb
        res[f"n_features_{stg}"] = len(fa)
    ids = A.index.intersection(B.index)
    feats = select_features(prod, "post_draft")
    bad, maxdiff = {}, 0.0
    for c in feats + ["y"]:
        x, z = A.loc[ids, c].astype(float).values, B.loc[ids, c].astype(float).values
        same = (x == z) | (np.isnan(x) & np.isnan(z))
        if not same.all():
            bad[c] = int((~same).sum())
            fin = np.isfinite(x) & np.isfinite(z)
            if fin.any():
                maxdiff = max(maxdiff, float(np.abs(x[fin] - z[fin]).max()))
    res["mismatch_cols"] = bad
    res["max_abs_diff"] = maxdiff
    # (c) 的收缩列: 有 >= 3 场的格子可以从线上的 (胜率, 场数) 反推胜场, 核对公式
    chk = []
    for c, sc in zip(WR_COLS, SHR_COLS):
        g = A.loc[ids, c.replace("_wr", "_games")].astype(float).values
        w = A.loc[ids, c].astype(float).values
        ok = np.isfinite(w) & (g >= 3)
        want = (np.round(w[ok] * g[ok]) + K_SHRINK / 2) / (g[ok] + K_SHRINK)
        chk.append(float(np.abs(B.loc[ids, sc].values[ok] - want).max()) if ok.any() else 0.0)
    res["shrink_formula_maxdiff"] = max(chk)
    ok = (not bad and not len(only_a) and not len(only_b)
          and all(res[f"features_equal_{s}"] for s in STAGES) and res["shrink_formula_maxdiff"] < 1e-12)
    res["ok"] = bool(ok)
    log(f"  核对 B0 矩阵: 线上 {len(prod)} 局 / 向量化 {len(M)} 局, 只在一边 {len(only_a)}/{len(only_b)}, "
        f"不相等的列 {bad or '无'}, 特征列表相同 {[res[f'features_equal_{s}'] for s in STAGES]}, "
        f"收缩公式偏差 {res['shrink_formula_maxdiff']:.1e} → {'通过' if ok else '失败'}")
    return res


# ══════════════════════════════════════════════════════════
#  (d) 赛区内 Elo
# ══════════════════════════════════════════════════════════

def elo_inputs(team_rows: pd.DataFrame) -> dict:
    """每个联赛一个池。返回 gameid -> (蓝方 Elo − 红方 Elo), 只用更早 UTC 日期的比赛。

    · 初始 1500, K = 20, 胜负 0/1, 不带比分差。
    · 每跨过一个日历年 (UTC) 交界, 池内所有队向池内均值回归 50% (跨过 n 个交界就是 0.5^n)。
      新队从 1500 进、更新零和, 所以池内均值恒为 1500 (浮点误差内) —— 照字面算均值, 不假设它。
    · 同一天的比赛: 先给当天每局记下"当天开始前"的评分作输入, 再按 (时间, gameid) 依次更新。
    """
    out = {}
    tt = team_rows.sort_values(["gameid", "participantid"], kind="stable")
    tt = tt[tt.groupby("gameid")["gameid"].transform("size") == 2]
    blue = tt.groupby("gameid").nth(0).set_index("gameid")
    red = tt.groupby("gameid").nth(1).set_index("gameid").loc[blue.index]
    G = pd.DataFrame({"gameid": blue.index.values, "league": blue["league"].values, "date": blue["date"].values,
                      "b": blue["teamname"].values, "r": red["teamname"].values,
                      "y": blue["result"].astype(int).values})
    G = G.sort_values(["date", "gameid"], kind="stable")
    G["day"] = G["date"].dt.normalize()
    for lg, g in G.groupby("league", sort=False):
        R: dict = {}
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
                out[gid] = R.get(bt, ELO_INIT) - R.get(rt, ELO_INIT)
            for gid, bt, rt, yy in rows:
                rb, rr = R.get(bt, ELO_INIT), R.get(rt, ELO_INIT)
                e = 1.0 / (1.0 + 10 ** ((rr - rb) / 400.0))
                R[bt], R[rt] = rb + ELO_K * (yy - e), rr - ELO_K * (yy - e)
    return out


# ══════════════════════════════════════════════════════════
#  构建全部输入 (可缓存)
# ══════════════════════════════════════════════════════════

def build_all(files, log, cache: Path | None):
    fp = data_fingerprint(files)
    cf = cache / f"g1_inputs_{fp}.pkl" if cache else None
    if cf and cf.exists():
        log(f"  读缓存 {cf.name}")
        with cf.open("rb") as fh:
            return pickle.load(fh)
    t0 = time.time()
    raw = load_raw(files)
    gc.collect()
    log(f"  读 CSV {len(raw)} 行, {time.time() - t0:.0f}s, 工作集 {mem_mb()[0]:.0f} MB")

    # 次级联赛矩阵: 先建 (这样它的峰值内存不被后面的东西盖住), 计时 + 内存
    mm = minor_mask(raw)
    mraw = raw[mm]
    leagues = sorted(mraw["league"].unique())
    gc.collect()
    ws0, pk0 = mem_mb()
    t1 = time.time()
    parts, per_league = [], {}
    for lg in leagues:
        m = build_pool(mraw[mraw["league"] == lg])
        per_league[lg] = len(m)
        parts.append(m)
    N = pd.concat(parts, ignore_index=True)
    del parts
    t_minor = time.time() - t1
    ws1, pk1 = mem_mb()
    build_stats = {"minor_leagues": len(leagues), "minor_games_raw": int(mraw["gameid"].nunique()),
                   "minor_rows": len(N), "seconds": round(t_minor, 1),
                   "ws_before_mb": round(ws0, 0), "ws_after_mb": round(ws1, 0),
                   "peak_before_mb": round(pk0, 0), "peak_after_mb": round(pk1, 0),
                   "matrix_mb": round(N.memory_usage(deep=True).sum() / 2 ** 20, 1),
                   "minor_raw_rows": int(len(mraw)),
                   "minor_raw_mb": round(mraw.memory_usage(deep=True).sum() / 2 ** 20, 1),
                   "per_league": per_league}
    log(f"  次级联赛矩阵: {len(leagues)} 个联赛, {build_stats['minor_games_raw']} 局 → {len(N)} 行, "
        f"{t_minor:.0f}s; 工作集 {ws0:.0f} → {ws1:.0f} MB, 进程峰值 {pk0:.0f} → {pk1:.0f} MB, "
        f"矩阵本身 {build_stats['matrix_mb']} MB")
    del mraw
    gc.collect()

    t2 = time.time()
    M = build_pool(raw[raw["league"].isin(LEAGUES)])
    log(f"  四大矩阵 {len(M)} 局, {time.time() - t2:.0f}s")

    t3 = time.time()
    pool_rows = raw[(raw["position"] == "team") & (raw["league"].isin(LEAGUES) | mm)]
    elo = elo_inputs(pool_rows[["gameid", "league", "date", "participantid", "teamname", "result"]])
    log(f"  Elo: {pool_rows['league'].nunique()} 个池, {len(elo)} 局, {time.time() - t3:.0f}s")
    del raw, pool_rows
    gc.collect()
    obj = {"M": M, "N": N, "elo": elo, "build_stats": build_stats, "leagues": leagues, "fingerprint": fp}
    if cf:
        cf.parent.mkdir(parents=True, exist_ok=True)
        with cf.open("wb") as fh:
            pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return obj


# ══════════════════════════════════════════════════════════
#  指标
# ══════════════════════════════════════════════════════════

def brier(y, p):
    return (p - y) ** 2


def logloss(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def acc(y, p):
    return ((p > .5).astype(int) == y).astype(float)


def logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def slope(y, p):
    """y ~ a + b·logit(p) 的极大似然 (Newton), 返回 (b, b 的标准误)。b < 1 = 过度自信。"""
    z = logit(p)
    X = np.column_stack([np.ones_like(z), z])
    beta = np.zeros(2)
    for _ in range(100):
        mu = 1 / (1 + np.exp(-(X @ beta)))
        H = X.T @ (X * (mu * (1 - mu))[:, None])
        step = np.linalg.solve(H, X.T @ (y - mu))
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    mu = 1 / (1 + np.exp(-(X @ beta)))
    H = X.T @ (X * (mu * (1 - mu))[:, None])
    return float(beta[1]), float(np.sqrt(np.linalg.inv(H)[1, 1]))


def cluster_t(d, groups):
    """逐局均值差的系列赛聚类稳健 t (CR1)。"""
    d = np.asarray(d, float)
    mu = d.mean()
    s = pd.Series(d - mu).groupby(groups).sum().values
    G, N = len(s), len(d)
    if G < 2:
        return float("nan")
    se = math.sqrt((s ** 2).sum() * G / (G - 1)) / N
    return float(mu / se) if se > 0 else float("nan")


def series_key(df):
    d = pd.to_datetime(df["date"]).dt.normalize().astype(str).values
    a = np.where(df["blue_team"].values < df["red_team"].values, df["blue_team"].values, df["red_team"].values)
    b = np.where(df["blue_team"].values < df["red_team"].values, df["red_team"].values, df["blue_team"].values)
    return np.array([f"{x}|{l}|{u}|{v}" for x, l, u, v in zip(d, df["league"].values, a, b)])


def pair(base_blocks, cand_blocks):
    """(Δ = base − cand 的块均值, t, cand 更好的块数)。损失越小越好, 所以正 = 候选更好。"""
    b, c = np.asarray(base_blocks, float), np.asarray(cand_blocks, float)
    delta, _sd, t, _ = paired_t(-b, -c)
    return float(delta), float(t), int(((b - c) > 0).sum())


# ══════════════════════════════════════════════════════════
#  评估
# ══════════════════════════════════════════════════════════

def calibrate(kind, p_ca, y_ca, e_ca, p_te, e_te):
    """kind 'platt' = train.py 原样 (吃原始概率); 'elo' = (d): [logit(p_raw), diff_elo·ln10/400]。"""
    if kind == "platt":
        c = LogisticRegression(C=1e6).fit(p_ca.reshape(-1, 1), y_ca)
        return c.predict_proba(p_te.reshape(-1, 1))[:, 1], (float(c.coef_[0][0]), float(c.intercept_[0]))
    Xc = np.column_stack([logit(p_ca), e_ca * LN10_400])
    Xt = np.column_stack([logit(p_te), e_te * LN10_400])
    c = LogisticRegression(C=1e6, max_iter=1000).fit(Xc, y_ca)
    return c.predict_proba(Xt)[:, 1], tuple(float(v) for v in c.coef_[0]) + (float(c.intercept_[0]),)


def evaluate(inp, log, jobs: int, quick: bool = False):
    M, N, elo = inp["M"], inp["N"], inp["elo"]
    M = M.sort_values(["date", "gameid"], kind="stable").reset_index(drop=True)
    N = N.sort_values(["date", "gameid"], kind="stable").reset_index(drop=True)
    A = pd.concat([M.assign(_maj=True), N.assign(_maj=False)], ignore_index=True)
    A = A.sort_values(["date", "gameid"], kind="stable").reset_index(drop=True)
    y = A["y"].astype(int).values
    dates = A["date"].values
    is_maj = A["_maj"].values
    e = np.array([elo.get(g, np.nan) for g in A["gameid"].values])
    if np.isnan(e).any():
        raise RuntimeError(f"{int(np.isnan(e).sum())} 行没有 Elo —— 池子没覆盖到训练行")

    ev = np.flatnonzero(is_maj & (A["date"].values >= np.datetime64(EVAL_START)))
    blocks = np.array_split(ev, N_BLOCKS)
    seeds = SEEDS[:1] if quick else SEEDS
    if quick:
        blocks = blocks[-2:]
        ev = np.concatenate(blocks)
    log(f"  评估: 四大 {len(ev)} 局 ({A['date'].iloc[ev[0]].date()} → {A['date'].iloc[ev[-1]].date()}), "
        f"{len(blocks)} 块 × {[len(b) for b in blocks]}; 种子 {seeds}")

    arms_needed = {}   # 树的配置 (a, b, c) -> 需要的校准方式集合
    preds = {stg: {arm: np.full(len(A), np.nan) for arm in ARMS} for stg in STAGES}
    preds_diag = {stg: np.full(len(A), np.nan) for stg in STAGES}   # P, 校准层只用四大行 (口径敏感性)
    calib_coefs = {stg: {arm: [] for arm in ARMS} for stg in STAGES}
    block_meta = []
    for bi, te in enumerate(blocks):
        s_ts = dates[te[0]]
        block_meta.append({"block": bi, "start": str(pd.Timestamp(s_ts).date()),
                           "end": str(pd.Timestamp(dates[te[-1]]).date()), "n_test": len(te)})

    t0 = time.time()
    n_fit = 0
    for stg in STAGES:
        feats = select_features(M, stg)
        Xh = A[feats].astype(float).fillna(-999).values
        Xs = None
        if stg == "post_draft":
            Ah = A[feats].copy()
            for c, sc in zip(WR_COLS, SHR_COLS):
                Ah[c] = A[sc].values
            Xs = Ah.astype(float).fillna(-999).values
            del Ah
        cfg_arms = {}
        for arm, comp in ARMS.items():
            a, b_, c, d = ("a" in comp), ("b" in comp), ("c" in comp), ("d" in comp)
            if stg == "pre_draft":
                c = False      # (c) 只动 draft 列, pre_draft 看不到: 树完全相同
            cfg_arms.setdefault((a, b_, c), []).append((arm, d))
        log(f"\n  ── {stg}: {len(feats)} 个特征, {len(cfg_arms)} 种树配置 ──")
        for (a, b_, c), arm_list in cfg_arms.items():
            X = Xs if c else Xh
            # 复核: 只有显式给了 --jobs 才覆盖线程数; 默认和 train.py 一样不传 (线程数会改变随机流, 见文档开头)
            params = {**train.XGB, **(A_PARAMS if a else {}), **({"n_jobs": jobs} if jobs else {})}
            for bi, te in enumerate(blocks):
                s_ts = dates[te[0]]
                pool = np.flatnonzero((dates < s_ts) & (is_maj | b_))
                ntr = int(len(pool) * TREE_FRAC)
                tr, ca = pool[:ntr], pool[ntr:]
                ca_maj = ca[is_maj[ca]]
                outs = {arm: [] for arm, _ in arm_list}
                diag = []
                for sd in seeds:
                    mdl = XGBClassifier(**{**params, "random_state": sd})
                    mdl.fit(X[tr], y[tr], verbose=False)
                    n_fit += 1
                    p_ca = mdl.predict_proba(X[ca])[:, 1]
                    p_te = mdl.predict_proba(X[te])[:, 1]
                    for arm, d in arm_list:
                        p, coef = calibrate("elo" if d else "platt", p_ca, y[ca], e[ca], p_te, e[te])
                        outs[arm].append(p)
                        calib_coefs[stg][arm].append(coef)
                    if (a, b_, c) == (True, True, stg == "post_draft"):
                        pm_ca = p_ca[is_maj[ca]]
                        p, _ = calibrate("elo", pm_ca, y[ca_maj], e[ca_maj], p_te, e[te])
                        diag.append(p)
                for arm, ps in outs.items():
                    preds[stg][arm][te] = np.mean(ps, axis=0)
                if diag:
                    preds_diag[stg][te] = np.mean(diag, axis=0)
                if bi == 0 or bi == len(blocks) - 1:
                    log(f"    a={int(a)} b={int(b_)} c={int(c)} 块 {bi + 1}: 树 {len(tr)} 行 / 校准 {len(ca)} 行 "
                        f"(其中四大 {len(ca_maj)}), 臂 {[x for x, _ in arm_list]}  [{time.time() - t0:.0f}s, {n_fit} 次拟合]")
    return A, ev, blocks, block_meta, preds, preds_diag, calib_coefs, n_fit


def summarize(A, ev, blocks, preds, preds_diag, calib_coefs, log):
    y = A["y"].astype(int).values
    blk = np.full(len(A), -1)
    for bi, te in enumerate(blocks):
        blk[te] = bi
    sk = series_key(A)
    E = A.iloc[ev]
    subsets = {"全部": np.ones(len(ev), bool)}
    for lg in LEAGUES:
        subsets[lg] = (E["league"] == lg).values
    subsets["季后赛"] = (E["playoffs"] == 1).values
    subsets["常规赛"] = (E["playoffs"] == 0).values
    subsets["系列赛第 1 局"] = (E["game_no"] == 1).values
    subsets["第 2 局及以后"] = (E["game_no"] >= 2).values
    subsets["2026 年"] = (E["date"].dt.year == 2026).values

    def blockwise(fn, p, idx):
        return np.array([fn(y[idx[blk[idx] == bi]], p[idx[blk[idx] == bi]]).mean()
                         for bi in range(len(blocks)) if (blk[idx] == bi).any()])

    res = {}
    for stg in STAGES:
        P = preds[stg]
        R = {"arms": {}, "vs_B0": {}, "loo": {}, "add_one": {}, "subsets": {}, "diag_cal_majors_only": {}}
        for arm in ARMS:
            p = P[arm][ev]
            sl, se = slope(y[ev], p)
            R["arms"][arm] = {"brier": float(brier(y[ev], p).mean()), "logloss": float(logloss(y[ev], p).mean()),
                              "acc": float(acc(y[ev], p).mean()), "auc": float(roc_auc_score(y[ev], p)),
                              "slope": sl, "slope_se": se,
                              "brier_blocks": blockwise(brier, P[arm], ev).tolist()}
        R["constant_brier"] = float(np.mean([y[te].mean() * (1 - y[te].mean()) for te in blocks]))

        def cmp(base, cand, idx=ev):
            out = {}
            for nm, fn in (("brier", brier), ("logloss", logloss), ("acc", acc)):
                sign = -1 if nm == "acc" else 1           # acc 越大越好, 统一成"损失"
                bb = sign * blockwise(fn, P[base] if isinstance(base, str) else base, idx)
                cb = sign * blockwise(fn, P[cand] if isinstance(cand, str) else cand, idx)
                dlt, t, nb = pair(bb, cb)
                pb = P[base] if isinstance(base, str) else base
                pc = P[cand] if isinstance(cand, str) else cand
                dg = sign * (fn(y[idx], pb[idx]) - fn(y[idx], pc[idx]))
                out[nm] = {"delta": dlt, "t_block": t, "blocks_better": nb, "n_blocks": len(bb),
                           "t_series": cluster_t(dg, sk[idx])}
            return out

        R["vs_B0"] = {arm: cmp("B0", arm) for arm in ARMS if arm != "B0"}
        R["recipe_vs_P"] = cmp("P", RECIPE)          # 正 = 配方比 P 更好 (补充, 不参与判决)
        for x in COMPONENTS:
            R["loo"][x] = cmp(f"P-{x}", "P")["brier"]
            R["add_one"][x] = cmp("B0", f"B0+{x}")["brier"]
        for nm, msk in subsets.items():
            idx = ev[msk]
            if len(idx) < 20:
                continue
            row = {"n": int(len(idx))}
            for arm in ("B0", "P", RECIPE):
                p = P[arm][idx]
                sl, se = slope(y[idx], p)
                row[arm] = {"brier": float(brier(y[idx], p).mean()), "logloss": float(logloss(y[idx], p).mean()),
                            "acc": float(acc(y[idx], p).mean()), "auc": float(roc_auc_score(y[idx], p)),
                            "slope": sl, "slope_se": se}
            row["cmp"] = cmp("B0", "P", idx)
            row["cmp_recipe"] = cmp("B0", RECIPE, idx)
            R["subsets"][nm] = row
        pdg = preds_diag[stg]
        if np.isfinite(pdg[ev]).all():
            p = pdg[ev]
            sl, se = slope(y[ev], p)
            R["diag_cal_majors_only"] = {"brier": float(brier(y[ev], p).mean()),
                                         "logloss": float(logloss(y[ev], p).mean()),
                                         "acc": float(acc(y[ev], p).mean()), "auc": float(roc_auc_score(y[ev], p)),
                                         "slope": sl, "vs_B0": cmp("B0", pdg), "vs_P": cmp("P", pdg)}
        cc = calib_coefs[stg]
        R["calib_coef_mean"] = {arm: np.mean(np.array(cc[arm]), axis=0).tolist() for arm in ARMS}

        v = R["vs_B0"]["P"]
        R["pass"] = bool(v["brier"]["t_block"] > T_PASS and v["logloss"]["t_block"] > 0)
        R["drop"] = {x: bool(R["loo"][x]["t_block"] < T_KEEP) for x in COMPONENTS}
        R["recipe"] = [x for x in COMPONENTS if not R["drop"][x]]
        res[stg] = R
    return res


def report_md(res, block_meta, build_stats, verify, runtime, n_fit) -> str:
    L = []
    f4 = lambda v: f"{v:.4f}"
    L.append("## G1 Stage 1/2 组合包 —— 结果\n")
    L.append("测试块 (四大): " + ", ".join(f"{b['start']}→{b['end']} ({b['n_test']})" for b in block_meta) + "\n")
    for stg in STAGES:
        R = res[stg]
        v = R["vs_B0"]["P"]
        L.append(f"\n### {stg}: P {'过闸' if R['pass'] else '不过闸'}  "
                 f"(Brier t = {v['brier']['t_block']:+.2f}, log loss t = {v['logloss']['t_block']:+.2f}; "
                 f"常数 Brier {R['constant_brier']:.4f})\n")
        L.append("| 臂 | Brier | log loss | 准确率 | AUC | 斜率 | ΔBrier vs B0 (正=更好) | 块 t | 更好块 | 系列 t | ΔLL t | Δacc | acc t |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for arm in ARMS:
            a = R["arms"][arm]
            if arm == "B0":
                L.append(f"| B0 | {f4(a['brier'])} | {f4(a['logloss'])} | {a['acc']:.3f} | {a['auc']:.3f} | "
                         f"{a['slope']:.2f}±{a['slope_se']:.2f} | — | — | — | — | — | — | — |")
                continue
            c = R["vs_B0"][arm]
            L.append(f"| {arm}{' (补充)' if arm in SUPPLEMENTARY else ''} | {f4(a['brier'])} | {f4(a['logloss'])} | "
                     f"{a['acc']:.3f} | {a['auc']:.3f} | "
                     f"{a['slope']:.2f}±{a['slope_se']:.2f} | {c['brier']['delta']:+.4f} | {c['brier']['t_block']:+.2f} | "
                     f"{c['brier']['blocks_better']}/{c['brier']['n_blocks']} | {c['brier']['t_series']:+.2f} | "
                     f"{c['logloss']['t_block']:+.2f} | {c['acc']['delta']:+.4f} | {c['acc']['t_block']:+.2f} |")
        L.append("\n逐项剔除 (P 相对 P−{x}, 正 = 这一项有用) 与逐项加入 (B0+{x} 相对 B0):\n")
        L.append("| 项 | LOO ΔBrier | LOO 块 t | 更好块 | 简洁规则 | 加入 ΔBrier | 加入 块 t | 更好块 |")
        L.append("|---|---|---|---|---|---|---|---|")
        for x in COMPONENTS:
            lo, ad = R["loo"][x], R["add_one"][x]
            L.append(f"| {x} | {lo['delta']:+.4f} | {lo['t_block']:+.2f} | {lo['blocks_better']}/{lo['n_blocks']} | "
                     f"{'去掉' if R['drop'][x] else '保留'} | {ad['delta']:+.4f} | {ad['t_block']:+.2f} | "
                     f"{ad['blocks_better']}/{ad['n_blocks']} |")
        L.append(f"\n简洁规则给出的配方: {{{', '.join(R['recipe'])}}}\n")
        rv = R["recipe_vs_P"]
        L.append(f"补充 (不参与判决): 配方 {RECIPE} 相对 P —— ΔBrier {rv['brier']['delta']:+.4f} (正 = 配方更好), "
                 f"块 t {rv['brier']['t_block']:+.2f} ({rv['brier']['blocks_better']}/{rv['brier']['n_blocks']}), "
                 f"log loss 块 t {rv['logloss']['t_block']:+.2f}, 准确率 Δ {rv['acc']['delta']:+.4f}\n")
        L.append("切片 (B0 → P):\n")
        L.append("| 切片 | n | Brier B0 → P | Δ | 块 t | 系列 t | 准确率 B0 → P | AUC B0 → P | 斜率 B0 → P |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for nm, row in R["subsets"].items():
            b0, p = row["B0"], row["P"]
            c = row["cmp"]["brier"]
            L.append(f"| {nm} | {row['n']} | {f4(b0['brier'])} → {f4(p['brier'])} | {c['delta']:+.4f} | "
                     f"{c['t_block']:+.2f} | {c['t_series']:+.2f} | {b0['acc']:.3f} → {p['acc']:.3f} | "
                     f"{b0['auc']:.3f} → {p['auc']:.3f} | {b0['slope']:.2f} → {p['slope']:.2f} |")
        L.append(f"\n切片 (B0 → 配方 {RECIPE}, 补充):\n")
        L.append("| 切片 | n | Brier B0 → 配方 | Δ | 块 t | 系列 t | 准确率 B0 → 配方 | AUC B0 → 配方 | 斜率 B0 → 配方 |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for nm, row in R["subsets"].items():
            b0, p = row["B0"], row[RECIPE]
            c = row["cmp_recipe"]["brier"]
            L.append(f"| {nm} | {row['n']} | {f4(b0['brier'])} → {f4(p['brier'])} | {c['delta']:+.4f} | "
                     f"{c['t_block']:+.2f} | {c['t_series']:+.2f} | {b0['acc']:.3f} → {p['acc']:.3f} | "
                     f"{b0['auc']:.3f} → {p['auc']:.3f} | {b0['slope']:.2f} → {p['slope']:.2f} |")
        dg = R.get("diag_cal_majors_only")
        if dg:
            L.append(f"\n口径敏感性 (不参与判决): P 的校准层只用校准段里的四大行拟合 → Brier {f4(dg['brier'])}, "
                     f"相对 B0 块 t {dg['vs_B0']['brier']['t_block']:+.2f}, 相对 P (正 = 只用四大更好) "
                     f"ΔBrier {dg['vs_P']['brier']['delta']:+.4f} 块 t {dg['vs_P']['brier']['t_block']:+.2f}, "
                     f"斜率 {dg['slope']:.2f}\n")
        cc = R["calib_coef_mean"]
        L.append(f"校准层系数均值: B0 Platt (coef, intercept) {np.round(cc['B0'], 3).tolist()}; "
                 f"P [logit 系数, Elo 系数, 截距] {np.round(cc['P'], 3).tolist()}; "
                 f"B0+d {np.round(cc['B0+d'], 3).tolist()}; 配方 {np.round(cc[RECIPE], 3).tolist()}\n")
    bs = build_stats
    L.append(f"\n### 次级联赛矩阵的构建\n{bs['minor_leagues']} 个联赛, {bs['minor_games_raw']} 局原始 → "
             f"{bs['minor_rows']} 行 (前 3 场滚动统计为空的局按线上规则丢掉); 耗时 {bs['seconds']} s; "
             f"工作集 {bs['ws_before_mb']:.0f} → {bs['ws_after_mb']:.0f} MB, 进程峰值 {bs['peak_before_mb']:.0f} → "
             f"{bs['peak_after_mb']:.0f} MB; 矩阵 {bs['matrix_mb']} MB, 次级原始行 {bs['minor_raw_rows']} 行 "
             f"{bs['minor_raw_mb']} MB。\n")
    L.append(f"B0 矩阵核对: {json.dumps(verify, ensure_ascii=False, default=str)}\n")
    L.append(f"评估耗时 {runtime:.0f} s, XGB 拟合 {n_fit} 次。\n")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="输出目录 (result.json / report.md / preds.pkl / run.log)")
    ap.add_argument("--cache", default=None, help="缓存矩阵的目录 (按数据文件指纹作废)")
    ap.add_argument("--verify", default="fresh",
                    help="fresh = 现场跑 build_training_matrix 核对; skip; 或一个 build_training_matrix 产出的 pickle")
    ap.add_argument("--jobs", type=int, default=None,
                    help="XGB 线程数; 不给 = 和 train.py 一样用默认 (复核: 线程数会改变随机流)")
    ap.add_argument("--quick", action="store_true", help="冒烟测试: 最后 2 块、1 个种子。结果不能用于判决")
    ap.add_argument("--verify-only", action="store_true", help="只建 (或读缓存) 矩阵并核对 B0, 不评估")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = Log(out / "run.log")
    log(f"G1 Stage 1/2 组合包闸门  {time.strftime('%Y-%m-%d %H:%M:%S')}  quick={a.quick}")
    files = data_files()
    inp = build_all(files, log, Path(a.cache) if a.cache else None)

    if a.verify == "skip":
        verify = {"ok": None, "note": "skipped"}
        log("  ⚠ 跳过 B0 矩阵核对")
    else:
        if a.verify == "fresh":
            t0 = time.time()
            prod = build_training_matrix(train.DATA, verbose=False)
            log(f"  build_training_matrix: {len(prod)} 局, {time.time() - t0:.0f}s")
        else:
            with open(a.verify, "rb") as fh:
                prod = pickle.load(fh)
            log(f"  核对对象: {a.verify} ({len(prod)} 局)")
        verify = verify_majors(inp["M"], prod, log)
        verify["source"] = a.verify
        del prod
        gc.collect()
        if not verify["ok"]:
            raise SystemExit("B0 矩阵与线上不一致 —— 闸门的基线不是线上配方, 停止")
    if a.verify_only:
        (out / "verify.json").write_text(json.dumps(verify, ensure_ascii=False, indent=1, default=str),
                                         encoding="utf-8")
        return

    t0 = time.time()
    A, ev, blocks, block_meta, preds, preds_diag, coefs, n_fit = evaluate(inp, log, a.jobs, a.quick)
    runtime = time.time() - t0
    res = summarize(A, ev, blocks, preds, preds_diag, coefs, log)
    md = report_md(res, block_meta, inp["build_stats"], verify, runtime, n_fit)
    log("\n" + md)
    payload = {"protocol": {"A_PARAMS": A_PARAMS, "K_SHRINK": K_SHRINK, "ELO": [ELO_K, ELO_INIT, ELO_REG],
                            "EVAL_START": str(EVAL_START.date()), "N_BLOCKS": N_BLOCKS, "SEEDS": SEEDS,
                            "TREE_FRAC": TREE_FRAC, "T_PASS": T_PASS, "T_KEEP": T_KEEP, "XGB_B0": train.XGB,
                            "quick": a.quick, "n_jobs": a.jobs},
               "blocks": block_meta, "build_stats": inp["build_stats"], "minor_leagues": inp["leagues"],
               "verify": verify, "runtime_s": runtime, "n_fit": n_fit, "data_fingerprint": inp["fingerprint"],
               "results": res}
    (out / "result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    (out / "report.md").write_text(md, encoding="utf-8")
    keep = A[["gameid", "date", "league", "blue_team", "red_team", "y", "playoffs", "game_no", "_maj"]].iloc[ev]
    with (out / "preds.pkl").open("wb") as fh:
        pickle.dump({"games": keep.reset_index(drop=True),
                     "preds": {s: {arm: preds[s][arm][ev] for arm in ARMS} for s in STAGES},
                     "diag_cal_majors_only": {s: preds_diag[s][ev] for s in STAGES},
                     "block": np.concatenate([[bi] * len(te) for bi, te in enumerate(blocks)])}, fh)
    log(f"\n写出 {out}")


if __name__ == "__main__":
    main()
