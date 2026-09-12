"""
Stage 4 服务模块 — 局内预测 + 会话追踪
========================================
被 api.py 导入。两种用法:

  单次预测   POST /predict/ingame     给一个时点的局势, 返回当前胜率
  会话追踪   POST /session/start      开一场比赛
             POST /session/{id}/tick  每隔几分钟报一次数据
             GET  /session/{id}       看整条胜率曲线

会话模式是为「一边看比赛一边填」设计的 —— 自动数据源对 LPL 不可用
(腾讯自己的基础设施, Riot 的 livestats API 不覆盖), 所以手动输入是
现实选择。你本来就坐在那儿看, 屏幕上写着经济差和人头。

前置: python ingame_train.py  生成 artifacts/model_ingame.json
"""
from __future__ import annotations
import json, math, uuid, numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Literal
from xgboost import XGBClassifier

ART = Path(__file__).parent / "artifacts"
ROLES = ["top", "jng", "mid", "bot", "sup"]

# 调用方给不出 gold_total 时的兜底值 —— 各切片上**一队**总经济的中位数,
# 取自五年约 9 万条 OE 记录 (goldat10/15/20/25)。
#
# 原先这里写的是 1200 * T, 那是拍脑袋的, 比实测低 25~32%:
#     T=10 估 12000 / 实测 15918      T=20 估 24000 / 实测 34597
#     T=15 估 18000 / 实测 25037      T=25 估 30000 / 实测 43854
# 而 golddiff_norm = golddiff / gold_total 是**重要性最高的特征** (0.345),
# 分母估小就等于把领先幅度放大 1.33~1.46 倍 —— 模型以为的优势比实际大三四成,
# 而且全程不报错。自动跟播不受影响 (帧里有真实总经济), 手动填局面每次都踩。
#
# 复算: research/audit_gold_total.py
GOLD_TOTAL_MEDIAN = {10: 15918, 15: 25037, 20: 34597, 25: 43854}


class IngameModel:
    """variant="" 是完整模型; variant="_live" 是不含经验差特征的实时变体。

    实时变体存在的原因: lolesports 的帧里没有 XP 字段。给完整模型传一个
    编造的 xpdiff=0, 模型会吃到训练时没见过的常量, 而且 evidence() 会把它
    讲成「双方经验持平」—— 把缺失讲成实测结论。宁可换一个训练时就不看
    经验差的模型。
    """

    def __init__(self, variant: str = ""):
        self.variant = variant
        self.model = XGBClassifier()
        self.model.load_model(str(ART / f"model_ingame{variant}.json"))
        cfg = json.loads((ART / f"calib_ingame{variant}.json").read_text(encoding="utf-8"))
        self.a, self.b = cfg["coef"], cfg["intercept"]
        # 校准器。缺这个字段就是旧 artifacts, 按 Platt 处理 —— 换代码和换
        # 模型文件可以分开做, 不必同时到位。
        self.calibrator = cfg.get("calibrator", "platt")
        self.iso_x = np.asarray(cfg.get("iso_x") or [], dtype=float)
        self.iso_y = np.asarray(cfg.get("iso_y") or [], dtype=float)
        self.clip = float(cfg.get("clip", 0.01))
        if self.calibrator == "isotonic" and len(self.iso_x) < 2:
            # 声称是保序回归却没有映射表 —— 宁可退回 Platt, 也不要拿一个
            # 空表去插值 (np.interp 会把所有输入都映射成同一个常数, 于是
            # 每一场比赛都给同一个概率, 而且不报错)
            print(f"  calib_ingame{variant}.json 缺 iso_x/iso_y, 退回 Platt")
            self.calibrator = "platt"
        self.features = cfg["features"]
        self.scaling = cfg["scaling_index"]
        self.metrics = cfg["metrics"]
        self.slices = cfg["slices"]
        self.trained_through = cfg["trained_through"]
        self.has_xpdiff = cfg.get("has_xpdiff", "xpdiff" in self.features)
        self.importance = dict(zip(self.features, self.model.feature_importances_))

    @classmethod
    def load_pair(cls) -> tuple["IngameModel", Optional["IngameModel"]]:
        """(完整模型, 实时变体)。实时变体缺失时返回 None —— 老的 artifacts
        没有它, 此时实时路径应当拒绝服务而不是退回去编造 xpdiff。"""
        full = cls("")
        try:
            live = cls("_live")
        except Exception:
            live = None
        return full, live

    # ── 阵容强势期 ──
    def comp_scaling(self, champs: list[str]) -> tuple[float, int]:
        # 按 champion_key 对齐: 看板传来的是 Data Dragon id (LeeSin), 表里是训练用的
        # OE 显示名 (Lee Sin)。原样查的话名字带空格/撇号的英雄全被漏掉, 阵容强势期
        # 只按剩下几个算, 不报错。见 feature_store.CHAMP_ID_ALIAS
        from feature_store import champion_key
        idx = self.__dict__.get("_scaling_idx")
        if idx is None:
            idx = {champion_key(k): v for k, v in self.scaling.items()}
            self._scaling_idx = idx
        vals = [idx[k] for k in map(champion_key, champs) if k in idx]
        if len(vals) < 3:
            return float("nan"), len(vals)
        return float(np.mean(vals)), len(vals)

    def nearest_slice(self, minute: float) -> int:
        return min(self.slices, key=lambda t: abs(t - minute))

    # ── 组一行特征 ──
    def make_row(self, *, minute, golddiff, xpdiff: Optional[float], csdiff,
                 blue_kills, red_kills, gold_total=None,
                 blue_champs=None, red_champs=None,
                 pre_row=None, league="LPL") -> tuple[dict, list[str]]:
        warn = []
        T = self.nearest_slice(minute)
        if abs(T - minute) > 2.5:
            warn.append(f"你填的是第 {minute:.0f} 分钟, 但模型只在 10/15/20/25 分钟"
                        f"这几个时间点训练过。已按最接近的 {T} 分钟评估, "
                        f"相差 {abs(T-minute):.0f} 分钟, 精度会下降。")

        # LPL 在训练集里样本偏薄, 但**不是零** —— 早先这里写死"一场都没有",
        # 那是 Oracle's Elixir 2022-2025 不提供 LPL 的 at10/15/20/25 时期的
        # 实情。2026 赛季 OE 开始给了, 现在约 465 局, 占局内训练集 9.3%。
        # 陈旧的警告比没有警告更糟: 它会被下游 (Stage 3 的 agent) 当成事实
        # 反复引用, 而简报里同时还写着"覆盖 LPL", 自相矛盾。
        if league == "LPL":
            warn.append("LPL 在局内训练集里样本偏薄 (约 9%, 且全部来自 2026 年) "
                        "—— Oracle's Elixir 在 2022-2025 不提供 LPL 的分钟级快照。"
                        "结论在 LPL 上的可靠性弱于其他三大赛区。")
        # 概率被压扁的警告**不在这里**发 —— 见 clip_warning()。
        # 这里只有输入, 还没有输出, 按经济差去猜"会不会被压扁"必然过时。

        b_sc = r_sc = float("nan")
        if blue_champs and red_champs:
            b_sc, nb = self.comp_scaling(blue_champs)
            r_sc, nr = self.comp_scaling(red_champs)
            if math.isnan(b_sc) or math.isnan(r_sc):
                warn.append("这几个英雄的历史样本不足, 算不出阵容强势期 —— "
                            "「领先但阵容吃亏」这类隐患本次无法识别。")
        else:
            warn.append("没填双方阵容, 无法判断谁的阵容更耐拖 —— "
                        "同样的经济差在不同阵容下含义相反。")

        sc_diff = (b_sc - r_sc) if not (math.isnan(b_sc) or math.isnan(r_sc)) else float("nan")
        if gold_total is None:
            gold_total = GOLD_TOTAL_MEDIAN.get(T, 1750 * T)
            warn.append(f"没填蓝方总经济, 按第 {T} 分钟的实测中位数 {gold_total} 估算, "
                        f"相对经济差可能有偏。")

        # xpdiff=None 表示「拿不到」, 不是「等于 0」。实时接入就是这种情况。
        # 不往行里放这一项 —— 也就不会有任何关于经验差的论断被生成出来。
        if xpdiff is None:
            if self.has_xpdiff:
                warn.append("没有经验差数据, 而当前模型把它当作输入 —— "
                            "结果不可靠。实时数据应当走 live 变体模型。")
            else:
                warn.append("本次没有经验差数据(实时数据源不提供), 已改用"
                            "训练时就不含经验差的模型, 而不是用 0 顶替。")

        m = {
            "T": T,
            "golddiff": float(golddiff),
            "csdiff": float(csdiff),
            "killdiff": float(blue_kills - red_kills),
            "killsum": float(blue_kills + red_kills),
            "golddiff_norm": float(golddiff) / gold_total if gold_total else float("nan"),
            "b_scaling": b_sc, "r_scaling": r_sc,
            "scaling_diff": sc_diff,
            "scaling_sum": (b_sc + r_sc) if not math.isnan(sc_diff) else float("nan"),
            "is_lpl": int(league == "LPL"), "is_lck": int(league == "LCK"),
            "is_lec": int(league == "LEC"), "is_lcs": int(league == "LCS"),
        }
        if xpdiff is not None:
            m["xpdiff"] = float(xpdiff)

        # 交互项
        if not math.isnan(sc_diff):
            m["x_gold_scaling"] = float(golddiff) * sc_diff
            m["x_goldnorm_scaling"] = m["golddiff_norm"] * sc_diff
            m["lead_by_early_comp"] = float(golddiff) * (-sc_diff)
            # x_xp_scaling = xpdiff * scaling_diff, 没有 xpdiff 就没有它
            if xpdiff is not None:
                m["x_xp_scaling"] = float(xpdiff) * sc_diff

        # 赛前特征 (来自 feature_store 的队伍滚动统计)
        if pre_row:
            for k, v in pre_row.items():
                if k in self.features:
                    m[k] = v
        else:
            warn.append("没有可用的赛前队伍统计, 本次只看局内数据。")
        return m, warn

    def calibrate(self, raw: float) -> float:
        """原始输出 -> 校准概率。

        保序回归靠一张单调的分段表插值 (np.interp 在两端自动 clamp, 和训练
        时 IsotonicRegression(out_of_bounds="clip") 的行为一致)。
        """
        if self.calibrator == "isotonic":
            v = float(np.interp(raw, self.iso_x, self.iso_y))
            return float(np.clip(v, self.clip, 1.0 - self.clip))
        return 1.0 / (1.0 + math.exp(-(self.a * raw + self.b)))

    def clip_warning(self, cal: float) -> Optional[str]:
        """概率有没有真的被校准的上下限夹住。返回 None 表示没有。

        **必须看实际输出, 不能按经济差硬编码。** 早先这条警告写在 make_row
        里, 条件是"第 20 分钟后且经济差 >= 5000 就说概率被压扁了" —— 那在
        Platt 时代成立 (它顶格只到 0.94, 而 6k+ 领先的实测胜率 98%)。换成
        保序回归之后范围是 [0.01, 0.99], 领先 8000 给的是 95.9%, 离上限还
        差得远, 那句话就成了**陈旧断言**。
        而陈旧的警告比没有警告更糟: 它会被下游 (Stage 3 的 agent、前端文案)
        当成事实反复引用 —— 这个坑在同一个文件里的 LPL 警告上已经踩过一次。
        """
        lo, hi = self.metrics.get("p_min"), self.metrics.get("p_max")
        eps = 1e-6
        if hi is not None and cal >= hi - eps:
            return (f"概率已经顶到模型的表达上限 {hi:.0%} —— 这种局面的实际"
                    f"胜率可能更高。这是校准的固有上限, 不是判断失误。")
        if lo is not None and cal <= lo + eps:
            return (f"概率已经触到模型的表达下限 {lo:.0%} —— 这种局面的实际"
                    f"胜率可能更低。这是校准的固有上限, 不是判断失误。")
        return None

    def predict(self, row: dict) -> tuple[float, float]:
        x = np.array([[float(row.get(f, np.nan)) for f in self.features]])
        x = np.nan_to_num(x, nan=-999.0)
        raw = float(self.model.predict_proba(x)[0, 1])
        return raw, self.calibrate(raw)

    def evidence(self, row: dict, k: int = 6):
        out = []
        for f in self.features:
            v = row.get(f)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            out.append({"feature": f, "value": round(float(v), 3),
                        "importance": round(float(self.importance.get(f, 0)), 4)})
        out.sort(key=lambda d: -d["importance"])
        return out[:k]


# ══════════════════════════════════════════════════════════
#  会话: 一场比赛的多次打点
# ══════════════════════════════════════════════════════════

class SessionStore:
    """内存版。重启即失。够单人看比赛用。"""

    def __init__(self, max_sessions=200):
        self.s: dict[str, dict] = {}
        self.max = max_sessions

    def start(self, meta: dict) -> str:
        sid = uuid.uuid4().hex[:8]
        if len(self.s) >= self.max:
            oldest = min(self.s, key=lambda k: self.s[k]["created"])
            del self.s[oldest]
        self.s[sid] = {"id": sid, "created": datetime.utcnow().isoformat(),
                       "meta": meta, "ticks": []}
        return sid

    def get(self, sid: str) -> Optional[dict]:
        return self.s.get(sid)

    def tick(self, sid: str, entry: dict) -> Optional[dict]:
        sess = self.s.get(sid)
        if sess is None:
            return None
        sess["ticks"].append(entry)
        sess["ticks"].sort(key=lambda e: e["minute"])
        return sess

    def trajectory(self, sid: str) -> Optional[dict]:
        sess = self.s.get(sid)
        if sess is None:
            return None
        ticks = sess["ticks"]
        if not ticks:
            return {"session": sess, "trajectory": [], "summary": None}
        pts = [{"minute": t["minute"], "probability_blue": t["probability_blue"],
                "golddiff": t["golddiff"]} for t in ticks]
        first, last = ticks[0], ticks[-1]
        swings = []
        for i in range(1, len(ticks)):
            d = ticks[i]["probability_blue"] - ticks[i - 1]["probability_blue"]
            if abs(d) >= 0.10:
                swings.append({"from_minute": ticks[i-1]["minute"],
                               "to_minute": ticks[i]["minute"],
                               "delta": round(d, 3)})
        return {
            "session": {"id": sess["id"], "meta": sess["meta"],
                        "n_ticks": len(ticks)},
            "trajectory": pts,
            "summary": {
                "opening": first["probability_blue"],
                "latest": last["probability_blue"],
                "net_change": round(last["probability_blue"] - first["probability_blue"], 3),
                "peak": max(t["probability_blue"] for t in ticks),
                "trough": min(t["probability_blue"] for t in ticks),
                "big_swings": swings,
            },
        }
