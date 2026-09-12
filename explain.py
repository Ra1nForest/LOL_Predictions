"""
特征翻译层 —— 把模型的字段名变成人话
======================================
模型内部用 diff_avg_towers / b_bot_champ_wr / golddiff_norm 这种名字,
对看比赛的人毫无意义。这一层负责翻译:

    diff_avg_towers  = +3.2    →  G2 场均多推 3.2 座塔
    golddiff_norm    = 0.078   →  经济领先占场上总经济的 7.8%
    b_bot_champ_wr   = 0.722   →  G2 的 ADC 用这个英雄历史胜率 72%
    x_gold_scaling   = -120    →  经济领先 × 阵容强势期 (利于 FNC)

命名规则 (来自 feature_store.py / ingame_service.py)
----------------------------------------------------
    diff_X     蓝方减红方
    b_X / r_X  单方数值
    sum_X      双方之和 (节奏类指标)
    avg_X      近 10 场滚动均值
    pre_X      局内模型里引用的赛前特征
    x_A_B      交互项

为什么不用一个通用模板
----------------------
中文里每个指标的比较说法不一样: 塔是「多推」, 死亡是「少死」,
胜率是「高几个百分点」。用一个模板套所有指标会得到「G2场均推塔多
3.2座」这种不像人话的东西。所以每个指标带自己的句式。

翻译不出来的字段返回 None —— 宁可不展示, 也不要露出英文字段名。
"""
from __future__ import annotations
import math

ROLE_CN = {"top": "上单", "jng": "打野", "mid": "中单",
           "bot": "ADC", "sup": "辅助"}

# ══════════════════════════════════════════════════════════
#  滚动统计: 每个指标自己的句式
#
#  (中文名, 单位, 小数位, 极性, diff 句式, 单方句式, 合计句式)
#     极性  +1 越大越好   -1 越小越好   0 中性 (只说「更高/更长」)
#     {who}  diff 里占优的一方   {team} 单方   {v} 格式化后的数值
# ══════════════════════════════════════════════════════════

_P = "%"

STAT = {
 "kills": ("场均击杀", "", 1, +1,
           "{who} 场均多拿 {v} 个人头", "{team} 场均 {v} 个人头",
           "双方场均合计 {v} 个人头"),
 "deaths": ("场均死亡", "", 1, -1,
            "{who} 场均少死 {v} 次", "{team} 场均送 {v} 个人头",
            "双方场均合计送 {v} 个人头"),
 "assists": ("场均助攻", "", 1, +1,
             "{who} 场均多 {v} 个助攻", "{team} 场均 {v} 个助攻", None),
 "totalgold": ("场均团队经济", "k", 1, +1,
               "{who} 场均多出 {v} 经济", "{team} 场均团队经济 {v}", None),
 "damagetochampions": ("场均对英雄伤害", "k", 1, +1,
                       "{who} 场均多打 {v} 英雄伤害",
                       "{team} 场均对英雄伤害 {v}", None),
 "wardsplaced": ("场均插眼", "", 0, 0,
                 "{who} 场均多插 {v} 个眼", "{team} 场均插 {v} 个眼", None),
 "wardskilled": ("场均排眼", "", 0, +1,
                 "{who} 场均多排 {v} 个眼", "{team} 场均排 {v} 个眼", None),
 "dragons": ("场均小龙", "", 2, +1,
             "{who} 场均多拿 {v} 条小龙", "{team} 场均拿 {v} 条小龙",
             "双方场均合计 {v} 条小龙"),
 "barons": ("场均大龙", "", 2, +1,
            "{who} 场均多拿 {v} 条大龙", "{team} 场均拿 {v} 条大龙", None),
 "towers": ("场均推塔", "", 1, +1,
            "{who} 场均多推 {v} 座塔", "{team} 场均推 {v} 座塔",
            "双方场均合计推 {v} 座塔"),
 "cspm": ("每分钟补刀", "", 1, +1,
          "{who} 每分钟多 {v} 刀补兵", "{team} 每分钟 {v} 刀补兵", None),
 "golddiffat10": ("10 分钟经济差", "", 0, +1,
                  "{who} 的十分钟经济表现好 {v}",
                  "{team} 场均十分钟经济差 {v}", None),
 "golddiffat15": ("15 分钟经济差", "", 0, +1,
                  "{who} 的十五分钟经济表现好 {v}",
                  "{team} 场均十五分钟经济差 {v}", None),
 "firstblood": ("一血率", _P, 0, +1,
                "{who} 的一血率高 {v}", "{team} 一血率 {v}", None),
 "firstdragon": ("首条小龙率", _P, 0, +1,
                 "{who} 更常拿到首条小龙 (高 {v})",
                 "{team} 首条小龙率 {v}", None),
 "firsttower": ("一塔率", _P, 0, +1,
                "{who} 的一塔率高 {v}", "{team} 一塔率 {v}", None),
 "ckpm": ("每分钟总人头", "", 2, 0,
          "{who} 的比赛节奏更快 (每分钟多 {v} 个人头)",
          "{team} 场均每分钟 {v} 个人头",
          "双方节奏合计 {v} 人头/分钟"),
 "team_kpm": ("每分钟击杀", "", 2, +1,
              "{who} 每分钟多拿 {v} 个人头",
              "{team} 场均每分钟击杀 {v} 个",
              "双方每分钟击杀合计 {v} 个"),
 "rolling_wr": ("近十场胜率", _P, 0, +1,
                "{who} 近十场胜率高 {v}", "{team} 近十场胜率 {v}", None),
 "gamelen": ("场均时长", "", 1, 0,
             "{who} 的比赛平均长 {v} 分钟", "{team} 场均时长 {v} 分钟",
             "双方场均时长合计 {v} 分钟"),
 "game_kills": ("场均总人头", "", 1, 0,
                "{who} 的比赛人头数多 {v} 个", "{team} 场均总人头 {v} 个",
                "双方场均总人头合计 {v} 个"),
 "champ_wr": ("阵容英雄平均胜率", _P, 0, +1,
              "{who} 阵容里的英雄近期胜率高 {v}",
              "{team} 阵容英雄平均胜率 {v}", None),
 "comfort": ("英雄熟练度", "", 0, +1,
             "{who} 这套阵容的总使用场次多 {v} 场",
             "{team} 这套阵容累计使用 {v} 场", None),
}

# diff_pre_wr 是 diff_rolling_wr 在局内模型里的别名
ALIAS = {"wr": "rolling_wr", "kpm": "team_kpm"}

# ══════════════════════════════════════════════════════════
#  局内实时字段
# ══════════════════════════════════════════════════════════

LIVE = {
 "golddiff":    ("经济", "", 0),
 "xpdiff":      ("经验", "", 0),
 "csdiff":      ("补刀", " 刀", 0),
 "killdiff":    ("人头", " 个", 0),
}

# ══════════════════════════════════════════════════════════
#  元信息 / 交互项
# ══════════════════════════════════════════════════════════

META = {"playoffs": "季后赛", "is_lpl": "LPL", "is_lck": "LCK",
        "is_lec": "LEC", "is_lcs": "LCS"}

INTERACTION = {
 "x_gold_scaling":     "经济领先 × 阵容强势期",
 "x_goldnorm_scaling": "相对经济领先 × 阵容强势期",
 "x_xp_scaling":       "经验领先 × 阵容强势期",
 "lead_by_early_comp": "经济领先 × 对手前期成色",
}

INTERACTION_NOTE = ("同样的经济差, 在不同阵容下含义相反 —— "
                    "领先方若拿的是前期阵容, 优势会随时间流失。")


# ══════════════════════════════════════════════════════════
#  格式化
# ══════════════════════════════════════════════════════════

def _fmt(v: float, unit: str, dec: int) -> str:
    """数值 + 单位, 大数走 k, 百分比走 %。"""
    a = abs(v)
    if unit == _P:
        return f"{a * 100:.{dec}f}%"
    if unit == "k":
        return f"{a / 1000:.{dec}f}k"
    if a >= 10000:
        return f"{a / 1000:.1f}k{unit}"
    return f"{a:,.{dec}f}{unit}"


def _split(name: str):
    """diff_avg_towers → ('diff', 'towers')。认不出来返回 (None, name)。"""
    pre = None
    for p in ("diff_", "sum_", "b_", "r_"):
        if name.startswith(p):
            pre, name = p[:-1], name[len(p):]
            break
    for p in ("avg_", "pre_"):
        if name.startswith(p):
            name = name[len(p):]
    return pre, ALIAS.get(name, name)


# ══════════════════════════════════════════════════════════
#  主翻译函数
# ══════════════════════════════════════════════════════════

def explain(name: str, value, blue: str = "蓝方", red: str = "红方"):
    """把一个 (字段名, 值) 翻译成一句人话。None = 不值得展示。"""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    v = float(value)

    # ── 元信息 ──
    if name in META:
        if name.startswith("is_"):
            return None          # 赛区前端自己知道, 不占位
        return "季后赛" if v else None

    # ── 交互项 ──
    if name in INTERACTION:
        if abs(v) < 1e-9:
            return None
        side = blue if v > 0 else red
        return f"{INTERACTION[name]} —— 当前利于 {side}"

    # ── 阵容强势期 ──
    if name in ("scaling_diff", "b_scaling", "r_scaling", "scaling_sum"):
        return _scaling(name, v, blue, red)

    # ── 时间切片 ──
    if name == "T":
        return f"按第 {int(v)} 分钟的模型切片评估"

    # ── 局内实时 ──
    if name == "killsum":
        return f"双方已合计击杀 {int(v)} 个人头"
    if name == "golddiff_norm":
        if abs(v) < 1e-9:
            return "双方经济持平"
        side = blue if v > 0 else red
        return f"{side} 的经济领先占场上总经济的 {abs(v) * 100:.1f}%"
    if name in LIVE:
        cn, unit, dec = LIVE[name]
        if abs(v) < 1e-9:
            return f"双方{cn}持平"
        side = blue if v > 0 else red
        return f"{side} {cn}领先 {_fmt(v, unit, dec)}"

    # ── 分路熟练度差 ──
    if name.startswith("diff_") and name.endswith("_comfort"):
        role = name[5:-8]
        if role in ROLE_CN:
            if abs(v) < 1:
                return None
            who, cn = (blue if v > 0 else red), ROLE_CN[role]
            return f"{who} {cn} 对这个英雄比对位多 {abs(int(v))} 场经验"

    # ── 选手-英雄熟练度 ──
    r = _player_champ(name, v, blue, red)
    if r:
        return r

    # ── 滚动统计 ──
    pre, base = _split(name)
    if base == "streak":
        return _streak(pre, v, blue, red)
    if base not in STAT or pre is None:
        return None
    cn, unit, dec, pol, t_diff, t_side, t_sum = STAT[base]

    if pre == "diff":
        shown = _fmt(v, unit, dec)
        # 格式化之后是 0 的, 说「持平」比说「多 0.00 条」诚实
        if abs(v) < 1e-9 or not any(ch in "123456789" for ch in shown):
            return f"双方{cn}持平"
        if pol == +1:
            who = blue if v > 0 else red
        elif pol == -1:
            who = red if v > 0 else blue
        else:
            who = blue if v > 0 else red
        return t_diff.format(who=who, v=_fmt(v, unit, dec))

    if pre == "sum":
        if not t_sum:
            return None
        return t_sum.format(v=_fmt(v, unit, dec))

    team = blue if pre == "b" else red
    return t_side.format(team=team, v=_fmt(v, unit, dec))


def _scaling(name, v, blue, red):
    if name == "scaling_sum":
        return None
    if name in ("b_scaling", "r_scaling"):
        team = blue if name[0] == "b" else red
        tone = "偏后期" if v > 0.5 else "偏前期" if v < -0.5 else "前后期均衡"
        return f"{team} 的阵容{tone}"
    if abs(v) < 0.3:
        return "双方阵容的强势期相当"
    late, early = (blue, red) if v > 0 else (red, blue)
    return f"{late} 的阵容更偏后期, {early} 需要在前期建立优势"


def _streak(pre, v, blue, red):
    n = int(v)
    if pre == "diff":
        return None
    if n == 0:
        return None
    team = blue if pre == "b" else red
    return f"{team} 当前{'连胜' if n > 0 else '连败'} {abs(n)} 场"


def _player_champ(name, v, blue, red):
    for side, team in (("b_", blue), ("r_", red)):
        if not name.startswith(side):
            continue
        rest = name[2:]
        for role, cn in ROLE_CN.items():
            if rest == f"{role}_champ_wr":
                return f"{team} {cn} 用这个英雄历史胜率 {v * 100:.0f}%"
            if rest == f"{role}_champ_games":
                n = int(v)
                if n == 0:
                    return f"{team} {cn} 没用过这个英雄"
                return f"{team} {cn} 用过这个英雄 {n} 场"
            if rest == f"{role}_comfort":
                return None
    return None


# ══════════════════════════════════════════════════════════
#  批量 / 摘要
# ══════════════════════════════════════════════════════════

SKIP_IN_REASONS = {"T", "playoffs", "is_lpl", "is_lck", "is_lec", "is_lcs"}


def explain_evidence(items, blue="蓝方", red="红方", k=6):
    """把 API 的 evidence 列表整体翻译, 丢掉翻译不出来的。

    输入既接受 [{'feature':..,'value':..,'importance':..}] 也接受
    [(feature, value, importance)]。
    """
    out, seen, bases = [], set(), set()
    for it in items or []:
        # T / 赛区标记是模型的元信息, 不是这场比赛的事实。
        # 它们在响应里另有位置 (slice_used / league), 不占证据位。
        if isinstance(it, (tuple, list)):
            f, val, imp = it[0], it[1], it[2]
        else:
            f, val, imp = it.get("feature"), it.get("value"), it.get("importance", 0)
        if f in SKIP_IN_REASONS:
            continue
        txt = explain(f, val, blue, red)
        if not txt or txt in seen:
            continue
        # 同一个基础指标只留权重最高的那种切法 —— 不要同时出现
        # 「场均多 0.3 个助攻」和「场均 29.4 个助攻」
        base = _split(f)[1]
        if base in bases:
            continue
        bases.add(base)
        seen.add(txt)
        out.append({"text": txt, "weight": round(float(imp or 0), 4)})
        if len(out) >= k:
            break
    return out


TONE = [(0.55, "势均力敌"), (0.65, "{lead} 略占优"), (0.78, "{lead} 占优"),
        (0.88, "{lead} 优势明显"), (1.01, "{lead} 大幅领先")]


def summarize(prob, blue, red, stage=None, minute=None):
    """一句话总结, 给前端当标题。"""
    lead = blue if prob >= 0.5 else red
    p = max(prob, 1 - prob)
    tone = next(t for th, t in TONE if p < th).format(lead=lead)
    when = {"1_pre_draft": "赛前", "2_post_draft": "BP 结束",
            "4_ingame": "局内"}.get(stage, "")
    if minute is not None:
        when = f"第 {int(minute)} 分钟"
    return f"{when} · {tone}" if when else tone


def confidence_note(metrics: dict) -> str:
    """把模型指标翻译成「这个数字有多可信」。"""
    m = metrics or {}
    acc = m.get("accuracy_at_this_slice") or m.get("accuracy")
    base = m.get("baseline")
    ece = m.get("ece_cal", m.get("ece"))
    parts = []
    if acc:
        s = f"留出测试集准确率 {acc * 100:.0f}%"
        if base:
            s += f" (基线 {base * 100:.0f}%)"
        parts.append(s)
    if ece is not None:
        q = "良好" if ece < 0.03 else "可用" if ece < 0.06 else "偏差偏大"
        parts.append(f"概率校准{q} (ECE {ece:.3f})")
    return " · ".join(parts)


def range_note(p_min, p_max) -> str:
    """模型够不到的区间 —— 前端的死区就是这个。"""
    if p_min is None or p_max is None:
        return ""
    return (f"该模型的概率输出被校准限制在 {p_min * 100:.0f}%–{p_max * 100:.0f}%。"
            f"超出这个范围的一边倒对局, 它只会顶到边界, 不会给出更极端的数字。")
