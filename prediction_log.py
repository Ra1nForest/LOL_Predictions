"""
预测留档
========

把每一次线上预测和它后来的真实结果记下来。

为什么需要
----------
项目里所有准确率数字都来自**留出集** —— 拿历史数据切一块假装没见过。但
留出集会骗人: Stage 2 的留出集只显示 +0.0006 准确率, 而配对前进检验测出
真实效果是 +0.033 (research/gate_draft_lift.py)。

留出集更测不出**线上**的问题: 数据源改格式、上游改口径、赛区分布漂移、
特征算错 —— 这些都不报错, 只让预测慢慢变差, 而离线指标纹丝不动。今天
那个时钟偏 32 秒就是典型: 留出集再漂亮也测不出来, 因为训练和评估用的是
同一份 (错位的) 数据。

留档记的是真账: 我们说了多少, 后来发生了什么。

设计
----
· **绝不能影响预测本身**。写盘失败就算了, 任何异常都吞掉并打一行日志 ——
  这是旁路, 不是主路。
· **按 (game_id, 分钟) 去重**。前端 3 秒轮询一次, 不去重的话一局能写出
  几百行几乎相同的记录。整分钟粒度足够了。
· **只记有真实局内读数的预测**。赛前/BP 后的概率整局不变, 每局记一次即可,
  用 minute=None 标出。
· 结果**事后回填**, 不在预测时猜 —— 见 resolve()。

用法:
    from prediction_log import log_prediction
    log_prediction(...)                       # api.py 里调

    python prediction_log.py --resolve        # 回填比赛结果
    python prediction_log.py --score          # 算真账
"""
from __future__ import annotations

import json
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

# Windows 计划任务里控制台是 GBK, 而本文件输出含 ✓ ✗ ⚠ − 这类 GBK 编不了的
# 字符 —— 不设这个会在 print 那一刻抛 UnicodeEncodeError, 表现为任务"跑到一半
# 消失"。2026-08-30 实测: 不加时 daily_update 正好崩在 [3/5] 过闸的判决行,
# 训练全做完了、结果打不出来, 整个任务算失败。
for _s in (sys.stdout, sys.stderr):
    if _s is not None and hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

_ROOT = Path(__file__).parent
LOG = Path(os.environ.get("LOL_PRED_LOG") or (_ROOT / "predictions" / "log.jsonl"))

_lock = threading.Lock()
_seen: set[tuple] = set()          # (game_id, 整分钟) 已经记过的
_loaded = False


def _load_seen():
    """启动后第一次写入时把已有记录读进来, 避免重启后重复写同一分钟。"""
    global _loaded
    if _loaded:
        return
    _loaded = True
    if not LOG.exists():
        return
    try:
        with open(LOG, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                _seen.add((r.get("game_id"), r.get("minute_key")))
    except Exception as e:
        print(f"  ⚠ 预测留档: 读取已有记录失败 {type(e).__name__}: {e}", flush=True)


def _r4(v):
    """附加字段取 4 位小数。转不成数就记 None —— 附加字段出问题不能连累整条记录写不进去。"""
    try:
        return None if v is None else round(float(v), 4)
    except (TypeError, ValueError):
        return None


def log_prediction(*, match_id, game_id, league, blue, red, probability_blue,
                   source, minute=None, slice_used=None, game_number=None,
                   golddiff=None, blue_kills=None, red_kills=None,
                   ingame_probability_blue=None, blend_weight=None,
                   anchor_probability_blue=None, anchor_source=None,
                   trained_through=None):
    """记一条。任何异常都不往外抛 —— 这是旁路, 不能拖垮预测接口。

    后五个参数 (2026-10-17 加) 都可以不给, 不给就记 None —— 老的调用方式照旧能用:
      ingame_probability_blue  局内模型自己的数 (渐变之前)。看板第 3-15 分钟记的 probability_blue 是
                               渐变后的 (api._blend_prior), 拿它和留出集 (局内模型自己的数) 比不是同一个
                               口径; 没渐变时它就等于 probability_blue。赛前/BP 后那条为 None。
      blend_weight             局内模型在渐变里的权重 (0-1); None = 没渐变 (没有锚, 或不是局内那条)
      anchor_probability_blue  渐变的锚 (蓝方概率)
      anchor_source            锚从哪来: post_draft | pre_draft | xregion | None
      trained_through          产出 probability_blue 的那个模型的训练数据截止日 —— 重训前后的记录分得开,
                               线上真账才能和**同一代**模型的留出成绩对照
    """
    if probability_blue is None:
        return
    try:
        with _lock:
            _load_seen()
            # 整分钟去重: 3 秒一轮的轮询不该写出几百条几乎相同的记录
            mkey = None if minute is None else int(minute)
            key = (game_id, mkey)
            if key in _seen:
                return
            _seen.add(key)

            rec = {
                "t": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "match_id": match_id, "game_id": game_id,
                "game_number": game_number, "league": league,
                "blue": blue, "red": red,
                "minute": None if minute is None else round(float(minute), 1),
                "minute_key": mkey,
                "probability_blue": round(float(probability_blue), 4),
                # pre_draft | post_draft | ingame | xregion (跨赛区模型 C 的赛前数, 只在两队都在四大已知
                # 队伍里时记 —— 回填按 OE 队名配胜负; score() 的"按来源"把它单列一组)
                "source": source,
                "slice_used": slice_used,
                "golddiff": golddiff,
                "blue_kills": blue_kills, "red_kills": red_kills,
                # 2026-10-17 起才有的五个键, 含义见本函数的文档。更早的行没有它们 —— 读的地方一律 .get
                "ingame_probability_blue": _r4(ingame_probability_blue),
                "blend_weight": _r4(blend_weight),
                "anchor_probability_blue": _r4(anchor_probability_blue),
                "anchor_source": anchor_source,
                "trained_through": trained_through,
                "y": None,                     # 事后由 --resolve 回填
            }
            LOG.parent.mkdir(parents=True, exist_ok=True)
            with open(LOG, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"  ⚠ 预测留档失败(不影响预测) {type(e).__name__}: {e}", flush=True)


# ── 事后回填结果 ─────────────────────────────────────────────────────
# 按赛区配不上时的退路, **只给四大赛区的行**。lolesports 把赛区内的世界赛资格赛挂在本赛区底下 (标 "LPL"),
# OE 却把它登记成国际赛代码: 2026-09-17~19 IG / TES / JDG / WE 的 LPL 资格赛在 OE 里是 "WLDs", 留档里
# 193 条 (12 局) 因此一直回填不了。按赛区配不上时, 依次拿下面这几个 OE 代码再配一次 —— 配对规则和 find_oe
# 一模一样 (两队 + 局号 + 时间窗), 只换赛区名; 哪一个先配上就用哪一个。
# 显式列出, 不做模糊匹配: 就是 esports_feed.LEAGUE_IDS 里那几个国际赛在 OE 里的名字 (backfill_late.oe_league;
# DCGI 两边同名)。WLDs 排第一, 因为它是唯一实测到的落点。
INTL_OE_RETRY = ("WLDs", "MSI", "FST", "EWC", "DCGI")
# 退路的时间窗比主路 (window_h, 默认 48h) 窄得多。日更每次拉完数据就跑 resolve, 而 OE 晚一天收录: 国内局当天
# 还不在 OE 里, 于是退路先跑。如果同一对阵前一两天在国际赛上打过 (同蓝红方、同局号), 48h 窗口会把这条国内行
# **永久**配到那局国际赛的胜负上 —— y 一旦写上就不再回填, 也不报错。实测: OE 2025-26 里 2026-05-06 LCK
# DN SOOPers vs Hanwha Life 第 1/2 局, ±48h 内就有 05-05 标 EWC 的同对阵同局号 (相差 26.4h)。资格赛那 12 局
# 的记录时间和 OE 开局时刻相差都在 0.7h 以内, 12h 足够把它们全部配上。
INTL_RETRY_WINDOW_H = 12


def _rows():
    if not LOG.exists():
        return []
    out = []
    with open(LOG, "r", encoding="utf-8") as fh:
        for line in fh:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def resolve(window_h: float = 48) -> int:
    """给还没有结果的记录回填 y (蓝方是否获胜)。

    结果来自 Oracle's Elixir —— API 里没有逐局胜者字段, 而 OE 每行都有
    result。配对方式和 research/backfill_late.py 一致: 赛区 + **两队** +
    局号 + 就近日期。只按一队配会在同一天两场比赛时张冠李戴。

    window_h 是"记录时间"和"OE 里那局的开局时间"允许差多远。线上跟播时
    两者相差不过几十分钟, 但 OE 的 date 是开局时刻而记录时间是预测时刻,
    跨时区/跨日的比赛容易擦边, 所以默认给 48 小时。
    **补记很久以前的比赛要把它调大** —— 那种情况下记录时间和比赛时间可以
    差好几天。

    四大赛区的行按本赛区配不上时, 再按 INTL_OE_RETRY 里的国际赛代码各配一次 (规则不变, 只换赛区名) ——
    赛区内的世界赛资格赛 lolesports 标本赛区, OE 标 WLDs, 见 INTL_OE_RETRY 那段说明。
    """
    sys.path.insert(0, str(_ROOT))
    sys.path.insert(0, str(_ROOT / "research"))
    from datetime import datetime as _dt
    from backfill_late import oe_index, find_oe, oe_league
    from esports_feed import is_major

    rows = _rows()
    todo = [r for r in rows if r.get("y") is None]
    if not todo:
        print("没有待回填的记录")
        return 0
    print(f"待回填 {len(todo)} 条")
    oe = oe_index()

    n = n_intl = 0
    for r in rows:
        if r.get("y") is not None:
            continue
        try:
            when = _dt.fromisoformat(r["t"])
        except Exception:
            continue
        # 留档存的是上游原样的赛区名 ("Worlds"), 换 OE 名字要大小写无关 —— 见 oe_league。
        # DCGI 不进映射表: OE 登记它用的就是 "DCGI", oe_league 原样返回就配得上 (见 backfill_late.LEAGUE_TO_OE)
        lg = oe_league(r.get("league"))
        key = (when, r.get("blue"), r.get("red"), r.get("game_number") or 1)
        row = find_oe(oe, lg, *key, window_h=window_h)
        if row is None and is_major(r.get("league")):
            # 赛区内的世界赛资格赛: 见 INTL_OE_RETRY
            for alt in INTL_OE_RETRY:
                row = find_oe(oe, alt, *key, window_h=min(window_h, INTL_RETRY_WINDOW_H))
                if row is not None:
                    n_intl += 1
                    break
        if row is None:
            continue
        r["y"] = int(row["result"])
        n += 1

    if n:
        tmp = LOG.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp.replace(LOG)          # 原子替换, 中途崩了不会留半截文件
    print(f"回填 {n} 条" + (f" (其中 {n_intl} 条是四大赛区的行按国际赛代码配上的, 见 INTL_OE_RETRY)"
                           if n_intl else ""))
    return n


def _game_key(r) -> str:
    return r.get("game_id") or f"{r.get('match_id')}#{r.get('game_number')}"


# 留出集的对照数从这里读 (只读)。看板的局内概率来自 live 变体 (实时帧没有经验差), 所以对照它的 metrics。
INGAME_CALIB = _ROOT / "artifacts" / "calib_ingame_live.json"


def _heldout_per_T():
    """(按切片的留出成绩, 口径)。口径 "served" = per_T_served, "all" = 退回全赛区的 per_T, None = 读不到。

    线上只服务四大 + 国际赛, 而 per_T 是**全赛区**的测试段 (约 9400 局, 四大 + 国际只占约 22%)。次级联赛
    更好预测, 拿 per_T 对照会让线上看起来系统性偏低: 2026-10-17 用当时的模型量, 10/15/20/25 分钟 per_T 是
    71.1/75.1/79.2/82.6%, 同一测试段只取四大 + 国际赛是 68.1/72.4/75.6/80.7% —— 之前"线上固定时刻比留出
    低 5-7 个点"有一半是口径造成的。ingame_train.py 从那天起另存 per_T_served; 之前训练的 artifacts 没有,
    退回 per_T, score() 会把这一点印出来。
    """
    try:
        m = json.loads(INGAME_CALIB.read_text(encoding="utf-8"))["metrics"]
    except Exception:
        return None, None
    if m.get("per_T_served"):
        return m["per_T_served"], "served"
    return (m.get("per_T") or None), "all"


def score():
    """真账。和留出集的数字对照着看 —— 差得多就说明线上出了离线看不见的问题。

    **按局计分, 不按条。** 看板每局每分钟记一条: 2026-09-13 时 2064 条其实只有 116 局, 打 41 分钟
    的局记 41 条、20 分钟的只记 20 条左右。按条平均, 长局就被多算几遍 —— 而长局恰恰是拉锯和
    翻盘的局, 当时按条准确率 69.7%, 按局 75.7%; 样本量也显得比实际大 (同一局几十条共享一个胜负)。
    所以这里每条记录的权重是 1/本局条数, 每局合起来算一票; ± 是按局算的 95% 区间。
    另给固定时刻 (第 10/15/20/25 分钟各取一条) 的切面, 和留出集按切片的成绩同一口径, 并排印出来。
    留出数用四大 + 国际赛区的 per_T_served, 不用全赛区的 per_T (理由见 _heldout_per_T); 老 artifacts 没有
    per_T_served 时退回 per_T 并印一句提醒。
    旧的按条数字留一行, 只作对照。
    """
    rows = [r for r in _rows() if r.get("y") is not None and r.get("probability_blue") is not None]
    if not rows:
        print("还没有已知结果的记录, 先跑 --resolve")
        return
    from collections import defaultdict
    import numpy as np
    key = [_game_key(r) for r in rows]
    p = np.array([r["probability_blue"] for r in rows], dtype=float)
    y = np.array([r["y"] for r in rows], dtype=float)
    hit = ((p >= .5).astype(float) == y).astype(float)
    sq = (p - y) ** 2

    def by_game(v, mask=None):
        """每局先在自己的记录里平均, 再对局平均。返回 (均值, 95% 半宽, 局数)。"""
        d = defaultdict(list)
        for i, k in enumerate(key):
            if mask is None or mask[i]:
                d[k].append(v[i])
        g = np.array([np.mean(x) for x in d.values()])
        if not len(g):
            return float("nan"), float("nan"), 0
        half = 1.96 * g.std(ddof=1) / np.sqrt(len(g)) if len(g) > 1 else float("nan")
        return float(g.mean()), float(half), len(g)

    acc, acc_h, n_games = by_game(hit)
    brier, brier_h, _ = by_game(sq)
    yb, _, _ = by_game(y)
    base = max(yb, 1 - yb)
    print(f"\n{'=' * 64}")
    print(f"  线上真账: {n_games} 局 ({len(rows)} 条记录, 每局每分钟一条) —— 下面按局计分, 每局一票")
    print("=" * 64)
    print(f"  准确率 {acc:.3f} ± {acc_h:.3f}   基线 {base:.3f}   Brier {brier:.4f} ± {brier_h:.4f}")
    print(f"  (按条的旧口径: 准确率 {hit.mean():.3f}, Brier {sq.mean():.4f} —— 长局被重复计入)")

    # 固定时刻: 每局取离第 T 分钟最近、相差不超过 1 分钟的那一条。每局一条, 区间直接按条算就是按局算。
    # 对照的留出数见 _heldout_per_T: 优先四大 + 国际赛区的 per_T_served, 不拿全赛区的 per_T 比。
    ref, ref_kind = _heldout_per_T()
    half = lambda v: 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
    lp = np.clip(p, 1e-6, 1 - 1e-6)
    ll = -(y * np.log(lp) + (1 - y) * np.log(1 - lp))
    if ref_kind == "served":
        print("\n按固定时刻 (每局一条; 留出 = 局内模型 live 变体在测试段上四大 + 国际赛区的成绩, "
              "和线上服务的赛区同一口径; ± 为 95% 区间):")
    elif ref_kind == "all":
        print("\n按固定时刻 (每局一条; 留出 = 全赛区的 per_T):")
        print(f"  注意: {INGAME_CALIB.name} 里还没有 per_T_served (下一次重训才会写进去), 留出只好退回全赛区的 "
              "per_T。它含次级联赛, 比线上实际服务的四大 + 国际赛区系统性偏乐观 (2026-10 实测 10/15/20/25 分钟高 3.0/2.7/3.6/1.9 个点), "
              "线上比它低不能直接算成线上出了问题。")
    else:
        print(f"\n按固定时刻 (每局一条; 读不到 {INGAME_CALIB.name}, 不和留出集对照):")
    print("  第 3-15 分钟的线上数是渐变后的 (2026-09-13 起, api._blend_prior), 留出是局内模型自己的数; "
          "2026-10-17 起的记录另存了渐变前的读数, 有的话下面\"其中\"那一行拿它对照。")
    for T in (10, 15, 20, 25):
        best = {}
        for i, r in enumerate(rows):
            m = r.get("minute")
            if m is None or abs(m - T) > 1:
                continue
            k = key[i]
            if k not in best or abs(m - T) < abs(rows[best[k]]["minute"] - T):
                best[k] = i
        idx = list(best.values())
        if len(idx) >= 5:
            print(f"  第 {T:>2} 分钟  线上 {len(idx):>5} 局   准确率 {hit[idx].mean():.3f} ± {half(hit[idx]):.3f}   "
                  f"Brier {sq[idx].mean():.4f} ± {half(sq[idx]):.4f}   LL {ll[idx].mean():.4f}")
            h = (ref or {}).get(str(T))
            if h and ref_kind == "served":
                print(f"              留出 {h['n']:>5} 局   准确率 {h['accuracy']:.3f} ± {h['accuracy_ci95']:.3f}   "
                      f"Brier {h['brier']:.4f} ± {h['brier_ci95']:.4f}   "
                      f"LL {h['log_loss']:.4f} ± {h['log_loss_ci95']:.4f}")
            elif h and h.get("accuracy") is not None:
                print(f"              留出 {h.get('n', 0):>5} 局   准确率 {h['accuracy']:.3f}   (全赛区 per_T)")
            # 渐变段上同一批局: 局内模型自己的数 vs 渐变后记下的数
            raw = [i for i in idx if rows[i].get("ingame_probability_blue") is not None
                   and rows[i].get("blend_weight") is not None and rows[i]["blend_weight"] < 1]
            if len(raw) >= 5:
                pr = np.array([rows[i]["ingame_probability_blue"] for i in raw], dtype=float)
                yr = y[raw]
                print(f"              其中 {len(raw)} 局记了渐变前的读数: 局内模型自己 准确率 "
                      f"{((pr >= .5).astype(float) == yr).mean():.3f}   Brier {((pr - yr) ** 2).mean():.4f}"
                      f"   (同一批局渐变后 {hit[raw].mean():.3f} / {sq[raw].mean():.4f})")

    print("\n按来源:")
    for src in sorted({r["source"] for r in rows}):
        m = np.array([r["source"] == src for r in rows])
        a, ah, n = by_game(hit, m)
        b, _, _ = by_game(sq, m)
        if n >= 5:
            print(f"  {src:<12} {n:>4} 局 / {m.sum():>5} 条   准确率 {a:.3f} ± {ah:.3f}   Brier {b:.4f}")

    print("\n按赛区:")
    for lg in sorted({r["league"] for r in rows if r.get("league")}):
        m = np.array([r.get("league") == lg for r in rows])
        a, ah, n = by_game(hit, m)
        if n >= 5:
            print(f"  {lg:<8} {n:>4} 局 / {m.sum():>5} 条   准确率 {a:.3f} ± {ah:.3f}")

    # 校准: 说 70% 的那些, 是不是真有 70% 赢了。一局在某一档里有几条都只算一票
    print("\n校准 (按局: 说了多少 vs 实际赢了多少):")
    for lo in (0.0, .2, .4, .6, .8):
        hi = lo + .2
        m = (p >= lo) & (p < hi if hi < 1 else p <= 1)
        wy, _, n = by_game(y, m)
        wp, _, _ = by_game(p, m)
        if n >= 5:
            print(f"  预测 {lo:.0%}-{hi:.0%}  {n:>4} 局 / {m.sum():>5} 条   "
                  f"实际蓝方胜率 {wy:.1%}   (理想 ≈ {wp:.1%})")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolve", action="store_true", help="回填比赛结果")
    ap.add_argument("--score", action="store_true", help="算真账")
    ap.add_argument("--window-h", type=float, default=48,
                    help="记录时间与比赛时间允许相差多少小时 (默认 48)")
    a = ap.parse_args()
    if a.resolve:
        resolve(a.window_h)
    if a.score or not (a.resolve or a.score):
        score()
    return 0


if __name__ == "__main__":
    sys.exit(main())
