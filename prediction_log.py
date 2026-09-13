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


def log_prediction(*, match_id, game_id, league, blue, red, probability_blue,
                   source, minute=None, slice_used=None, game_number=None,
                   golddiff=None, blue_kills=None, red_kills=None):
    """记一条。任何异常都不往外抛 —— 这是旁路, 不能拖垮预测接口。"""
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
                "source": source,              # pre_draft | post_draft | ingame
                "slice_used": slice_used,
                "golddiff": golddiff,
                "blue_kills": blue_kills, "red_kills": red_kills,
                "y": None,                     # 事后由 --resolve 回填
            }
            LOG.parent.mkdir(parents=True, exist_ok=True)
            with open(LOG, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"  ⚠ 预测留档失败(不影响预测) {type(e).__name__}: {e}", flush=True)


# ── 事后回填结果 ─────────────────────────────────────────────────────
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
    """
    sys.path.insert(0, str(_ROOT))
    sys.path.insert(0, str(_ROOT / "research"))
    from datetime import datetime as _dt
    from backfill_late import oe_index, find_oe, LEAGUE_TO_OE

    rows = _rows()
    todo = [r for r in rows if r.get("y") is None]
    if not todo:
        print("没有待回填的记录")
        return 0
    print(f"待回填 {len(todo)} 条")
    oe = oe_index()

    n = 0
    for r in rows:
        if r.get("y") is not None:
            continue
        try:
            when = _dt.fromisoformat(r["t"])
        except Exception:
            continue
        lg = LEAGUE_TO_OE.get(r.get("league"), r.get("league"))
        row = find_oe(oe, lg, when, r.get("blue"), r.get("red"),
                      r.get("game_number") or 1, window_h=window_h)
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
    print(f"回填 {n} 条")
    return n


def _game_key(r) -> str:
    return r.get("game_id") or f"{r.get('match_id')}#{r.get('game_number')}"


def score():
    """真账。和留出集的数字对照着看 —— 差得多就说明线上出了离线看不见的问题。

    **按局计分, 不按条。** 看板每局每分钟记一条: 2026-09-13 时 2064 条其实只有 116 局, 打 41 分钟
    的局记 41 条、20 分钟的只记 20 条左右。按条平均, 长局就被多算几遍 —— 而长局恰恰是拉锯和
    翻盘的局, 当时按条准确率 69.7%, 按局 75.7%; 样本量也显得比实际大 (同一局几十条共享一个胜负)。
    所以这里每条记录的权重是 1/本局条数, 每局合起来算一票; ± 是按局算的 95% 区间。
    另给固定时刻 (第 10/15/20/25 分钟各取一条) 的切面, 和留出集的 per_T 同一口径, 可以直接对照。
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

    # 固定时刻: 每局取离第 T 分钟最近、相差不超过 1 分钟的那一条
    print("\n按固定时刻 (每局一条, 和留出集的 per_T 同一口径):")
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
            print(f"  第 {T:>2} 分钟  {len(idx):>4} 局   准确率 {hit[idx].mean():.3f}   "
                  f"Brier {sq[idx].mean():.4f}")

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
