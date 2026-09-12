"""
日更的运行留档
==============
`daily_update.py` 每跑完一轮写这里, `/health` 读这里。

为什么需要它
------------
日更失败是**完全静默**的: 它的 gate 设计得很好 (拉取失败就不往下走, 线上继续
用旧模型), 于是"坏了"和"没事"在外部看起来一模一样 —— 服务照常应答, 概率照常
输出, 只是数据悄悄停在几天前。

实测两次都是用户肉眼发现数据旧了才查出来的:
  · 2026-08-23 起连续两次失败
  · 2026-08-31 起连续七次失败, CSV 停更 5 天

`/health` 里本来就有 `data_freshness`, 但它回答的是"数据有多旧", 不是
**"更新流程还活着吗"**。两者会分叉: 赛区休赛时数据本来就旧 (LEC/LCS 实测
停在 08-30 是真没打), 而流程完全健康; 反过来流程死了好几天, 只要赛区也恰好
没比赛, `data_freshness` 依然报 fresh。所以必须单独记流程本身。

**拉取成功和整轮成功要分开记。**
这是从那次 OAuth 故障学到的: 当时失败的只有第一步 (拉数据), 而"数据没有变化
就正常退出 0"这条路径同样返回成功。只记一个布尔量的话, 两者混在一起, 看不出
是"上游没有新数据"还是"我们根本没连上上游"。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
STATUS_FILE = HERE / "update_status.json"

# 判级阈值。日更是**一天两次** (10:00 / 22:00), 所以间隔正常是 12 小时。
#   <= 26h   允许漏掉一次 + 余量
#   <= 72h   漏了几次, 但还不至于让模型不可用
#   >  72h   基本可以确定流程坏了
FRESH_H = 26
STALE_H = 72

_FMT = "%Y-%m-%dT%H:%M:%S"


def _now() -> str:
    return datetime.now().strftime(_FMT)


def read() -> dict:
    """原样读回留档。文件不存在或读坏了都返回 {} —— 这个模块**绝不能**
    因为自己的问题把 /health 弄挂: 它是来报告故障的, 不是来制造故障的。"""
    try:
        # utf-8-sig 而不是 utf-8: 有 BOM 就吃掉, 没有也照常读。
        # record() 自己写的没有 BOM, 但这台机器上 PowerShell 的
        # `Out-File -Encoding utf8` 是**带 BOM** 的 —— 手工改过一次这个文件,
        # json.loads 就抛异常, 被下面的兜底吞成"没有留档", 而这个文件的
        # 全部意义就是暴露故障。用 utf-8 读会让它自己变成一个静默故障。
        return json.loads(STATUS_FILE.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def record(outcome: str, *, stage: str | None = None, reason: str | None = None,
           fetch_ok: bool = False, exit_code: int | None = None) -> dict:
    """记一轮运行的结局。outcome 是 "ok" 或 "failed"。

    合并逻辑在这里而不在调用方: `last_success` / `last_fetch_ok` 要**跨失败
    保留**, 否则一失败就把"上次好的时候"擦掉, 而那恰恰是最需要的信息。
    `last_failure` 反过来在成功后也保留 —— 想知道"上次是怎么坏的"。
    """
    prev = read()
    now = _now()
    ok = outcome == "ok"

    cur = {
        "last_run": now,
        "outcome": outcome,
        "exit_code": exit_code,
        "last_success": now if ok else prev.get("last_success"),
        "last_fetch_ok": now if fetch_ok else prev.get("last_fetch_ok"),
        "consecutive_failures": 0 if ok else int(prev.get("consecutive_failures") or 0) + 1,
        "last_failure": ({"time": now, "stage": stage, "reason": reason}
                         if not ok else prev.get("last_failure")),
    }
    try:
        STATUS_FILE.write_text(json.dumps(cur, ensure_ascii=False, indent=1),
                               encoding="utf-8")
    except Exception as e:                        # 写不进去也不能中断日更
        print(f"  (留档写入失败, 不影响更新: {type(e).__name__}: {e})")
    return cur


def _hours_since(ts: str | None) -> float | None:
    if not ts:
        return None
    try:
        return round((datetime.now() - datetime.strptime(ts, _FMT)).total_seconds() / 3600, 1)
    except Exception:
        return None


def summary() -> dict:
    """给 /health 用: 留档 + 算好的判级。

    判级看的是 **last_fetch_ok**, 不是 last_success —— "数据没有变化" 也算
    整轮成功, 但它证明不了我们连得上上游; 而拉取成功证明得了。
    """
    s = read()
    if not s:
        return {"level": "unknown", "message": "还没有任何日更留档 —— "
                "要么从没跑过, 要么跑的是旧版本 (不写留档)。"}

    h = _hours_since(s.get("last_fetch_ok"))
    fails = int(s.get("consecutive_failures") or 0)

    if h is None:
        level = "unknown"
        msg = "从未成功拉取过数据。"
    elif h <= FRESH_H:
        level, msg = "fresh", None
    elif h <= STALE_H:
        level = "stale"
        msg = f"已经 {h} 小时没有成功拉取数据 (正常间隔 12 小时)。"
    else:
        level = "very_stale"
        msg = (f"已经 {h} 小时没有成功拉取数据 —— 日更流程基本可以确定是坏的。"
               f"看 logs/update.log。")

    fail = s.get("last_failure") or {}
    if fails:
        why = fail.get("reason") or "原因未记录"
        where = fail.get("stage") or "未知步骤"
        msg = (f"{msg + ' ' if msg else ''}"
               f"连续失败 {fails} 次, 最近一次停在「{where}」: {why}")

    return {
        "level": level,
        "message": msg,
        "last_run": s.get("last_run"),
        "outcome": s.get("outcome"),
        "last_success": s.get("last_success"),
        "last_fetch_ok": s.get("last_fetch_ok"),
        "hours_since_fetch_ok": h,
        "consecutive_failures": fails,
        # 失败详情始终带上 —— 成功之后也留着, 用来回答"上次是怎么坏的"
        "last_failure": s.get("last_failure"),
    }
