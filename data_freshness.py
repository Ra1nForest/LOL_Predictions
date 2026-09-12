"""
数据陈旧告警
============

用旧数据跑出来的结果**看着完全正常** —— 脚本不报错, 图表照画, 指标照出,
只是悄悄少了几天的比赛。2026-08-20 就栽过一次: 本地 CSV 停在 08-16 而云端
已到 08-18, 回填因此少配了几十局, 是靠人问"OE 没更新吗"才发现的。

所以每个读 CSV 的入口都喊一嗓子。判据用**数据最后一天距今多少天**, 不是
文件的修改时间 —— 重新下载一份同样陈旧的文件, mtime 会变新, 骗过所有人。

**真值在服务器上**: 本机会关机、脚本可能没跑, 所以本地副本随时可能落后。
拉取方式见下面 SYNC_HINT。
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

# 超过这个天数就警告。OE 每天更新, 云端日更每天 12:20 UTC 拉一次,
# 所以正常情况下本地最多落后一两天。
STALE_DAYS = float(os.environ.get("LOL_STALE_DAYS", 3))

# 服务器地址不写进代码 (仓库是公开的): 从环境变量取, 没设就只给出命令的样子
_SRV = os.environ.get("LOL_SERVER", "<user>@<server>")
SYNC_HINT = (f"同步: scp -i <key> {_SRV}:"
             "/home/ubuntu/lol/service/data/*.csv data/")


def check(last_date, label="数据", quiet=False) -> float:
    """last_date 是数据里最后一场比赛的日期。返回距今天数。"""
    if last_date is None:
        if not quiet:
            print(f"  ⚠ {label}: 判断不出最后日期")
        return float("nan")
    try:
        import pandas as pd
        ts = pd.Timestamp(last_date)
        if ts.tzinfo is not None:
            ts = ts.tz_convert(None)
        days = (datetime.now(timezone.utc).replace(tzinfo=None) - ts.to_pydatetime()).days
    except Exception:
        return float("nan")

    if quiet:
        return days
    if days > STALE_DAYS:
        print(f"  ⚠ {label}截至 {ts.date()}, 距今 {days} 天 —— "
              f"云端很可能已经更新, 结果会少算这几天的比赛")
        print(f"    {SYNC_HINT}")
    else:
        print(f"  {label}截至 {ts.date()}, 距今 {days} 天  ✓")
    return days


def check_frame(df, col="date", label="数据", quiet=False) -> float:
    """从 DataFrame 里取最后日期再判。"""
    try:
        import pandas as pd
        s = pd.to_datetime(df[col], errors="coerce").dropna()
        return check(s.max() if len(s) else None, label, quiet)
    except Exception:
        return float("nan")
