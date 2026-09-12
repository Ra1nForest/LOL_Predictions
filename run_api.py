"""Windows 上启动 API 的入口。存在的理由是三个坑, 每个都会静默出事。

1. **不能用 `pythonw.exe`。** 它把 `sys.stdout`/`sys.stderr` 置为 None, 而
   uvicorn 启动时第一件事就是写日志 —— 立刻崩, 退出码 1。又因为没有控制台,
   错误无处可见, 表现为"任务跑了、失败了、查不出原因"。

2. **不能用 `cmd.exe /c python ... > log` 那种包装。** 计划任务的
   `schtasks /end` 只终止 cmd 外壳, python 子进程被孤儿化后继续占着 8000
   端口。于是 `daily_update.py` 换完 artifacts 去重启服务, 重启"成功"了,
   探活也过了 —— 应答的却是那个还在跑旧模型的孤儿。这和 2026-08 服务器上
   `NoNewPrivileges` 导致的那个 bug 是同一个形状: 每一步都报成功, 线上悄悄
   用着过时的模型。

3. 所以这里用单进程: 计划任务直接拥有它, `/end` 能真正杀掉; 日志由 Python
   自己重定向到文件, 不依赖外部 shell。

用法::

    python run_api.py                      # 默认 127.0.0.1:8000
    python run_api.py --port 8010 --no-log # 前台调试
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--log", default=str(HERE / "logs" / "predict.log"))
    ap.add_argument("--no-log", action="store_true",
                    help="不重定向, 日志留在终端 (调试用)")
    a = ap.parse_args()

    if not a.no_log:
        p = Path(a.log)
        p.parent.mkdir(parents=True, exist_ok=True)
        # 行缓冲 + errors="replace": Windows 控制台/文件的编码问题不该让
        # 服务本身挂掉。
        f = open(p, "a", buffering=1, encoding="utf-8", errors="replace")
        sys.stdout = f
        sys.stderr = f
        print(f"\n{'=' * 60}\n启动 {datetime.now():%Y-%m-%d %H:%M:%S}  "
              f"{a.host}:{a.port}  pid={__import__('os').getpid()}\n{'=' * 60}",
              flush=True)

    import uvicorn
    uvicorn.run("api:app", host=a.host, port=a.port, workers=1,
                log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
