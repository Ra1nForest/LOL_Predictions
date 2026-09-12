"""计划任务的统一启动器: 无窗口 + 落日志。

为什么需要它 —— 2026-08-30 实测出来的三选一困境:

    python.exe  直接跑      -> 有控制台窗口, 每 2 分钟闪一次
    cmd.exe /c ... > log    -> 能落日志, 但 cmd 自己就是个窗口, 照样闪
                               (计划任务的 -Hidden 压不住它)
    pythonw.exe 直接跑      -> 没窗口, 但 sys.stdout 是 None, 脚本一 print 就崩

这个文件是第四条路: 用 pythonw 启动 (无窗口), 在**执行目标脚本之前**先把
sys.stdout/stderr 接到日志文件上, 于是三样全都有。

用法::

    pythonw run_task.py <日志名> <脚本> [参数...]
    pythonw run_task.py update daily_update.py --service LoL-Predict
    python  run_task.py collect collect_live.py --stdout   # 调试: 不重定向
"""
from __future__ import annotations

import os
import runpy
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAX_BYTES = 20 * 1024 * 1024        # 单个日志超过 20MB 就轮转一次


def _rotate(p: Path) -> None:
    """够大就改名成 .1, 只留一代。日志无人看管会长到占满磁盘。"""
    try:
        if p.exists() and p.stat().st_size > MAX_BYTES:
            old = p.with_suffix(p.suffix + ".1")
            if old.exists():
                old.unlink()
            p.rename(old)
    except OSError:
        pass                         # 轮转失败不该拖垮任务本身


def main() -> int:
    argv = sys.argv[1:]
    if len(argv) < 2:
        # 注意: pythonw 下这句也打不出来, 但 python 下能看到。
        print(__doc__)
        return 2

    name, script, rest = argv[0], argv[1], argv[2:]
    to_stdout = "--stdout" in rest
    if to_stdout:
        rest = [a for a in rest if a != "--stdout"]

    if not to_stdout:
        d = HERE / "logs"
        d.mkdir(parents=True, exist_ok=True)
        lf = d / f"{name}.log"
        _rotate(lf)
        f = open(lf, "a", buffering=1, encoding="utf-8", errors="replace")
        sys.stdout = f
        sys.stderr = f

    print(f"\n{'=' * 66}")
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {script} {' '.join(rest)}"
          f"   pid={os.getpid()}")
    print("=" * 66, flush=True)

    target = HERE / script
    if not target.exists():
        print(f"找不到脚本: {target}", flush=True)
        return 2

    # 目标脚本用 sys.argv 解析参数, 且大多按 __main__ 分支跑, 所以用
    # run_path(run_name="__main__") 而不是 import。
    sys.argv = [str(target)] + rest
    os.chdir(HERE)
    try:
        runpy.run_path(str(target), run_name="__main__")
        return 0
    except SystemExit as e:
        return int(e.code or 0)
    except Exception:
        import traceback
        traceback.print_exc()
        print(f"[{datetime.now():%H:%M:%S}] 异常退出", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
