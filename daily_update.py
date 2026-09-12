"""
每日无人值守更新
=================
拉新数据 → 训到暂存目录 → 指标过闸 → 换上线 → 重启服务 → 验活。

    python daily_update.py            完整流程
    python daily_update.py --dry-run  只拉数据和训练, 不换不重启
    python daily_update.py --force    数据没变也强制重训

设计前提: 没有人会看日志
-------------------------
「全自动」的风险不在下载, 在于**没人注意到指标变坏了**。所以这里不假设
有人事后检查, 而是让流水线自己把关:

  1. 新模型训到 artifacts_staging/, 不碰线上的 artifacts/
  2. 和当前线上指标逐项比对, 超过阈值就**拒绝上线**, 保留旧模型
  3. 只有过闸才原子替换并重启服务
  4. 换完探活 /health, 起不来就回滚

一次失败的更新应当留下「线上还是昨天那个能用的模型」, 而不是
「线上是个刚训出来、没人看过的模型」。

阈值 (见 GUARD) 是**退化**的容忍上限, 不是要求变好。模型指标本来就会
随验证集切分小幅波动 —— 实测同一份代码两次重训, pre/post_draft 的准确率
差异能有 1.2pp, 而闸门证明那只是噪声。阈值必须宽到容得下这种波动,
又要窄到能拦住真正的故障 (比如少加载了一年数据)。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Windows 计划任务里控制台是 GBK, 而本文件输出含 ✓ ✗ ⚠ − 这类 GBK 编不了的
# 字符 —— 不设这个会在 print 那一刻抛 UnicodeEncodeError, 表现为任务"跑到一半
# 消失"。2026-08-30 实测: 不加时 daily_update 正好崩在 [3/5] 过闸的判决行,
# 训练全做完了、结果打不出来, 整个任务算失败。
for _s in (sys.stdout, sys.stderr):
    if _s is not None and hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).parent
ART = HERE / "artifacts"
STAGING = HERE / "artifacts_staging"
PY = sys.executable

# 允许的退化幅度。超过就不上线。
GUARD = {
    "accuracy_drop": 0.030,      # 准确率最多掉 3 个百分点
    "ece_rise": 0.030,           # 校准误差最多涨 0.030
    "min_games": 8000,           # 训练场次不得低于这个数 (防少加载数据)
    "min_snapshots": 18000,      # Stage 4 快照数下限
}

ARTIFACT_FILES = [
    "model_pre_draft.json", "calib_pre_draft.json",
    "model_post_draft.json", "calib_post_draft.json",
    "model_ingame.json", "calib_ingame.json",
    "model_ingame_live.json", "calib_ingame_live.json",
    "summary.json",
]


def log(msg=""):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}" if msg else "", flush=True)


def run(cmd, env=None, timeout=3600) -> tuple[int, str]:
    e = dict(os.environ)
    if env:
        e.update(env)
    # Windows: 不加 CREATE_NO_WINDOW 的话, 每个子进程 (train.py /
    # ingame_train.py / schtasks) 都会闪一个控制台窗口 —— 即便父进程是
    # pythonw 且计划任务设了 Hidden。capture_output 只重定向句柄, 不阻止
    # 系统给控制台子系统程序分配窗口。
    flags = subprocess.CREATE_NO_WINDOW if sys.platform.startswith("win") else 0
    p = subprocess.run(cmd, cwd=HERE, env=e, timeout=timeout,
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", creationflags=flags)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


# ── 重启与探活 (跨平台) ───────────────────────────────────────────────
# 服务端用 systemd, 本地 Windows 用计划任务。这一段刻意写成平台感知而不是
# "失败就跳过" —— 因为跳过的后果是**静默使用旧模型**: artifacts 已经换了,
# 进程内存里还是旧的, 而日志上每一步都写着成功。
# 2026-08 服务器上就是这么坏了十天: NoNewPrivileges 让 sudo 失效, 日更连续
# 19 次走到最后一步才失败, 线上一直跑着六天前的模型。

IS_WIN = sys.platform.startswith("win")


def health_ok(url="http://127.0.0.1:8000/health", timeout=5) -> bool:
    """探活。用标准库而不是 curl —— Windows 上不保证有 curl, 而
    "curl 不存在" 和 "服务没起来" 会返回同样的失败, 分不出来。"""
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return 200 <= r.status < 300
    except Exception:
        return False


def service_exists(name: str) -> bool:
    if IS_WIN:
        rc, out = run(["schtasks", "/query", "/tn", name], timeout=30)
        return rc == 0
    rc, _ = run(["systemctl", "cat", name], timeout=30)
    return rc == 0


def service_restart(name: str) -> tuple[int, str]:
    if IS_WIN:
        # /end 对没在跑的任务会返回非 0, 那不是错误, 忽略即可。
        run(["schtasks", "/end", "/tn", name], timeout=60)
        time.sleep(2)
        return run(["schtasks", "/run", "/tn", name], timeout=60)
    return run(["sudo", "systemctl", "restart", name], timeout=120)


# ── 指标 ──────────────────────────────────────────────────────────────
def read_metrics(d: Path) -> dict:
    """从 artifacts 目录读出用于比对的指标。缺文件返回 {}。"""
    out = {}
    s = d / "summary.json"
    if s.exists():
        j = json.loads(s.read_text(encoding="utf-8"))
        for stage in ("pre_draft", "post_draft"):
            if stage in j:
                out[stage] = {"accuracy": j[stage]["accuracy"],
                              "ece": j[stage]["ece_cal"],
                              "n": j[stage]["n_train"] + j[stage]["n_cal"]
                                   + j[stage]["n_test"]}
    for name, key in (("calib_ingame.json", "ingame"),
                      ("calib_ingame_live.json", "ingame_live")):
        f = d / name
        if f.exists():
            m = json.loads(f.read_text(encoding="utf-8"))["metrics"]
            out[key] = {"accuracy": m["accuracy"], "ece": m["ece"],
                        "n": None}
    return out


def gate(old: dict, new: dict, games: int | None,
         snapshots: int | None) -> list[str]:
    """返回拒绝上线的理由。空列表 = 过闸。"""
    bad = []

    if games is not None and games < GUARD["min_games"]:
        bad.append(f"训练场次 {games} 低于下限 {GUARD['min_games']} "
                   f"—— 很可能有一年的数据没加载进来")
    if snapshots is not None and snapshots < GUARD["min_snapshots"]:
        bad.append(f"局内快照 {snapshots} 低于下限 {GUARD['min_snapshots']}")

    for stage, n in new.items():
        if stage not in old:
            continue          # 新增的模型, 没有可比对象
        o = old[stage]
        da = o["accuracy"] - n["accuracy"]
        de = n["ece"] - o["ece"]
        if da > GUARD["accuracy_drop"]:
            bad.append(f"{stage} 准确率 {o['accuracy']:.4f} → {n['accuracy']:.4f} "
                       f"(掉 {da*100:.1f}pp, 上限 {GUARD['accuracy_drop']*100:.0f}pp)")
        if de > GUARD["ece_rise"]:
            bad.append(f"{stage} ECE {o['ece']:.4f} → {n['ece']:.4f} "
                       f"(涨 {de:.4f}, 上限 {GUARD['ece_rise']:.3f})")

    missing = [f for f in ARTIFACT_FILES if not (STAGING / f).exists()]
    if missing:
        bad.append(f"暂存目录缺文件: {', '.join(missing)}")
    return bad


def show(old: dict, new: dict):
    keys = sorted(set(old) | set(new))
    log(f"  {'':<12}{'旧准确率':>10}{'新准确率':>10}{'Δ':>9}   "
        f"{'旧ECE':>8}{'新ECE':>8}{'Δ':>9}")
    for k in keys:
        o, n = old.get(k), new.get(k)
        if not n:
            log(f"  {k:<12}{'(新版缺失)':>28}")
            continue
        if not o:
            log(f"  {k:<12}{'—':>10}{n['accuracy']:>10.4f}{'新增':>9}")
            continue
        log(f"  {k:<12}{o['accuracy']:>10.4f}{n['accuracy']:>10.4f}"
            f"{(n['accuracy']-o['accuracy'])*100:>+8.2f}pp   "
            f"{o['ece']:>8.4f}{n['ece']:>8.4f}{n['ece']-o['ece']:>+9.4f}")


# ── 主流程 ────────────────────────────────────────────────────────────
def _oauth_hint(out: str) -> str | None:
    """拉取输出里认得出来的已知故障, 翻译成一句能照做的话。

    这个失败**发生过两次**且完全静默 (见 update_status 的模块注释), 而它在
    日志里的样子是三十行 google.auth 的 traceback —— 留档里塞一整段 traceback
    等于没记。认得出来的就写清楚, 认不出来的再退回截原文。
    """
    if "invalid_grant" in out or "Token has been expired or revoked" in out:
        return ("Google OAuth token 已过期或被吊销 —— 跑 "
                "`python fetch_data.py --auth --force` 重新授权; "
                "若每 7 天复发一次, 说明 Cloud 项目仍是 Testing 状态")
    if "配额" in out or "quota" in out.lower():
        return "Drive 配额被打满 (匿名模式共用文件所有者的额度)"
    return None


def _tail_reason(out: str, n: int = 2) -> str:
    """截最后几行非空输出当原因。留档要能一眼看完, 不塞整段 traceback。"""
    lines = [l.strip() for l in out.strip().splitlines() if l.strip()]
    return " / ".join(lines[-n:])[:400] if lines else "无输出"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="不换不重启")
    ap.add_argument("--force", action="store_true", help="数据没变也重训")
    ap.add_argument("--service", default="lol-predict", help="systemd 单元名 (Windows 上是计划任务名)")
    ap.add_argument("--no-restart", action="store_true", help="换了但不重启服务")
    a = ap.parse_args()

    code, stage, reason, fetch_ok = _pipeline(a)

    # 留档。**--dry-run 不记** —— 它不替换也不重启, 记进去会把"上次真正成功
    # 更新"的时间冲掉, 而那个时间正是判断流程死活的依据。
    if not a.dry_run:
        try:
            import update_status
            update_status.record("ok" if code == 0 else "failed",
                                 stage=stage, reason=reason,
                                 fetch_ok=fetch_ok, exit_code=code)
        except Exception as e:
            log(f"      (留档失败, 不影响更新结果: {type(e).__name__}: {e})")
    return code


def _pipeline(a) -> tuple[int, str | None, str | None, bool]:
    """跑完整流程。返回 (退出码, 失败停在哪一步, 原因, 拉取是否成功)。

    抽出来是为了让**所有**出口都经过同一处留档 —— 这里有九个 return、五种
    退出码, 在每个 return 旁边各写一次 record() 迟早会漏掉一个, 而漏掉的那个
    恰好就是没人注意到的那种失败。
    """
    log("=" * 66)
    log("每日更新")
    log("=" * 66)

    # 1. 拉数据
    log("[1/5] 拉取数据")
    rc, out = run([PY, "fetch_data.py", "--retry", "3"], timeout=7200)
    for line in out.strip().splitlines():
        log(f"      {line}")
    if rc != 0 and "已更新" not in out:
        log("      拉取失败, 且没有任何文件更新 —— 本次到此为止, 线上不动")
        return 1, "拉取数据", (_oauth_hint(out) or _tail_reason(out)), False
    changed = "已更新" in out
    if not changed and not a.force:
        log("      数据没有变化, 不需要重训。")
        # 拉取本身是成功的 —— 这一点必须记下来, 它是判断"连不连得上上游"的
        # 唯一证据。上游没有新数据和我们连不上上游, 结局都是"不重训"。
        return 0, None, None, True

    # 2. 训到暂存
    log("[2/5] 训练到暂存目录 (不碰线上 artifacts/)")
    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True)
    env = {"LOL_ARTIFACTS": str(STAGING)}

    games = snapshots = None
    for script in ("train.py", "ingame_train.py"):
        log(f"      {script} …")
        rc, out = run([PY, script], env=env, timeout=7200)
        if rc != 0:
            log(f"      {script} 失败 (rc={rc}), 线上保持不变:")
            for line in out.strip().splitlines()[-15:]:
                log(f"        {line}")
            return 1, f"训练 ({script})", _tail_reason(out), True
        for line in out.splitlines():
            s = line.strip()
            if s.endswith("games"):
                try: games = int(s.split()[0])
                except Exception: pass
            if "个快照" in s and "→" in s:
                try: snapshots = int(s.split("→")[1].split()[0])
                except Exception: pass
    log(f"      训练场次 {games}   局内快照 {snapshots}")

    # 3. 过闸
    log("[3/5] 指标过闸")
    old, new = read_metrics(ART), read_metrics(STAGING)
    show(old, new)
    reasons = gate(old, new, games, snapshots)
    if reasons:
        log("      ✗ 未过闸, 拒绝上线。线上仍是原来的模型:")
        for r in reasons:
            log(f"        · {r}")
        log(f"      新模型留在 {STAGING.name}/ 供人工查看, 没有被丢弃。")
        return 2, "指标过闸", "; ".join(reasons)[:400], True
    log("      ✓ 过闸")

    if a.dry_run:
        log("[4/5] --dry-run: 不替换, 不重启")
        return 0, None, None, True

    # 4. 原子替换
    # 预测留档的结果回填 —— 必须放在数据刷新之后: 比赛打完当天 OE 还没收录,
    # 要等下一次日更把新 CSV 拉下来才能查到 result。放这儿正好每天补一次。
    # 失败不影响主流程, 它是旁路。
    try:
        from prediction_log import resolve as _resolve_preds
        n = _resolve_preds()
        log(f"      预测留档: 回填 {n} 条比赛结果")
    except Exception as e:
        log(f"      预测留档回填失败(不影响更新): {type(e).__name__}: {e}")

    log("[4/5] 替换线上 artifacts")
    backup = HERE / "artifacts_prev"
    if backup.exists():
        shutil.rmtree(backup)
    if ART.exists():
        shutil.copytree(ART, backup)     # 留一份回滚用
    for f in ARTIFACT_FILES:
        shutil.copy2(STAGING / f, ART / f)
    log(f"      已替换 {len(ARTIFACT_FILES)} 个文件 (旧的备份在 artifacts_prev/)")

    # 5. 重启 + 探活
    if a.no_restart:
        log("[5/5] --no-restart: 跳过")
        return 0, None, None, True

    # 服务可能压根没装 (本项目就有过这种状态: unit 文件不存在、8000 端口
    # 无监听)。那不算失败 —— 新模型已经就位, 下次服务起来自然会加载。
    if not service_exists(a.service):
        log(f"[5/5] 系统里没有 {a.service} 这个服务/任务, 跳过重启。")
        log("      新模型已就位; 服务启动时会加载它。")
        log("      注意: 服务此刻仍在用旧模型, 直到它下次启动。")
        return 0, None, None, True

    log(f"[5/5] 重启 {a.service} 并探活")
    rc, out = service_restart(a.service)
    if rc != 0:
        log(f"      重启失败: {out.strip()[:300]}")
        # 这一条格外要紧: artifacts 已经换了, 进程内存里还是旧的 —— 服务器上
        # 就这么静默跑了十天旧模型 (见文件上方"重启与探活"那段注释)。
        return 3, "重启服务", f"{_tail_reason(out)} —— artifacts 已替换但服务没重启, 线上仍在用旧模型", True

    for i in range(60):
        time.sleep(5)
        if health_ok():
            log(f"      服务已就绪 (约 {(i+1)*5} 秒)")
            log("完成。")
            return 0, None, None, True
    # 起不来 -> 回滚
    log("      探活超时, 回滚到上一版 artifacts")
    for f in ARTIFACT_FILES:
        if (backup / f).exists():
            shutil.copy2(backup / f, ART / f)
    service_restart(a.service)
    log("      已回滚并重启。请人工检查。")
    return 4, "重启后探活", "新模型换上后服务起不来, 已回滚到上一版 artifacts", True


if __name__ == "__main__":
    sys.exit(main())
