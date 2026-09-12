"""
把网站用的模型发布到 GitHub Pages
==================================
`daily_update.py --publish-web` 在新模型换上线之后调它; 也可以手动跑:

    python tools/publish_web.py            导出 → 对账 → 提交 → 推送
    python tools/publish_web.py --no-push  只导出、对账、提交, 不推

只在本机用。服务器上没有 Node (对账跑不了), 也不是 git 仓库。

四步, 以及每一步为什么这么保守
------------------------------
1. `tools/export_web_model.py` 从 artifacts/ 和五年 CSV 导出 frontend/public/web/*.json,
   顺带生成黄金用例。
2. `node frontend/scripts/golden.mjs` 逐位对账。**不过就不发布**, 并把导出的文件还原 ——
   网站上挂一个两份实现对不上的模型, 比挂一个旧模型更糟: 旧的至少对得上。
3. 只提交 frontend/public/web/ 下的文件。工作区里别的改动 (改到一半的代码) 一律不碰。
4. 推送前确认本地没有**别的**未推送提交。这是无人值守的推送, 推上去就公开了、收不
   回来, 它只该发布它自己产出的东西。之前自动提交过但没推成功的 (比如断网), 只要也只
   动了 public/web, 就一起推。远端比本地新 (有人在别处推过) 也拒绝 —— 不替人合并。

git 用凭据管理器里存的登录 (第一次手动 push 时弹窗登录的那个)。它失效时这里**不会
弹窗** (GCM_INTERACTIVE=never, 计划任务里弹了也没人点), 而是失败, 由 daily_update
记进 update_status.json 的 web_publish, /health 里看得见。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# 同 daily_update: 输出里有 GBK 编不了的字符, 计划任务里不设会在 print 时崩
for _s in (sys.stdout, sys.stderr):
    if _s is not None and hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
WEB = "frontend/public/web"
BRANCH = "main"
IS_WIN = sys.platform.startswith("win")


class PublishError(Exception):
    pass


def sh(cmd: list, timeout: int = 600, check: bool = True) -> str:
    env = dict(os.environ, PYTHONIOENCODING="utf-8",
               GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="never")
    # 不加 CREATE_NO_WINDOW, 每个 git / node 子进程都会闪一个控制台窗口 (见 daily_update.run)
    flags = subprocess.CREATE_NO_WINDOW if IS_WIN else 0
    p = subprocess.run(cmd, cwd=ROOT, env=env, timeout=timeout, capture_output=True,
                       text=True, encoding="utf-8", errors="replace", creationflags=flags)
    out = ((p.stdout or "") + (p.stderr or "")).strip()
    if check and p.returncode != 0:
        tail = " / ".join(out.splitlines()[-3:])[:400]
        what = " ".join([Path(str(cmd[0])).stem] + [str(c) for c in cmd[1:3]])
        raise PublishError(f"{what} 失败 (rc={p.returncode}): {tail or '无输出'}")
    return out


def git(*args: str, **kw) -> str:
    return sh(["git", *args], **kw)


def only_web_paths(paths) -> bool:
    """这些路径是不是全在 public/web 下 —— 自动流程只准推它自己产出的文件。"""
    paths = [p.strip() for p in paths if p.strip()]
    return bool(paths) and all(p.startswith(WEB + "/") for p in paths)


def publish(push: bool = True) -> str:
    if git("rev-parse", "--is-inside-work-tree", check=False) != "true":
        raise PublishError("不在 git 仓库里")
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    if branch != BRANCH:
        raise PublishError(f"当前分支是 {branch}, 不是 {BRANCH} —— 不在别的分支上自动提交")
    node = shutil.which("node")
    if not node:
        raise PublishError("找不到 node, 没法对账")

    try:
        print("[1/4] 导出网站模型", flush=True)
        out = sh([sys.executable, "tools/export_web_model.py"], timeout=1800)
        for line in out.splitlines()[-7:]:
            print(f"      {line}")
        print("[2/4] 浏览器版逐位对账 (test:golden)", flush=True)
        out = sh([node, "frontend/scripts/golden.mjs"], timeout=900)
        for line in out.splitlines()[-4:]:
            print(f"      {line}")
    except (PublishError, subprocess.TimeoutExpired):
        # 导出了一半或对不上的文件不能留在工作区, 否则下次手动提交会把它带上去
        git("checkout", "--", WEB, check=False)
        raise

    print("[3/4] 提交", flush=True)
    if not git("status", "--porcelain", "--", WEB):
        return "网站模型没有变化, 不需要发布"
    git("add", "--", WEB)
    teams = json.loads((ROOT / WEB / "teams.json").read_text(encoding="utf-8"))
    ig = json.loads((ROOT / WEB / "ingame_live.json").read_text(encoding="utf-8"))
    title = f"网站模型更新: 数据截至 {teams.get('data_through')}"
    body = (f"局内模型训练至 {ig.get('trained_through')}。"
            f"由 daily_update.py --publish-web 自动提交, 已通过 test:golden 逐位对账。")
    # 带路径的 commit 只提交这些路径, 别处已暂存的改动原样留着
    git("commit", "-q", "-m", title, "-m", body, "--", WEB)
    head = git("rev-parse", "--short", "HEAD")
    if not push:
        return f"已提交 {head}, 按 --no-push 没有推送"

    print("[4/4] 推送", flush=True)
    git("fetch", "-q", "origin", BRANCH, timeout=120)
    behind = int(git("rev-list", "--count", f"HEAD..origin/{BRANCH}"))
    if behind:
        raise PublishError(f"远端比本地多 {behind} 个提交 —— 不替人合并, 先手动 git pull; "
                           f"新模型已提交在本地 ({head}), 下次会一起推")
    ahead = git("rev-list", f"origin/{BRANCH}..HEAD").split()
    for c in ahead:
        files = git("diff-tree", "--no-commit-id", "--name-only", "-r", c).splitlines()
        if not only_web_paths(files):
            raise PublishError(f"本地有别的未推送提交 ({c[:7]} 改了 {WEB} 以外的文件) —— "
                               f"自动流程只推它自己的提交, 请手动 push")
    git("push", "-q", "origin", f"HEAD:{BRANCH}", timeout=180)
    return f"已推送 {len(ahead)} 个提交, 最新 {head}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-push", action="store_true", help="只导出、对账、提交, 不推送")
    a = ap.parse_args()
    # 最后一行固定以"结果:"开头, daily_update 从这里取一句话记进留档
    try:
        print(f"结果: {publish(push=not a.no_push)}")
        return 0
    except (PublishError, subprocess.TimeoutExpired) as e:
        print(f"结果: 失败 —— {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
