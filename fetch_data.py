"""
Oracle's Elixir 数据自动更新
=============================
从 Oracle's Elixir 的公开 Google Drive 文件夹拉取最新的赛季 CSV 到 data/。

    python fetch_data.py                只更新当年 + 上一年
    python fetch_data.py --years 2025 2026
    python fetch_data.py --all          全部 2014-2026
    python fetch_data.py --list         只列出远端文件, 不下载
    python fetch_data.py --retry 6      配额被打满时按小时退避重试
    python fetch_data.py --auth         一次性授权 (见下方「用自己的账号」)

上游情况 (实测 2026-08-16)
--------------------------
· Google Drive 是**唯一**分发渠道。oracleselixir.com/tools/downloads 上没有
  直链, 页面本身就写着「Access files via Google Drive」。

· 官网原文: "Data files are updated ONCE PER DAY. There is no value in
  downloading the files more frequently than this." 所以每天一次足够,
  不要做成高频轮询。

· **匿名下载有配额, 而且会被打满。** 实测四个文件 (含 2014 那个没人下的)
  全部返回一个标题为 "Google Drive - Quota exceeded" 的 HTML 页面, 说明
  配额是文件夹所有者账号级的, 不是单文件热度。配额通常 24 小时内恢复。

  这就是本脚本最重要的防线: 那个错误页是 HTTP **200**, 内容是 HTML。
  直接写进 data/ 就会用一个 2KB 的网页覆盖掉 58MB 的训练数据, 而且
  feature_store 对读不出内容的 CSV 是静默跳过的 —— 一次成功的「更新」
  可能悄悄让模型少一年数据。所以: 先下到临时文件, 校验通过才替换。


用自己的账号 (推荐, 匿名配额基本指望不上)
------------------------------------------
匿名下载共享的是文件所有者的配额, 打满了谁都下不了。用自己的账号认证后,
额度算你自己的。做法:

  1. 打开 https://console.cloud.google.com/  新建一个项目 (名字随便)
  2. 「API 和服务」→「库」→ 搜 Google Drive API → 启用
  3. 「API 和服务」→「OAuth 同意屏幕」→ 用户类型选「外部」, 填个应用名,
     测试用户里把自己的 Gmail 加进去
  4. 「凭据」→「创建凭据」→「OAuth 客户端 ID」→ 应用类型选「桌面应用」
  5. 下载那个 JSON, 改名放成本目录下的  google_credentials.json
  6. 跑一次  python fetch_data.py --auth   浏览器会弹出授权页, 点同意

之后 token 存在 google_token.json, 会自动续期, 不用再管。

两个文件都是凭据, **不要提交进 git, 权限收到 600**。本脚本只读它们,
不会把内容打印出来。申请的是 drive.readonly 这个只读权限, 拿不到写权限。

如果认证后仍然撞配额 (下别人的文件, 额度限制依然存在, 只是宽得多),
脚本会自动改走「先复制到你自己的云端硬盘, 再下载自己的副本, 下完删掉」
—— 自己拥有的文件没有这个限制。这条路需要 drive.file 权限, 见 SCOPES。
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

FOLDER_ID = "1gLSw0RLjBbtaNy0dgnGQDAZOHIgCe-HH"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36")
HEADERS = {"User-Agent": UA}

_HERE = Path(__file__).parent
DATA_DIR = _HERE / "data"
NAME_RE = re.compile(r"^(\d{4})_LoL_esports_match_data_from_OraclesElixir\.csv$")

# 凭据。两个都是机密, 不进 git。脚本只读, 不打印内容。
CRED_FILE = _HERE / "google_credentials.json"   # 你从 Cloud Console 下载的
TOKEN_FILE = _HERE / "google_token.json"        # --auth 之后自动生成

# drive.readonly 够读公开文件; drive.file 只能碰本程序自己创建的文件,
# 是「复制到自己盘再下载」那条兜底路径需要的。两个都不含全盘写权限。
SCOPES = ["https://www.googleapis.com/auth/drive.readonly",
          "https://www.googleapis.com/auth/drive.file"]

# 校验用: 真正的文件必须以这几列开头
EXPECTED_HEAD = "gameid,datacompleteness,url,league,year"
MIN_BYTES = 1_000_000          # 最小的 2014 文件也有几十 MB, 1MB 是宽松下限


class FetchError(RuntimeError):
    pass


class QuotaExceeded(FetchError):
    """Drive 的下载配额被打满。等一段时间会恢复。"""


# ── 认证 ──────────────────────────────────────────────────────────────
def _secure(path: Path):
    """尽量把权限收到 600。Windows 上 chmod 基本无效, 只做提示。"""
    try:
        os.chmod(path, 0o600)
    except Exception:
        pass


def authorize(force: bool = False):
    """一次性 OAuth。会打开浏览器让你点同意, 然后把 token 存到本地。

    这一步必须由文件的主人自己跑 —— 它会用到你的 Google 账号。
    """
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        raise FetchError("缺依赖, 先跑: pip install -r requirements.txt")

    if not CRED_FILE.exists():
        raise FetchError(
            f"找不到 {CRED_FILE.name}。\n"
            f"    去 https://console.cloud.google.com/ 建项目 → 启用 Drive API →\n"
            f"    凭据 → OAuth 客户端 ID → 桌面应用 → 下载 JSON →\n"
            f"    改名成 {CRED_FILE.name} 放到 {CRED_FILE.parent}")

    if TOKEN_FILE.exists() and not force:
        print(f"{TOKEN_FILE.name} 已存在。要重新授权就加 --force。")
        return

    flow = InstalledAppFlow.from_client_secrets_file(str(CRED_FILE), SCOPES)
    creds = flow.run_local_server(port=0, prompt="consent")
    TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")
    _secure(TOKEN_FILE)
    print(f"授权完成, token 已存到 {TOKEN_FILE.name}")
    print("这是凭据文件, 不要提交进 git。")


def load_creds():
    """有 token 就返回凭据, 没有返回 None (走匿名)。"""
    if not TOKEN_FILE.exists():
        return None
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
    except ImportError:
        print("  装了 token 但缺 google 库, 退回匿名模式")
        return None

    creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except Exception as e:
            # **过期的 token 比没有 token 更糟**: 没有 token 时下面会走匿名,
            # 而这里原来直接把异常抛穿, 整个抓取崩掉 —— 日志里只有三十行
            # google.auth 的 traceback, 看不出"要重新授权"这件事。
            #
            # 这个失败是**注定会周期性发生**的: Cloud Console 里应用处于
            # 「测试」状态时, refresh token 只活 7 天。实测 2026-08-30 授权,
            # 08-31 起连续 7 次日更全挂在这里, CSV 停更 5 天才被发现。
            #
            # 治本是把 Cloud 项目发布成「生产」(见文件顶部说明); 在那之前
            # 至少让日志一眼能看懂, 并退回匿名试一把。
            print(f"  token 刷新失败 ({type(e).__name__}: {e})")
            print("  ** 需要重新授权 **:  python fetch_data.py --auth --force")
            print("     根治: 把 Google Cloud 项目从「测试」发布为「生产」,")
            print("     否则 refresh token 每 7 天就会失效一次。")
            print("  先退回匿名模式再试 (共用配额, 经常也是满的)")
            return None
        TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")
        _secure(TOKEN_FILE)
    return creds if creds and creds.valid else None


def drive_service(creds):
    from googleapiclient.discovery import build
    return build("drive", "v3", credentials=creds, cache_discovery=False)


# ── 列目录 ────────────────────────────────────────────────────────────
def list_folder(folder_id: str = FOLDER_ID, creds=None) -> dict[str, str]:
    """{文件名: fileId}。

    有凭据就走正经 API; 没有就从公开文件夹的 HTML 里抠 —— fileId 在
    data-id 属性上, 文件名在紧随其后的 aria-label。后者靠页面结构, Drive
    改版就会失效, 所以能用 API 就用 API。
    """
    if creds is not None:
        svc = drive_service(creds)
        out, token = {}, None
        while True:
            resp = svc.files().list(
                q=f"'{folder_id}' in parents and trashed=false",
                fields="nextPageToken, files(id, name, size, modifiedTime)",
                pageSize=200, pageToken=token,
                supportsAllDrives=True, includeItemsFromAllDrives=True).execute()
            for f in resp.get("files", []):
                out[f["name"]] = f["id"]
            token = resp.get("nextPageToken")
            if not token:
                break
        if out:
            return out
        raise FetchError("API 列目录返回空 —— 检查 Drive API 是否已启用")

    r = requests.get(f"https://drive.google.com/drive/folders/{folder_id}",
                     headers=HEADERS, timeout=40)
    if "accounts.google.com" in r.url:
        raise FetchError("文件夹不可匿名访问 (被重定向到登录页)")
    r.raise_for_status()

    out: dict[str, str] = {}
    for m in re.finditer(r'data-id="([A-Za-z0-9_-]{20,60})"', r.text):
        tail = r.text[m.end(): m.end() + 1500]
        nm = re.search(r'aria-label="([^"]+?\.csv)\s', tail)
        if nm:
            out.setdefault(nm.group(1), m.group(1))
    if not out:
        raise FetchError("没能从文件夹页面解析出文件 —— Drive 的页面结构可能改了")
    return out


# ── 下载 ──────────────────────────────────────────────────────────────
def _looks_like_error_page(chunk: bytes) -> str | None:
    head = chunk[:2048].lower()
    if b"<html" not in head and b"<!doctype html" not in head:
        return None
    t = re.search(rb"<title>([^<]*)</title>", chunk[:4096])
    title = t.group(1).decode("utf-8", "replace") if t else "未知 HTML 页面"
    return title


def _validate_and_place(tmp: Path, dest: Path) -> int:
    """临时文件通过三关校验才原子替换。不过就删掉临时文件, 原文件不动。"""
    n = tmp.stat().st_size
    with open(tmp, "rb") as fh:
        first = fh.read(4096)

    title = _looks_like_error_page(first)
    if title:
        tmp.unlink(missing_ok=True)
        if "quota" in title.lower():
            raise QuotaExceeded(f"Drive 返回「{title}」—— 下载配额已满")
        raise FetchError(f"拿到的是网页而不是文件: {title}")

    head = first[:len(EXPECTED_HEAD)].decode("utf-8", "replace")
    if not head.startswith(EXPECTED_HEAD):
        tmp.unlink(missing_ok=True)
        raise FetchError(f"表头不对, 不是 Oracle's Elixir 的数据: {head!r}")

    if n < MIN_BYTES:
        tmp.unlink(missing_ok=True)
        raise FetchError(f"文件只有 {n} 字节, 太小, 不像完整数据")

    os.replace(tmp, dest)
    return n


def download_authed(file_id: str, dest: Path, creds) -> int:
    """用凭据下载。先直下; 撞配额就复制到自己的盘再下自己的副本。

    别人的文件即使认证后也有配额 (只是宽得多), 而自己拥有的文件没有 ——
    所以兜底路径是「复制 → 下载副本 → 删副本」。
    """
    from googleapiclient.http import MediaIoBaseDownload
    from googleapiclient.errors import HttpError

    svc = drive_service(creds)
    tmp = dest.with_suffix(dest.suffix + ".tmp")

    def _pull(fid):
        req = svc.files().get_media(fileId=fid, supportsAllDrives=True)
        with open(tmp, "wb") as fh:
            dl = MediaIoBaseDownload(fh, req, chunksize=8 << 20)
            done = False
            while not done:
                _, done = dl.next_chunk()

    try:
        _pull(file_id)
        return _validate_and_place(tmp, dest)
    except (QuotaExceeded, HttpError) as e:
        tmp.unlink(missing_ok=True)
        msg = str(e)
        if not isinstance(e, QuotaExceeded) and "quota" not in msg.lower() \
                and "rateLimit" not in msg:
            raise FetchError(f"下载失败: {msg}") from e

        print(f"       直下撞配额, 改走「复制到自己的云端硬盘」")
        copy_id = None
        try:
            copy = svc.files().copy(
                fileId=file_id, supportsAllDrives=True,
                body={"name": f"_oe_tmp_{dest.stem}"}).execute()
            copy_id = copy["id"]
            _pull(copy_id)
            return _validate_and_place(tmp, dest)
        finally:
            tmp.unlink(missing_ok=True)
            if copy_id:
                try:
                    svc.files().delete(fileId=copy_id,
                                       supportsAllDrives=True).execute()
                except Exception as ce:
                    print(f"       注意: 云端副本没删掉 ({copy_id}): {ce}")


def download(file_id: str, dest: Path, session: requests.Session | None = None) -> int:
    """下到 dest.tmp, 校验后原子替换。返回字节数。

    校验三关, 任何一关不过就保留原文件不动:
      1. 首块不是 HTML (配额页/确认页都是 HTML, 且 HTTP 200)
      2. 首行以已知表头开头
      3. 大小不小于 MIN_BYTES
    """
    s = session or requests.Session()
    url = (f"https://drive.usercontent.google.com/download"
           f"?id={file_id}&export=download&confirm=t")

    with s.get(url, headers=HEADERS, stream=True, timeout=120) as r:
        r.raise_for_status()
        it = r.iter_content(1 << 16)
        first = next(it, b"")

        title = _looks_like_error_page(first)
        if title:
            if "quota" in title.lower():
                raise QuotaExceeded(f"Drive 返回「{title}」—— 匿名下载配额已满")
            raise FetchError(f"Drive 返回的是网页而不是文件: {title}")

        head = first[:len(EXPECTED_HEAD)].decode("utf-8", "replace")
        if not head.startswith(EXPECTED_HEAD):
            raise FetchError(f"表头不对, 拿到的不是 Oracle's Elixir 的数据: {head!r}")

        tmp = dest.with_suffix(dest.suffix + ".tmp")
        n = 0
        with open(tmp, "wb") as fh:
            fh.write(first); n += len(first)
            for chunk in it:
                fh.write(chunk); n += len(chunk)

    if n < MIN_BYTES:
        tmp.unlink(missing_ok=True)
        raise FetchError(f"文件只有 {n} 字节, 太小, 不像完整数据")

    os.replace(tmp, dest)      # 原子替换, 中途失败不会留下半个文件
    return n


# ── 主流程 ────────────────────────────────────────────────────────────
def update(years: list[int], data_dir: Path = DATA_DIR,
           retries: int = 0, retry_gap: int = 3600) -> int:
    data_dir.mkdir(parents=True, exist_ok=True)

    creds = load_creds()
    if creds:
        print("认证模式 (用你自己的 Google 账号, 配额算你的)")
    else:
        print("匿名模式 —— 共用文件所有者的配额, 经常是满的。")
        print("  用自己的账号: python fetch_data.py --auth  (见文件顶部说明)")

    print("列出远端文件…")
    remote = list_folder(creds=creds)
    print(f"  远端 {len(remote)} 个文件")

    sess = requests.Session()
    changed = failed = 0

    for y in years:
        name = f"{y}_LoL_esports_match_data_from_OraclesElixir.csv"
        fid = remote.get(name)
        dest = data_dir / name
        before = dest.stat().st_size if dest.exists() else 0

        if not fid:
            print(f"  {y}  远端没有这个文件, 跳过")
            continue

        for attempt in range(retries + 1):
            try:
                n = (download_authed(fid, dest, creds) if creds
                     else download(fid, dest, sess))
                if n == before:
                    print(f"  {y}  {n:,} 字节 · 与本地相同, 无变化")
                else:
                    d = n - before
                    print(f"  {y}  {n:,} 字节 · 已更新 ({d:+,})")
                    changed += 1
                break
            except QuotaExceeded as e:
                if attempt < retries:
                    print(f"  {y}  {e}; {retry_gap//60} 分钟后重试 "
                          f"({attempt + 1}/{retries})")
                    time.sleep(retry_gap)
                else:
                    print(f"  {y}  {e}")
                    print(f"       本地文件保持不变 ({before:,} 字节)")
                    failed += 1
            except FetchError as e:
                print(f"  {y}  失败: {e}")
                print(f"       本地文件保持不变 ({before:,} 字节)")
                failed += 1
                break

    print()
    if changed:
        print(f"{changed} 个文件有更新 —— 需要重跑 train.py 和 ingame_train.py, "
              f"否则模型和数据不同步。")
    elif not failed:
        print("没有文件发生变化, 不需要重训。")
    return failed


def main():
    now = datetime.now().year
    ap = argparse.ArgumentParser(description="从 Oracle's Elixir 更新赛季 CSV")
    ap.add_argument("--years", nargs="+", type=int,
                    help=f"要更新的年份, 默认 {now-1} {now}")
    ap.add_argument("--all", action="store_true", help="全部年份")
    ap.add_argument("--list", action="store_true", help="只列出远端文件")
    ap.add_argument("--retry", type=int, default=0,
                    help="配额满时的重试次数, 每次间隔一小时")
    ap.add_argument("--auth", action="store_true",
                    help="一次性 OAuth 授权 (需要 google_credentials.json)")
    ap.add_argument("--force", action="store_true", help="配合 --auth: 重新授权")
    a = ap.parse_args()

    if a.auth:
        try:
            authorize(force=a.force)
            return 0
        except FetchError as e:
            print(f"授权失败: {e}")
            return 1

    if a.list:
        for n, f in sorted(list_folder(creds=load_creds()).items()):
            print(f"  {n:<60} {f}")
        return 0

    years = list(range(2014, now + 1)) if a.all else (a.years or [now - 1, now])
    return update(years, retries=a.retry)


if __name__ == "__main__":
    sys.exit(main())
