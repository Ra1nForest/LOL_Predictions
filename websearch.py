"""
检索后端 — 给 agent 一个真实的信息源
=====================================
支持三家, 按环境变量自动选择 (免费额度都够单人用):

    TAVILY_API_KEY    tavily.com        1000 次/月, 为 LLM 优化, 首选
    BRAVE_API_KEY     brave.com/search/api   2000 次/月
    SERPER_API_KEY    serper.dev        2500 次一次性

都没设置 → 退化成「无检索」模式, 服务照常跑, agent 回到只能说「无可补充」。

设计原则
--------
检索结果**只用于呈现事实**, 不用于自动调整概率。这条从 advisory-only
那次实测继承下来 —— 三个 agent 拿不到模型之外的信息时会编造理由把
校准过的概率往错误方向推 3pp。给了信息源之后风险变成另一种: 引用
一篇过期文章然后言之凿凿。所以:

    · 每条检索结果强制带 URL 和日期
    · 超过 N 天的结果标记为「陈旧」
    · agent 引用时必须指明来源, 不能凭空断言
"""
from __future__ import annotations
import os, re, json, time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone

STALE_DAYS = 45          # 超过这个天数标记为陈旧


@dataclass
class Hit:
    title: str
    url: str
    snippet: str
    published: str | None = None
    age_days: int | None = None
    stale: bool = False
    source: str = ""

    def line(self) -> str:
        age = ""
        if self.age_days is not None:
            age = f" [{self.age_days}天前{'·陈旧' if self.stale else ''}]"
        return f"· {self.title}{age}\n  {self.snippet[:220]}\n  {self.url}"


def _age(pub: str | None):
    if not pub:
        return None, False
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%Y/%m/%d",
                "%a, %d %b %Y %H:%M:%S %Z", "%a, %d %b %Y %H:%M:%S %z",
                "%Y-%m-%dT%H:%M:%SZ"):
        try:
            dt = datetime.strptime(pub[:len(fmt) + 6].strip(), fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            d = (datetime.now(timezone.utc) - dt).days
            # 负数 = 日期在未来。**必须单独挡掉** —— 它不是"更新鲜", 而是
            # 这条结果根本不可信 (日期解析错了, 或者是赛程预告类文章)。
            # 早先只判 d > STALE_DAYS, 于是 -11 天顺利通过: 8月20日那天,
            # 一条"8月30日 DK 2-1 击败 HLE"被当成既成事实进了简报, 裁定
            # 还给它盖了"站住了"。日期在未来 = 事情还没发生, 不能当证据。
            if d < -1:                      # 留一天余量给时区
                return d, True
            return d, d > STALE_DAYS
        except Exception:
            continue
    return None, False


# ── 后端 ──────────────────────────────────────────────

def _tavily(q, n, key):
    import requests
    # topic="news" 是**必须的** —— basic/advanced 搜索返回的结果里根本没有
    # published_date 字段 (实测: 只有 content/id/raw_content/score/title/url)。
    # 没有日期 -> 下面的 _age 返回 None -> 全部被当成"不陈旧", 于是一篇旧闻
    # 能被 agent 当成当前事实言之凿凿。真栽过一次: 2026-08 的辩论里引用旧文
    # 断言"JackeyLove 医疗休假", 而他 2026-03 就已经回归首发了。
    r = requests.post("https://api.tavily.com/search", timeout=25, json={
        "api_key": key, "query": q, "max_results": n, "topic": "news",
        "days": 365, "search_depth": "basic", "include_answer": False})
    r.raise_for_status()
    out = []
    for x in r.json().get("results", [])[:n]:
        d, st = _age(x.get("published_date"))
        out.append(Hit(x.get("title", "")[:160], x.get("url", ""),
                       (x.get("content") or "")[:400],
                       x.get("published_date"), d, st, "tavily"))
    return out


def _brave(q, n, key):
    import requests
    r = requests.get("https://api.search.brave.com/res/v1/web/search",
                     timeout=25, params={"q": q, "count": n},
                     headers={"X-Subscription-Token": key,
                              "Accept": "application/json"})
    r.raise_for_status()
    out = []
    for x in r.json().get("web", {}).get("results", [])[:n]:
        d, st = _age(x.get("age") or x.get("page_age"))
        out.append(Hit(x.get("title", "")[:160], x.get("url", ""),
                       re.sub("<[^>]+>", "", x.get("description", ""))[:400],
                       x.get("age"), d, st, "brave"))
    return out


def _serper(q, n, key):
    import requests
    r = requests.post("https://google.serper.dev/search", timeout=25,
                      headers={"X-API-KEY": key, "Content-Type": "application/json"},
                      json={"q": q, "num": n})
    r.raise_for_status()
    out = []
    for x in r.json().get("organic", [])[:n]:
        d, st = _age(x.get("date"))
        out.append(Hit(x.get("title", "")[:160], x.get("link", ""),
                       (x.get("snippet") or "")[:400],
                       x.get("date"), d, st, "serper"))
    return out


BACKENDS = [("TAVILY_API_KEY", _tavily), ("BRAVE_API_KEY", _brave),
            ("SERPER_API_KEY", _serper)]


def available() -> str | None:
    for env, _ in BACKENDS:
        if os.environ.get(env):
            return env.replace("_API_KEY", "").lower()
    return None


class Search:
    def __init__(self, max_calls: int = 12):
        self.backend = None
        for env, fn in BACKENDS:
            k = os.environ.get(env)
            if k:
                self.backend, self.key, self.name = fn, k, env.replace("_API_KEY", "").lower()
                break
        self.calls = 0
        self.max_calls = max_calls
        self.cache: dict[str, list[Hit]] = {}

    @property
    def enabled(self) -> bool:
        return self.backend is not None

    def query(self, q: str, n: int = 4) -> list[Hit]:
        if not self.enabled:
            return []
        if q in self.cache:
            return self.cache[q]
        if self.calls >= self.max_calls:
            return []
        try:
            self.calls += 1
            hits = self.backend(q, n, self.key)
        except Exception as e:
            # 静默返回空列表会让「key 无效」和「搜索无结果」长得一样。
            # 记下来, 至少能在日志里看到真相。
            self.last_error = f"{type(e).__name__}: {str(e)[:120]}"
            print(f"[websearch] 查询失败 ({q[:40]}): {self.last_error}", flush=True)
            hits = []
        self.cache[q] = hits
        return hits

    # ── 结构化预检索 ──
    def scout(self, blue: str, red: str, league: str) -> tuple[list[Hit], list[str]]:
        """比赛前跑的固定查询。返回 (去重后的结果, 用过的查询)."""
        yr = datetime.now().year
        qs = [
            f"{blue} roster change {yr}",
            f"{red} roster change {yr}",
            f"{league} {yr} roster news transfer",
            f"{blue} vs {red} {league} {yr}",
        ]
        seen, out = set(), []
        for q in qs:
            for h in self.query(q, 3):
                if h.url in seen:
                    continue
                seen.add(h.url); out.append(h)
        return out, qs

    def block(self, hits: list[Hit], title="外部检索结果") -> str:
        if not self.enabled:
            return ("外部检索: 未启用 (没有配置 TAVILY_API_KEY / BRAVE_API_KEY / "
                    "SERPER_API_KEY)。你没有模型之外的信息源。")
        if not hits:
            return f"{title}: 检索无结果。"
        # 三类, 不是两类。**日期未知 ≠ 新鲜** —— 早先把 age_days 为 None 的
        # 归进"新鲜", 结果一篇旧闻被 agent 当成当前事实: 2026-08 的辩论里
        # 断言"JackeyLove 医疗休假", 而他 2026-03 就已经回归首发。
        # 拿不准的必须让 agent 看见"拿不准", 而不是替它做乐观假设。
        fresh = [h for h in hits
                 if h.age_days is not None and not h.stale]
        # "来自未来"和"太旧"不可信的理由完全不同, 要分开说
        future = [h for h in hits
                  if h.age_days is not None and h.age_days < -1]
        stale = [h for h in hits
                 if h.age_days is not None and h.stale and h.age_days >= -1]
        undated = [h for h in hits if h.age_days is None]

        s = [f"{title} (来自 {self.name}, 共 {len(hits)} 条):"]
        if fresh:
            s.append(f"\n【{STALE_DAYS} 天内】")
            for h in fresh[:8]:
                s.append(h.line())
        if stale:
            s.append(f"\n【已超过 {STALE_DAYS} 天】引用前请核对是否仍然成立:")
            for h in stale[:4]:
                s.append(h.line())
        if future:
            s.append("\n【日期在未来】这些结果的日期晚于今天 —— 说明日期解析有误, "
                     "或者是赛程预告一类的前瞻文章。**里面描述的比赛结果尚未发生, "
                     "绝不能当既成事实引用**:")
            for h in future[:4]:
                s.append(h.line())
        if undated:
            s.append("\n【日期不明】**不要**用这些断言当前状态 —— "
                     "它们可能是几年前的旧闻。只能作为线索:")
            for h in undated[:4]:
                s.append(h.line())
        return "\n".join(s)


SEARCH_REQ = re.compile(r"^\s*SEARCH:\s*(.+?)\s*$", re.M | re.I)


def extract_requests(text: str, limit: int = 2) -> list[str]:
    """从 agent 输出里抓 `SEARCH: xxx` 形式的检索请求。"""
    return [m.group(1)[:120] for m in SEARCH_REQ.finditer(text or "")][:limit]


def strip_requests(text: str) -> str:
    return SEARCH_REQ.sub("", text or "").strip()
