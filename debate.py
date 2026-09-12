"""
多轮辩论 — 不是三个观点然后摘要
================================
旧版的问题: 三个 agent 并行发言, 协调者压成一段摘要。分歧被抹平了,
没被挑战过的说法和经受住反驳的说法看起来一样可信。

新协议
------
  Round 0  预检索      服务端跑结构化查询, 结果注入简报
  Round 1  独立分析    三个 agent 各自表态, 可以发 `SEARCH: xxx` 要资料
  Round 2  交叉质证    每个 agent 看到另外两人的原话, 被要求明确
                       「反驳哪一条」「让步哪一条」「维持哪一条」
                       新的检索结果也在这一轮注入
  Round 3  裁定        协调者只做一件事: 报告哪些主张活了下来

关键设计
--------
· 概率不变。Round 3 不产出数字 —— 这条从 advisory-only 的实测继承。
· 每条主张标注状态: survived / conceded / contested / refuted
· external_fact 必须带 URL, 否则降级为 reinterpretation
· 分歧不被抹平 —— contested 就明确写着 contested

为什么这比摘要好: 一条被两个 agent 挑战过仍然站住的说法, 比一条
从没被质疑的说法可信得多。旧版把这个区别丢了。
"""
from __future__ import annotations
import os, re, json, time
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import websearch as WS

BASE_URL = "https://integrate.api.nvidia.com/v1"

_DEFAULTS = {
    # 四个角色刻意用四个**不同血统**的模型 —— 同源模型盲区相同, 交叉质证会
    # 退化成三个人一起点头。
    #
    # 这份默认值通常被 agent_config.json 覆盖 (见 _models()), 那份由
    # research/screen_agents.py 实测生成 —— 它测的是"会不会编造 / 会不会
    # 失控输出 / JSON 合不合规", 而不只是"能不能返回文字"。
    #
    # 2026-08-20 的教训: 探活时如果给的 max_tokens 太小 (200), 推理型模型
    # 会把 token 全花在内部推理上, 返回空 content, 看起来像"模型坏了"。
    # 真实调用给的是 1200, 这些模型都正常。**别用小 max_tokens 判死刑。**
    "验证者": "stepfun-ai/step-3.7-flash",
    "补充者": "openai/gpt-oss-120b",
    "质疑者": "nvidia/nemotron-3-ultra-550b-a55b",
    "协调者": "google/gemma-4-31b-it",
}


JSON_SCORE: dict[str, float] = {}       # 由 _models() 填充, 兜底排序用


def _models() -> dict:
    p = Path(__file__).with_name("agent_config.json")
    if p.exists():
        try:
            cfg = json.loads(p.read_text(encoding="utf-8"))
            for k, v in cfg.items():
                if "model" in v:
                    JSON_SCORE[v["model"]] = float(v.get("json_score", 0.0))
            return {k: v["model"] for k, v in cfg.items() if "model" in v}
        except Exception:
            pass
    return dict(_DEFAULTS)


MODELS = _models()

# ── 熔断 ──────────────────────────────────────────────
# NIM 免费额度会整段整段地降级: 实测 google/gemma-4-31b-it 从筛选时的
# 3.2 秒掉到「3 次挂 2 次, 成功那次 26 秒」, 而同期 nemotron 稳定在 5 秒。
# 没有熔断的话, 每一场辩论都要先在死掉的主模型上烧满重试时间才轮到兜底 ——
# 直播时用户干等 60 秒, 而这 60 秒是**可以预知**的浪费: 上一次刚失败过。
#
# 用时间窗而不是永久拉黑, 因为降级也会自己好: 窗口一过就再试一次, 通了
# 就自动恢复。这样配置不用改, 两个方向都自愈。
VERDICT_RESERVE = 120    # 攥给裁定的秒数, 辩论阶段不许动
_TRIP_AFTER = 2          # 连续失败几次就熔断
_TRIP_SEC = 300          # 熔断多久
_FAILS: dict[str, list] = {}      # model -> [连续失败次数, 恢复时间戳]


def _tripped(model: str) -> bool:
    st = _FAILS.get(model)
    return bool(st and st[0] >= _TRIP_AFTER and time.monotonic() < st[1])


def _note(model: str, ok: bool):
    if ok:
        _FAILS.pop(model, None)
        return
    st = _FAILS.setdefault(model, [0, 0.0])
    st[0] += 1
    if st[0] >= _TRIP_AFTER:
        st[1] = time.monotonic() + _TRIP_SEC
        print(f"[debate] {model} 连续失败 {st[0]} 次, 熔断 {_TRIP_SEC} 秒",
              flush=True)

_COT = ["we need to", "we should", "the instruction", "the user wants",
        "let me", "i should", "we can say", "so we can",
        "但我们需要", "指令要求", "用户要求"]


def _clean(txt: str) -> tuple[str, str | None]:
    if not txt:
        return "", "空回复"
    for a, b in (("<think>", "</think>"), ("<thinking>", "</thinking>")):
        if a in txt:
            txt = (txt.split(b)[-1] if b in txt else txt.split(a)[0]).strip()
    txt = txt.strip()
    if not txt:
        return "", "仅有思维链"
    low = txt.lower()
    if sum(1 for m in _COT if m in low) >= 2 and len(txt) > 150:
        return txt, "疑似思维链泄漏"
    return txt, None


# ══════════════════════════════════════════════════════════
#  Prompts
# ══════════════════════════════════════════════════════════

_SEARCH_NOTE = (
    "\n\n如果你需要额外资料, 单独起一行写 `SEARCH: <查询词>` (最多两条), "
    "下一轮会把结果给你。没有需要就不要写。"
)

R1 = {
    "验证者": ("你审查一个量化模型的预测是否自洽。给你的是预测概率、权重最高的"
              "特征证据、以及外部检索结果。判断: 预测是否由证据支撑, 有无内部矛盾。"
              "不要编造简报里没有的事实。150 字以内, 用用户的语言。"),
    "补充者": ("detailed thinking off\n\n"
              "你只报告模型结构上看不到的**外部事实**: 有名有姓有日期的转会、"
              "具体的版本改动、已确认的缺阵。\n"
              "规则: 每条必须引用检索结果里的 URL。检索里没有的, 不要说。"
              "简报里已有的信息(数据滞后、英雄无记录)不归你管。\n"
              "确实没有可引用的外部事实 → 只输出「无可补充」。"
              "禁止用「若/如果」揣测。150 字以内, 用用户的语言。"),
    "质疑者": ("你论证这个预测为什么可能是错的。找: 过拟合迹象、小样本特征、"
              "优势是否集中在单一特征、置信度是否过高、数据时效问题。"
              "具体、对抗性, 不要空泛。150 字以内, 用用户的语言。"),
}

R2 = ("这是第二轮。你已经看到另外两位的原话。\n\n"
      "对他们的每一条实质主张, 明确表态。用这个格式, 一行一条:\n"
      "  反驳 | <对方的主张> | <理由>\n"
      "  让步 | <你自己之前的哪条主张> | <为什么改口>\n"
      "  维持 | <你自己的主张> | <面对质疑为何仍成立>\n\n"
      "只针对实质分歧。同意的地方不必逐条附和。\n"
      "如果对方引用了检索结果, 检查那条资料的日期是否仍然有效。\n"
      "200 字以内, 用用户的语言。")

R3 = ("你是裁定者。给你三位 agent 的两轮发言。\n\n"
      "**你不调整概率。** 模型经过留出集校准, agent 没有独立的定价依据。\n"
      "你的工作是裁定每条主张的状态。严格输出 JSON, 不要 markdown 围栏:\n"
      '{"claims": [{"claim": "<25字内>", "raised_by": "<谁>", '
      '"status": "survived"|"conceded"|"contested"|"refuted", '
      '"kind": "external_fact"|"reinterpretation"|"error", '
      '"evidence_url": "<有则填, 无则空字符串>", "note": "<20字内>"}], '
      '"consensus": "<各方一致同意的部分, 60字内>", '
      '"open_disputes": ["<仍有分歧的点, 各20字内>"], '
      '"agent_errors": ["<agent 对简报的事实性误读>"]}\n\n'
      "status 定义:\n"
      "  survived  被质疑过但站住了 —— 最可信\n"
      "  conceded  提出者自己让步了\n"
      "  contested 双方各执一词, 未收敛\n"
      "  refuted   被有效反驳\n\n"
      "kind 定义:\n"
      "  external_fact  具名、带日期、且 evidence_url 非空的外部事件。"
      "没有 URL 的一律不算 external_fact。\n"
      "  reinterpretation  对简报内已有证据的重读。一个「类别」的盲区"
      "(如『转会会影响结果』)算这一类, 不算 external_fact。\n"
      "  error  该 agent 说错了简报里的内容\n\n"
      "任何带具体数字的断言 (胜率、翻盘率、百分比区间), 若 agent 没有标注来源或说明推导过程, 一律列入 agent_errors —— 即使那个数字看起来合理。无来源的定量断言不是论据。\n"
            "核对每个数字声明。常见错误: 混淆特征重要性与样本量; 把 ECE"
      "(无方向的绝对偏差均值)当成有方向的高估; 误读特征取值; 算错区间距离。\n"
      "用用户的语言。")


@dataclass
class Turn:
    agent: str
    model: str
    round: int
    text: str
    error: str | None = None
    searches: list = field(default_factory=list)


class Debate:
    def __init__(self, search: WS.Search | None = None, timeout: int = 90,
                 total_budget: int = 300):
        """timeout: 单次调用给多久。**给宽** —— 这些是推理模型, 真在想的时候
        本来就慢, 卡死一个太短的限制等于把正常输出当故障砍掉。

        total_budget: 整场辩论的总时限 (默认 5 分钟)。超了就直接返回超时错误,
        不再往下试 —— 免得前端转一个永远转不完的圈。两个限制各管一头:
        单次限制防"一个连接挂死拖住整轮", 总时限防"每一步都慢但都没超, 累计
        跑到十分钟"。
        """
        from openai import OpenAI
        key = os.environ.get("NVIDIA_API_KEY", "")
        if not key:
            raise RuntimeError("NVIDIA_API_KEY 未设置")
        self.cli = OpenAI(base_url=BASE_URL, api_key=key, max_retries=0)
        self.search = search or WS.Search()
        self.timeout = timeout
        self.total_budget = total_budget
        self._deadline = None       # run() 开跑时设
        # 辩论阶段要给裁定留出时间。没有这个预留的话, 两轮辩论可以把预算吃光,
        # 最后一步反而没得跑 —— 而裁定失败等于整场零输出, 是最不该省的一步。
        self._reserve = 0

    def _left(self) -> float:
        """还剩多少秒可用。没设总时限就当无限。"""
        if self._deadline is None:
            return 1e9
        return self._deadline - time.monotonic() - self._reserve

    @property
    def timed_out(self) -> bool:
        return self._left() <= 5

    def _call(self, model, system, user, max_tok, temp, tries=2, per_try=None):
        """带重试的单次调用。返回 (文本, 错误)。

        NIM 免费额度的延迟方差极大 —— 同一个 stepfun-ai/step-3.7-flash
        实测一次 6.9 秒、一次 57.7 秒。一次超时就放弃, 等于把偶发拥堵当成
        永久故障。重试一次通常就过去了。

        用短超时 + 重试, 而不是长超时死等: 后者会让一个卡住的模型拖着整轮
        不放, 而前三轮是并发的, 最慢的那个决定整轮耗时。
        """
        if _tripped(model):
            return "", "近期连续失败, 已熔断 (跳过, 不占用时间)"
        if self.timed_out:
            return "", f"整场辩论已超过 {self.total_budget} 秒总时限"
        # 单次限制给宽 (默认 90 秒), 但不能超过剩余总预算 —— 否则最后一次
        # 调用可以把总时限捅穿。
        per_try = per_try or self.timeout
        last = ""
        for i in range(tries):
            budget = int(min(per_try, max(10, self._left())))
            try:
                r = self.cli.chat.completions.create(
                    model=model, timeout=budget, max_tokens=max_tok,
                    temperature=temp,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}])
                txt = r.choices[0].message.content or ""
                if txt.strip():
                    _note(model, True)
                    return txt, None
                last = "空回复"
            except Exception as e:
                last = f"{type(e).__name__}: {str(e)[:120]}"
            if i < tries - 1:
                if self.timed_out:
                    last = f"{last} (剩余总时限不足, 不再重试)"
                    break
                time.sleep(1.5 * (i + 1))
        _note(model, False)
        return "", last

    def _ask(self, agent, system, user, max_tok=1200, temp=0.6, rnd=1):
        model = MODELS.get(agent, _DEFAULTS.get(agent, ""))
        raw, err = self._call(model, system, user, max_tok, temp)
        if err:
            return Turn(agent, model, rnd, "", err[:160])
        txt, cerr = _clean(raw)
        reqs = WS.extract_requests(txt)
        return Turn(agent, model, rnd, WS.strip_requests(txt), cerr, reqs)

    def _parallel(self, jobs):
        out = {}
        with ThreadPoolExecutor(max_workers=3) as ex:
            futs = {ex.submit(self._ask, *j): j[0] for j in jobs}
            for f in as_completed(futs):
                t = f.result(); out[t.agent] = t
        return [out[a] for a, *_ in jobs if a in out]

    # ── 主流程 ──
    def run(self, brief: str, blue: str, red: str, league: str,
            do_search: bool = True) -> dict:
        log = {"search_enabled": self.search.enabled, "rounds": {}}
        self._deadline = time.monotonic() + self.total_budget
        # 两轮辩论期间假装"只剩这么多", 把 VERDICT_RESERVE 秒攥在手里留给裁定。
        # 裁定挂掉等于整场零输出, 是最不该被前面两轮吃掉时间的一步。
        self._reserve = VERDICT_RESERVE

        # Round 0
        hits, queries = ([], [])
        if do_search and self.search.enabled:
            hits, queries = self.search.scout(blue, red, league)
        sblock = self.search.block(hits)
        log["rounds"]["0_search"] = {
            "queries": queries,
            "hits": [{"title": h.title, "url": h.url, "age_days": h.age_days,
                      "stale": h.stale} for h in hits],
        }

        full_brief = f"{brief}\n\n{sblock}"

        # Round 1
        r1 = self._parallel([(a, R1[a] + _SEARCH_NOTE, full_brief) for a in R1])
        log["rounds"]["1_positions"] = [
            {"agent": t.agent, "model": t.model, "text": t.text,
             "error": t.error, "search_requests": t.searches} for t in r1]

        # 执行 agent 请求的检索
        extra = []
        if do_search and self.search.enabled:
            asked = [q for t in r1 for q in t.searches]
            for q in asked[:4]:
                extra += self.search.query(q, 3)
        if extra:
            log["rounds"]["1b_agent_searches"] = {
                "queries": [q for t in r1 for q in t.searches][:4],
                "hits": [{"title": h.title, "url": h.url,
                          "age_days": h.age_days, "stale": h.stale} for h in extra]}

        # Round 2
        def others(me):
            return "\n\n".join(f"【{t.agent}】\n{t.text}"
                               for t in r1 if t.agent != me and not t.error)

        extra_block = ("\n\n" + self.search.block(extra, "本轮新增检索")) if extra else ""
        r2 = self._parallel([
            (a, R2, f"{full_brief}{extra_block}\n\n=== 第一轮 ===\n"
                    f"你自己说的:\n{next((t.text for t in r1 if t.agent==a), '')}\n\n"
                    f"另外两位:\n{others(a)}")
            for a in R1])
        log["rounds"]["2_crossexam"] = [
            {"agent": t.agent, "text": t.text, "error": t.error} for t in r2]

        # Round 3
        transcript = []
        for t in r1 + r2:
            if t.error and not t.text:
                continue
            transcript.append(f"【{t.agent} · 第{t.round}轮】\n{t.text}")
        verdict = self._verdict(full_brief + extra_block, "\n\n".join(transcript))
        log["rounds"]["3_verdict"] = verdict

        # 汇总
        claims = verdict.get("claims", [])
        log["summary"] = {
            "advisory_only": True,
            "note": "本层不修改概率。",
            "n_claims": len(claims),
            "survived": [c for c in claims if c.get("status") == "survived"],
            "contested": [c for c in claims if c.get("status") == "contested"],
            "external_facts": [c for c in claims
                               if c.get("kind") == "external_fact"
                               and c.get("evidence_url")],
            "consensus": verdict.get("consensus", ""),
            "open_disputes": verdict.get("open_disputes", []),
            "agent_errors": verdict.get("agent_errors", []),
            "search_calls": self.search.calls,
        }
        return log

    def _verdict(self, brief, transcript):
        """协调者失败 = 整场辩论零输出, 所以这里比三个辩手更需要兜底。

        前三轮各自失败只是少一个视角, 裁定失败则前面全白跑 —— 用户看到的
        就是一句「裁定失败: Request timed out」。所以除了 _call 的重试,
        还按顺序换模型再试。候选取自另外三个 agent (它们此刻刚答过话,
        是活的), 排掉协调者自己。
        """
        prompt = f"{brief}\n\n=== 辩论记录 ===\n{transcript}\n\n输出 JSON。"
        primary = MODELS.get("协调者", _DEFAULTS["协调者"])
        chain, seen = [], {primary}
        for a in ("补充者", "质疑者", "验证者"):
            m = MODELS.get(a, _DEFAULTS.get(a, ""))
            if m and m not in seen:
                seen.add(m); chain.append(m)
        # 按 json_score 降序。裁定这一步要的是**能解析的 JSON**, 不是文采 ——
        # 第一次上线时按 agent 顺序排, 结果 json_score=0.0 的 gpt-oss 排在最前,
        # 兜底"成功"了却吐出个没有 consensus 字段的壳, 面板照样是空的。
        chain.sort(key=lambda m: -JSON_SCORE.get(m, 0.0))

        # 总时限。健康时裁定 3-10 秒就回来了; 拖过 45 秒的都是卡住的连接,
        # 不是"在慢慢想"。没有这个闸, 主模型 2 次 + 三个兜底最坏能跑 375 秒,
        # 前端只会看见一个永远转不完的圈 —— 比直接报错更难受。
        # 到这一步把预留放开 —— 攥着的那 120 秒就是留给它的。
        self._reserve = 0
        why, err = None, "未调用"
        partial = partial_model = None
        for i, model in enumerate([primary] + chain):
            if self.timed_out:
                err = f"{err} (整场已达 {self.total_budget} 秒总时限)"
                break
            # 每个候选**只给一次**机会。协调者不参与前两轮, 所以熔断在这之前
            # 不会触发 —— 主模型要是重试两次, 一个挂死的模型就能吃掉 180 秒,
            # 后面三个兜底全饿死。换模型本身就是重试, 而且换的是活的那个。
            raw_txt, err = self._call(model, R3, prompt, 1500, 0.2, tries=1)
            if i == 0:
                why = err                      # 成功后 err 会被覆写成 None
            if not raw_txt:
                continue

            # **拿到 JSON ≠ 拿到裁定。** 校验必须在循环里, 否则一个
            # json_score=0.0 的兜底模型返回 `{"claims":[...]}`(没有
            # consensus 字段)也算"成功", 链条就此停住, 面板照样空白 ——
            # 实测 gpt-oss-120b 就是这样, 耗了 99 秒换来一个空结论。
            d = self._parse_verdict(raw_txt)
            if d is None:
                err = "返回的不是可解析的 JSON"
                continue
            if not (d.get("consensus") or "").strip():
                # 只有 claims 没有结论句。这**有用但不完整** —— 面板的主读数
                # 就是那句结论。实测 gpt-oss-120b (json_score 0.0) 反复这样。
                # 所以先留着当保底, 继续往下找能给出结论的; 找不到再回头用它,
                # 并如实标明缺了什么, 绝不替它编一句结论出来。
                if d.get("claims") and partial is None:
                    partial, partial_model = d, model
                err = "JSON 合法但没有 consensus 结论句"
                continue

            if i:
                print(f"[debate] 协调者 {primary} 失败({why}), "
                      f"已改用 {model} 裁定", flush=True)
                d["verdict_model"] = model      # 前端要能看出这不是主模型判的
            return d

        if partial is not None:
            partial["verdict_model"] = partial_model
            partial["consensus"] = ("（本轮没有模型给出总结句，以下主张由 "
                                    f"{partial_model} 逐条列出，未经汇总。）")
            print(f"[debate] 无模型给出结论句, 采用 {partial_model} 的逐条主张",
                  flush=True)
            return partial
        # 超时和"模型答不出来"要分开说 —— 前者你重试一次多半就好了,
        # 后者重试也是白搭。混成一句"裁定失败"会让人不知道该怎么办。
        msg = (f"复核超时: 整场超过 {self.total_budget} 秒仍未完成裁定 "
               f"(最后一次: {err})。免费额度拥堵时会这样, 稍后重试。"
               if self.timed_out else f"裁定失败: {err}")
        return {"claims": [], "consensus": msg,
                "open_disputes": [], "agent_errors": []}

    def _parse_verdict(self, raw_txt: str):
        """把模型输出解析成裁定 dict。解析不出来返回 None (交给调用方换模型)。"""
        try:
            txt = re.sub(r"^```(?:json)?|```$", "", raw_txt.strip(),
                         flags=re.M).strip()
            m = re.search(r"\{.*\}", txt, re.S)
            raw = m.group(0) if m else txt
            try:
                d = json.loads(raw)
            except json.JSONDecodeError:
                # agent 原文里的引号常常没转义。清理一遍再试。
                fixed = re.sub(r"[\u201c\u201d\u2018\u2019]", "'", raw)
                fixed = re.sub(r",\s*([}\]])", r"\1", fixed)      # 尾逗号
                fixed = fixed.replace("\n", " ")
                d = json.loads(fixed)
            if not isinstance(d, dict):
                return None
            # external_fact 必须有 URL, 否则降级
            for c in d.get("claims", []):
                if isinstance(c, dict) and c.get("kind") == "external_fact" \
                        and not c.get("evidence_url"):
                    c["kind"] = "reinterpretation"
                    c["note"] = (c.get("note", "") + " [无URL, 降级]").strip()
            return d
        except Exception:
            return None
