"""
Agent 复核层 (Stage 3)
========================
拿 ML 的定量输出, 交给多个 LLM 做定性审查。

设计原则:
  · 三个 agent 用三个不同的底层模型 — 避免同源盲区
  · 它们看到的是「预测 + 特征证据」, 不是原始数据 — 它们的工作是审查, 不是重算
  · 协调者只能在 ±0.10 内调整概率, 且必须给出理由
  · agent 无法访问模型看不到的信息时, 应当明确说「无可补充」而不是编造

需要 NVIDIA NIM 免费 API key: https://build.nvidia.com
  export NVIDIA_API_KEY="nvapi-..."
"""
from __future__ import annotations
import os, json, re
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "https://integrate.api.nvidia.com/v1"

# 默认配置 — 若存在 agent_config.json 则以其为准 (由 discover_models.py 生成)
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

def _load_models() -> dict:
    p = Path(__file__).with_name("agent_config.json")
    if p.exists():
        try:
            cfg = json.loads(p.read_text(encoding="utf-8"))
            return {k: v["model"] for k, v in cfg.items() if "model" in v}
        except Exception:
            pass
    return dict(_DEFAULTS)

_MODELS = _load_models()

_PROMPTS = {
    "验证者": ("You audit a quantitative model's prediction. You are given the "
              "predicted probability and the top feature evidence. Your job: judge "
              "whether the prediction follows from the evidence, and flag any "
              "internal inconsistency. Do NOT invent facts about teams or players "
              "you were not given. If the evidence supports the prediction, say so "
              "plainly. Under 150 words. Reply in the user's language."),
    "补充者": ("You report ONLY externally-known facts the model could not see: a named "
              "player transfer with a date, a specific patch and what it changed, a "
              "confirmed absence. \n\n"
              "Decision rule — apply it directly, do not deliberate in your output:\n"
              "  Do you know a specific named dated event? → state it in one sentence.\n"
              "  Otherwise → output exactly: 无可补充\n\n"
              "Anything already in the brief (data staleness, missing champion history) "
              "is NOT yours to report — another agent covers it. A general category "
              "('roster changes matter') is not a fact. Never speculate with '若/如果'.\n"
              "Output the answer only. No reasoning, no restating these instructions."),
    "质疑者": ("You argue the case AGAINST the model's prediction. Look for: "
              "overfitting signals, small-sample features, whether the favoured team's "
              "edge rests on one dominant feature, and whether the confidence level is "
              "justified. Be concrete and adversarial, not vague. "
              "Under 150 words. Reply in the user's language."),
}

AGENTS = [dict(id=i, name=n, model=_MODELS.get(n, _DEFAULTS[n]), system=_PROMPTS[n])
          for i, n in (("verifier", "验证者"), ("context", "补充者"), ("critic", "质疑者"))]

# 协调者不再有调整概率的权限。
#
# 理由 (实测得出, 见 README):
#   三个不联网的 agent 拿不到任何模型没有的具名事实, 所以永远不存在
#   正当的调整依据。给它调整权时, 它接受了一个含两处事实错误的论证
#   (误读 playoffs=0 的含义; 把 ECE 当成有方向的偏差), 把一个经过
#   留出集验证的校准概率往错误方向推了 3pp —— 可靠性表显示该区间
#   本应略微上调。
#
# 现在它的岗位是: 归纳风险点, 标注哪些是模型自身证据的重读、
# 哪些是真正的外部事实。概率一律沿用 Stage 2。
COORDINATOR = dict(
    name="协调者", model=_MODELS.get("协调者", _DEFAULTS["协调者"]),
    system=("You receive a calibrated model prediction plus three agent reviews "
            "(verifier, context, critic). You do NOT adjust the probability — the "
            "model is calibrated on a held-out test set and the agents have no "
            "information source the model lacks.\n\n"
            "Your job is to classify each concern raised. Output STRICT JSON only, "
            "no markdown fence:\n"
            '{"risk_flags": [{"issue": "<under 25 words>", '
            '"kind": "reinterpretation"|"external_fact"|"error", '
            '"severity": "low"|"medium"|"high"}], '
            '"summary": "<under 80 words>", '
            '"agent_errors": ["<any factual mistake an agent made, under 20 words>"]}\n\n'
            "kind definitions:\n"
            "  reinterpretation = a re-reading of evidence already in the brief "
            "(e.g. 'edge rests on one feature', 'sample size is small'). Useful to "
            "surface, but the model already priced this in.\n"
            "  external_fact = a specific, named, dated event outside the data "
            "(e.g. 'team X replaced its top laner on 2026-07-20'). A general "
            "CATEGORY of blind spot ('roster changes matter and the model can't see "
            "them') is NOT an external_fact — it is a reinterpretation.\n"
            "  error = the agent stated something factually wrong about the brief.\n\n"
            "Fact-check every numeric claim an agent made against the brief. Common "
            "mistakes to look for:\n"
            "  - confusing feature IMPORTANCE with sample size or effect size\n"
            "  - treating ECE (a signed-magnitude-free average) as directional bias\n"
            "  - misreading a feature value (e.g. playoffs=0 means NOT a playoff "
            "game, not 'no playoff adjustment')\n"
            "  - arithmetic errors about distances or ranges\n"
            "  - claiming an agent output is an answer when it is leaked reasoning\n"
            "List each under agent_errors, quoting the wrong number. If an agent's "
            "output was flagged with an error, note that too. "
            "Write in the user's language."))


# reasoning model 会把内部思考当答案吐出来。这些是元审议的标志 ——
# 模型在讨论「我该怎么回答」而不是在回答。
_COT_MARKERS = [
    "we need to", "we should", "the instruction", "the user wants",
    "let me", "i should", "we can say", "so we can", "however we need",
    "但我们需要", "指令要求", "我们应该回答", "用户要求",
]

def _clean(txt: str) -> tuple[str, str | None]:
    """
    剥掉 <think> 块。若剩余内容仍像思维链, 返回 error。
    reasoning model 的思考不是答案, 混进审查记录里会污染下游。
    """
    if not txt:
        return "", "空回复"
    # 常见的显式思考标记
    for a, b in (("<think>", "</think>"), ("<thinking>", "</thinking>"),
                 ("<reasoning>", "</reasoning>")):
        if a in txt:
            tail = txt.split(b)[-1] if b in txt else ""
            txt = tail.strip() or txt.split(a)[0].strip()
    txt = txt.strip()
    if not txt:
        return "", "仅有思维链, 无答案"
    low = txt.lower()
    hits = sum(1 for m in _COT_MARKERS if m in low)
    # 两个以上元审议标志 + 较长 => 判定为思维链泄漏
    if hits >= 2 and len(txt) > 150:
        return txt, f"疑似思维链泄漏 ({hits} 个元审议标志)"
    return txt, None


@dataclass
class AgentResult:
    id: str; name: str; model: str; text: str; error: str | None = None


def _client():
    key = os.environ.get("NVIDIA_API_KEY", "")
    if not key:
        raise RuntimeError("NVIDIA_API_KEY 未设置。免费申请: https://build.nvidia.com")
    from openai import OpenAI
    return OpenAI(base_url=BASE_URL, api_key=key)


def build_brief(pred: dict, top_features: list[tuple[str, float, float]],
                lang: str = "zh") -> str:
    """把 ML 输出打包成给 agent 看的简报。"""
    lines = [
        f"比赛: {pred['blue_team']} (蓝方) vs {pred['red_team']} (红方)  [{pred['league']}]",
        f"阶段: {pred['stage']}",
        f"模型预测: 蓝方胜率 {pred['probability']:.1%}",
        f"",
        f"模型元信息:",
        f"  · 该模型在留出测试集上的准确率 {pred['model_accuracy']:.1%} (基线 {pred['model_baseline']:.1%})",
        f"  · 概率已做 Platt 校准, ECE {pred['model_ece']:.3f}",
        f"  · 概率输出范围受限于 [{pred['p_range'][0]:.2f}, {pred['p_range'][1]:.2f}]",
        f"",
        f"权重最高的特征证据:",
    ]
    for name, val, imp in top_features:
        lines.append(f"  · {name} = {val:+.3f}   (重要性 {imp:.4f})")
    if pred.get("warnings"):
        lines.append("")
        lines.append("数据警告:")
        for w in pred["warnings"]:
            lines.append(f"  · {w}")
    lines += [
        "",
        "模型结构性看不到的东西: 转会与阵容变动、版本改动的即时影响、",
        "选手当日状态、临场 BP 博弈、赛程与旅途疲劳、场外消息。",
    ]
    return "\n".join(lines)


def run_agents(brief: str, timeout: int = 150) -> list[AgentResult]:
    """三个 reviewer 并发跑 — 串行要 2-3 分钟, 并发约等于最慢那个。"""
    cli = _client()

    def one(a):
        try:
            r = cli.chat.completions.create(
                model=a["model"], timeout=timeout, max_tokens=1500, temperature=0.6,
                messages=[{"role": "system", "content": a["system"]},
                          {"role": "user", "content": brief}])
            txt, err = _clean(r.choices[0].message.content or "")
            return AgentResult(a["id"], a["name"], a["model"], txt, err)
        except Exception as e:
            return AgentResult(a["id"], a["name"], a["model"], "", str(e))

    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(one, a): a["id"] for a in AGENTS}
        done = {f.result().id: f.result() for f in as_completed(futs)}
    return [done[a["id"]] for a in AGENTS]      # 保持固定顺序


def summarize(brief: str, reviews: list[AgentResult]) -> dict:
    """归纳风险点并分类。不返回概率 —— 概率由 Stage 2 决定。"""
    cli = _client()
    parts = []
    for r in reviews:
        if r.error and not r.text:
            parts.append(f"【{r.name}】(调用失败: {r.error})")
        elif r.error:
            parts.append(f"【{r.name}】⚠ 输出异常: {r.error}\n{r.text[:600]}")
        else:
            parts.append(f"【{r.name}】\n{r.text}")
    body = "\n\n".join(p for p in parts if "调用失败" not in p)
    if not body:
        return {"risk_flags": [], "summary": "所有 agent 调用失败, 无审查结果。",
                "agent_errors": [], "ok": False}
    try:
        r = cli.chat.completions.create(
            model=COORDINATOR["model"], timeout=120, max_tokens=800, temperature=0.2,
            messages=[{"role": "system", "content": COORDINATOR["system"]},
                      {"role": "user",
                       "content": f"{brief}\n\n=== Agent 复核 ===\n{body}\n\n输出 JSON。"}])
        txt = r.choices[0].message.content.strip()
        txt = re.sub(r"^```(?:json)?|```$", "", txt, flags=re.M).strip()
        m = re.search(r"\{.*\}", txt, re.S)
        d = json.loads(m.group(0) if m else txt)
        flags = d.get("risk_flags", []) or []
        if not isinstance(flags, list):
            flags = []
        return {"risk_flags": flags[:8],
                "summary": str(d.get("summary", ""))[:600],
                "agent_errors": (d.get("agent_errors") or [])[:5],
                "ok": True}
    except Exception as e:
        return {"risk_flags": [], "summary": f"归纳失败: {e}",
                "agent_errors": [], "ok": False}


def review(pred: dict, top_features: list, verbose: bool = False) -> dict:
    """
    Advisory-only 复核。

    返回结构里没有任何概率字段 —— 调用方的 final 概率必须来自 Stage 2。
    """
    brief = build_brief(pred, top_features)
    if verbose:
        print(brief, "\n" + "=" * 60)
    reviews = run_agents(brief)
    if verbose:
        for r in reviews:
            print(f"\n【{r.name}】 ({r.model})")
            print(r.error and f"  ERROR: {r.error}" or r.text)
    summ = summarize(brief, reviews)
    return {
        "advisory_only": True,
        "note": "本层不修改概率。agent 无模型之外的信息源, 概率沿用 Stage 2。",
        "brief": brief,
        "reviews": [dict(id=r.id, name=r.name, model=r.model,
                         text=r.text, error=r.error) for r in reviews],
        "risk_flags": summ["risk_flags"],
        "summary": summ["summary"],
        "agent_errors": summ["agent_errors"],
    }
