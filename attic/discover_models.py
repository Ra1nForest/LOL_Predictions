"""
模型探测 — 按角色实测, 不按「快」或「聪明」
==============================================
四个 agent 的岗位要求不同, 而且都可以直接测:

  协调者  必须稳定吐出可解析 JSON       → JSON 合规率 (3 次)
  补充者  没信息时必须说「无可补充」      → 克制度 (是否编造)
  验证者  审查推理链, 要简洁不跑题       → 指令遵循 (字数/语言)
  质疑者  同上

延迟只测中位数, 且只作为同分时的决胜项 —— 单次延迟主要反映
endpoint 的冷热状态, 不是模型属性。

用法:  python discover_models.py
"""
import os, sys, json, re, time, statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = "https://integrate.api.nvidia.com/v1"
MAX_ADJUST = 0.10

# 候选池 —— 全部来自 /v1/models 的真实 ID (python list_models.py 可复查)
# 每行一个独立训练血统, 多 agent 的前提是血统互异
CANDIDATES = [
    ("deepseek-ai/deepseek-v4-pro",            "deepseek",  "DeepSeek V4 Pro, MoE 1M ctx"),
    ("qwen/qwen3-next-80b-a3b-instruct",       "qwen",      "Qwen3-Next 80B (397B 版未部署)"),
    ("z-ai/glm-5.2",                           "glm",       "GLM-5.2, agentic 旗舰"),
    # moonshotai/kimi-k2.6 已移除 — NVIDIA 侧未部署 (function_id=None), 非账号问题
    ("deepseek-ai/deepseek-v4-flash",          "deepseek",  "DeepSeek V4 Flash — 同血统的快速版"),
    ("minimaxai/minimax-m3",                   "minimax",   "MiniMax M3, 多模态 MoE"),
    ("meta/llama-3.3-70b-instruct",            "llama",     "Llama 3.3 70B dense"),
    ("openai/gpt-oss-120b",                    "gptoss",    "GPT-OSS 120B MoE reasoning"),
    ("nvidia/nemotron-3-super-120b-a12b",      "nemotron",  "Nemotron 3 Super, Mamba-Transformer"),
    ("mistralai/mistral-medium-3.5-128b",      "mistral",   "Mistral Medium 3.5"),
    ("google/gemma-4-31b-it",                  "gemma",     "Gemma 4 31B dense"),
    ("thinkingmachines/inkling",               "thinking",  "Inkling, Mamba 混合 256-expert MoE"),
    ("bytedance/seed-oss-36b-instruct",        "bytedance", "Seed-OSS 36B"),
    ("stepfun-ai/step-3.7-flash",              "stepfun",   "Step 3.7 Flash, 稀疏 MoE"),
]

# ── 探针 ────────────────────────────────────────────────

JSON_SYS = (f"You receive a model prediction and agent reviews. Output STRICT JSON only, "
            f"no markdown fence:\n"
            f'{{"adjusted_probability": <float>, "adjustment": <float>, '
            f'"confidence": "low"|"medium"|"high", "reasoning": "<under 40 words>"}}\n'
            f"adjustment must be within ±{MAX_ADJUST}. Adjust only if an agent raised a "
            f"concrete verifiable factor. Vague concern → adjustment 0.0.")
JSON_USR = ("模型预测: 蓝方胜率 0.620\n\n【验证者】证据与预测一致, 无异常。\n"
            "【补充者】无可补充。\n【质疑者】优势集中在单一特征上, 建议谨慎。\n\n输出 JSON。")

RESTRAINT_SYS = ("You identify what a pre-match statistical model structurally cannot see: "
                 "roster changes, patch shifts, scheduling, player form outside recorded stats. "
                 "Only raise a factor if you can name it concretely. If you have no verifiable "
                 "information beyond what the model already used, reply exactly '无可补充' and "
                 "nothing else — do not speculate. Under 150 words. Reply in the user's language.")
RESTRAINT_USR = ("比赛: Aurora Collective (蓝方) vs Northwind Syndicate (红方)  [LPL]\n"
                 "日期: 2026-08-14\n模型预测: 蓝方胜率 58.3%\n\n"
                 "证据: diff_rolling_wr = +0.20, diff_avg_towers = +1.4\n\n"
                 "请指出模型结构上看不到的因素。")

ADHERE_SYS = ("You audit a prediction's internal consistency. Be concise and concrete. "
              "Under 150 words. Reply in the user's language.")
ADHERE_USR = ("模型预测蓝方 62%, 主要依据是近十场胜率差 +0.20。\n"
              "请判断这个推理是否自洽。")

# 编造标志 —— 出现具体的、不可能知道的断言
FABRICATION = ["转会", "换人", "退役", "受伤", "生病", "教练", "版本更新", "补丁",
               "首发", "替补", "签约", "roster", "injur", "transfer", "benched"]


def _call(cli, model, sys_p, usr_p, max_tok=350, temp=0.2, timeout=120):
    # 自己控制重试: 只对瞬时错误重 1 次, 固定间隔, 失败可见。
    # 不用 SDK 的 max_retries —— 它是指数退避且静默, 出问题时看起来像卡死。
    TRANSIENT = ("503", "429", "502", "504", "timeout", "timed out", "connection")
    t0 = time.time()
    last = None
    for attempt in (1, 2):
        try:
            r = cli.chat.completions.create(
                model=model, max_tokens=max_tok, temperature=temp, timeout=timeout,
                messages=[{"role": "system", "content": sys_p},
                          {"role": "user", "content": usr_p}])
            return (r.choices[0].message.content or "").strip(), time.time() - t0
        except Exception as e:
            last = e
            if attempt == 2 or not any(k in str(e).lower() for k in TRANSIENT):
                raise
            time.sleep(4)
    raise last


def score_json(txt):
    t = re.sub(r"^```(?:json)?|```$", "", txt, flags=re.M).strip()
    m = re.search(r"\{.*\}", t, re.S)
    if not m:
        return 0, "无 JSON"
    try:
        d = json.loads(m.group(0))
    except Exception:
        return 0, "JSON 解析失败"
    need = {"adjusted_probability", "adjustment", "confidence", "reasoning"}
    if not need.issubset(d):
        return 0, f"缺字段 {sorted(need - set(d))}"
    try:
        adj = float(d["adjustment"])
    except Exception:
        return 0, "adjustment 非数字"
    if abs(adj) > MAX_ADJUST + 1e-9:
        return 0.5, f"越界 {adj:+.3f}"
    fenced = txt.strip().startswith("```")
    return (0.8, "带 fence") if fenced else (1.0, "OK")


def score_restraint(txt):
    t = txt.strip()
    said = ("无可补充" in t) or ("no additional" in t.lower())
    hits = [k for k in FABRICATION if k.lower() in t.lower()]
    if said and len(t) < 60:
        return 1.0, "克制"
    if said:
        return 0.7, "说了但话多"
    if hits:
        return 0.0, f"编造: {'/'.join(hits[:3])}"
    return 0.4, "未按要求"


def score_adherence(txt):
    n = len(re.findall(r"[\u4e00-\u9fff]", txt)) + len(re.findall(r"\b\w+\b", txt))
    zh = len(re.findall(r"[\u4e00-\u9fff]", txt)) > 5
    s = 1.0
    note = []
    if n > 260:
        s -= 0.4; note.append(f"过长({n})")
    if not zh:
        s -= 0.4; note.append("未用中文")
    if not txt.strip():
        return 0.0, "空回复"
    return max(0.0, s), "/".join(note) or "OK"


BUDGET = 300        # 单个模型总预算 (秒) — 慢模型 (llama 50s/次) 也要跑得完

def bench(cli, model, lineage, desc):
    lat, out = [], dict(model=model, lineage=lineage, desc=desc, ok=False)
    t_start = time.time()
    def over():
        return time.time() - t_start > BUDGET
    try:
        # JSON 合规 x3
        js, jn = [], []
        for _ in range(2):
            if js and over(): break          # 已有样本且超预算 -> 提前停
            txt, t = _call(cli, model, JSON_SYS, JSON_USR, 250, 0.0)
            lat.append(t); s, n = score_json(txt); js.append(s); jn.append(n)
        # 克制度
        txt, t = _call(cli, model, RESTRAINT_SYS, RESTRAINT_USR, 300, 0.4)
        lat.append(t); rs, rn = score_restraint(txt)
        # 指令遵循
        txt, t = _call(cli, model, ADHERE_SYS, ADHERE_USR, 350, 0.5)
        lat.append(t); as_, an = score_adherence(txt)

        out.update(ok=True, json_score=statistics.mean(js), json_note=jn[0],
                   restraint=rs, restraint_note=rn,
                   adherence=as_, adherence_note=an,
                   latency=statistics.median(lat), lat_spread=max(lat) - min(lat))
    except Exception as e:
        msg = str(e)
        out["err"] = ("401 未授权" if "401" in msg else "403 需单独注册" if "403" in msg else
                      "404 不存在" if "404" in msg else "429 限流 (降并发重试)" if "429" in msg else
                      "503 服务繁忙" if "503" in msg else
                      "超时 (>45s)" if "timeout" in msg.lower() else msg[:44])
    return out


def main():
    key = os.environ.get("NVIDIA_API_KEY", "")
    print("=" * 78); print("  NIM 模型探测 — 按角色实测"); print("=" * 78)
    if not key:
        print("\n  ✗ NVIDIA_API_KEY 未设置")
        print('  PowerShell 临时: $env:NVIDIA_API_KEY="nvapi-..."'); sys.exit(1)
    try:
        from openai import OpenAI
    except ImportError:
        print("\n  ✗ pip install openai"); sys.exit(1)
    # max_retries=0 关键: SDK 默认重试 2 次 + 指数退避, 遇到 429/5xx 会静默卡很久
    cli = OpenAI(base_url=BASE, api_key=key, max_retries=0)

    print(f"\n  Key {key[:12]}...{key[-4:]}")
    print(f"  {len(CANDIDATES)} 个候选 x 4 次调用, 并发 3 路, 单次上限 120s")
    print(f"  预计 5-10 分钟 (慢模型需要时间, 别中断)\n")

    res = []
    pending = {m for m, _, _ in CANDIDATES}
    t_all = time.time()
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(bench, cli, m, l, d): m for m, l, d in CANDIDATES}
        for fu in as_completed(futs, timeout=1500):
            r = fu.result(); res.append(r); pending.discard(r['model'])
            if r["ok"]:
                print(f"  ✓ {r['model']:<42} JSON {r['json_score']:.2f}  "
                      f"克制 {r['restraint']:.1f}  遵循 {r['adherence']:.1f}  "
                      f"{r['latency']:>5.1f}s")
            else:
                print(f"  ✗ {r['model']:<42} {r['err']}")
            if pending and len(pending) <= 4:
                print(f"      still running: {', '.join(sorted(pending))}")
    print(f"\n  探测耗时 {time.time()-t_all:.0f}s")

    ok = [r for r in res if r["ok"]]
    bad = [r for r in res if not r["ok"]]
    if not ok:
        print("\n  无可用模型"); sys.exit(1)

    print(f"\n{'=' * 78}"); print(f"  完整结果 ({len(ok)}/{len(CANDIDATES)} 可用)"); print(f"{'=' * 78}")
    print(f"  {'模型':<42}{'JSON':>6}{'克制':>6}{'遵循':>6}{'延迟':>8}{'抖动':>7}")
    print(f"  {'─' * 76}")
    for r in sorted(ok, key=lambda x: -(x["json_score"] + x["restraint"] + x["adherence"])):
        print(f"  {r['model']:<42}{r['json_score']:>6.2f}{r['restraint']:>6.1f}"
              f"{r['adherence']:>6.1f}{r['latency']:>7.1f}s{r['lat_spread']:>6.1f}s")
    print(f"\n  抖动 = 同一模型多次调用的延迟极差。大 = 冷启动影响, 该数字不可靠。")

    for r in sorted(ok, key=lambda x: -x["restraint"]):
        if r["restraint"] < 1.0:
            print(f"    {r['model']:<42}克制度 {r['restraint']:.1f} — {r['restraint_note']}")

    if bad:
        print(f"\n  不可用:")
        for r in bad: print(f"    {r['model']:<42}{r['err']}")
        if any("403" in (r.get("err") or "") for r in bad):
            print(f"    403 → build.nvidia.com 打开模型页, 点 \"Try API\" 注册")

    # ── 按角色择优, 血统不重复 ──
    used, pick = set(), {}

    def take(role, keyfn):
        cands = [r for r in ok if r["lineage"] not in used]
        if not cands:
            cands = [r for r in ok if r["model"] not in {v["model"] for v in pick.values()}]
        if not cands: return
        best = max(cands, key=keyfn)
        used.add(best["lineage"]); pick[role] = best

    # 协调者优先: JSON 合规是硬约束, 解析失败整条链路失效
    take("协调者", lambda r: (r["json_score"], -r["latency"]))
    # 补充者次之: 克制度是唯一防幻觉的闸
    take("补充者", lambda r: (r["restraint"], r["adherence"], -r["latency"]))
    # 两个 reviewer: 指令遵循 + 速度
    take("验证者", lambda r: (r["adherence"], -r["latency"]))
    take("质疑者", lambda r: (r["adherence"], -r["latency"]))

    print(f"\n{'=' * 78}"); print(f"  推荐配置"); print(f"{'=' * 78}")
    why = {"协调者": "JSON 合规最高 — 解析失败会让整个 Stage 3 失效",
           "补充者": "克制度最高 — 最不容易编造模型看不到的「事实」",
           "验证者": "指令遵循最好", "质疑者": "指令遵循次优"}
    for role in ["验证者", "补充者", "质疑者", "协调者"]:
        r = pick.get(role)
        if not r: print(f"  {role:<8} — 可用模型不足"); continue
        print(f"  {role:<8} {r['model']}")
        print(f"           {why[role]}")
        print(f"           JSON {r['json_score']:.2f} / 克制 {r['restraint']:.1f} / "
              f"遵循 {r['adherence']:.1f} / {r['latency']:.1f}s / 血统 {r['lineage']}")

    par = max((pick[x]["latency"] for x in ["验证者","补充者","质疑者"] if x in pick), default=0)
    par += pick["协调者"]["latency"] if "协调者" in pick else 0
    print(f"\n  Stage 3 预计耗时 ≈ {par:.0f}s (三个 reviewer 并发 + 协调者串行)")

    cfg = {role: {"model": r["model"], "lineage": r["lineage"],
                  "json_score": round(r["json_score"], 2),
                  "restraint": r["restraint"], "adherence": r["adherence"],
                  "latency_s": round(r["latency"], 1)}
           for role, r in pick.items()}
    with open("agent_config.json", "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2, ensure_ascii=False)
    print(f"\n  已写入 agent_config.json")
    print(f"{'=' * 78}")


if __name__ == "__main__":
    main()