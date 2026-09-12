"""
Agent 筛选 v2 —— 加上「会不会失控输出」这一项
================================================
上一版 (finalize_agents.py) 测了 JSON 合规、克制度、指令遵循, 但漏了
现在最痛的失败模式: **思维链泄漏 / token 失控**。

nemotron-3-super 每次都能过原来那套测试, 却在真实的两轮辩论里吐出
三千字英文自言自语 ("We need to respond in Chinese... The instruction:
'Only raise a factor if...'"), 把 max_tokens 撑爆, 还污染了下游的裁定。

原因: 原来的探针是单轮简单任务, 触发不了 reasoning model 的思考模式。
真实场景里 agent 要看别人的发言、要权衡、要决定说不说 —— 这时候才暴露。

本版改动
--------
1. 候选池从 /v1/models 动态拉取, 不再手打 (上次 13 个手打错了 2 个)
2. 新增 leak 探针: 用真实的辩论 prompt, 测输出长度和元审议标志
3. leak 是硬门槛 —— 泄漏一次就取消 reviewer 资格, 不看别的分
4. 每项重复多次 (上次克制度只测 1 次, gpt-oss 在两轮之间从 0.0 翻到 1.0)

用法:
  python screen_agents.py --list      只看可用模型
  python screen_agents.py             跑完整筛选
  python screen_agents.py --quick     只测已知候选, 省时间
"""
import os, sys, json, re, time, statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = "https://integrate.api.nvidia.com/v1"
MAX_TOK = 600            # 探针给的预算 —— 正常答案 150 字够了
LEAK_LEN = 900           # 超过这个字符数视为失控
CONC = 3

# 排除非对话模型
SKIP = re.compile(r"embed|rerank|guard|safety|parse|nemoretriever|nvclip|"
                  r"ocr|translate|vila|neva|kosmos|fuyu|deplot|detector|"
                  r"diffusion|vision|laguna|-vl-|nemoguard|calibration|"
                  r"diffusion|vision|laguna|-vl-|nemoguard|calibration|"
                  r"reward|code-|codegemma|codellama|codestral|starcoder", re.I)

# 元审议标志 —— 模型在讨论「我该怎么回答」而不是在回答
COT = ["we need to", "we should", "the instruction", "the user wants",
       "let me", "i should", "we can say", "so we can", "however we",
       "the task is", "i need to", "但我们需要", "指令要求", "用户要求",
       "我需要", "让我", "首先我"]

FABRICATE = ["转会", "换人", "退役", "受伤", "教练", "版本更新", "补丁",
             "首发", "替补", "签约", "roster", "injur", "transfer",
             "benched", "patch", "coach"]


# ══════════════════════════════════════════════════════════
#  探针 1: 泄漏 / 失控  ← 新增, 最重要
# ══════════════════════════════════════════════════════════
# 用真实的第二轮辩论 prompt, 这是 nemotron 炸掉的那个场景

LEAK_SYS = ("你只报告模型结构上看不到的**外部事实**: 有名有姓有日期的转会、"
            "具体的版本改动、已确认的缺阵。\n"
            "规则: 每条必须引用检索结果里的 URL。检索里没有的, 不要说。\n"
            "确实没有可引用的外部事实 → 只输出「无可补充」。\n"
            "禁止用「若/如果」揣测。150 字以内, 用用户的语言。\n"
            "只输出答案。不要推理过程, 不要复述这些指令。")

LEAK_USR = """比赛: {b} (蓝方) vs {r} (红方)  [{lg}]
模型预测: 蓝方胜率 {p:.1%}

权重最高的特征证据:
  · diff_avg_towers = +3.200   (重要性 0.0473)
  · diff_rolling_wr = +0.300   (重要性 0.0433)
  · diff_avg_golddiffat15 = +1002.8  (重要性 0.0134)

外部检索结果: 检索无结果。

数据警告:
  · {lg} 数据截至 2026-06-14 (50 天前) — 可能整段赛程缺失

=== 另外两位的发言 ===
【验证者】证据与预测方向一致, 推塔差 +3.2 是主要驱动。未发现内部矛盾。
【质疑者】优势高度集中在 diff_avg_towers 单一特征上, 且该赛区数据已滞后
50 天。若期间发生阵容变动, 队伍统计描述的是一支已不存在的队伍。建议
将置信度下调至 60% 区间。

请指出模型结构上看不到的因素。"""

PROBES = [
    ("Aurora Collective", "Northwind Syndicate", "LPL", 0.583),
    ("Verdant Pact", "Ironhold Vanguard", "LCK", 0.471),
    ("Solstice Republic", "Cobalt Meridian", "LEC", 0.624),
]



def _strip_think(txt):
    """剥掉所有形式的思考标记。两个打分函数必须用同一套逻辑,
    否则会对同一段文本给出矛盾的判定。"""
    t = (txt or "").strip()
    for a, b in (("<think>", "</think>"), ("<thinking>", "</thinking>"),
                 ("<reasoning>", "</reasoning>")):
        if a in t:
            t = (t.split(b)[-1] if b in t else "").strip()
    return t


def score_leak(txt):
    """返回 (0/1, 说明)。任何泄漏或超长都是 0。"""
    t = _strip_think(txt)
    if not t:
        return 0.0, "只有思维链"
    # 「无可补充」只有 4 字符, 是本任务的标准正确答案。
    # 长度下界必须排在它后面, 否则会把满分答案判成失败。
    if "无可补充" in t and len(t) < 200:
        return 1.0, f"干净({len(t)}字符)"
    if len(t) < 6:
        return 0.0, f"回复过短({len(t)}字符)"
    low = t.lower()
    hits = [m for m in COT if m in low]
    if len(hits) >= 2:
        return 0.0, f"泄漏({len(hits)}个标志)"
    if len(t) > LEAK_LEN:
        return 0.0, f"失控({len(t)}字符)"
    return 1.0, f"干净({len(t)}字符)"


def score_restraint(txt):
    t = _strip_think(txt)
    if not t:
        return 0.0, "只有思维链"
    said = "无可补充" in t
    fab = [k for k in FABRICATE if k.lower() in t.lower()]
    if said and len(t) < 80:
        return 1.0, "克制"
    if len(t) < 6:
        return 0.0, f"回复过短({len(t)}字符)"
    if said:
        return 0.7, "说了但话多"
    if fab:
        return 0.0, f"编造({'/'.join(fab[:2])})"
    return 0.4, "未按要求"


# ══════════════════════════════════════════════════════════
#  探针 2: JSON 合规 (协调者用)
# ══════════════════════════════════════════════════════════

JSON_SYS = ('你是裁定者。严格输出 JSON, 不要 markdown 围栏:\n'
            '{"claims":[{"claim":"<25字内>","raised_by":"<谁>",'
            '"status":"survived"|"conceded"|"contested"|"refuted",'
            '"kind":"external_fact"|"reinterpretation"|"error"}],'
            '"consensus":"<60字内>","open_disputes":["<20字内>"],'
            '"agent_errors":["<20字内>"]}\n'
            '只输出 JSON。')

JSON_USR = """模型预测: 蓝方胜率 0.762

【验证者】证据与预测一致, 推塔差 +3.2 主导。
【补充者】无可补充。
【质疑者】优势集中在单一特征, 且数据滞后 50 天, 建议下调至 60%。
【验证者·二轮】反驳 | 数据滞后 | 该赛区已更新至 8 月 2 日, 滞后说法不成立。
【质疑者·二轮】让步 | 数据滞后 | 确认已更新, 撤回此点。维持 | 单特征依赖 |
diff_avg_towers 重要性 0.0473 是第二名的 1.1 倍, 集中度仍然偏高。

输出 JSON。"""


def score_json(txt):
    t = re.sub(r"^```(?:json)?|```$", "", (txt or "").strip(), flags=re.M).strip()
    for a, b in (("<think>", "</think>"),):
        if a in t:
            t = (t.split(b)[-1] if b in t else "").strip()
    m = re.search(r"\{.*\}", t, re.S)
    if not m:
        return 0.0, "无 JSON"
    try:
        d = json.loads(m.group(0))
    except Exception:
        return 0.0, "解析失败"
    if not {"claims", "consensus"}.issubset(d):
        return 0.3, "缺字段"
    if not isinstance(d.get("claims"), list):
        return 0.3, "claims 非列表"
    fenced = (txt or "").strip().startswith("```")
    return (0.8, "带围栏") if fenced else (1.0, "OK")


# ══════════════════════════════════════════════════════════

def lineage(model):
    """从模型 ID 推训练血统 —— 多 agent 的前提是血统互异"""
    p = model.split("/")[0].lower()
    n = model.lower()
    for key, lin in [("deepseek", "deepseek"), ("qwen", "qwen"),
                     ("z-ai", "glm"), ("glm", "glm"), ("moonshot", "kimi"),
                     ("minimax", "minimax"), ("meta", "llama"),
                     ("openai", "gptoss"), ("mistral", "mistral"),
                     ("google", "gemma"), ("stepfun", "stepfun"),
                     ("bytedance", "bytedance"), ("01-ai", "yi"),
                     ("ai21", "jamba"), ("thinkingmachines", "inkling"),
                     ("ibm", "granite"), ("microsoft", "phi"),
                     ("databricks", "dbrx"), ("writer", "palmyra"),
                     ("upstage", "solar"), ("zyphra", "zamba"),
                     ("sarvam", "sarvam"), ("poolside", "poolside")]:
        if key in p or key in n:
            return lin
    if "nemotron" in n or p == "nvidia":
        return "nemotron"
    return p


def call(cli, model, sys_p, usr_p, temp=0.4, timeout=150):
    t0 = time.time()
    r = cli.chat.completions.create(
        model=model, max_tokens=MAX_TOK, temperature=temp, timeout=timeout,
        messages=[{"role": "system", "content": sys_p},
                  {"role": "user", "content": usr_p}])
    return (r.choices[0].message.content or ""), time.time() - t0


def bench(cli, model):
    out = dict(model=model, lineage=lineage(model), ok=False)
    lats = []
    try:
        # 泄漏 + 克制 (同一个探针测两件事, 省调用)
        leaks, rests, notes = [], [], []
        for b, r, lg, p in PROBES:
            txt, t = call(cli, model, LEAK_SYS,
                          LEAK_USR.format(b=b, r=r, lg=lg, p=p))
            lats.append(t)
            ls, ln = score_leak(txt)
            rs, rn = score_restraint(txt)
            leaks.append(ls); rests.append(rs); notes.append(f"{ln}/{rn}")
            if ls == 0.0 and len(leaks) >= 2 and sum(leaks) == 0:
                break                       # 连续泄漏, 不用再测
        # JSON x2
        js, jn = [], []
        for _ in range(2):
            txt, t = call(cli, model, JSON_SYS, JSON_USR, temp=0.1)
            lats.append(t)
            s, n = score_json(txt); js.append(s); jn.append(n)

        out.update(ok=True,
                   leak=statistics.mean(leaks),
                   restraint=statistics.mean(rests),
                   json_score=statistics.mean(js),
                   notes=notes, json_note=jn[0],
                   latency=statistics.median(lats),
                   n_probes=len(leaks))
    except Exception as e:
        msg = str(e)
        out["err"] = ("401 未授权" if "401" in msg else
                      "403 需开通" if "403" in msg else
                      "404 未部署" if "404" in msg else
                      "429 限流" if "429" in msg else
                      "503 繁忙" if "503" in msg else
                      "超时" if "timeout" in msg.lower() else msg[:40])
    return out


def main():
    key = os.environ.get("NVIDIA_API_KEY", "")
    if not key:
        print("NVIDIA_API_KEY 未设置"); sys.exit(1)
    from openai import OpenAI
    cli = OpenAI(base_url=BASE, api_key=key, max_retries=0)

    print("=" * 76)
    print("  Agent 筛选 v2 —— 含「失控输出」硬门槛")
    print("=" * 76)

    try:
        ids = sorted(m.id for m in cli.models.list().data)
    except Exception as e:
        print(f"拉取模型列表失败: {e}"); sys.exit(1)
    cands = [i for i in ids if not SKIP.search(i)]

    if "--list" in sys.argv:
        by = {}
        for c in cands:
            by.setdefault(lineage(c), []).append(c)
        print(f"\n  {len(cands)} 个候选, {len(by)} 个血统\n")
        for lin in sorted(by):
            print(f"  [{lin}]")
            for m in by[lin]:
                print(f"      {m}")
        sys.exit(0)

    if "--quick" in sys.argv:
        keep = ("deepseek-v4-pro", "deepseek-v4-flash", "qwen3-next",
                "glm-5.2", "minimax-m3", "minimax-m2.7", "llama-3.3-70b",
                "gpt-oss-120b", "nemotron-3-super", "gemma-4-31b",
                "mistral-medium", "inkling", "seed-oss", "step-3.7")
        cands = [c for c in cands if any(k in c for k in keep)]

    # 每个血统最多测 2 个 —— 但要按「可能好用」排序, 不能按字母序。
    # 字母序会让 deepseek-v4-pro 排在 deepseek-coder-6.7b 后面被挤掉。
    def pref(m):
        n = m.lower()
        score = 0
        # 参数量: 大的优先 (30B 以下基本做不了这个任务)
        import re as _re
        sizes = [int(x) for x in _re.findall(r"(\d+)b(?![a-z])", n)]
        if sizes:
            big = max(sizes)
            score += 300 if big >= 100 else 200 if big >= 60 else \
                     120 if big >= 30 else -200
        else:
            score += 100          # 没标参数量的通常是旗舰
        # 版本号: 新的优先
        vers = [float(x) for x in _re.findall(r"[-/](\d+\.\d+)", n)]
        if vers:
            score += max(vers) * 10
        # 明显不适合的
        for bad, pen in [("mini", -150), ("nano", -150), ("small", -100),
                         ("tiny", -200), ("chat", -30), ("v0.1", -80),
                         ("2b", -200), ("-7b", -180), ("-8b", -150)]:
            if bad in n: score += pen
        for good, bon in [("instruct", 30), ("pro", 60), ("ultra", 50),
                          ("large", 40), ("flash", 20), ("super", 30)]:
            if good in n: score += bon
        return -score            # 降序

    seen, pool = {}, []
    for c in sorted(cands, key=pref):
        lin = lineage(c)
        seen[lin] = seen.get(lin, 0) + 1
        if seen[lin] <= 2:
            pool.append(c)
    pool.sort()

    print(f"\n  {len(pool)} 个模型 x 最多 5 次调用, 并发 {CONC}")
    print(f"  预计 5-12 分钟\n")

    res = []
    with ThreadPoolExecutor(max_workers=CONC) as ex:
        futs = [ex.submit(bench, cli, m) for m in pool]
        for fu in as_completed(futs, timeout=1800):
            r = fu.result(); res.append(r)
            if r["ok"]:
                mark = "✓" if r["leak"] == 1.0 else "✗"
                print(f"  {mark} {r['model']:<44} 失控{r['leak']:.2f} "
                      f"克制{r['restraint']:.2f} JSON{r['json_score']:.2f} "
                      f"{r['latency']:>5.1f}s")
            else:
                print(f"  · {r['model']:<44} {r['err']}")

    ok = [r for r in res if r.get("ok")]
    if not ok:
        print("\n  没有可用模型"); sys.exit(1)

    # ── 硬门槛: 失控 = 1.0 ──
    clean = [r for r in ok if r["leak"] == 1.0]

    print(f"\n{'=' * 76}")
    print(f"  完整结果 ({len(ok)}/{len(pool)} 可调用, {len(clean)} 个无失控)")
    print(f"{'=' * 76}")
    print(f"  {'模型':<44}{'失控':>7}{'克制':>7}{'JSON':>7}{'延迟':>8}")
    print(f"  {'─' * 74}")
    for r in sorted(ok, key=lambda x: (-x["leak"], -x["restraint"], x["latency"])):
        g = "✓" if r["leak"] == 1.0 else "✗"
        print(f"  {g} {r['model']:<42}{r['leak']:>7.2f}{r['restraint']:>7.2f}"
              f"{r['json_score']:>7.2f}{r['latency']:>7.1f}s")
        if r["leak"] < 1.0:
            bad = [n for n in r["notes"] if "泄漏" in n or "失控" in n or "思维链" in n]
            if bad: print(f"      {' | '.join(bad[:3])}")

    if len(clean) < 4:
        print(f"\n  ⚠ 只有 {len(clean)} 个通过失控门槛, 少于 4 个角色。")
        print(f"     可以让某个模型兼任两个角色, 但会削弱交叉验证。")

    # ── 分配 ──
    used, pick = set(), {}
    def take(role, keyfn, need_json=False):
        c = [r for r in clean if r["lineage"] not in used
             and (not need_json or r["json_score"] >= 0.9)]
        if not c:
            c = [r for r in clean if r["model"] not in
                 {v["model"] for v in pick.values()}
                 and (not need_json or r["json_score"] >= 0.9)]
        if not c: return
        b = max(c, key=keyfn); used.add(b["lineage"]); pick[role] = b

    take("协调者", lambda r: (r["json_score"], -r["latency"]), need_json=True)
    take("补充者", lambda r: (r["restraint"], -r["latency"]))
    take("验证者", lambda r: (r["restraint"], -r["latency"]))
    take("质疑者", lambda r: (r["restraint"], -r["latency"]))

    print(f"\n{'=' * 76}")
    print(f"  推荐配置")
    print(f"{'=' * 76}")
    why = {"协调者": "JSON 合规 + 无失控 —— 解析失败会让整个裁定层静默失效",
           "补充者": "克制度最高 —— 唯一的防幻觉闸门",
           "验证者": "无失控 + 克制达标", "质疑者": "无失控 + 克制达标"}
    for role in ["验证者", "补充者", "质疑者", "协调者"]:
        r = pick.get(role)
        if not r:
            print(f"  {role:<8} — 候选不足"); continue
        print(f"  {role:<8} {r['model']}")
        print(f"           {why[role]}")
        print(f"           失控 {r['leak']:.2f} / 克制 {r['restraint']:.2f} / "
              f"JSON {r['json_score']:.2f} / {r['latency']:.1f}s / {r['lineage']}")

    if pick:
        par = max((pick[x]["latency"] for x in
                   ["验证者", "补充者", "质疑者"] if x in pick), default=0)
        par = par * 2 + pick.get("协调者", {}).get("latency", 0)
        print(f"\n  辩论预计耗时 ≈ {par:.0f}s (两轮并发 + 裁定)")
        cfg = {role: {"model": r["model"], "lineage": r["lineage"],
                      "leak": r["leak"], "restraint": round(r["restraint"], 2),
                      "json_score": round(r["json_score"], 2),
                      "latency_s": round(r["latency"], 1)}
               for role, r in pick.items()}
        # **绝对路径**。原来写的是 Path("agent_config.json") —— 相对当前目录。
        # 从 research/ 下跑就写进 research/, 而读它的 debate.py 在上一层, 于是
        # 筛选结果一次都没生效, 脚本却照样打印"重启服务生效"。
        # 2026-08-20 就这样白跑了一次: 服务用的还是三周前那份配置。
        out = Path(__file__).resolve().parent.parent / "agent_config.json"
        out.write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        print(f"  已写入 {out} (重启服务生效)")
    print("=" * 76)


if __name__ == "__main__":
    main()
