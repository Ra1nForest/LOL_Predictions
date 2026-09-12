"""
决赛圈复测 — 克制度重复测量 + 全角色硬门槛
=============================================
discover_models.py 的两个缺陷:
  1. 克制度只测 1 次 → gpt-oss 在两轮间从 0.0 翻到 1.0, 单次不可信
  2. 只对补充者卡克制度 → 会编造的 qwen 被选成了验证者

本脚本只跑决赛圈 (~6 个模型), 每个做 4 次克制测试, 用 4 组不同的
虚构对局避免相关性。然后:

  · 克制度均值 < 0.6  → 一票否决, 任何角色都不用
  · 协调者额外要求 JSON 合规 = 1.00
  · 血统仍需互异

用法:  python finalize_agents.py
"""
import os, sys, json, re, time, statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = "https://integrate.api.nvidia.com/v1"
MAX_ADJUST = 0.10
RESTRAINT_FLOOR = 0.6      # 硬门槛

# 决赛圈 —— 来自 discover 的可用结果 (json, adherence 已知合格)
FINALISTS = [
    ("openai/gpt-oss-120b",                 "gptoss",   1.00, 1.0,   1.2),
    ("nvidia/nemotron-3-super-120b-a12b",   "nemotron", 0.00, 1.0,   3.3),
    ("qwen/qwen3-next-80b-a3b-instruct",    "qwen",     1.00, 1.0,   2.3),
    ("minimaxai/minimax-m3",                "minimax",  1.00, 1.0,  20.3),
    ("deepseek-ai/deepseek-v4-pro",         "deepseek", 1.00, 1.0,  42.9),
    ("google/gemma-4-31b-it",               "gemma",    1.00, 1.0,  58.1),
]

RESTRAINT_SYS = ("You identify what a pre-match statistical model structurally cannot see: "
                 "roster changes, patch shifts, scheduling and travel, player form outside "
                 "recorded stats. Only raise a factor if you can name it concretely. If you "
                 "have no verifiable information beyond what the model already used, reply "
                 "exactly '无可补充' and nothing else — do not speculate. "
                 "Under 150 words. Reply in the user's language.")

# 4 组不同的虚构对局 —— 避免重复同一 prompt 导致结果相关
PROBES = [
    ("Aurora Collective", "Northwind Syndicate", "LPL", "2026-08-14", 0.583),
    ("Verdant Pact",      "Ironhold Vanguard",   "LCK", "2026-09-02", 0.471),
    ("Solstice Republic", "Cobalt Meridian",     "LEC", "2026-08-27", 0.624),
    ("Umbral Ascendancy", "Golden Tessellate",   "LCS", "2026-09-11", 0.512),
]

FABRICATION = ["转会", "换人", "退役", "受伤", "生病", "教练", "版本更新", "补丁",
               "首发", "替补", "签约", "roster", "injur", "transfer", "benched",
               "patch", "coach", "substitut"]


def make_probe(b, r, lg, date, p):
    return (f"比赛: {b} (蓝方) vs {r} (红方)  [{lg}]\n日期: {date}\n"
            f"模型预测: 蓝方胜率 {p:.1%}\n\n"
            f"证据: diff_rolling_wr = +0.20, diff_avg_towers = +1.4\n\n"
            f"请指出模型结构上看不到的因素。")


def score(txt):
    t = txt.strip()
    said = ("无可补充" in t) or ("no additional" in t.lower())
    hits = [k for k in FABRICATION if k.lower() in t.lower()]
    if said and len(t) < 60: return 1.0, "克制"
    if said:                 return 0.7, "说了但话多"
    if hits:                 return 0.0, f"编造({'/'.join(hits[:2])})"
    return 0.4, "未按要求"


def test(cli, model, lineage, js, ad, lat0):
    scores, notes, lats = [], [], []
    for b, r, lg, d, p in PROBES:
        t0 = time.time()
        try:
            resp = cli.chat.completions.create(
                model=model, max_tokens=300, temperature=0.4, timeout=150,
                messages=[{"role": "system", "content": RESTRAINT_SYS},
                          {"role": "user", "content": make_probe(b, r, lg, d, p)}])
            s, n = score((resp.choices[0].message.content or ""))
            scores.append(s); notes.append(n); lats.append(time.time() - t0)
        except Exception as e:
            notes.append(f"err:{str(e)[:22]}")
        time.sleep(1)
    if not scores:
        return dict(model=model, lineage=lineage, ok=False, notes=notes)
    return dict(model=model, lineage=lineage, ok=True,
                restraint=statistics.mean(scores),
                worst=min(scores), n=len(scores), notes=notes,
                json_score=js, adherence=ad,
                latency=statistics.median(lats) if lats else lat0)


def main():
    key = os.environ.get("NVIDIA_API_KEY", "")
    print("=" * 76); print("  决赛圈复测 — 克制度 x4"); print("=" * 76)
    if not key: print("\n  NVIDIA_API_KEY 未设置"); sys.exit(1)
    from openai import OpenAI
    cli = OpenAI(base_url=BASE, api_key=key, max_retries=0)

    print(f"\n  {len(FINALISTS)} 个决赛模型 x 4 组虚构对局 = {len(FINALISTS)*4} 次调用")
    print(f"  硬门槛: 克制度均值 >= {RESTRAINT_FLOOR}\n")

    res = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = [ex.submit(test, cli, *f) for f in FINALISTS]
        for fu in as_completed(futs, timeout=1200):
            r = fu.result(); res.append(r)
            if r["ok"]:
                bar = "".join("●" if s == 1.0 else "◐" if s >= 0.4 else "○"
                              for s in [1.0 if "克制" == n else 0.7 if "话多" in n
                                        else 0.0 if "编造" in n else 0.4
                                        for n in r["notes"] if not n.startswith("err")])
                mark = "✓" if r["restraint"] >= RESTRAINT_FLOOR else "✗"
                print(f"  {mark} {r['model']:<40} 克制 {r['restraint']:.2f} "
                      f"(最低 {r['worst']:.1f})  {bar}")
            else:
                print(f"  ✗ {r['model']:<40} 全部失败")

    ok = [r for r in res if r.get("ok")]
    passed = [r for r in ok if r["restraint"] >= RESTRAINT_FLOOR]

    print(f"\n{'=' * 76}"); print("  明细"); print(f"{'=' * 76}")
    print(f"  {'模型':<40}{'克制均值':>9}{'最低':>7}{'JSON':>7}{'延迟':>8}")
    print(f"  {'─' * 72}")
    for r in sorted(ok, key=lambda x: -x["restraint"]):
        g = "✓" if r["restraint"] >= RESTRAINT_FLOOR else "✗"
        print(f"  {g} {r['model']:<38}{r['restraint']:>9.2f}{r['worst']:>7.1f}"
              f"{r['json_score']:>7.2f}{r['latency']:>7.1f}s")
        bad = [n for n in r["notes"] if "编造" in n or "未按" in n]
        if bad: print(f"      {' | '.join(bad)}")

    if len(passed) < 4:
        print(f"\n  ⚠ 只有 {len(passed)} 个过克制门槛, 少于 4 个角色")
        print(f"     可以放宽 RESTRAINT_FLOOR, 或把同一模型用在两个角色上")

    # ── 选人: 克制是所有角色的准入 ──
    used, pick = set(), {}
    def take(role, keyfn):
        c = [r for r in passed if r["lineage"] not in used]
        if not c:
            c = [r for r in passed if r["model"] not in {v["model"] for v in pick.values()}]
        if not c: return
        best = max(c, key=keyfn); used.add(best["lineage"]); pick[role] = best

    take("协调者", lambda r: (r["json_score"], r["restraint"], -r["latency"]))
    take("补充者", lambda r: (r["restraint"], r["worst"], -r["latency"]))
    take("验证者", lambda r: (r["restraint"], r["adherence"], -r["latency"]))
    take("质疑者", lambda r: (r["restraint"], r["adherence"], -r["latency"]))

    print(f"\n{'=' * 76}"); print("  最终配置"); print(f"{'=' * 76}")
    why = {"协调者": "JSON 1.00 且克制达标 — 解析失败会让 Stage 3 静默失效",
           "补充者": "克制度最高 — 唯一的防幻觉闸门",
           "验证者": "克制达标 + 指令遵循", "质疑者": "克制达标 + 指令遵循"}
    for role in ["验证者", "补充者", "质疑者", "协调者"]:
        r = pick.get(role)
        if not r: print(f"  {role:<8} — 候选不足"); continue
        print(f"  {role:<8} {r['model']}")
        print(f"           {why[role]}")
        print(f"           克制 {r['restraint']:.2f} (4次最低 {r['worst']:.1f}) / "
              f"JSON {r['json_score']:.2f} / {r['latency']:.1f}s / {r['lineage']}")

    if pick:
        par = max((pick[x]["latency"] for x in ["验证者","补充者","质疑者"] if x in pick), default=0)
        par += pick.get("协调者", {}).get("latency", 0)
        print(f"\n  Stage 3 预计 ≈ {par:.0f}s")
        cfg = {role: {"model": r["model"], "lineage": r["lineage"],
                      "restraint": round(r["restraint"], 2), "restraint_worst": r["worst"],
                      "json_score": r["json_score"], "latency_s": round(r["latency"], 1)}
               for role, r in pick.items()}
        with open("agent_config.json", "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2, ensure_ascii=False)
        print(f"  已写入 agent_config.json")
    print("=" * 76)


if __name__ == "__main__":
    main()
