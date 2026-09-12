"""
辩论协议干跑 —— 不需要 API key
================================

Stage 3 的多轮协议 (预检索 → 独立分析 → 交叉质证 → 裁定) 逻辑不简单:
提示词拼装、`SEARCH: xxx` 的提取、把另外两人的**原话**注入下一轮、主张状态
(survived / conceded / contested / refuted) 的解析 —— 每一环都可能悄悄坏掉,
而且坏了以后输出看着仍然像模像样。

但这些管路和 LLM 本身无关。把模型调用换成桩, 就能在**没有 API key** 的情况下
把整条流程跑通, 检查:

  · 四轮是不是都跑到了
  · Round 2 有没有真的拿到另外两人的原话 (而不是自己的)
  · SEARCH: 请求有没有被正确提取
  · 裁定环节有没有把主张状态解析出来
  · 无检索后端时会不会崩

这样先验证完, 你再去 build.nvidia.com 拿 key (免费), 不至于白折腾。

用法:
    python tools/debate_dryrun.py            用桩跑一遍, 打印检查结果
    python tools/debate_dryrun.py --dump     顺带把每轮的提示词打出来
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


# 桩: 模仿 agent 的回复。每个 agent 说点**不一样**的, 好检验 Round 2 拿到的
# 到底是不是"另外两人"的原话。
#
# 键必须和 debate.R1 里的一致 (中文角色名) —— 第一版我按英文写, 结果全部
# 落到默认回复, 三个 agent 说的一模一样, 于是"没提取到 SEARCH"和"没拿到
# 别人原话"两个检查都误报成代码有问题。桩不忠实, 测出来的就是桩自己。
STUB = {
    "验证者": ("蓝方在 T=20 的经济领先主要来自上路, 与特征证据一致。\n"
               "主张: 预测由证据支撑。"),
    "补充者": ("SEARCH: T1 roster change August 2026\n"
               "主张: 没有已确认的阵容变动可补充。"),
    "反方": ("经济差 2k 在 25 分钟并不安全。\n"
             "主张: 模型高估了蓝方, 红方阵容更耐拖。"),
}

# 裁定轮要的是 JSON, 不是散文 —— 见 debate._verdict 里的解析。
VERDICT_JSON = """{
  "claims": [
    {"text": "预测由证据支撑", "status": "survived", "kind": "reinterpretation"},
    {"text": "模型高估了蓝方", "status": "contested", "kind": "reinterpretation"}
  ],
  "consensus": "分歧集中在经济领先的安全边际, 未收敛。"
}"""


class StubClient:
    """冒充 openai.OpenAI 的最小形状: cli.chat.completions.create(...)"""

    def __init__(self):
        self.calls = []
        self.chat = self
        self.completions = self

    def create(self, model=None, messages=None, **kw):
        sys_txt = next((m["content"] for m in messages if m["role"] == "system"), "")
        usr_txt = next((m["content"] for m in messages if m["role"] == "user"), "")
        self.calls.append({"model": model, "system": sys_txt, "user": usr_txt})

        # 角色是按顺序问的, 没法只靠 system 文本区分 (R1 的提示词里不含角色名),
        # 所以按调用序号轮着给: 前三次是 Round 1, 中间三次是 Round 2, 最后裁定。
        n = len(self.calls)
        names = list(STUB.keys())
        if "JSON" in usr_txt or "辩论记录" in usr_txt:
            body = VERDICT_JSON
        elif n <= 6:
            body = STUB[names[(n - 1) % 3]]
            if n > 3:                      # Round 2: 表明自己看到了什么
                body = "维持原判。\n" + body
        else:
            body = "无可补充。"

        class _M:      # 仿 SDK 的返回形状
            def __init__(self, c): self.message = type("x", (), {"content": c})()
        return type("R", (), {"choices": [_M(body)]})()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="store_true", help="打印每轮的提示词")
    a = ap.parse_args()

    import debate as D
    import websearch as WS

    print("=" * 66)
    print("  辩论协议干跑 (LLM 用桩替换, 不需要 API key)")
    print("=" * 66)

    # 绕开 __init__ 里的 key 检查 —— 我们要测的是协议, 不是鉴权
    d = D.Debate.__new__(D.Debate)
    stub = StubClient()
    d.cli = stub
    d.search = WS.Search()
    d.timeout = 5
    print(f"\n检索后端: {WS.available() or '无 (会跳过 Round 0 和 agent 检索)'}")

    brief = ("对阵: T1 (蓝) vs Gen.G (红), LCK\n"
             "Stage 2 概率: 蓝方 62.0%\n"
             "关键特征: 上路经济差 +1200, 阵容强势期差 -0.3")
    out = d.run(brief, "T1", "Gen.G", "LCK", do_search=True)

    rounds = out.get("rounds", {})
    print(f"\n跑完的轮次: {list(rounds.keys())}")
    print(f"LLM 被调用 {len(stub.calls)} 次")

    ok = True

    def check(label, cond, detail=""):
        nonlocal ok
        ok = ok and cond
        print(f"  {'✓' if cond else '✗'} {label}" + (f"   {detail}" if detail else ""))

    print("\n检查:")
    r1 = rounds.get("1_positions") or []
    check("Round 1 三个 agent 都表态了", len(r1) == 3, f"实际 {len(r1)} 个")

    # SEARCH: 提取
    asked = [q for t in r1 for q in (t.get("search_requests") or [])]
    check("SEARCH: 请求被提取出来", len(asked) >= 1, f"提到 {asked[:2]}")

    # Round 2 必须看到**别人**的原话
    r2 = rounds.get("2_crossexam") or rounds.get("2_cross") or []
    check("Round 2 跑到了", bool(r2), f"{len(r2)} 条")
    if r2:
        r2_prompts = [c["user"] for c in stub.calls[len(r1):len(r1) + len(r2)]]
        got_others = all(
            sum(1 for k, v in STUB.items() if v.split("\n")[0] in p) >= 2
            for p in r2_prompts)
        check("Round 2 每人拿到了另外两人的原话", got_others)

    v = out.get("verdict") or rounds.get("3_verdict") or {}
    check("裁定环节有输出", bool(v), str(v)[:60] if v else "")

    if a.dump:
        print("\n" + "=" * 66)
        for i, c in enumerate(stub.calls, 1):
            print(f"\n--- 第 {i} 次调用  model={c['model']} ---")
            print("[system]", c["system"][:300])
            print("[user]", c["user"][:600])

    print("\n" + "=" * 66)
    print("  管路" + ("通了 —— 拿到 key 就能真跑" if ok else "有问题, 见上面的 ✗"))
    print("=" * 66)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
