"""
单模型诊断 — 分辨 404 是 SDK 问题还是 endpoint 问题
=====================================================
同一个模型用两种方式调:
  A. raw requests  — 完全照抄 NVIDIA 页面给的写法
  B. openai SDK    — 我们服务里实际用的写法

A 通 B 不通  → SDK 参数问题, 改调用方式
两个都不通   → endpoint 当前不可用, 过一会儿再试
两个都通     → 之前是瞬时故障, 直接加回候选池

用法:  python test_kimi.py [模型名]
       默认测 moonshotai/kimi-k2.6
"""
import os, sys, json, time
import requests

MODEL = sys.argv[1] if len(sys.argv) > 1 else "moonshotai/kimi-k2.6"
URL = "https://integrate.api.nvidia.com/v1/chat/completions"
KEY = os.environ.get("NVIDIA_API_KEY", "")

print("=" * 62)
print(f"  单模型诊断: {MODEL}")
print("=" * 62)
if not KEY:
    print("\n  NVIDIA_API_KEY 未设置"); sys.exit(1)

MSG = [{"role": "user", "content": "用中文回复两个字: 正常"}]

# ── A. raw requests, 照抄官方样例 ──
print("\n  [A] raw requests (官方样例写法)")
t0 = time.time()
try:
    r = requests.post(URL,
        headers={"Authorization": f"Bearer {KEY}", "Accept": "application/json"},
        json={"messages": MSG, "model": MODEL, "max_tokens": 16384,
              "seed": 0, "stream": False, "temperature": 1, "top_p": 1},
        timeout=90)
    dt = time.time() - t0
    print(f"      HTTP {r.status_code}   {dt:.1f}s")
    if r.status_code == 200:
        d = r.json()
        txt = d["choices"][0]["message"]["content"].strip()[:60]
        print(f"      ✓ «{txt}»")
        a_ok = True
    else:
        print(f"      ✗ {r.text[:200]}")
        a_ok = False
except Exception as e:
    print(f"      ✗ {type(e).__name__}: {str(e)[:150]}")
    a_ok = False

time.sleep(2)

# ── B. openai SDK, 我们服务里的写法 ──
print("\n  [B] openai SDK (服务里的写法)")
t0 = time.time()
try:
    from openai import OpenAI
    cli = OpenAI(base_url="https://integrate.api.nvidia.com/v1",
                 api_key=KEY, max_retries=0)
    r = cli.chat.completions.create(
        model=MODEL, messages=MSG, max_tokens=300, temperature=0.2, timeout=90)
    dt = time.time() - t0
    txt = (r.choices[0].message.content or "").strip()[:60]
    print(f"      ✓ {dt:.1f}s   «{txt}»")
    b_ok = True
except Exception as e:
    dt = time.time() - t0
    print(f"      ✗ {dt:.1f}s   {type(e).__name__}: {str(e)[:180]}")
    b_ok = False

# ── C. 若 A 通 B 不通, 逐个排查参数 ──
if a_ok and not b_ok:
    print("\n  [C] 参数排查 — A 通 B 不通, 逐个试")
    from openai import OpenAI
    cli = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=KEY, max_retries=0)
    variants = [
        ("大 max_tokens", dict(max_tokens=16384, temperature=1, top_p=1)),
        ("加 seed",       dict(max_tokens=16384, temperature=1, top_p=1, seed=0)),
        ("最小参数",       dict()),
    ]
    for name, kw in variants:
        try:
            r = cli.chat.completions.create(model=MODEL, messages=MSG, timeout=90, **kw)
            print(f"      ✓ {name}")
            break
        except Exception as e:
            print(f"      ✗ {name}: {str(e)[:90]}")

print("\n" + "=" * 62)
if a_ok and b_ok:
    print("  两种都通 → 之前是瞬时故障")
    print(f"  把 {MODEL} 加回 discover_models.py 的 CANDIDATES")
elif a_ok and not b_ok:
    print("  A 通 B 不通 → SDK 参数问题, 看 [C] 哪个变体能过")
elif not a_ok and not b_ok:
    print("  两种都不通 → endpoint 当前不可用")
    print("  隔十几分钟再试。NIM 免费端点会下线/重部署。")
print("=" * 62)
