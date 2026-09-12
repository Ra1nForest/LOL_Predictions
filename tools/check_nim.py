"""
NIM 连通性自检
================
在跑 Stage 3 之前先跑这个, 确认四个模型都能调通。

常见问题:
  401  key 没设置或拼错
  403  该模型家族未注册 → 去 build.nvidia.com 打开这个模型的页面, 点 "Try API"
  404  模型名写错了
  429  超过 40 req/min

用法:
  Windows PowerShell:  $env:NVIDIA_API_KEY="nvapi-xxxx"; python check_nim.py
  Windows CMD:         set NVIDIA_API_KEY=nvapi-xxxx && python check_nim.py
  Mac/Linux:           export NVIDIA_API_KEY="nvapi-xxxx" && python check_nim.py
"""
import os, sys, time

BASE = "https://integrate.api.nvidia.com/v1"

MODELS = [
    ("验证者",  "deepseek-ai/deepseek-r1"),
    ("补充者",  "qwen/qwen3-235b-a22b"),
    ("质疑者",  "meta/llama-3.3-70b-instruct"),
    ("协调者",  "nvidia/llama-3.3-nemotron-super-49b-v1"),
]

key = os.environ.get("NVIDIA_API_KEY", "")
print("=" * 62)
print("  NIM 连通性自检")
print("=" * 62)

if not key:
    print("\n  ✗ NVIDIA_API_KEY 未设置")
    print("\n  PowerShell:  $env:NVIDIA_API_KEY=\"nvapi-xxxx\"")
    print("  CMD:         set NVIDIA_API_KEY=nvapi-xxxx")
    print("  永久设置:     setx NVIDIA_API_KEY \"nvapi-xxxx\"  (需重开终端)")
    sys.exit(1)

if not key.startswith("nvapi-"):
    print(f"\n  ⚠ key 不是以 nvapi- 开头, 可能复制错了: {key[:12]}...")

print(f"\n  Key: {key[:12]}...{key[-4:]}  ({len(key)} 字符)")

try:
    from openai import OpenAI
except ImportError:
    print("\n  ✗ 缺少 openai 库:  pip install openai")
    sys.exit(1)

cli = OpenAI(base_url=BASE, api_key=key)
ok, fail = [], []

print(f"\n  逐个测试 (每次间隔 2 秒避免触发限流)\n")
for role, model in MODELS:
    print(f"  {role:<8} {model:<44}", end="", flush=True)
    t0 = time.time()
    try:
        r = cli.chat.completions.create(
            model=model, max_tokens=20, temperature=0,
            messages=[{"role": "user", "content": "回复两个字: 正常"}],
            timeout=60)
        txt = (r.choices[0].message.content or "").strip().replace("\n", " ")[:20]
        print(f" ✓ {time.time()-t0:>5.1f}s  «{txt}»")
        ok.append(model)
    except Exception as e:
        msg = str(e)
        code = ("401 未授权" if "401" in msg else
                "403 该模型未注册" if "403" in msg else
                "404 模型名错误" if "404" in msg else
                "429 触发限流" if "429" in msg else
                msg[:60])
        print(f" ✗ {code}")
        fail.append((model, code))
    time.sleep(2)

print(f"\n{'=' * 62}")
print(f"  {len(ok)}/{len(MODELS)} 可用")
if fail:
    print(f"\n  失败的模型:")
    for m, c in fail:
        print(f"    {m}")
        print(f"      {c}")
    if any("403" in c for _, c in fail):
        print(f"\n  403 的解决办法:")
        print(f"    NIM 部分模型家族需要单独注册。")
        print(f"    去 build.nvidia.com 搜到那个模型 → 打开页面 → 点右侧 \"Try API\"")
        print(f"    注册一次之后 key 就能调它了。")
    print(f"\n  也可以在 agents.py 里把失败的模型换成可用的。")
else:
    print(f"\n  全部通过, 可以跑 Stage 3 了:")
    print(f"    uvicorn api:app --port 8000")
    print(f"    然后 POST /predict 时加 \"agent_review\": true")
print(f"{'=' * 62}")
