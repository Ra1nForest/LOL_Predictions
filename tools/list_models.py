"""
列出 NIM 上所有可调用的模型 ID
================================
目录页的卡片标题 ≠ API 模型字符串。别猜, 直接问。

用法:
  python list_models.py            列出全部
  python list_models.py kimi       只看含 kimi 的
  python list_models.py glm qwen   多个关键词
"""
import os, sys

key = os.environ.get("NVIDIA_API_KEY", "")
if not key:
    print("NVIDIA_API_KEY 未设置"); sys.exit(1)

from openai import OpenAI
cli = OpenAI(base_url="https://integrate.api.nvidia.com/v1",
             api_key=key, max_retries=0)

try:
    ids = sorted(m.id for m in cli.models.list().data)
except Exception as e:
    print(f"拉取失败: {e}"); sys.exit(1)

kw = [a.lower() for a in sys.argv[1:]]
hit = [i for i in ids if not kw or any(k in i.lower() for k in kw)]

print(f"共 {len(ids)} 个模型" + (f", 匹配 {' / '.join(kw)}: {len(hit)} 个" if kw else ""))
print("─" * 60)
for i in hit:
    print(f"  {i}")

if kw and not hit:
    print("  (无匹配)")
    print("\n  所有发布者前缀:")
    for p in sorted({i.split("/")[0] for i in ids if "/" in i}):
        print(f"    {p}")
