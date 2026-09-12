"""
跨年列语义审计
================
找出在 2025 / 2026 之间语义发生变化的列。

三类破坏, 危险程度递增:
  A. 列缺失      — 一年有一年没。明显, 容易发现。
  B. 全空        — 列存在但全是 NaN。也比较明显。
  C. 僵尸列 ★    — 列存在、非空、但一年是变量另一年是常数。
                   最危险: 看起来完全正常, 模型会把它当年份标签用。
                   例: atakhans 在 2026 恒为 0

用法: python research/audit_columns.py   (从任何目录跑都行)
"""
import pandas as pd, numpy as np, warnings, sys
from pathlib import Path
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from multiyear_gate import data_path

TARGET = ["LPL","LCK","LEC","LCS"]
F25 = data_path(2025)
F26 = data_path(2026)

print("="*72)
print("  跨年列语义审计  2025 vs 2026")
print("="*72)

for f in (F25, F26):
    if not Path(f).exists():
        print(f"  缺少 {f}"); sys.exit(1)

def load(f, label):
    d = pd.read_csv(f, low_memory=False)
    d["league"] = d["league"].replace({"LTA N":"LCS","LTA S":"CBLOL"})
    d = d[d["league"].isin(TARGET)]
    t = d[d["position"]=="team"]
    p = d[d["position"]!="team"]
    print(f"  {label}: {len(t)} 队伍行, {len(p)} 选手行, {t['gameid'].nunique()} 场")
    return t

A = load(F25, "2025")
B = load(F26, "2026")

def profile(s):
    """返回 (填充率, 唯一值数, 均值, 标准差)"""
    fill = s.notna().mean()
    v = pd.to_numeric(s, errors="coerce")
    if v.notna().sum() == 0:
        return fill, s.nunique(dropna=True), None, None
    return fill, v.nunique(dropna=True), v.mean(), v.std()

cols = sorted(set(A.columns) | set(B.columns))
missing, empty, zombie, shifted, ok = [], [], [], [], []

for c in cols:
    inA, inB = c in A.columns, c in B.columns
    if not (inA and inB):
        missing.append((c, "仅2025" if inA else "仅2026")); continue

    fa, ua, ma, sa = profile(A[c])
    fb, ub, mb, sb = profile(B[c])

    if fa < 0.01 or fb < 0.01:
        if not (fa < 0.01 and fb < 0.01):
            empty.append((c, fa, fb))
        continue

    # 僵尸列: 一年是变量, 另一年是常数
    const_a = (ua <= 1) or (sa is not None and sa == 0)
    const_b = (ub <= 1) or (sb is not None and sb == 0)
    if const_a != const_b:
        zombie.append((c, fa, fb, ma, mb, ua, ub)); continue

    # 均值大幅漂移 (数值列)
    if ma is not None and mb is not None:
        pooled = np.nanmean([sa or 0, sb or 0])
        if pooled > 0:
            shift = abs(ma - mb) / pooled
            if shift > 0.5:
                shifted.append((c, ma, mb, shift)); continue
    ok.append(c)

print(f"\n{'='*72}")
print(f"  ★★★ C 类: 僵尸列 — 最危险")
print(f"{'='*72}")
if zombie:
    print(f"  {'列名':<28}{'25填充':>8}{'26填充':>8}{'25均值':>10}{'26均值':>10}{'25唯一':>7}{'26唯一':>7}")
    print(f"  {'─'*70}")
    for c, fa, fb, ma, mb, ua, ub in zombie:
        msa = f"{ma:.3f}" if ma is not None else "—"
        msb = f"{mb:.3f}" if mb is not None else "—"
        print(f"  {c:<28}{fa:>8.1%}{fb:>8.1%}{msa:>10}{msb:>10}{ua:>7}{ub:>7}")
    print(f"\n  → 这些列必须处理: 加年份交互项 / 分年填 NaN / 直接删除")
else:
    print("  未发现")

print(f"\n{'='*72}")
print(f"  A 类: 列缺失")
print(f"{'='*72}")
if missing:
    for c, w in missing: print(f"  {c:<40} {w}")
else:
    print("  未发现")

print(f"\n{'='*72}")
print(f"  B 类: 一年全空")
print(f"{'='*72}")
if empty:
    for c, fa, fb in empty:
        print(f"  {c:<32} 2025 {fa:>6.1%}   2026 {fb:>6.1%}")
else:
    print("  未发现")

print(f"\n{'='*72}")
print(f"  D 类: 均值大幅漂移 (>0.5 个标准差) — 可能是玩法变化, 未必是 bug")
print(f"{'='*72}")
if shifted:
    print(f"  {'列名':<32}{'2025':>12}{'2026':>12}{'漂移(SD)':>10}")
    print(f"  {'─'*66}")
    for c, ma, mb, sh in sorted(shifted, key=lambda x:-x[3])[:25]:
        print(f"  {c:<32}{ma:>12.3f}{mb:>12.3f}{sh:>10.2f}")
else:
    print("  未发现")

# ── 专项检查: blue_firstpick ──
print(f"\n{'='*72}")
print(f"  专项: firstPick × side")
print(f"{'='*72}")
for name, d in [("2025", A), ("2026", B)]:
    if "firstPick" not in d.columns: continue
    bl = d[d["side"]=="Blue"]
    fp = pd.to_numeric(bl["firstPick"], errors="coerce")
    print(f"  {name}: 蓝方拿首选比例 {fp.mean():.3f}   ({len(bl)} 场)")
print(f"\n  若 2025 ≈ 1.000 而 2026 ≈ 0.335 → blue_firstpick 是僵尸列, 必须处理")

print(f"\n{'='*72}")
print(f"  正常列: {len(ok)} 个")
print(f"{'='*72}")
