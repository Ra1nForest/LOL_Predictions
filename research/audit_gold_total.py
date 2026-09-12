"""gold_total 缺失时的估算 1200*T 准不准?

golddiff_norm = golddiff / gold_total 是**重要性最高的特征** (0.345)。
估错 gold_total, 这个特征就系统性偏掉, 而且不报错。
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
import ingame_model as IM

files = sorted((_ROOT / "data").glob("20*_from_OraclesElixir.csv"))
df = IM.load([str(f) for f in files])
teams = df[df["position"] == "team"]

print(f"{'T':>4}{'样本':>8}{'实际 goldat 中位':>18}{'均值':>10}"
      f"{'估算 1200*T':>14}{'偏差':>10}")
for T in IM.SLICES:
    col = f"goldat{T}"
    if col not in teams.columns:
        print(f"{T:>4}   没有这一列"); continue
    v = pd.to_numeric(teams[col], errors="coerce").dropna()
    if not len(v):
        continue
    est = 1200 * T
    med = v.median()
    print(f"{T:>4}{len(v):>8}{med:>18,.0f}{v.mean():>10,.0f}"
          f"{est:>14,}{100*(est-med)/med:>9.0f}%")

print("\n影响: golddiff_norm = golddiff / gold_total")
print("gold_total 估低 -> golddiff_norm 被放大 -> 模型以为领先幅度比实际大")
for T in IM.SLICES:
    col = f"goldat{T}"
    if col not in teams.columns:
        continue
    v = pd.to_numeric(teams[col], errors="coerce").dropna()
    if not len(v):
        continue
    med = v.median()
    gd = 3000.0
    print(f"  T={T:>2}: 领先 3000 时  真实 norm={gd/med:.4f}   "
          f"估算 norm={gd/(1200*T):.4f}   放大 {(gd/(1200*T))/(gd/med):.2f}x")
