"""
Draft 校验 — 提交前先查名字
=============================
选手名/英雄名对不上时, API 不会报错, 只会在 warnings 里提一句,
然后那个位置的熟练度特征全是缺失值。提交前先跑这个。

用法:
  python check_draft.py                    校验下面 DRAFT 里的内容
  python check_draft.py --team "Bilibili"  列出该队最近首发
  python check_draft.py --champ 凤凰        模糊搜英雄 (英文关键词)
"""
import sys, difflib
from pathlib import Path
import pandas as pd

_DATA = Path(__file__).parent.parent / "data"
FILES = [str(_DATA / f"{y}_LoL_esports_match_data_from_OraclesElixir.csv")
         for y in (2025, 2026)]
ROLES = ["top", "jng", "mid", "bot", "sup"]

# ── 待校验的 draft ──
BLUE_TEAM, RED_TEAM, LEAGUE = "Bilibili Gaming", "Anyone's Legend", "LPL"
DRAFT = {
    "blue": {                       # BLG
        "top": ("Wenbo",   "???"),      # 狼母 — 待确认
        "jng": ("Xun",     "Naafiri"),  # 那娅菲利
        "mid": ("knight",  "Anivia"),   # 凤凰
        "bot": ("Viper",   "Ezreal"),   # EZ
        "sup": ("ON",      "Karma"),    # 卡尔马
    },
    "red": {                        # AL
        "top": ("Breathe", "Gnar"),     # 纳尔
        "jng": ("Tarzan",  "Vi"),       # 蔚
        "mid": ("Shanks",  "Ryze"),     # 瑞兹
        "bot": ("Hope",    "Lucian"),   # 卢锡安
        "sup": ("Kael",    "Milio"),    # 明烛
    },
}

# ── 加载 ──
import os
dfs = [pd.read_csv(f, low_memory=False) for f in FILES if os.path.exists(f)]
if not dfs:
    print("找不到 CSV"); sys.exit(1)
d = pd.concat(dfs, ignore_index=True)
d["league"] = d["league"].replace({"LTA N": "LCS", "LTA S": "CBLOL"})
d["date"] = pd.to_datetime(d["date"], errors="coerce")
p = d[d["position"].isin(ROLES)].copy()

ALL_TEAMS = sorted(p["teamname"].dropna().unique())
ALL_PLAYERS = set(p["playername"].dropna())
ALL_CHAMPS = sorted(p["champion"].dropna().unique())

# ── 子命令 ──
if "--team" in sys.argv:
    kw = sys.argv[sys.argv.index("--team") + 1].lower()
    hits = [t for t in ALL_TEAMS if kw in t.lower()]
    if not hits:
        print(f"无匹配。所有队伍:\n  " + "\n  ".join(ALL_TEAMS)); sys.exit()
    for t in hits:
        sub = p[p["teamname"] == t].sort_values("date")
        last = sub.groupby("position").last()
        print(f"\n{t}   (最后一场 {sub['date'].max().date()})")
        for r in ROLES:
            if r in last.index:
                print(f"  {r:<4} {last.loc[r,'playername']:<14} 最近用 {last.loc[r,'champion']}")
    sys.exit()

if "--champ" in sys.argv:
    kw = sys.argv[sys.argv.index("--champ") + 1].lower()
    hits = [c for c in ALL_CHAMPS if kw in c.lower()]
    print("\n".join(hits) if hits else f"无匹配。共 {len(ALL_CHAMPS)} 个英雄")
    sys.exit()

# ── 校验 ──
print("=" * 66)
print(f"  Draft 校验   {BLUE_TEAM} (蓝) vs {RED_TEAM} (红)")
print("=" * 66)

for tm in (BLUE_TEAM, RED_TEAM):
    if tm not in ALL_TEAMS:
        near = difflib.get_close_matches(tm, ALL_TEAMS, n=3, cutoff=0.4)
        print(f"\n  ✗ 队名不存在: {tm}")
        if near: print(f"     可能是: {' | '.join(near)}")

bad = 0
for side, label in (("blue", BLUE_TEAM), ("red", RED_TEAM)):
    print(f"\n  {label}")
    for role in ROLES:
        player, champ = DRAFT[side].get(role, ("?", "?"))
        line, notes = f"    {role:<4} {player:<14} {champ:<14}", []

        if player not in ALL_PLAYERS:
            near = difflib.get_close_matches(player, list(ALL_PLAYERS), n=2, cutoff=0.6)
            notes.append(f"✗ 选手不存在" + (f" → {'/'.join(near)}" if near else ""))
        if champ not in ALL_CHAMPS:
            near = difflib.get_close_matches(champ, ALL_CHAMPS, n=3, cutoff=0.4)
            notes.append(f"✗ 英雄不存在" + (f" → {'/'.join(near)}" if near else ""))

        if not notes:
            n = len(p[(p["playername"] == player) & (p["champion"] == champ)])
            if n == 0:
                notes.append("⚠ 该选手无此英雄记录 (熟练度会缺失)")
            else:
                wr = p[(p["playername"] == player) & (p["champion"] == champ)]["result"].mean()
                notes.append(f"✓ {n} 场, 胜率 {wr:.0%}")
        if any("✗" in x for x in notes): bad += 1
        print(line + "  ".join(notes))

print(f"\n{'=' * 66}")
if bad:
    print(f"  {bad} 处错误 — 改完再提交")
    print(f"  查英雄: python check_draft.py --champ <英文关键词>")
    print(f"  查队伍: python check_draft.py --team <队名关键词>")
else:
    print(f"  全部通过, 可以提交")
print("=" * 66)
