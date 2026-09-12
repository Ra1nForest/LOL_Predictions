/**
 * lolesports 的名字 → Oracle's Elixir 的名字 (训练只见过后者)。
 *
 * feature_store.champion_key 和 esports_feed.oe_player_name 的浏览器版。原样拿去查表的话,
 * 选手 (IGTheShy vs TheShy) 一个都查不到, 英雄 (LeeSin vs Lee Sin) 查不到 13% —— 不报错,
 * 只是 BP 后的熟练度恒为 0、局内阵容强势期少算几个英雄。
 *
 * 别名表由 Python 导出 (各模型文件里的 champion_alias), 不在这里抄一份 —— 两边各写一份,
 * 迟早有一边漏改。由 scripts/golden.mjs 第 4 组核对。
 */

/** feature_store.champion_key: 小写、只留字母数字, 再过一遍别名 */
export function championKey(name: string | null | undefined, alias: Map<string, string>): string {
  const k = String(name ?? "").toLowerCase().replace(/[^a-z0-9]/g, "");
  return alias.get(k) ?? k;
}

/**
 * esports_feed.oe_player_name: 只剥这场比赛两队的简称, 长的优先; 都不是前缀就原样返回。
 * 排序要稳定 (同长度保持原顺序) —— Python 的 sorted(reverse=True) 也是稳定的。
 */
export function oePlayerName(summoner: string | null | undefined, codes: (string | null | undefined)[]): string {
  const s = (summoner ?? "").trim();
  const cs = codes.filter((c): c is string => !!c).sort((a, b) => b.length - a.length);
  for (const c of cs) {
    if (s.startsWith(c) && s.length > c.length) return s.slice(c.length).trim();
  }
  return s;
}
