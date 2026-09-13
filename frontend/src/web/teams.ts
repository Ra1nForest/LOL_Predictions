/**
 * 队伍表 —— 浏览器里代替服务器的 FeatureStore。
 *
 * 表由 tools/export_web_model.py 从 FeatureStore.team_snapshot() 原样导出, 所以
 * "此刻"的队伍状态和服务器完全相同; 这里只做查表、队名映射和日期换算。
 *
 * 队名映射 (mapTeam) 是 esports_feed.map_team 的移植, 规矩一样: **不模糊匹配**。
 * 显式别名 → 已知队名原样 → 归一化后精确相等, 三条都不中就返回 null。
 * 模糊匹配曾把 "Beijing JDG Esports" 配成 "LNG Esports" —— 看着合理, 是另一支队,
 * 而且不报错, 只会拿错队伍的历史去算特征。
 */

export interface TeamsFile {
  format: string;
  /** FeatureStore.roll_cols 的顺序, 即每队 v 数组的下标 */
  roll_cols: string[];
  /** 这些列在赛前模型里还要算双方合计 (sum_X) */
  sum_feats: string[];
  /** 模型队名 → {v: 各列滚动统计 (null = NaN), n: 历史场数, last: 最后一场日期} */
  teams: Record<string, { v: (number | null)[]; n: number; last: string }>;
  /** 赛区 → FeatureStore.known_teams(赛区) */
  known: Record<string, string[]>;
  /** 赛区 → 该赛区数据截至哪天 (FeatureStore.league_last) */
  league_last: Record<string, string>;
  /** lolesports 队名 → 模型队名 (esports_feed.TEAM_ALIASES) */
  aliases: Record<string, string>;
  data_through: string;
  generated_at: string;
}

export interface Snapshot {
  values: Map<string, number>;
  nGames: number;
  lastDate: string;
}

const own = (o: object, k: string) => Object.prototype.hasOwnProperty.call(o, k);

/** FeatureStore.team_snapshot(team) —— 不认识的队返回 null */
export function snapshot(t: TeamsFile, name: string | null | undefined): Snapshot | null {
  if (!name || !own(t.teams, name)) return null;
  const e = t.teams[name]!;
  const values = new Map<string, number>();
  t.roll_cols.forEach((c, i) => values.set(c, e.v[i] ?? NaN));
  return { values, nGames: e.n, lastDate: e.last };
}

export function knownTeams(t: TeamsFile, league: string): string[] {
  return own(t.known, league) ? t.known[league]! : [];
}

/** esports_feed._norm: 小写后只留 a-z0-9 */
export function norm(s: string | null | undefined): string {
  return (s ?? "").toLowerCase().replace(/[^a-z0-9]/g, "");
}

/** esports_feed.map_team: API 队名 → 模型队名。映射不出来返回 null, 不猜。 */
export function mapTeam(t: TeamsFile, apiName: string | null | undefined, known: string[] | null): string | null {
  if (!apiName || apiName === "TBD") return null;
  if (own(t.aliases, apiName)) return t.aliases[apiName]!;
  if (known && known.length > 0) {
    if (known.includes(apiName)) return apiName;
    const byNorm = new Map<string, string>();
    for (const k of known) byNorm.set(norm(k), k); // 同名归一化时后者覆盖前者, 和 dict 推导式一样
    return byNorm.get(norm(apiName)) ?? null;
  }
  return apiName;
}

/** 本地日期 YYYY-MM-DD —— 服务器用的是 pd.Timestamp.now().normalize(), 也是本地日期 */
export function localToday(d: Date = new Date()): string {
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`;
}

/** 两个 YYYY-MM-DD 之间差几天 (后减前), 即 (a.normalize() - b.normalize()).days */
export function daysBetween(later: string, earlier: string): number {
  const ms = (s: string) => {
    const [y, m, d] = s.split("-").map(Number);
    return Date.UTC(y!, m! - 1, d!);
  };
  return Math.round((ms(later) - ms(earlier)) / 86_400_000);
}

export interface Staleness {
  league: string;
  ok: boolean;
  days: number | null;
  level: "fresh" | "stale" | "very_stale" | "unknown";
  last_date?: string;
  message: string | null;
}

/** FeatureStore.staleness(league) —— 阈值 8 / 14 天的理由见那边 (周末才打的赛区) */
export function staleness(t: TeamsFile, league: string, today: string): Staleness {
  if (!own(t.league_last, league)) {
    return { league, ok: false, days: null, level: "unknown", message: `${league} 无数据` };
  }
  const last = t.league_last[league]!;
  const days = daysBetween(today, last);
  let level: Staleness["level"];
  let message: string | null;
  if (days <= 8) {
    level = "fresh";
    message = null;
  } else if (days < 14) {
    level = "stale";
    message = `${league} 数据截至 ${last} (${days} 天前) — 滚动窗口缺少最近比赛, 预测偏向旧状态`;
  } else {
    level = "very_stale";
    message =
      `${league} 数据截至 ${last} (${days} 天前) — ` +
      `可能整段赛程缺失, 若期间有转会或换人, 队伍统计描述的是` +
      `一支已不存在的阵容, 预测不可靠`;
  }
  return { league, ok: level === "fresh", days, level, last_date: last, message };
}
