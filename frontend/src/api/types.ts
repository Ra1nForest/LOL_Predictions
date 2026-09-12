/**
 * 后端响应的类型定义。
 *
 * 这一层是有意写厚的: 这个项目出过的 bug 几乎都是"不报错, 只是显示错数字"
 * —— 后端多一个字段、改一个名字, 旧前端会静默地读到 undefined 然后画出一个
 * 看着合理的画面。TypeScript 挡不住后端改字段, 但能保证**前端每一处引用都
 * 声明过它期待什么**, 改名时编译期就红。
 *
 * 凡是后端可能给不出的字段一律标成 `| null`, 而不是可选 —— 逼调用方显式处理,
 * 不给"忘了判空"留口子。
 */

export interface TeamRef {
  name: string;
  code: string;
  /** 映射到模型认识的队名; 映射不出来是 null (后端绝不模糊匹配) */
  model_name: string | null;
  known: boolean;
  game_wins: number | null;
  image: string | null;
}

export interface Match {
  match_id: string;
  league: string;
  /** 转播开始时刻, **不是开局** —— 实测能差 67 分钟 */
  start_time: string;
  state: "unstarted" | "inProgress" | "completed" | string;
  best_of: number | null;
  predictable: boolean;
  teams: TeamRef[];
  unknown_teams?: string[];
  between_games?: boolean;
}

export interface MatchListResponse {
  count: number;
  matches: Match[];
}

/** 一方的目标资源读数 */
export interface TeamObjectives {
  gold: number;
  kills: number;
  towers: number;
  inhibitors: number;
  barons: number;
  dragons: number;
  dragon_types: string[];
  cs: number;
}

export interface Item {
  id: number;
  icon: string;
}

export interface Player {
  participant_id: number;
  side: "blue" | "red";
  role: string | null;
  summoner_name: string | null;
  champion: string | null;
  champion_icon: string | null;
  level: number;
  kills: number;
  deaths: number;
  assists: number;
  cs: number;
  gold: number;
  kill_participation: number | null;
  damage_share: number | null;
  wards_placed: number | null;
  wards_destroyed: number | null;
  current_health: number | null;
  max_health: number | null;
  /** 六格普通装备。后端已经把饰品摘出去了 —— 帧给的是混装数组。 */
  items: Item[];
  /** 饰品 (侦察守卫/远见改造/神谕透镜…)。没买时为 null。 */
  trinket?: Item | null;
  /** AD/AP/护甲/魔抗/攻速/暴击/吸血/韧性。后端一直在发, 类型里以前没有。 */
  stats?: PlayerStats;
  /** 符文。后端已经把 id 解析成名字和图标 —— UI 不放领域逻辑。 */
  perks?: Perks;
  /** 技能加点顺序, 形如 ["Q","E","Q","W",...]。 */
  abilities?: string[];
}

export interface PlayerStats {
  attack_damage: number | null;
  ability_power: number | null;
  armor: number | null;
  magic_resist: number | null;
  attack_speed: number | null;
  /** 0–1 */
  crit: number | null;
  life_steal: number | null;
  /** 0–1 */
  tenacity: number | null;
}

export interface Perk {
  id: number;
  /** 解析不出来时为 null —— 前端要退回显示 id, 不能崩 */
  name: string | null;
  icon: string | null;
  /** true = 小符文碎片, 它没有独立图标 */
  shard?: boolean;
}

export interface Perks {
  style?: Perk;
  sub_style?: Perk;
  perks?: Perk[];
}

export interface Lane {
  lane: string;
  gold_diff: number;
  blue_gold: number;
  red_gold: number;
}

export interface LiveState {
  game_id: string;
  game_number: number | null;
  /** in_game | paused | finished —— 帧里的 gameState, 权威信号 */
  game_state: string;
  /** 局内分钟数, **已扣掉暂停** */
  minute: number | null;
  frame_time: string;
  in_progress: boolean;
  teams: { blue: TeamObjectives; red: TeamObjectives };
  stale_seconds: number;
  /** 这一局累计暂停了多少秒 (minute 已经减掉了它) */
  paused_seconds: number;
  /** 帧不再推进: gameState 还写着 in_game 但数据流断了 */
  stalled: boolean;
  note?: string;
  players: Player[];
  lanes: Lane[];
}

export interface ModelCard {
  accuracy_at_this_slice: number | null;
  high_conf_share_at_this_slice: number | null;
  overall_accuracy: number | null;
  ece: number | null;
  /** 校准能表达的下限 —— 界面上要把够不到的区间画成死区 */
  p_min: number | null;
  p_max: number | null;
  trained_through: string | null;
  note: string | null;
  range_note: string | null;
}

export interface Reason {
  text: string;
  weight: number;
}

export interface Prediction {
  blue_team: string;
  red_team: string;
  league: string;
  minute: number | null;
  slice_used: number | null;
  probability_blue: number | null;
  probability_raw: number | null;
  summary: string | null;
  /** **可能不存在。** 后端有三条路径返回不带 model 的 prediction:
   *  开局不足 10 分钟的 too_early、以及两个 {"error": …} 分支
   *  (api.py 的 out["prediction"] 赋值处)。原来这里写成必填, 于是
   *  `pr?.model.note` 这种写法在 too_early 时抛 TypeError, **整个 React 树
   *  卸载, 页面全白** —— 实测 2026-09-01 LCK 第 2 局开局那一刻。 */
  model?: ModelCard;
  warnings: string[];
  pregame_probability_blue: number | null;
  postdraft_probability_blue: number | null;
  shift_from_pregame: number | null;
  shift_note: string | null;
  reasons: Reason[];
  disclaimer: string | null;
  /** 开局不足 3 分钟时后端只给赛前/BP 后概率 */
  too_early?: boolean;
  note?: string;
  source?: string;
  error?: string;
}

export interface TimelinePoint {
  minute: number;
  golddiff: number;
  blue_kills: number;
  red_kills: number;
  /** 早期切片算不出时是 null —— 曲线要断开, 不能当 0 画 */
  probability_blue: number | null;
}

export interface GameRef {
  number: number;
  game_id: string;
  persisted_state: string;
  playable: boolean;
  sides: { blue?: string; red?: string };
}

export interface BoardResponse {
  match_id: string;
  match: Match | null;
  side_note: string | null;
  side_warning: string | null;
  games: GameRef[];
  selected_game_id: string | null;
  ddragon_version: string | null;
  live: LiveState | null;
  prediction: Prediction | null;
  timeline: TimelinePoint[];
  note: string | null;
}

export interface StageMetrics {
  stage: string;
  n_features: number;
  accuracy: number;
  baseline: number;
  lift: number;
  brier: number;
  ece_cal: number;
  p_min: number;
  p_max: number;
}

export interface HealthResponse {
  status: string;
  stages: Record<string, StageMetrics>;
  data?: unknown;
}

/* ── Stage 3: 多轮辩论 ─────────────────────────────────────────
   **这一层永远不改概率。** 数字始终来自 Stage 2 或 Stage 4, 辩论只输出
   经过交叉质证后仍然站得住的主张。见 README "why it is advisory-only"。 */

export type ClaimStatus = "survived" | "contested" | "conceded" | "refuted";

export interface Claim {
  claim: string;
  status: ClaimStatus | string;
  kind?: string;
  note?: string;
  /** external_fact 必须带 URL, 没有的后端会自动降级 */
  evidence_url?: string;
}

export interface Verdict {
  claims: Claim[];
  consensus?: string;
  open_disputes?: string[];
  agent_errors?: string[];
  /** 主协调模型挂了时, 代为裁定的那个 */
  verdict_model?: string;
}

export interface Turn {
  agent: string;
  text: string;
  error?: string | null;
}

export interface DebateLog {
  rounds: {
    "1_positions"?: Turn[];
    "2_rebuttal"?: Turn[];
    "3_verdict"?: Verdict;
  };
  verdict?: Verdict;
  summary?: {
    consensus?: string;
    n_claims?: number;
    search_calls?: number;
  };
}

export interface DebateResponse {
  blue_team: string;
  red_team: string;
  league: string;
  model_probability_blue: number;
  ingame: {
    probability_blue: number;
    pregame: number;
    shift: number;
  } | null;
  warnings: string[];
  debate: DebateLog;
  final: {
    probability_blue: number;
    probability_source: string;
    note: string;
  };
  disclaimer: string;
}

export interface DebateRequestBody {
  blue_team: string;
  red_team: string;
  league: string;
  playoffs?: boolean;
  search?: boolean;
  ingame?: Record<string, number | null> | null;
  blue_champions?: string[] | null;
  red_champions?: string[] | null;
  players?: unknown[] | null;
  lanes?: unknown[] | null;
}

/* ── Stage 1/2: 手动预测 ───────────────────────────────────── */

export interface PredictRequestBody {
  blue_team: string;
  red_team: string;
  league: string;
  playoffs?: boolean;
  draft?: {
    blue: Record<string, { player: string; champion: string }>;
    red: Record<string, { player: string; champion: string }>;
  } | null;
}

/** /predict 的模型卡和局内那份**不一样** —— 这里是 Stage 1/2 的整体指标 */
export interface PreModelCard {
  accuracy: number;
  baseline: number;
  lift: number;
  ece_cal: number;
  p_min: number;
  p_max: number;
  note: string | null;
  range_note: string | null;
}

export interface StageBlock {
  probability_blue: number;
  probability_raw: number;
  summary: string | null;
  reasons: Reason[];
  model: PreModelCard;
  /** 只有 2_post_draft 有: 相对赛前那一段的变动 */
  delta_from_stage1?: number;
}

/**
 * 注意键名带序号前缀 —— `stages["1_pre_draft"]` / `stages["2_post_draft"]`,
 * 不是 pre_draft / post_draft。
 *
 * 这几个类型最初是我凭猜测写的, 结果和后端对不上: TypeScript 只保证前端
 * **内部**一致, 管不了和后端的契约。现在的定义是照真实响应抄的。
 */
export interface PredictResponse {
  blue_team: string;
  red_team: string;
  league: string;
  playoffs: boolean;
  stages: Record<string, StageBlock>;
  final: {
    probability_blue: number;
    probability_red: number;
    probability_source: string;
    summary: string | null;
    note: string | null;
    implied_fair_odds_blue?: number;
    implied_fair_odds_red?: number;
  };
  data_quality?: {
    league_data_through?: string;
    days_old?: number;
    level?: string;
    reliable?: boolean;
  };
  warnings: string[];
  disclaimer?: string;
}

export const STAGE_PRE = "1_pre_draft";
export const STAGE_POST = "2_post_draft";

export interface TeamsResponse {
  league?: string;
  count: number;
  teams: string[];
}
