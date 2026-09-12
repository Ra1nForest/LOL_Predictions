/**
 * 界面语言 —— 跟随系统语言: 系统语言是中文就显示中文, 其余一律英文。
 * 地址栏加 ?lang=en / ?lang=zh 可以强制 (调试用, 界面上不露出)。
 *
 * 两类文字分开处理:
 *
 *   T(zh, vars)  界面上写死的字。**以中文原文作键**查英文表: 组件里读到的仍然是中文,
 *                符合本项目"界面文案用中文写"的约定, 也不必另起一套 key 名。
 *
 *   tx(text)     程序生成的句子 —— 胜率解析、提醒、模型说明、报错。生成它们的
 *                src/web/*.ts 和服务端的 api.py 是逐字对账过的 (test:golden 拿中文
 *                原文和 Python 比), 所以**不去改生成端**, 而是显示前按句式翻:
 *                中文句式编译成正则, 抓出队名和数字, 填进英文句式。
 *                认不出的句子**原样显示中文** —— 宁可露出一句中文, 也不要猜着翻错。
 *
 * 英文模式下 client.ts 把取回的响应整个过一遍 localize(), 组件不用逐个处理;
 * 服务器版和静态版的响应都走这一处。句式表是照着 explain.json 的文案和各处写死的
 * 句子抄的: 那边改了措辞, 这边就退回显示中文, 不会出错 —— 开发模式下控制台会
 * 列出没翻到的句子, 照着补上即可。
 */
export type Lang = "zh" | "en";

function detect(): Lang {
  try {
    const q = new URLSearchParams(window.location.search).get("lang");
    if (q === "zh" || q === "en") return q;
  } catch {
    /* 不在浏览器里 */
  }
  const nav = typeof navigator === "undefined" ? undefined : navigator;
  const first = (nav?.languages?.[0] ?? nav?.language ?? "zh").toLowerCase();
  return first.startsWith("zh") ? "zh" : "en";
}

export const LANG: Lang = detect();
export const EN = LANG === "en";

/** <html lang> 和标签页标题跟着语言走 */
export function applyDocumentLang(): void {
  document.documentElement.lang = EN ? "en" : "zh-CN";
  if (EN) document.title = "Win Probability · LoL";
}

const missed = new Set<string>();
function miss(s: string) {
  // env?. —— Node 直接跑这个文件 (检查翻译覆盖的脚本) 时没有 import.meta.env
  if (import.meta.env?.DEV && !missed.has(s)) {
    missed.add(s);
    console.warn(`[i18n] 没有英文: ${s}`);
  }
}

const fillVars = (s: string, vars?: Record<string, string | number>) =>
  vars ? Object.entries(vars).reduce((a, [k, v]) => a.replaceAll(`{${k}}`, String(v)), s) : s;

// ══════════════════════════════════════════════════════════
//  界面文字
// ══════════════════════════════════════════════════════════

const UI: Record<string, string> = {
  // 首页
  胜率预测: "Win Probability",
  "四大赛区比赛的实时胜率预测。": "Live win probabilities for LPL, LCK, LEC and LCS matches.",
  正在进行: "Live now",
  当前没有进行中的比赛: "No matches are live right now",
  即将开始: "Upcoming",
  近期没有排期: "Nothing scheduled soon",
  "今天 {hm}": "Today {hm}",
  "明天 {hm}": "Tomorrow {hm}",
  "缺少历史数据, 无法预测": "Not enough history to predict",
  无历史数据: "No history",
  进行中: "Live",
  已结束: "Finished",
  暂停中: "Paused",
  刷新: "Refresh",
  "加载中…": "Loading…",
  // 看板
  "← 返回": "← Back",
  "第 {n} 局": "Game {n}",
  "第 {n} 分钟": "Minute {n}",
  蓝方: "Blue",
  红方: "Red",
  数据中断: "Feed interrupted",
  状态未知: "Unknown state",
  "期间暂停约 {n} 分钟": "Paused for about {n} min",
  "数据停留在 {n} 分钟前": "Data stuck {n} min ago",
  "数据落后约 {n} 分钟": "Data about {n} min behind",
  局间休息: "Between games",
  尚未开始: "Not started",
  "训练数据截至 {date}": "Trained on data through {date}",
  "{team} 占优": "{team} favored",
  "已计入双方阵容, 局内模型尚未启用": "Both drafts counted · in-game model not active yet",
  尚未开打: "Not started yet",
  暂无预测数据: "No prediction yet",
  胜率走势: "Win probability",
  经济差: "Gold difference",
  准确率约: "Accuracy ≈",
  模型说明: "About the model",
  " 与 {n} 条提醒": " & {n} notes",
  // 赛前页
  "{league} 数据截至 {date}（{days} 天前）": "{league} data through {date} ({days} days ago)",
  "已计入 BP 的预测": "Includes the draft",
  "赛前预测，尚未开打": "Pre-match prediction · not started",
  // 概率刻度 / 走势图 / 经济图
  胜率解析: "Why this number",
  胜率: "Win prob.",
  斜线区间超出模型输出范围: "Hatched areas are beyond the model's output range",
  赛前: "Pre-match",
  "BP 后": "After draft",
  还没有胜率数据: "No win-probability data yet",
  "{team} 胜率超过 {n}% 时只显示到此": "Capped here when {team} is above {n}%",
  势均力敌: "Even",
  "暂停 / 数据中断 —— 胜负未定": "Paused / feed interrupted — undecided",
  这一局已结束: "This game is over",
  还没有经济数据: "No gold data yet",
  "{team} 领先": "{team} ahead",
  "相对权重 {n}%": "Relative weight {n}%",
  击杀: "Kills",
  "峰值 {v}": "Peak {v}",
  // 记分板
  记分板: "Scoreboard",
  左右滑动看全部: "Swipe for more",
  经济持平: "Gold even",
  持平: "Even",
  总经济: "Total gold",
  推塔: "Towers",
  小龙: "Dragons",
  大龙: "Barons",
  "水晶 (抑制器)": "Inhibitors",
  "{name} — 点开看详情": "{name} — tap for details",
  阵亡: "Dead",
  饰品: "Trinket",
  "饰品 {id}": "Trinket {id}",
  "{n} 刀": "{n} CS",
  火龙: "Infernal Drake",
  地龙: "Mountain Drake",
  海龙: "Ocean Drake",
  风龙: "Cloud Drake",
  法龙: "Hextech Drake",
  化龙: "Chemtech Drake",
  远古龙: "Elder Dragon",
  // 选手详情
  选手详情: "Player details",
  "{n} 级": "Level {n}",
  关闭: "Close",
  上单: "Top",
  打野: "Jungle",
  中单: "Mid",
  ADC: "ADC",
  辅助: "Support",
  补刀: "CS",
  经济: "Gold",
  参团率: "Kill part.",
  伤害占比: "Damage share",
  视野: "Wards",
  属性: "Stats",
  符文: "Runes",
  "技能加点 ({n} 级)": "Skill order ({n} levels)",
  攻击力: "Attack damage",
  法强: "Ability power",
  护甲: "Armor",
  魔抗: "Magic resist",
  攻速: "Attack speed",
  暴击: "Crit chance",
  吸血: "Life steal",
  韧性: "Tenacity",
  // 主题
  主题: "Theme",
  亮色: "Light",
  暗色: "Dark",
  跟随系统: "Match system",
};

/** 界面上写死的字: 中文原文作键。vars 填 {name} 占位符, 中英文都填 */
export function T(zh: string, vars?: Record<string, string | number>): string {
  if (!EN) return fillVars(zh, vars);
  const en = UI[zh];
  if (en === undefined) miss(zh);
  return fillVars(en ?? zh, vars);
}

// ══════════════════════════════════════════════════════════
//  程序生成的句子
// ══════════════════════════════════════════════════════════

/** 整句一对一的词条 —— 也用来翻句式里抓出来的片段 (统计项名、位置、交互项…) */
const TERMS: Record<string, string> = {
  季后赛: "Playoffs",
  势均力敌: "Evenly matched",
  赛前: "Pre-match",
  "BP 结束": "After draft",
  局内: "In-game",
  非投注建议: "Not betting advice",
  // 统计项 (explain.json 的 stat / live 名字, 出现在"双方{名字}持平"里)
  场均击杀: "Average kills",
  场均死亡: "Average deaths",
  场均助攻: "Average assists",
  场均团队经济: "Average team gold",
  场均对英雄伤害: "Average damage to champions",
  场均插眼: "Average wards placed",
  场均排眼: "Average wards cleared",
  场均小龙: "Average dragons",
  场均大龙: "Average Barons",
  场均推塔: "Average towers",
  每分钟补刀: "CS per minute",
  "10 分钟经济差": "Gold difference at 10 min",
  "15 分钟经济差": "Gold difference at 15 min",
  一血率: "First-blood rate",
  首条小龙率: "First-dragon rate",
  一塔率: "First-tower rate",
  每分钟总人头: "Kills per minute (both teams)",
  每分钟击杀: "Kills per minute",
  近十场胜率: "Win rate over the last 10 games",
  场均时长: "Average game length",
  场均总人头: "Average total kills",
  阵容英雄平均胜率: "Champion win rate",
  英雄熟练度: "Champion experience",
  经济: "Gold",
  经验: "XP",
  补刀: "CS",
  人头: "Kills",
  // 交互项 (explain.json 的 interaction)
  "经济领先 × 阵容强势期": "Gold lead × comp scaling",
  "相对经济领先 × 阵容强势期": "Relative gold lead × comp scaling",
  "经验领先 × 阵容强势期": "XP lead × comp scaling",
  "经济领先 × 对手前期成色": "Gold lead × opponent's early-game strength",
  // 位置
  上单: "top laner",
  打野: "jungler",
  中单: "mid laner",
  ADC: "ADC",
  辅助: "support",
  // 小符文碎片 (feed.ts / esports_feed 的 STAT_SHARDS)
  成长生命值: "Scaling Health",
  护甲: "Armor",
  魔法抗性: "Magic Resist",
  攻击速度: "Attack Speed",
  技能急速: "Ability Haste",
  适应之力: "Adaptive Force",
  移动速度: "Move Speed",
  生命值: "Health",
  抗性成长: "Scaling Resists",
  韧性和减速抗性: "Tenacity and Slow Resist",
  // 报错里的名词
  队伍列表: "the team list",
  "AI 辩论": "AI debate",
};

/**
 * [中文句式, 英文句式]。{名字} 抓任意文字, {名字:甲|乙} 只抓列出的几种。
 * 从上往下试, 整句匹配 —— 更具体的放前面。
 */
const RULES: [string, string][] = [
  // ── 胜率解析: 队伍统计 (explain.json 的 stat) ──
  ["{who} 场均多拿 {v} 个人头", "{who} averages {v} more kills per game"],
  ["{team} 场均 {v} 个人头", "{team} averages {v} kills per game"],
  ["双方场均合计 {v} 个人头", "The two teams combined average {v} kills per game"],
  ["{who} 场均少死 {v} 次", "{who} averages {v} fewer deaths per game"],
  ["{team} 场均送 {v} 个人头", "{team} averages {v} deaths per game"],
  ["双方场均合计送 {v} 个人头", "The two teams combined average {v} deaths per game"],
  ["{who} 场均多 {v} 个助攻", "{who} averages {v} more assists per game"],
  ["{team} 场均 {v} 个助攻", "{team} averages {v} assists per game"],
  ["{who} 场均多出 {v} 经济", "{who} averages {v} more gold per game"],
  ["{team} 场均团队经济 {v}", "{team} averages {v} team gold per game"],
  ["{who} 场均多打 {v} 英雄伤害", "{who} deals {v} more damage to champions per game"],
  ["{team} 场均对英雄伤害 {v}", "{team} averages {v} damage to champions"],
  ["{who} 场均多插 {v} 个眼", "{who} places {v} more wards per game"],
  ["{team} 场均插 {v} 个眼", "{team} places {v} wards per game"],
  ["{who} 场均多排 {v} 个眼", "{who} clears {v} more wards per game"],
  ["{team} 场均排 {v} 个眼", "{team} clears {v} wards per game"],
  ["{who} 场均多拿 {v} 条小龙", "{who} takes {v} more dragons per game"],
  ["{team} 场均拿 {v} 条小龙", "{team} takes {v} dragons per game"],
  ["双方场均合计 {v} 条小龙", "The two teams combined take {v} dragons per game"],
  ["{who} 场均多拿 {v} 条大龙", "{who} takes {v} more Barons per game"],
  ["{team} 场均拿 {v} 条大龙", "{team} takes {v} Barons per game"],
  ["{who} 场均多推 {v} 座塔", "{who} takes {v} more towers per game"],
  ["{team} 场均推 {v} 座塔", "{team} takes {v} towers per game"],
  ["双方场均合计推 {v} 座塔", "The two teams combined take {v} towers per game"],
  ["{who} 每分钟多 {v} 刀补兵", "{who} gets {v} more CS per minute"],
  ["{team} 每分钟 {v} 刀补兵", "{team} gets {v} CS per minute"],
  ["{who} 的十分钟经济表现好 {v}", "{who}'s gold difference at 10 min is better by {v}"],
  ["{team} 场均十分钟经济差 {v}", "{team}'s average gold difference at 10 min is {v}"],
  ["{who} 的十五分钟经济表现好 {v}", "{who}'s gold difference at 15 min is better by {v}"],
  ["{team} 场均十五分钟经济差 {v}", "{team}'s average gold difference at 15 min is {v}"],
  ["{who} 的一血率高 {v}", "{who}'s first-blood rate is {v} higher"],
  ["{team} 一血率 {v}", "{team}'s first-blood rate is {v}"],
  ["{who} 更常拿到首条小龙 (高 {v})", "{who} takes first dragon more often ({v} higher)"],
  ["{team} 首条小龙率 {v}", "{team}'s first-dragon rate is {v}"],
  ["{who} 的一塔率高 {v}", "{who}'s first-tower rate is {v} higher"],
  ["{team} 一塔率 {v}", "{team}'s first-tower rate is {v}"],
  ["{who} 的比赛节奏更快 (每分钟多 {v} 个人头)", "{who} plays faster games ({v} more kills per minute)"],
  ["{team} 场均每分钟 {v} 个人头", "{team}'s games average {v} kills per minute"],
  ["双方节奏合计 {v} 人头/分钟", "Combined pace: {v} kills per minute"],
  ["{who} 每分钟多拿 {v} 个人头", "{who} gets {v} more kills per minute"],
  ["{team} 场均每分钟击杀 {v} 个", "{team} averages {v} kills per minute"],
  ["双方每分钟击杀合计 {v} 个", "The two teams combined get {v} kills per minute"],
  ["{who} 近十场胜率高 {v}", "{who}'s win rate over the last 10 games is {v} higher"],
  ["{team} 近十场胜率 {v}", "{team}'s win rate over the last 10 games is {v}"],
  ["{who} 的比赛平均长 {v} 分钟", "{who}'s games run {v} min longer on average"],
  ["{team} 场均时长 {v} 分钟", "{team}'s games average {v} min"],
  ["双方场均时长合计 {v} 分钟", "Combined average game length: {v} min"],
  ["{who} 的比赛人头数多 {v} 个", "{who}'s games have {v} more total kills"],
  ["{team} 场均总人头 {v} 个", "{team}'s games average {v} total kills"],
  ["双方场均总人头合计 {v} 个", "The two teams' games combined average {v} total kills"],
  ["{who} 阵容里的英雄近期胜率高 {v}", "{who}'s champions have a {v} higher recent win rate"],
  ["{team} 阵容英雄平均胜率 {v}", "{team}'s champions average a {v} win rate"],
  ["{who} 这套阵容的总使用场次多 {v} 场", "{who}'s players have {v} more total games on these champions"],
  ["{team} 这套阵容累计使用 {v} 场", "{team}'s players have {v} total games on these champions"],

  // ── 胜率解析: 局内和其余 (explain.ts 里写死的句子) ──
  ["双方经济持平", "Gold is even"],
  ["双方{cn}持平", "{cn}: even"],
  ["{team} 补刀领先 {v} 刀", "{team} leads in CS by {v}"],
  ["{team} 人头领先 {v} 个", "{team} leads in kills by {v}"],
  ["{team} 经济领先 {v}", "{team} leads in gold by {v}"],
  ["{team} 经验领先 {v}", "{team} leads in XP by {v}"],
  ["{team} {cn:经济|经验|补刀|人头}领先 {v}", "{team} leads in {cn} by {v}"],
  ["{team} 的经济领先占场上总经济的 {p}%", "{team}'s gold lead is {p}% of all gold on the map"],
  ["双方已合计击杀 {n} 个人头", "{n} kills so far in total"],
  ["按第 {n} 分钟的模型切片评估", "Evaluated with the {n}-minute model slice"],
  ["{x} —— 当前利于 {team}", "{x} — currently favors {team}"],
  ["{team} 的阵容偏后期", "{team}'s comp leans late-game"],
  ["{team} 的阵容偏前期", "{team}'s comp leans early-game"],
  ["{team} 的阵容前后期均衡", "{team}'s comp is balanced across the game"],
  ["双方阵容的强势期相当", "Both comps peak at about the same time"],
  ["{late} 的阵容更偏后期, {early} 需要在前期建立优势", "{late}'s comp scales better; {early} needs to get ahead early"],
  ["{team} 当前连胜 {n} 场", "{team} is on a {n}-game win streak"],
  ["{team} 当前连败 {n} 场", "{team} is on a {n}-game losing streak"],
  ["{team} {role:上单|打野|中单|ADC|辅助} 用这个英雄历史胜率 {p}%", "{team}'s {role} has a {p}% win rate on this champion"],
  ["{team} {role:上单|打野|中单|ADC|辅助} 没用过这个英雄", "{team}'s {role} has never played this champion"],
  ["{team} {role:上单|打野|中单|ADC|辅助} 用过这个英雄 {n} 场", "{team}'s {role} has played this champion {n} times"],
  ["{team} {role:上单|打野|中单|ADC|辅助} 对这个英雄比对位多 {n} 场经验", "{team}'s {role} has {n} more games on their champion than their lane opponent"],

  // ── 一句话总结 (explain.json 的 tone, 前面的"赛前 · / 第 N 分钟 ·"由 " · " 拆开单独翻) ──
  ["{lead} 略占优", "{lead} slightly favored"],
  ["{lead} 优势明显", "{lead} clearly favored"],
  ["{lead} 大幅领先", "{lead} well ahead"],
  ["{lead} 占优", "{lead} favored"],
  ["第 {n} 分钟", "Minute {n}"],

  // ── 模型说明 (explain.confidence_note / range_note, ingame 的免责声明) ──
  ["留出测试集准确率 {a}% (基线 {b}%)", "Held-out test accuracy {a}% (baseline {b}%)"],
  ["留出测试集准确率 {a}%", "Held-out test accuracy {a}%"],
  ["概率校准良好 (ECE {e})", "Well calibrated (ECE {e})"],
  ["概率校准可用 (ECE {e})", "Calibration acceptable (ECE {e})"],
  ["概率校准偏差偏大 (ECE {e})", "Calibration is off (ECE {e})"],
  ["该模型的概率输出被校准限制在 {lo}%–{hi}%", "Calibration limits this model's output to {lo}%–{hi}%"],
  ["超出这个范围的一边倒对局, 它只会顶到边界, 不会给出更极端的数字", "In games more one-sided than that it stops at the edge rather than going further"],
  ["局内模型在留出测试集上准确率约 {a}, 25 分钟时约 {b}", "The in-game model is about {a} accurate on the held-out test set, about {b} at 25 minutes"],
  ["局内模型在留出测试集上准确率约 {a}", "The in-game model is about {a} accurate on the held-out test set"],
  ["概率经保序回归校准 (ECE ≈ {e})", "Probabilities are calibrated with isotonic regression (ECE ≈ {e})"],
  ["概率经Platt校准 (ECE ≈ {e})", "Probabilities are Platt-calibrated (ECE ≈ {e})"],
  ["概率来自留出测试集校准的统计模型, 非投注建议", "Probabilities come from a statistical model calibrated on a held-out test set; not betting advice"],
  ["模型概率输出范围有限, 对一边倒的对局判别力不足", "The model's output range is limited, so it separates very one-sided games poorly"],
  ["walk-forward 估计的期望 lift 约 +0.13 ± 0.10, 单场预测不确定性大", "Walk-forward estimated lift is about +0.13 ± 0.10; any single prediction is very uncertain"],
  ["⚠ 本次预测基于 {d} 天前的数据, 结果不可靠", "⚠ This prediction uses data from {d} days ago and is unreliable"],
  ["⚠ 数据滞后 {d} 天, 未反映最近比赛", "⚠ Data is {d} days behind and misses recent matches"],
  ["Stage 3 为 advisory-only, 不影响此概率", "Stage 3 is advisory-only and doesn't change this probability"],

  // ── 提醒 (ingame.ts / stage1.ts / teams.ts) ──
  ["你填的是第 {m} 分钟, 但模型只在 10/15/20/25 分钟这几个时间点训练过", "This is minute {m}, but the model was only trained at minutes 10/15/20/25"],
  ["已按最接近的 {t} 分钟评估, 相差 {d} 分钟, 精度会下降", "It was evaluated at the nearest one ({t} min), {d} min away, so accuracy drops"],
  ["LPL 在局内训练集里样本偏薄 (约 9%, 且全部来自 2026 年) —— Oracle's Elixir 在 2022-2025 不提供 LPL 的分钟级快照", "LPL is thin in the in-game training set (about 9%, all from 2026) — Oracle's Elixir had no minute-level LPL snapshots for 2022–2025"],
  ["结论在 LPL 上的可靠性弱于其他三大赛区", "Results are less reliable for LPL than for the other three leagues"],
  ["这几个英雄的历史样本不足, 算不出阵容强势期 —— 「领先但阵容吃亏」这类隐患本次无法识别", "Not enough history on these champions to compute comp scaling — risks like \"ahead but out-scaled\" can't be flagged this time"],
  ["没填双方阵容, 无法判断谁的阵容更耐拖 —— 同样的经济差在不同阵容下含义相反", "No lineups, so it can't tell whose comp scales better — the same gold lead means opposite things for different comps"],
  ["没填蓝方总经济, 按第 {t} 分钟的实测中位数 {g} 估算, 相对经济差可能有偏", "Blue team's total gold is missing; estimated from the median at minute {t} ({g}), so the relative gold lead may be off"],
  ["没有经验差数据, 而当前模型把它当作输入 —— 结果不可靠", "No XP data, but this model uses it as an input — the result is unreliable"],
  ["实时数据应当走 live 变体模型", "Live data should use the live model variant"],
  ["本次没有经验差数据(实时数据源不提供), 已改用训练时就不含经验差的模型, 而不是用 0 顶替", "No XP data this time (the live feed doesn't provide it), so a model trained without XP is used instead of filling in 0"],
  ["没有可用的赛前队伍统计, 本次只看局内数据", "No pre-match team stats available; using in-game data only"],
  ["概率已经顶到模型的表达上限 {p} —— 这种局面的实际胜率可能更高", "The probability has hit the model's upper limit of {p} — the real win rate here may be higher"],
  ["概率已经触到模型的表达下限 {p} —— 这种局面的实际胜率可能更低", "The probability has hit the model's lower limit of {p} — the real win rate here may be lower"],
  ["这是校准的固有上限, 不是判断失误", "That's a built-in calibration limit, not a misjudgment"],
  ["赛前特征不可用: {msg}", "Pre-match features unavailable: {msg}"],
  ["相比赛前, 局势已向 {team} 移动 {n} 个百分点", "Since pre-match, the game has swung {n} points toward {team}"],
  ["{nm} 仅有 {n} 场历史, 统计不稳定", "{nm} has only {n} games of history; stats are unstable"],
  ["{nm} 最后一场是 {d} ({g} 天前), 近期状态未知", "{nm}'s last game was on {d} ({g} days ago); recent form unknown"],
  ["{lg} 数据截至 {d} ({n} 天前) — 滚动窗口缺少最近比赛, 预测偏向旧状态", "{lg} data runs through {d} ({n} days ago) — the rolling window is missing recent matches, so predictions lean on older form"],
  ["{lg} 数据截至 {d} ({n} 天前) — 可能整段赛程缺失, 若期间有转会或换人, 队伍统计描述的是一支已不存在的阵容, 预测不可靠", "{lg} data runs through {d} ({n} days ago) — a whole stretch of the schedule may be missing; if rosters changed, the stats describe a lineup that no longer exists, so predictions are unreliable"],
  ["{lg} 无数据", "No data for {lg}"],

  // ── 看板的说明 (board.ts / api.py) ──
  ["开局第 {m} 分钟, 局内模型最早在第 10 分钟的切片上训练过", "Minute {m}; the in-game model's earliest slice is minute 10"],
  ["刚开局 (局内时间还没同步出来), 局内模型最早在第 10 分钟的切片上训练过", "The game just started (the in-game clock hasn't synced yet); the in-game model's earliest slice is minute 10"],
  ["这里显示的是BP 后的概率 —— 它不看局内数据, 但一直有效", "Showing the after-draft probability — it ignores in-game data but stays valid"],
  ["这里显示的是赛前的概率 —— 它不看局内数据, 但一直有效", "Showing the pre-match probability — it ignores in-game data but stays valid"],
  ["这一局的数据流已经 {n} 分钟没有更新了 —— 帧里还写着 {state}, 但它停在 {t}, 不是实时战况", "This game's feed hasn't updated for {n} min — the frames still say {state} but stopped at {t}; this isn't live"],
  ["赛程也已经把这一局标为结束", "The schedule has also marked this game as finished"],
  ["局内分钟数 {m} 超出局内模型的输入范围 (3-60)", "In-game minute {m} is outside the in-game model's input range (3–60)"],
  // 选边说明: 界面上目前不显示 (见 Board.tsx 的 side_warning 注释), 但响应里有, 一并翻
  ["以帧里的 gameState 为准, persisted_state 会滞后", "Frame gameState is authoritative; persisted_state lags behind"],
  ["按本局选边调整了左右", "Left/right adjusted to this game's sides"],
  ["两个官方源对本局选边说法不一: 实时帧说蓝方是 {b}, 赛程接口说是 {p}", "The two official sources disagree on sides: live frames say blue is {b}, the schedule says {p}"],
  ["页面上的数字全部按实时帧分组, 因此以帧为准; 若转播画面显示的是另一边, 以转播为准", "All numbers follow the live frames; if the broadcast shows the other side, trust the broadcast"],
  ["本局蓝方是 {b}, 但和赛程里的队名都对不上 —— 左右未调整, 可能是反的", "Blue side this game is {b}, which matches neither scheduled team — left/right not adjusted and may be reversed"],
  ["拿不到本局的队伍对照, 左右可能不准", "Couldn't match this game's teams; left/right may be wrong"],
  ["这一局的帧里没有选边信息, 左右可能不准", "This game's frames have no side info; left/right may be wrong"],
  ["本局还没有实时帧, 无法确认选边 —— 左右按赛程顺序排, 不代表蓝红", "No live frames yet, so sides can't be confirmed — teams are in schedule order, not blue/red"],
  ["{what}: {err}", "{what}: {err}"],
  ["取队名失败, 无法确认本局选边", "Couldn't get team names, so sides can't be confirmed"],
  ["取帧选边失败", "Couldn't read sides from frames"],
  ["查找比赛信息失败", "Couldn't find match info"],
  ["头条取阵容失败, 本次预测不含阵容强势期", "Couldn't get lineups; this prediction has no comp scaling"],
  ["曲线取赛前特征失败, 曲线会偏离头条", "Couldn't get pre-match features for the chart; it may differ from the headline"],
  ["曲线取阵容失败, 曲线会偏离头条", "Couldn't get lineups for the chart; it may differ from the headline"],

  // ── 报错 ──
  ["请求超时", "Request timed out"],
  ["请求格式不对: {x}", "Bad request: {x}"],
  ["静态版不提供{x}", "Not available on the static site: {x}"],
  ["对局详情拉取失败: {msg}", "Couldn't load game details: {msg}"],
  ["模型文件 {f} 下载失败: HTTP {s}", "Couldn't download model file {f}: HTTP {s}"],
  ["{url} 取数失败: {x}", "{url} request failed: {x}"],
  ["未知赛区 {x}", "Unknown league {x}"],
  ["still loading", "Still loading"],
];

interface Rule {
  re: RegExp;
  names: string[];
  en: string;
}

const esc = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

function compile([zh, en]: [string, string]): Rule {
  const names: string[] = [];
  let src = "";
  let last = 0;
  for (const m of zh.matchAll(/\{(\w+)(?::([^}]+))?\}/g)) {
    src += esc(zh.slice(last, m.index));
    names.push(m[1]!);
    // 占位符不许跨过句号: 不然三句连写的免责声明会被第一句的句式整个吞下,
    // 后两句塞进最后一个占位符里, 译文顺序全乱 (实测过)
    src += m[2] ? `(${m[2].split("|").map(esc).join("|")})` : "([^。]+?)";
    last = m.index! + m[0].length;
  }
  src += esc(zh.slice(last));
  return { re: new RegExp(`^${src}$`), names, en };
}

const COMPILED = RULES.map(compile);
const CJK = /[㐀-鿿]/;

function translate(s: string, depth = 0): string {
  const t = s.trim();
  if (!CJK.test(t) || depth > 5) return s;
  // " · " 拼起来的 (一句话总结、模型说明): 逐段翻
  if (t.includes(" · ")) return t.split(" · ").map((p) => translate(p, depth + 1)).join(" · ");
  const term = TERMS[t];
  if (term !== undefined) return term;
  for (const r of COMPILED) {
    const m = r.re.exec(t);
    if (m) return r.names.reduce((out, n, i) => out.replaceAll(`{${n}}`, translate(m[i + 1]!, depth + 1)), r.en);
  }
  // 几句连在一起的 (提醒、免责声明): 按句号拆开逐句翻
  const parts = t.split("。").map((p) => p.trim()).filter(Boolean);
  if (parts.length > 1 || t.endsWith("。")) {
    return parts
      .map((p) => {
        const x = translate(p, depth + 1);
        return CJK.test(x) ? `${x}。` : /[.!?]$/.test(x) ? x : `${x}.`;
      })
      .join(" ");
  }
  miss(t);
  return s;
}

const cache = new Map<string, string>();

/** 程序生成的一句话 → 当前语言。中文模式原样返回 */
export function tx(s: string): string {
  if (!EN || !CJK.test(s)) return s;
  let out = cache.get(s);
  if (out === undefined) {
    if (cache.size > 5000) cache.clear();
    out = translate(s);
    cache.set(s, out);
  }
  return out;
}

/** 整个响应里带中文的字符串都过一遍 tx。中文模式原样返回 (不复制) */
export function localize<V>(v: V): V {
  if (!EN) return v;
  const walk = (x: unknown): unknown => {
    if (typeof x === "string") return tx(x);
    if (Array.isArray(x)) return x.map(walk);
    if (x && typeof x === "object") {
      const o: Record<string, unknown> = {};
      for (const [k, y] of Object.entries(x)) o[k] = walk(y);
      return o;
    }
    return x;
  };
  return walk(v) as V;
}
