/**
 * 闸门: 逐秒胜率要不要平滑 (每一秒取它和前 45 秒的平均, 在对数几率上)?
 * =====================================================================
 * 走势图改成每秒一个点之后 (web/feed.fineRows), 胜率会一段不动、一下跳几个百分点, 来回抖:
 * 局内模型是 XGBoost, 输出对经济差是分段常数 —— 经济差在某个分界线 (比如 1250) 上下晃一百块,
 * 胜率就来回跳 7 个百分点 (2026-09-13 AL vs BLG 第 2 局, 人头一直 3:0)。另外 12.5/17.5/22.5 分钟
 * nearest_slice 换切片也会跳一个台阶。以前每分钟一个点, 这些都藏住了。
 *
 * 候选: 只看过去的滑动平均 (logit 空间), 窗口 45 秒 —— **在看结果之前定的**; 30 / 60 秒只列出来
 * 做参考, 不拿来挑。它只会做在静态站的显示层 (逐秒走势 + 直播回放), 看板本身和 Python 不动,
 * 所以用 web 的实现直接算。
 *
 * 考卷: backfill/snapshots.jsonl 里的四大赛区局 —— 全部落在 Stage 2 / Stage 4 的按日期留出测试段里
 * (research/gate_anchor.py 核过), 两份线上模型都没见过。lolesports 只留约 95 天, 只取最近 80 天的,
 * 随机抽 N 局, 每局按线上同一套逐秒走势重算 (fineTimeline), 只看第 3 分钟起 (之前是 BP 后/赛前概率,
 * 本来就平)。渐变的先验用赛前概率 (这里拿不到 BP 名单) —— 原始和平滑两边一样, 不影响比较。
 * 赛前特征按"今天"取 (fineTimeline 的口径), 两边同样受影响, 也不影响比较。
 *
 * 判据: 每局在段内按秒平均 Brier, 逐局配对, |t| > 2.5 才算显著; 平滑"不显著变差"即可采纳
 * (它是为了显示, 不是为了更准 —— 同 gate_anchor 的 C2)。另报每秒平均波动和 >=4pp 跳变的次数。
 *
 *     node research/gate_smoothing.mjs [--games 150] [--seed 1]
 *
 * **结果 (2026-09-13, 150 局): 明显变差, 不上线。**
 *
 *     分钟段    原始 Brier   平滑45s        t
 *     3-15      0.1970       0.1982      -2.23
 *     15-25     0.1581       0.1589      -1.56
 *     25+       0.0961       0.1007      -5.74
 *     全部      0.1616       0.1637     -13.25      (30s / 60s 一样: -13.28 / -13.01)
 *     每局 >=4pp 跳变  20.7 -> 0.0;  每秒平均波动 0.24 -> 0.07 个百分点
 *
 * 只看过去的平滑必然落后: 胜率在朝结果走的时候 (团战、大龙、终局前), 平滑值总停在几十秒前,
 * 而那恰恰是最确定的一段 —— 几乎每一局都吃这个亏。好看换不准, 不换。
 * 想去掉抖动得在**特征空间**里做, 不能在时间上做 (比如对经济差 ±几十块取平均、相邻切片之间插值),
 * 那样没有滞后; 要做同样得过闸。
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const WEB = "file:///" + path.join(ROOT, "frontend", "src", "web").replace(/\\/g, "/");
const { Feed } = await import(`${WEB}/feed.ts`);
const { fineTimeline } = await import(`${WEB}/board.ts`);
const { loadIngame } = await import(`${WEB}/ingame.ts`);
const { loadStage1 } = await import(`${WEB}/stage1.ts`);
const { loadStage2 } = await import(`${WEB}/stage2.ts`);
const { knownTeams, mapTeam } = await import(`${WEB}/teams.ts`);

const arg = (k, d) => {
  const i = process.argv.indexOf(`--${k}`);
  return i > 0 ? Number(process.argv[i + 1]) : d;
};
const N = arg("games", 150);
const SEED = arg("seed", 1);
const WINDOWS = [45, 30, 60]; // 第一个是候选, 其余只作参考
const SEGS = [["3-15", 3, 15], ["15-25", 15, 25], ["25+", 25, 999], ["全部", 3, 999]];
const MAJORS = new Set(["LPL", "LCK", "LEC", "LCS"]);
const T_ACCEPT = 2.5;

const J = (n) => JSON.parse(readFileSync(path.join(ROOT, "frontend", "public", "web", n), "utf8"));
const teams = J("teams.json");
const models = { ingame: loadIngame(J("ingame_live.json")), s1: loadStage1(J("stage1_pre.json")), s2: loadStage2(J("stage2_post.json")), ex: J("explain.json") };
const today = new Date().toISOString().slice(0, 10);

// 考卷
const games = new Map();
for (const line of readFileSync(path.join(ROOT, "backfill", "snapshots.jsonl"), "utf8").split("\n")) {
  if (!line.trim()) continue;
  const r = JSON.parse(line);
  if (!MAJORS.has(r.league)) continue;
  const g = games.get(r.game_id) ?? { id: r.game_id, league: r.league, date: r.date, blue: r.blue, red: r.red, y: r.y, last: 0 };
  g.last = Math.max(g.last, Date.parse(r.frame_t));
  games.set(r.game_id, g);
}
const cutoff = Date.now() - 80 * 86400_000;
let pool = [...games.values()].filter((g) => Date.parse(g.date.replace(" ", "T") + "Z") > cutoff);
let s = SEED;
const rnd = () => ((s = (s * 1103515245 + 12345) % 2147483648) / 2147483648);
pool = pool.map((g) => [rnd(), g]).sort((a, b) => a[0] - b[0]).map(([, g]) => g).slice(0, N);
console.log(`考卷: 四大赛区回溯 ${games.size} 局, 最近 80 天 ${pool.length} 局参与`);

const logit = (p) => Math.log(Math.min(Math.max(p, 1e-6), 1 - 1e-6) / (1 - Math.min(Math.max(p, 1e-6), 1 - 1e-6)));
const sig = (z) => 1 / (1 + Math.exp(-z));
/** 只看过去的滑动平均: 每个点取 (t - w 秒, t] 里所有点 logit 的平均 */
function smooth(pts, w) {
  const out = new Array(pts.length);
  let lo = 0;
  let sum = 0;
  for (let i = 0; i < pts.length; i++) {
    sum += logit(pts[i].p);
    while (pts[i].m - pts[lo].m > w / 60) sum -= logit(pts[lo++].p);
    out[i] = sig(sum / (i - lo + 1));
  }
  return out;
}

const feed = new Feed({ teams, runeLocale: "zh_CN" });
const per = []; // 每局: { seg: { raw, s45, s30, s60 } Brier }, 波动
let done = 0;
const t0 = Date.now();
async function one(g) {
  const kn = knownTeams(teams, g.league);
  const blue = mapTeam(teams, g.blue, kn);
  const red = mapTeam(teams, g.red, kn);
  if (!blue || !red) return;
  let pts = [];
  try {
    await fineTimeline(feed, teams, models, today, { gameId: g.id, blue, red, league: g.league, upto: g.last + 60_000, prior: null }, (p) => { pts = p; });
  } catch (e) {
    return;
  }
  pts = pts.filter((p) => p.minute >= 3 && p.probability_blue != null).map((p) => ({ m: p.minute, p: p.probability_blue }));
  if (pts.length < 300) return;
  const cand = { raw: pts.map((p) => p.p) };
  for (const w of WINDOWS) cand[`s${w}`] = smooth(pts, w);
  const rec = { y: g.y, seg: {}, vol: {}, jumps: {} };
  for (const [lab, lo, hi] of SEGS) {
    rec.seg[lab] = {};
    for (const [k, arr] of Object.entries(cand)) {
      let b = 0, n = 0;
      pts.forEach((p, i) => { if (p.m >= lo && p.m < hi) { b += (arr[i] - g.y) ** 2; n++; } });
      if (n) rec.seg[lab][k] = b / n;
    }
  }
  for (const [k, arr] of Object.entries(cand)) {
    let v = 0, j = 0;
    for (let i = 1; i < arr.length; i++) { const d = Math.abs(arr[i] - arr[i - 1]); v += d; if (d >= 0.04) j++; }
    rec.vol[k] = v / (arr.length - 1);
    rec.jumps[k] = j;
  }
  per.push(rec);
}
// 同时 3 局 (每局内部 fineRows 自己还有 8 路并发)
const queue = [...pool];
await Promise.all(Array.from({ length: 3 }, async () => {
  while (queue.length) {
    await one(queue.shift());
    if (++done % 25 === 0) console.log(`  ${done}/${pool.length}  (${((Date.now() - t0) / 1000).toFixed(0)}s)`);
  }
}));

function pairedT(a, b) {
  const d = a.map((x, i) => b[i] - x);
  const m = d.reduce((s, x) => s + x, 0) / d.length;
  const sd = Math.sqrt(d.reduce((s, x) => s + (x - m) ** 2, 0) / (d.length - 1));
  return [m, sd > 0 ? m / (sd / Math.sqrt(d.length)) : 0];
}
console.log(`\n用上 ${per.length} 局 (队名对不上或帧取不到的跳过)。Brier 越小越好; t > 0 = 平滑更好, |t| > ${T_ACCEPT} 才算显著`);
for (const [lab] of SEGS) {
  const rows = per.filter((r) => r.seg[lab]?.raw !== undefined);
  const raw = rows.map((r) => r.seg[lab].raw);
  const line = [`  第 ${lab.padEnd(5)} 分钟 (${rows.length} 局)  原始 ${(raw.reduce((s, x) => s + x, 0) / raw.length).toFixed(4)}`];
  for (const w of WINDOWS) {
    const sm = rows.map((r) => r.seg[lab][`s${w}`]);
    const [, t] = pairedT(sm, raw); // t > 0: 平滑的 Brier 更小
    line.push(`平滑${w}s ${(sm.reduce((s, x) => s + x, 0) / sm.length).toFixed(4)} (t ${t >= 0 ? "+" : ""}${t.toFixed(2)})${w === WINDOWS[0] ? " ←候选" : ""}`);
  }
  console.log(line.join("   "));
}
const avg = (k, f) => per.reduce((s, r) => s + r[f][k], 0) / per.length;
console.log(`\n  每秒平均波动 (百分点):  原始 ${(avg("raw", "vol") * 100).toFixed(2)}   ` + WINDOWS.map((w) => `平滑${w}s ${(avg(`s${w}`, "vol") * 100).toFixed(2)}`).join("   "));
console.log(`  每局 >=4pp 跳变次数:   原始 ${avg("raw", "jumps").toFixed(1)}   ` + WINDOWS.map((w) => `平滑${w}s ${avg(`s${w}`, "jumps").toFixed(1)}`).join("   "));
