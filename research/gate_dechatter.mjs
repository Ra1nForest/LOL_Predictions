/**
 * 闸门 G3: 在**特征空间**里给逐秒胜率去抖 —— 经济差 ±200 邻域平均 (只为显示)
 * ===========================================================================
 * 为什么要做: 局内模型是 XGBoost, 输出对经济差是分段常数。经济差在某个分界线上下晃一百块,
 * 胜率就来回跳几个百分点 (人头、塔、龙都没变)。逐秒走势 (web/feed.fineRows) 把这个抖动全露了出来。
 * 在时间上平滑已经被否掉 (research/gate_smoothing.mjs: 只看过去的滑动平均必然落后, 全局 Brier
 * t = −13.25)。那里留的出路是在特征空间里做 —— 对同一时刻、邻近的经济差取平均, 不看过去, 没有滞后。
 * 这是那条出路的闸门。
 *
 * **协议在看结果之前定死 (2026-10-17), 评估数据上什么都不调:**
 *
 *   候选  每个评估点, 把经济差换成 [gd − 200, gd + 200] 里等距的 9 个值 (−200, −150, …, +200),
 *         由经济差派生的特征 (golddiff_norm, x_gold_scaling, x_goldnorm_scaling, lead_by_early_comp)
 *         全由 buildRow (= ingame_service.make_row) 从挪过的经济差重算; 其余输入不动 —— 蓝方总经济
 *         gold_total 是独立输入, 不是由经济差派生的, 也不动。9 个**校准后**的局内概率 (live 变体)
 *         取算术平均, 再原样走 3–15 分钟渐变 (blendPrior), 最后同看板一样 round 到 4 位。
 *   基线  看板现在的数: web/board.ts 的 fineTimeline (逐秒走势) 原样的 probability_blue。
 *         候选那一路是在这里按 curveCtx / curveProb 重搭的; 挪动量全取 0 时必须和基线**逐点相等**,
 *         否则整个对比不作数 (脚本里有这条自检, 不等就退出)。
 *   考卷  backfill/snapshots.jsonl 里四大赛区 + 国际赛 (feed.LEAGUE_IDS 里四大之外的那几个) 的局,
 *         日期 ≥ 2026-01-15 —— Stage 4 的测试段, 树和保序表都没见过 (G2 之前的导出才成立, 见 requireUnseen)。洗牌和 gate_smoothing 是同一个
 *         LCG, seed 1, 取前 160 局; 帧取不到、第 3 分钟起不到 300 个点的跳过 (同 gate_smoothing)。
 *         每局第 3 分钟起的每一个逐秒点都算。
 *   看板口径 (boardPlan)  四大 / 同母赛区的国际赛: 带赛前特征, 渐变的锚 = 赛前 (Stage 1) 概率 ——
 *         拿不到 BP 名单 (回填里没有战队简称, draftFromMeta 要它), 同 gate_smoothing。跨赛区或有队名
 *         不认识: 不带赛前特征; 模型 C 要赛事内重放, 回填里没有 match id, 所以不渐变 (= 看板拿不到 C 时
 *         的样子)。两边永远是同一个锚, 只有去抖不同。赛前特征按"今天"取 (fineTimeline 的口径), 两边一样。
 *   主判据  每局所有逐秒点的平均 Brier; 逐局配对, **系列赛层面**的 t (同一系列赛的局先平均再配对。
 *         系列赛 = 同赛区、同一对队伍、相邻两局开赛间隔 ≤ 12 小时连成一串)。t > 0 = 候选更好。
 *         **采纳 iff t > −2.5** (不显著变差 —— 显示用的改动, 规则和 gate_anchor 的 C2 一样, 用户已同意
 *         按这条判)。同时报 t 是否 > +2.5 (显著变好)、逐局 t、分钟段。
 *   次要  每局无事件的 ≥4pp / ≥2pp 跳变次数 (事件 = 相邻两点之间人头、塔、龙、大龙、水晶任何一项有变化);
 *         逐分钟回填上的同一个候选 (Python, research/gate_dechatter_backfill.py); 9 倍树求值的 CPU 成本。
 *
 *     node research/gate_dechatter.mjs [--games 160] [--seed 1] [--cache DIR] [--out per_game.json] [--ingame FILE]
 *
 * --ingame FILE: 局内模型用这份导出, 不用 public/web/ingame_live.json。G2 之后那份是服务模型, 考卷那段它拟合时
 * 见过, 脚本会拒绝 (requireUnseen) —— 复现时指向 G2 之前的导出。
 *
 * --cache DIR: 把 feed.lolesports.com 的 200/204 响应存到 DIR (打完的局, 帧不会再变), 重跑不再打上游。
 *
 * **结果 (2026-10-17, 160 局 / 133 个系列赛 / 283404 个逐秒点; 自检: 挪动 0 时逐点等于看板): 采纳。**
 *
 *     分钟段   局 / 系列    基线 Brier   候选       Δ(基线−候选, 逐局)   逐局 t   系列赛 t
 *     3-15     160 / 133    0.20074     0.20058    +0.00015            +2.04    +1.96
 *     15-25    160 / 133    0.14834     0.14825    +0.00009            +1.24    +0.95
 *     25+      154 / 128    0.08279     0.08268    +0.00011            +2.40    +1.93
 *     全部     160 / 133    0.15451     0.15440    +0.00012            +2.73    +2.39   ← 主判据
 *     (子集, 全部分钟: 四大 123 局 / 100 系列 系列赛 t +1.78; 国际赛 37 局 / 33 系列 +1.60)
 *
 *   主判据系列赛 t = +2.39 > −2.5 → 不显著变差, 采纳。方向是变好, 但没到 +2.5, 不能说"更准"。
 *   跳变 (每局平均): 无事件 >=4pp 25.9 → 7.6 (逐局配对 t +15.1, 160 局里没有一局变多);
 *   无事件 >=2pp 79.5 → 38.5 (t +25.2); 全部 >=4pp 35.0 → 15.6。候选和看板逐点差平均 0.47pp, 最大 5.7pp。
 *   选边: 抽到的局里能和 OE 第 10/15 分钟经济差对照的 135 局, 没有一局反号。
 *   逐分钟回填 (gate_dechatter_backfill.py, 看板口径): 看板赛区 756 局 / 298 系列 系列赛 t +2.51
 *   (逐局 +2.36), 全部赛区 1663 局 +1.74; 所有子集、所有分钟段最差的是 25+ 分钟 −0.91。同向。
 *   CPU (本机 Node 24, 单线程, 平均每局 1771 个逐秒点, 不含取帧): 现在整条走势 252 ms (每点 142 µs,
 *   大头是组响应和文案); 概率算 1 次 40 µs/点, 9 次 341 µs/点 —— 每局多 533 ms, 整条走势的计算约 ×3.1。
 *   直播中每秒新增一个点, 多约 0.3 ms。
 *
 * **已上线 (2026-10-17):** api._ingame_core(dechatter=True) / web ingameResponse({dechatter: true}), 只在看板上
 * (头条、曲线、逐秒走势、直播回放), 9 个挪动量和求和顺序同这里 (ingame_service.DECHATTER_SHIFTS)。上线之后
 * fineTimeline 默认就是候选那一路, 所以这里的基线显式传 dechatter: false —— 重跑仍是"上线之前的看板 vs 候选"。
 */
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const WEB = "file:///" + path.join(ROOT, "frontend", "src", "web").replace(/\\/g, "/");
const { Feed, FEED, LEAGUE_IDS, MAJORS } = await import(`${WEB}/feed.ts`);
const { fineTimeline, boardPlan, pregameLeague, knownFor } = await import(`${WEB}/board.ts`);
const { loadIngame, buildRow, predictRow, blendPrior, ingameResponse } = await import(`${WEB}/ingame.ts`);
const { loadStage1, preContext } = await import(`${WEB}/stage1.ts`);
const { loadStage2 } = await import(`${WEB}/stage2.ts`);
const { mapTeam } = await import(`${WEB}/teams.ts`);
const { pyRound } = await import(`${WEB}/pyfmt.ts`);

const argv = process.argv;
const opt = (k, d) => {
  const i = argv.indexOf(`--${k}`);
  return i > 0 ? argv[i + 1] : d;
};
const N = Number(opt("games", 160));
const SEED = Number(opt("seed", 1));
const CACHE = opt("cache", null);
const OUT = opt("out", null);
const SINCE = "2026-01-15"; // Stage 4 测试段的起点
const K = 200; // ±200 金
const NPTS = 9;
const SHIFTS = Array.from({ length: NPTS }, (_, i) => -K + (2 * K * i) / (NPTS - 1)); // −200 … +200, 全是整数
const T_ACCEPT = 2.5;
const SEGS = [["3-15", 3, 15], ["15-25", 15, 25], ["25+", 25, 999], ["全部", 3, 999]];
const EVENT_COLS = ["blue_kills", "red_kills", "blue_towers", "red_towers", "blue_dragons", "red_dragons",
  "blue_barons", "red_barons", "blue_inhibitors", "red_inhibitors"];
const BOARD_LEAGUES = new Set(LEAGUE_IDS.map(([k]) => k)); // 四大 + 国际赛, 大写
/** 跳变阈值的比较: 概率已 round 到 4 位, 差值是 1e-4 的整数倍加浮点残差 */
const ge = (d, th) => d >= th - 1e-9;

// ── 上游缓存 (可选) ───────────────────────────────────────────
function cachedFetch(dir) {
  mkdirSync(dir, { recursive: true });
  return async (url, init) => {
    const u = String(url);
    if (!u.startsWith(FEED)) return fetch(url, init);
    const f = path.join(dir, createHash("sha1").update(u).digest("hex") + ".json");
    if (existsSync(f)) {
      const { status, body } = JSON.parse(readFileSync(f, "utf8"));
      return new Response(status === 204 ? null : body, { status });
    }
    const r = await fetch(url, init);
    if (r.status !== 200 && r.status !== 204) return r;
    const body = r.status === 200 ? await r.text() : null;
    writeFileSync(f, JSON.stringify({ status: r.status, body }));
    return new Response(body, { status: r.status });
  };
}

const J = (n) => JSON.parse(readFileSync(path.join(ROOT, "frontend", "public", "web", n), "utf8"));
const teams = J("teams.json");
/**
 * ingame_train.require_unseen 的同一道检查: G2 起 public/web/ingame_live.json 导出的是 Stage 4 的服务模型 (带
 * serving_fit), 截至 serving_fit.iso.through 的每一局它拟合时都见过 —— 拿它在那之前的局上打分是样本内, 不报错,
 * 只偏乐观。复现上线前的结论: --ingame 指向 G2 之前的导出, 例如
 *   git show 203c887:frontend/public/web/ingame_live.json > old_ingame_live.json
 */
function requireUnseen(file, firstDay, who) {
  const sf = file.serving_fit;
  if (sf && firstDay <= sf.iso.through) {
    console.error(
      `${who}: 这份局内模型是 G2 的服务模型, 拟合时见过截至 ${sf.iso.through} 的每一局 (树 ${sf.trees.from}~${sf.trees.through}, ` +
        `保序表 ${sf.iso.from}~${sf.iso.through}); 考卷从 ${firstDay} 开始, 是样本内打分。用 --ingame 指向 G2 之前的导出`,
    );
    process.exit(2);
  }
}
const ingameFile = opt("ingame", null) ? JSON.parse(readFileSync(opt("ingame", null), "utf8")) : J("ingame_live.json");
requireUnseen(ingameFile, SINCE, "gate_dechatter");
const models = { ingame: loadIngame(ingameFile), s1: loadStage1(J("stage1_pre.json")), s2: loadStage2(J("stage2_post.json")), ex: J("explain.json") };
const today = new Date().toISOString().slice(0, 10);

// ── 考卷 (同 gate_smoothing 的取法) ──────────────────────────
const games = new Map();
for (const line of readFileSync(path.join(ROOT, "backfill", "snapshots.jsonl"), "utf8").split("\n")) {
  if (!line.trim()) continue;
  const r = JSON.parse(line);
  if (!BOARD_LEAGUES.has(String(r.league).toUpperCase())) continue;
  const g = games.get(r.game_id) ?? { id: r.game_id, league: r.league, date: r.date, blue: r.blue, red: r.red, y: r.y, last: 0 };
  g.last = Math.max(g.last, Date.parse(r.frame_t));
  games.set(r.game_id, g);
}
let pool = [...games.values()].filter((g) => g.date >= SINCE);
const nPool = pool.length;
let s = SEED;
const rnd = () => ((s = (s * 1103515245 + 12345) % 2147483648) / 2147483648);
pool = pool.map((g) => [rnd(), g]).sort((a, b) => a[0] - b[0]).map(([, g]) => g).slice(0, N);
console.log(`考卷: 四大 + 国际赛回填 ${games.size} 局, 日期 ≥ ${SINCE} 的 ${nPool} 局, 抽 ${pool.length} 局 (seed ${SEED})`);

const feed = new Feed({ teams, runeLocale: "zh_CN", ...(CACHE ? { fetchImpl: cachedFetch(CACHE) } : {}) });
// fineTimeline 交出去的点不带塔/龙/大龙/水晶, 判"无事件"要用 —— 截住同一批行 (点和行一一对应)
const rowsOf = new Map();
const origFine = feed.fineRows.bind(feed);
feed.fineRows = (gid, upto, onBatch, batch) =>
  origFine(gid, upto, (rows) => { rowsOf.set(gid, rows); onBatch?.(rows); }, batch);

/** board.ts 的 lineups (未导出): 元数据凑齐 5 + 5 个英雄才给 */
async function lineups(gameId) {
  const meta = await feed.gameMetadata(gameId);
  const bl = [], rd = [];
  for (const v of meta.values()) {
    if (v.side === "blue" && v.champion) bl.push(v.champion);
    if (v.side === "red" && v.champion) rd.push(v.champion);
  }
  return bl.length === 5 && rd.length === 5 ? [bl, rd] : null;
}

/** curveCtx + curveProb 的候选版: 9 个挪过的经济差各算一次校准概率, 平均后再渐变。shifts=[0] 即基线 */
function makeScorer(plan, preRow, preP, ln) {
  const m = models.ingame;
  const anchor = plan.preLeague !== null ? preP : plan.xr; // prior (BP 后) = null → 退回赛前, 同 ingameResponse
  return (row, shifts) => {
    const minute = Math.min(60, Math.max(3, row.minute));
    let sum = 0;
    for (const d of shifts) {
      const b = buildRow(m, preRow, {
        minute,
        golddiff: row.golddiff + d,
        csdiff: row.csdiff ?? 0,
        blueKills: row.blue_kills,
        redKills: row.red_kills,
        goldTotal: row.blue_gold,
        blueChamps: ln ? ln[0] : null,
        redChamps: ln ? ln[1] : null,
        league: plan.league,
      });
      sum += predictRow(m, b.row).cal;
    }
    let p = sum / shifts.length;
    const blend = plan.preLeague !== null || plan.xr !== null;
    if (blend && anchor !== null && anchor !== undefined) [p] = blendPrior(anchor, p, minute);
    return pyRound(p, 4);
  };
}

const per = [];
const kept = []; // 留给 CPU 计时
let mismatch = 0;
let skipped = { names: 0, frames: 0, short: 0 };
const t0 = Date.now();
async function one(g) {
  const kn = knownFor(teams, g.league);
  const bm = mapTeam(teams, g.blue, kn);
  const rm = mapTeam(teams, g.red, kn);
  const plan = boardPlan([{ api: g.blue, model: bm }, { api: g.red, model: rm }], g.league, pregameLeague(teams, bm, rm, g.league), null);
  if (!plan) { skipped.names++; return; }
  let pts = [];
  try {
    await fineTimeline(feed, teams, models, today,
      { gameId: g.id, blue: plan.blue, red: plan.red, league: plan.league, preLeague: plan.preLeague, upto: g.last + 60_000, prior: null, xr: plan.xr, dechatter: false },
      (p) => { pts = p; });
  } catch (e) {
    skipped.frames++;
    return;
  }
  const rows = rowsOf.get(g.id) ?? [];
  rowsOf.delete(g.id);
  if (rows.length !== pts.length) throw new Error(`${g.id}: 行 ${rows.length} 和点 ${pts.length} 对不上`);
  // curveCtx 的赛前上下文和阵容: 同一份取法 (preContext 抛错 = 没有赛前特征, ingameResponse 里也是这样)
  let preRow = null, preP = null;
  if (plan.preLeague !== null) {
    try {
      const c = preContext(models.s1, teams, plan.blue, plan.red, plan.preLeague, today);
      preRow = c.preRow;
      preP = c.preP;
    } catch { /* 同 ingameResponse: 赛前特征不可用 */ }
  }
  let ln = null;
  try { ln = await lineups(g.id); } catch { /* 同 curveCtx */ }
  const score = makeScorer(plan, preRow, preP, ln);

  const idx = [];
  pts.forEach((p, i) => { if (p.minute >= 3 && p.probability_blue != null) idx.push(i); });
  if (idx.length < 300) { skipped.short++; return; }
  const base = [], cand = [], ev = [], mins = [];
  let prev = null;
  for (const i of idx) {
    const r = rows[i];
    const b0 = score(r, [0]);
    if (b0 !== pts[i].probability_blue) mismatch++;
    base.push(pts[i].probability_blue);
    cand.push(score(r, SHIFTS));
    mins.push(pts[i].minute);
    ev.push(prev === null ? true : EVENT_COLS.some((c) => (r[c] ?? 0) !== (prev[c] ?? 0)));
    prev = r;
  }
  kept.push({ plan, preRow, preP, ln, rows: idx.map((i) => rows[i]) });

  const rec = { id: g.id, league: g.league, date: g.date, blue: g.blue, red: g.red, y: g.y,
    major: MAJORS.has(String(g.league).toUpperCase()),
    kind: plan.preLeague !== null ? (MAJORS.has(String(g.league).toUpperCase()) ? "domestic" : "intl_home") : "no_pre",
    n: idx.length, seg: {}, jumps: {} };
  for (const [lab, lo, hi] of SEGS) {
    let bb = 0, bc = 0, n = 0;
    mins.forEach((m, k) => { if (m >= lo && m < hi) { bb += (base[k] - g.y) ** 2; bc += (cand[k] - g.y) ** 2; n++; } });
    if (n) rec.seg[lab] = { base: bb / n, cand: bc / n, n };
  }
  for (const [nm, arr] of [["base", base], ["cand", cand]]) {
    let j4 = 0, j2 = 0, j4all = 0, vol = 0;
    for (let k = 1; k < arr.length; k++) {
      const d = Math.abs(arr[k] - arr[k - 1]);
      vol += d;
      if (ge(d, 0.04)) j4all++;
      if (!ev[k]) { if (ge(d, 0.04)) j4++; if (ge(d, 0.02)) j2++; }
    }
    rec.jumps[nm] = { j4, j2, j4all, vol: vol / (arr.length - 1) };
  }
  let mx = 0, ma = 0;
  base.forEach((b, k) => { const d = Math.abs(cand[k] - b); mx = Math.max(mx, d); ma += d; });
  rec.diff = { max: mx, mean: ma / base.length };
  per.push(rec);
}

const queue = [...pool];
let done = 0;
await Promise.all(Array.from({ length: 3 }, async () => {
  while (queue.length) {
    await one(queue.shift());
    if (++done % 20 === 0) console.log(`  ${done}/${pool.length}  (${((Date.now() - t0) / 1000).toFixed(0)}s)`);
  }
}));
if (mismatch) {
  console.error(`\n自检失败: 挪动量全取 0 时有 ${mismatch} 个点和看板的数不等 —— 候选那一路搭错了, 对比不作数`);
  process.exit(2);
}
per.sort((a, b) => (a.date < b.date ? -1 : a.date > b.date ? 1 : 0));

// ── 统计 ─────────────────────────────────────────────────────
/** harness.paired_t: 返回 [Δ, t]; 传进来的是"好坏"值 (越大越好) */
function pairedT(baseline, candidate) {
  const d = candidate.map((x, i) => x - baseline[i]);
  const n = d.length;
  const m = d.reduce((a, x) => a + x, 0) / n;
  const sd = Math.sqrt(d.reduce((a, x) => a + (x - m) ** 2, 0) / (n - 1));
  const t = sd > 1e-12 ? m / (sd / Math.sqrt(n)) : Math.abs(m) <= 1e-12 ? 0 : Infinity * Math.sign(m);
  return [m, t];
}
/** 系列赛: 同赛区 + 同一对队伍, 按开赛时间排, 相邻两局间隔 ≤ 12 小时连成一串 */
function seriesOf(recs) {
  const byPair = new Map();
  for (const r of recs) {
    const k = `${String(r.league).toUpperCase()}|${[r.blue, r.red].sort().join("|")}`;
    if (!byPair.has(k)) byPair.set(k, []);
    byPair.get(k).push(r);
  }
  const out = new Map();
  for (const [k, rs] of byPair) {
    rs.sort((a, b) => Date.parse(a.date.replace(" ", "T") + "Z") - Date.parse(b.date.replace(" ", "T") + "Z"));
    let sid = 0, last = -Infinity;
    for (const r of rs) {
      const t = Date.parse(r.date.replace(" ", "T") + "Z");
      if (t - last > 12 * 3600_000) sid++;
      last = t;
      out.set(r.id, `${k}#${sid}`);
    }
  }
  return out;
}
const sid = seriesOf(per);
function compare(recs, seg) {
  const rs = recs.filter((r) => r.seg[seg]);
  const gb = rs.map((r) => -r.seg[seg].base), gc = rs.map((r) => -r.seg[seg].cand);
  const [dg, tg] = pairedT(gb, gc);
  const bySer = new Map();
  for (const r of rs) {
    const k = sid.get(r.id);
    const a = bySer.get(k) ?? { b: 0, c: 0, n: 0 };
    a.b += r.seg[seg].base; a.c += r.seg[seg].cand; a.n++;
    bySer.set(k, a);
  }
  const ss = [...bySer.values()];
  const [ds, ts] = pairedT(ss.map((a) => -a.b / a.n), ss.map((a) => -a.c / a.n));
  const mb = rs.reduce((a, r) => a + r.seg[seg].base, 0) / rs.length;
  const mc = rs.reduce((a, r) => a + r.seg[seg].cand, 0) / rs.length;
  return { nG: rs.length, nS: ss.length, mb, mc, dg, tg, ds, ts };
}
const sgn = (x) => (x >= 0 ? "+" : "") + x.toFixed(2);
console.log(`\n用上 ${per.length} 局 (跳过: 队伍不成对 ${skipped.names}, 帧取不到 ${skipped.frames}, 不到 300 点 ${skipped.short}); `
  + `${new Set(per.map((r) => sid.get(r.id))).size} 个系列赛; 自检 (挪动 0 = 看板) 全部逐点相等`);
const kinds = {};
for (const r of per) kinds[r.kind] = (kinds[r.kind] ?? 0) + 1;
console.log(`  口径: ${Object.entries(kinds).map(([k, v]) => `${k} ${v}`).join(", ")};  逐秒点 ${per.reduce((a, r) => a + r.n, 0)}`);
console.log(`\nBrier 越小越好; t > 0 = 候选 (经济差 ±${K}, ${NPTS} 点平均) 更好。采纳 iff 系列赛 t > −${T_ACCEPT}`);
const subsets = [["全部", per], ["四大", per.filter((r) => r.major)], ["国际赛", per.filter((r) => !r.major)]];
for (const [lab, recs] of subsets) {
  if (recs.length < 3) continue;
  console.log(`\n  [${lab}]`);
  for (const [seg] of SEGS) {
    const c = compare(recs, seg);
    if (c.nG < 3) continue;
    console.log(`    分钟 ${seg.padEnd(5)}  ${String(c.nG).padStart(3)} 局 / ${String(c.nS).padStart(3)} 系列  基线 ${c.mb.toFixed(5)}  候选 ${c.mc.toFixed(5)}`
      + `  Δ(基线−候选) 逐局 ${c.dg >= 0 ? "+" : ""}${c.dg.toFixed(5)} t ${sgn(c.tg)}   系列赛 ${c.ds >= 0 ? "+" : ""}${c.ds.toFixed(5)} t ${sgn(c.ts)}`);
  }
}
const prim = compare(per, "全部");
console.log(`\n主判据 (全部局, 全部分钟, 系列赛 t): ${sgn(prim.ts)}  → 不显著变差 (t > −${T_ACCEPT}): ${prim.ts > -T_ACCEPT ? "是 → 采纳" : "否 → 不采纳"};`
  + `  显著变好 (t > +${T_ACCEPT}): ${prim.ts > T_ACCEPT ? "是" : "否"}  (逐局 t ${sgn(prim.tg)})`);

const avgJ = (k, f) => per.reduce((a, r) => a + r.jumps[k][f], 0) / per.length;
console.log(`\n跳变 (相邻两个逐秒点; 无事件 = 人头/塔/龙/大龙/水晶都没变):`);
console.log(`  每局无事件 >=4pp:  基线 ${avgJ("base", "j4").toFixed(2)}   候选 ${avgJ("cand", "j4").toFixed(2)}`);
console.log(`  每局无事件 >=2pp:  基线 ${avgJ("base", "j2").toFixed(2)}   候选 ${avgJ("cand", "j2").toFixed(2)}`);
console.log(`  每局全部 >=4pp:    基线 ${avgJ("base", "j4all").toFixed(2)}   候选 ${avgJ("cand", "j4all").toFixed(2)}`);
console.log(`  每步平均波动 (pp): 基线 ${(avgJ("base", "vol") * 100).toFixed(3)}   候选 ${(avgJ("cand", "vol") * 100).toFixed(3)}`);
const jd4 = per.map((r) => r.jumps.base.j4 - r.jumps.cand.j4);
const jd2 = per.map((r) => r.jumps.base.j2 - r.jumps.cand.j2);
const [, tj4] = pairedT(per.map(() => 0), jd4);
const [, tj2] = pairedT(per.map(() => 0), jd2);
console.log(`  (逐局配对: 无事件 >=4pp 少 ${(jd4.reduce((a, x) => a + x, 0) / per.length).toFixed(2)} 次 t ${sgn(tj4)};  >=2pp 少 ${(jd2.reduce((a, x) => a + x, 0) / per.length).toFixed(2)} 次 t ${sgn(tj2)};`
  + `  无事件 >=4pp 变多的局 ${jd4.filter((x) => x < 0).length} / ${per.length})`);
console.log(`  候选 − 基线 |Δp|: 每局平均 ${(per.reduce((a, r) => a + r.diff.mean, 0) / per.length * 100).toFixed(3)}pp, 最大 ${(Math.max(...per.map((r) => r.diff.max)) * 100).toFixed(2)}pp`);

// ── CPU 成本: 帧都在内存里之后单独计时 (不含网络) ─────────────────
{
  /** 一局的三种算法各跑一遍, 返回毫秒 */
  function timeGame(k) {
    const score = makeScorer(k.plan, k.preRow, k.preP, k.ln);
    const preCtx = k.preRow ? { preRow: k.preRow, preP: k.preP, warnings: [] } : null;
    const out = { resp: 0, one: 0, nine: 0 };
    let a = performance.now();
    for (const r of k.rows) {
      const base = {
        blue: k.plan.blue, red: k.plan.red, league: k.plan.league,
        state: { minute: Math.min(60, Math.max(3, r.minute)), golddiff: r.golddiff, csdiff: r.csdiff ?? 0, blueKills: r.blue_kills, redKills: r.red_kills, goldTotal: r.blue_gold },
        blueChamps: k.ln ? k.ln[0] : null, redChamps: k.ln ? k.ln[1] : null,
      };
      // 和 curveProb 同一组参数 (这里 xr 恒为 null)
      ingameResponse(models.ingame, models.s1, teams, models.ex,
        k.plan.preLeague !== null ? { ...base, preCtx, blend: true, blendWith: null, preLeague: k.plan.preLeague } : { ...base, includePregame: false }, today);
    }
    out.resp = performance.now() - a;
    a = performance.now();
    for (const r of k.rows) score(r, [0]);
    out.one = performance.now() - a;
    a = performance.now();
    for (const r of k.rows) score(r, SHIFTS);
    out.nine = performance.now() - a;
    return out;
  }
  // 先空跑几局让 JIT 热起来 (不计), 否则排在第一个计时的那一项吃亏
  for (const k of kept.slice(0, 3)) timeGame(k);
  let nPts = 0;
  const tm = { resp: 0, one: 0, nine: 0 };
  for (const k of kept) {
    const o = timeGame(k);
    nPts += k.rows.length;
    for (const f of ["resp", "one", "nine"]) tm[f] += o[f];
  }
  const g = kept.length;
  console.log(`
CPU (本机 Node ${process.version}, 单线程, ${g} 局 / ${nPts} 个逐秒点, 平均每局 ${(nPts / g).toFixed(0)} 点, 不含取帧):`);
  console.log(`  看板现在每点 (ingameResponse 整个响应): ${(tm.resp / nPts * 1000).toFixed(1)} µs, 每局 ${(tm.resp / g).toFixed(0)} ms`);
  console.log(`  只算概率 1 次 (buildRow + 500 棵树 + 保序): ${(tm.one / nPts * 1000).toFixed(1)} µs, 每局 ${(tm.one / g).toFixed(0)} ms`);
  console.log(`  候选 9 次: ${(tm.nine / nPts * 1000).toFixed(1)} µs, 每局 ${(tm.nine / g).toFixed(0)} ms`
    + `  → 每局多 ${((tm.nine - tm.one) / g).toFixed(0)} ms (整条逐秒走势的计算 ×${((tm.resp + tm.nine - tm.one) / tm.resp).toFixed(2)})`);
}
if (OUT) {
  writeFileSync(OUT, JSON.stringify(per.map((r) => ({ ...r, series: sid.get(r.id) })), null, 1));
  console.log(`\n逐局结果写到 ${OUT}`);
}
