/**
 * 差分测试: 浏览器版数据层 + 看板 (src/web/feed.ts, board.ts) 对 Python 逐字段比对。
 *
 * **标准答案来自一个全新的 Python 进程** (tools/board_oracle.py), 不是正在跑的服务。
 * 服务身上攒着直播时的缓存 (开局零点、暂停观测), 比赛结束后不重算 —— 拿它比, 测的是
 * 服务的历史状态, 不是代码。实测它就在 LEC GX vs NAVI 第 3 局上差了一分钟, 而全新的
 * Python 和浏览器版逐项一致。
 *
 * 为什么只挑**已经打完**的比赛: 帧定型了, 两边拉到的是同一批上游数据, 输出理应
 * 完全相同 —— 不同就是移植错了。进行中的比赛两边取数差几秒, 本来就会不一样。
 *
 * 不参与比对的字段 (都有理由, 不是为了让测试变绿):
 *   live.stale_seconds                     "此刻减帧时刻", 两边取数的时刻不同
 *
 *   node scripts/diff-board.mjs                 最近 4 天, 每赛区最多 3 场
 *   node scripts/diff-board.mjs --days 7 --per 4
 */
import { execFile } from "node:child_process";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { buildBoard, liveList, upcomingList } from "../src/web/board.ts";
import { Feed, LEAGUE_IDS } from "../src/web/feed.ts";
import { loadIngame } from "../src/web/ingame.ts";
import { loadStage1 } from "../src/web/stage1.ts";
import { loadStage2 } from "../src/web/stage2.ts";
import { knownTeams, localToday } from "../src/web/teams.ts";

const here = dirname(fileURLToPath(import.meta.url));
const read = (p) => JSON.parse(readFileSync(join(here, p), "utf8"));
const arg = (k, d) => {
  const i = process.argv.indexOf(k);
  return i > 0 ? Number(process.argv[i + 1]) : d;
};
const DAYS = arg("--days", 4);
const PER = arg("--per", 3);
const ORACLE = join(here, "..", "..", "tools", "board_oracle.py");

const teams = read("../public/web/teams.json");
const models = {
  ingame: loadIngame(read("../public/web/ingame_live.json")),
  s1: loadStage1(read("../public/web/stage1_pre.json")),
  s2: loadStage2(read("../public/web/stage2_post.json")),
  ex: read("../public/web/explain.json"),
};
const feed = new Feed({ teams });
const today = localToday();

/** 所有不同之处 (最多 limit 条) */
function diffs(js, py, path = "$", out = [], limit = 8) {
  if (out.length >= limit) return out;
  if (typeof js === "number" && typeof py === "number") {
    if (js !== py) out.push({ path, js, py });
    return out;
  }
  if (js === null || py === null || typeof js !== "object" || typeof py !== "object") {
    if (js !== py) out.push({ path, js, py });
    return out;
  }
  if (Array.isArray(js) !== Array.isArray(py)) {
    out.push({ path, js: "(类型不同)", py: "(类型不同)" });
    return out;
  }
  if (Array.isArray(js)) {
    if (js.length !== py.length) out.push({ path: `${path}.length`, js: js.length, py: py.length });
    for (let i = 0; i < Math.min(js.length, py.length); i++) diffs(js[i], py[i], `${path}[${i}]`, out, limit);
    return out;
  }
  const keys = new Set([...Object.keys(js), ...Object.keys(py)]);
  for (const k of [...keys].sort()) {
    if (!(k in js)) out.push({ path: `${path}.${k}`, js: "(无此键)", py: py[k] });
    else if (!(k in py)) out.push({ path: `${path}.${k}`, js: js[k], py: "(无此键)" });
    else diffs(js[k], py[k], `${path}.${k}`, out, limit);
  }
  return out;
}

function normalize(b) {
  const o = JSON.parse(JSON.stringify(b));
  if (o.live) delete o.live.stale_seconds;
  return o;
}

// ── 挑比赛: 最近 DAYS 天、已完赛, 每赛区最多 PER 场 ──
const now = Date.now();
const targets = [];
for (const [lg] of LEAGUE_IDS) {
  const ms = (await feed.schedule(lg, knownTeams(teams, lg)))
    .filter((m) => m.state === "completed" && now - Date.parse(m.start_time) < DAYS * 86_400_000)
    .sort((a, b) => (a.start_time < b.start_time ? 1 : -1))
    .slice(0, PER);
  for (const m of ms) {
    for (const g of await feed.games(m.match_id)) {
      if (g.id && g.state === "completed") {
        targets.push({ m, g, label: `${m.league} ${m.teams.map((t) => t.code).join(" vs ")} 第${g.number}局` });
      }
    }
  }
}
console.log(`最近 ${DAYS} 天、每赛区至多 ${PER} 场已完赛的系列赛, 共 ${targets.length} 局`);
console.log("浏览器版和全新 Python 进程同时开算 (Python 要先加载五年 CSV) …\n");

// ── 两边同时算 ──
const oracleP = promisify(execFile)(
  "python",
  [ORACLE, ...targets.map((x) => `${x.m.match_id}:${x.g.id}`), "--lists"],
  { encoding: "utf8", maxBuffer: 512 * 1024 * 1024, env: { ...process.env, PYTHONIOENCODING: "utf-8" } },
);
const t0 = Date.now();
const jsBoards = [];
for (const x of targets) {
  const t = Date.now();
  try {
    jsBoards.push({ ok: await buildBoard(feed, teams, models, today, x.m.match_id, x.g.id, true, 60), ms: Date.now() - t });
  } catch (e) {
    jsBoards.push({ err: e.message, ms: Date.now() - t });
  }
}
const [jsLive, jsUp] = await Promise.all([liveList(feed, teams), upcomingList(feed, teams, 24, 5)]);
console.log(`浏览器版 ${((Date.now() - t0) / 1000).toFixed(1)} 秒算完, 等 Python …`);
let py;
try {
  py = JSON.parse((await oracleP).stdout);
} catch (e) {
  console.error(`Python 参照程序失败: ${e.message}\n${e.stderr ?? ""}`.slice(0, 3000));
  process.exit(2);
}
console.log();

let bad = 0;
targets.forEach((x, i) => {
  const js = jsBoards[i];
  const ref = py[`${x.m.match_id}:${x.g.id}`];
  if (js.err || ref?.__error__) {
    const same = js.err && ref?.__error__;
    if (!same) bad++;
    console.log(`${same ? "~" : "✗"} ${x.label}  浏览器: ${js.err ?? "正常"}  Python: ${ref?.__error__ ?? "正常"}`);
    return;
  }
  const b = js.ok;
  const d = diffs(normalize(b), normalize(ref));
  const brief = b.live
    ? `第${b.live.minute}分钟 暂停${b.live.paused_seconds}s 曲线${b.timeline.length}点 P(蓝)=${b.prediction?.probability_blue}`
    : "(无帧)";
  if (d.length) {
    bad++;
    console.log(`✗ ${x.label}  ${brief}  (${js.ms}ms)`);
    for (const e of d) console.log(`    ${e.path}\n      浏览器: ${JSON.stringify(e.js)}\n      Python: ${JSON.stringify(e.py)}`);
  } else {
    console.log(`✓ ${x.label}  ${brief}  (${js.ms}ms)`);
  }
});

// 列表: 进行中的比赛两边取数有先后, 仅供参考, 不计入判决
console.log("\n── 列表 (仅供参考) ──");
for (const [name, js, ref] of [
  ["live", jsLive, py.__live__],
  ["upcoming", jsUp, py.__upcoming__],
]) {
  const d = diffs(js, ref, "$", [], 5);
  console.log(`${d.length ? "≠" : "="} ${name}  浏览器 ${js.count} 场 / Python ${ref.count} 场`);
  for (const e of d) console.log(`    ${e.path}  浏览器 ${JSON.stringify(e.js)}  Python ${JSON.stringify(e.py)}`);
}

console.log(`\n${targets.length} 局里 ${bad} 局不一致`);
process.exit(bad ? 1 : 0);
