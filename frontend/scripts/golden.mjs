/**
 * 黄金测试: 浏览器版模型 (src/web/) 必须和 Python 线上逐位一致。三组:
 *
 *   1. 局内特征/概率   IngameModel.make_row + predict
 *                      输入是 collected/ 的真实直播帧, 外加专打边界的合成局面
 *   2. 局内整条响应     api._ingame_core —— 看板头条和走势图每个点用的就是它
 *   3. 赛前整条响应     POST /predict —— 赛前页面
 *   4. BP 后            api._board_draft + api._predict_core —— 看板那条"BP 后"参考线。
 *                      名字对齐 (选手去战队前缀、英雄 id 归一化) 和特征向量逐位核对
 *   5. 国际赛         api.pregame_league (赛前/BP 后按哪个赛区算)、api._known_for (映射队名用
 *                      哪份表)、esports_feed.map_team (空表必须返回 null) 逐条对账; 外加看板的
 *                      api._pregame_reason (为什么没有赛前/BP 后)、api._early_prediction (开局头
 *                      3 分钟那一段 —— test:diff 只取终局帧, 碰不到它)、api._board_warnings
 *
 * 局内的国际赛用例 (src 以 "syn:intl:" 打头) 输入里多两个键, 取法和 api._ingame_core 一字不差:
 *   include_pregame: false   不带赛前特征 (看板在跨赛区 / 队名不认识时这样调)
 *   pre_league               赛前特征按它算, 不给就按 league (国际赛同母赛区时是两队的母赛区)
 *
 * 期望值由 tools/export_web_model.py 调线上真函数生成, 不另写一份。
 *
 *   python tools/export_web_model.py
 *   npm run test:golden
 *
 * 判据:
 *   特征向量  逐位相等          特征构造全是 float64 运算, 两边没有理由差一个比特
 *   概率      |Δ| < 1e-6        树和 sigmoid 在 float32 上, JS 只能模拟到一两个 ulp
 *   响应      逐字段相等        文案、round 过的数字、键的有无, 一处都不许差 ——
 *                              界面和 Stage 3 都会原样引用它们
 */
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { buildRow, ingameResponse, loadIngame, predictRow } from "../src/web/ingame.ts";
import { loadStage1, preContext, predictResponse } from "../src/web/stage1.ts";
import { draftVector, loadStage2, makeDraftRow, predictCore } from "../src/web/stage2.ts";
import { boardWarnings, draftFromMeta, earlyPrediction, knownFor, pregameLeague, pregameReason } from "../src/web/board.ts";
import { MAJORS } from "../src/web/feed.ts";
import { knownTeams, mapTeam } from "../src/web/teams.ts";
import { oePlayerName } from "../src/web/names.ts";

const here = dirname(fileURLToPath(import.meta.url));
const read = (p) => JSON.parse(readFileSync(join(here, p), "utf8"));
const TOL = 1e-6;

let model, s1, s2, teams, ex, gi, gs, g2, g5;
try {
  model = loadIngame(read("../public/web/ingame_live.json"));
  s1 = loadStage1(read("../public/web/stage1_pre.json"));
  s2 = loadStage2(read("../public/web/stage2_post.json"));
  teams = read("../public/web/teams.json");
  ex = read("../public/web/explain.json");
  gi = read("golden/ingame_cases.json");
  gs = read("golden/stage1_cases.json");
  g2 = read("golden/stage2_cases.json");
  g5 = read("golden/intl_cases.json");
} catch (e) {
  console.error(`读不到导出文件: ${e.message}\n先跑: python tools/export_web_model.py`);
  process.exit(2);
}
if (
  JSON.stringify(gi.features) !== JSON.stringify(model.forest.features) ||
  JSON.stringify(g2.features) !== JSON.stringify(s2.forest.features)
) {
  console.error("黄金用例和模型文件的特征列表不一致 —— 两者不是同一次导出, 重新导出");
  process.exit(2);
}

/** 第一处不同 → {path, js, py}; 相同 → null。键的有无也算不同。 */
function diff(js, py, path = "$") {
  if (typeof js === "number" && typeof py === "number") return js === py ? null : { path, js, py };
  if (js === null || py === null || typeof js !== "object" || typeof py !== "object") {
    return js === py ? null : { path, js, py };
  }
  if (Array.isArray(js) !== Array.isArray(py)) return { path, js, py };
  if (Array.isArray(js)) {
    if (js.length !== py.length) return { path: `${path}.length`, js: js.length, py: py.length };
    for (let i = 0; i < js.length; i++) {
      const d = diff(js[i], py[i], `${path}[${i}]`);
      if (d) return d;
    }
    return null;
  }
  const kj = Object.keys(js).sort();
  const kp = Object.keys(py).sort();
  if (kj.join("|") !== kp.join("|")) {
    return { path: `${path} 的键`, js: kj.filter((k) => !kp.includes(k)), py: kp.filter((k) => !kj.includes(k)) };
  }
  for (const k of kj) {
    const d = diff(js[k], py[k], `${path}.${k}`);
    if (d) return d;
  }
  return null;
}
const plain = (o) => JSON.parse(JSON.stringify(o));

// ── 1. 局内特征 / 概率 ─────────────────────────────────────
const f1 = { features: [], warnings: [], slice: [], prob: [] };
let maxRaw = 0;
let maxCal = 0;
let exactRaw = 0;
for (const c of gi.cases) {
  const i = c.in;
  let pre = null;
  // include_pregame=false: 不带赛前特征; 否则按 pre_league (没给就按 league) 算 —— 同 _ingame_core
  if (i.blue && i.red && i.include_pregame !== false) {
    try {
      pre = preContext(s1, teams, i.blue, i.red, i.pre_league ?? i.league, gi.today).preRow;
    } catch {
      pre = null; // 和 _pre_context 抛异常时一样: 不带赛前特征
    }
  }
  const b = buildRow(model, pre, {
    minute: i.minute,
    golddiff: i.golddiff,
    csdiff: i.csdiff,
    blueKills: i.blue_kills,
    redKills: i.red_kills,
    goldTotal: i.gold_total,
    blueChamps: i.blue_champs,
    redChamps: i.red_champs,
    league: i.league,
  });
  const p = predictRow(model, b.row);
  const bad = [];
  for (let k = 0; k < p.x.length; k++) {
    if (!Object.is(p.x[k], c.x[k])) bad.push({ f: model.forest.features[k], js: p.x[k], py: c.x[k] });
  }
  if (bad.length) f1.features.push({ src: c.src, bad: bad.slice(0, 4) });
  if (JSON.stringify(b.warnings) !== JSON.stringify(c.warnings)) {
    f1.warnings.push({ src: c.src, js: b.warnings, py: c.warnings });
  }
  if (b.T !== c.T) f1.slice.push({ src: c.src, js: b.T, py: c.T });
  const dr = Math.abs(p.raw - c.raw);
  const dc = Math.abs(p.cal - c.cal);
  if (dr === 0) exactRaw++;
  maxRaw = Math.max(maxRaw, dr);
  maxCal = Math.max(maxCal, dc);
  if (!(dr < TOL) || !(dc < TOL)) f1.prob.push({ src: c.src, raw: [p.raw, c.raw], cal: [p.cal, c.cal] });
}

// ── 2. 局内整条响应 ─────────────────────────────────────────
const f2 = [];
let n2 = 0;
for (const c of gi.cases) {
  if (!c.resp) continue;
  n2++;
  const i = c.in;
  const out = ingameResponse(
    model,
    s1,
    teams,
    ex,
    {
      blue: i.blue,
      red: i.red,
      league: i.league,
      state: {
        minute: i.minute,
        golddiff: i.golddiff,
        csdiff: i.csdiff,
        blueKills: i.blue_kills,
        redKills: i.red_kills,
        goldTotal: i.gold_total,
      },
      blueChamps: i.blue_champs,
      redChamps: i.red_champs,
      includePregame: i.include_pregame ?? true,
      preLeague: i.pre_league ?? null,
    },
    gi.today,
  );
  const d = diff(plain(out), c.resp);
  if (d) f2.push({ src: c.src, ...d });
}

// ── 3. 赛前整条响应 ─────────────────────────────────────────
const f3 = [];
for (const c of gs.cases) {
  const tag = `${c.in.blue_team} vs ${c.in.red_team} (${c.in.league}${c.in.playoffs ? ", 季后赛" : ""})`;
  try {
    const out = predictResponse(s1, teams, ex, c.in, gs.today);
    if (c.error) {
      f3.push({ src: tag, path: "应当报错", js: "没报错", py: c.error });
      continue;
    }
    const d = diff(plain(out), c.resp);
    if (d) f3.push({ src: tag, ...d });
  } catch (e) {
    if (!c.error || e.message !== c.error) f3.push({ src: tag, path: "报错", js: e.message, py: c.error ?? "(不该报错)" });
  }
}

// ── 4. BP 后 ────────────────────────────────────────────────
const f4 = { names: [], draft: [], x: [], p: [] };
let max4 = 0;
let n4p = 0;
for (const c of g2.names) {
  const js = oePlayerName(c.in[0], c.in[1]);
  if (js !== c.out) f4.names.push({ in: c.in, js, py: c.out });
}
for (const c of g2.cases) {
  const d = draftFromMeta(c.in.players, c.in.codes);
  const dd = diff(plain(d), c.draft);
  if (dd) {
    f4.draft.push({ src: c.src, ...dd });
    continue;
  }
  if (!d) continue;
  try {
    const x = draftVector(s2, makeDraftRow(s2, teams, c.in.blue, c.in.red, c.in.league, d, g2.today));
    if (c.error) {
      f4.p.push({ src: c.src, path: "应当报错", js: "没报错", py: c.error });
      continue;
    }
    const bad = [];
    for (let k = 0; k < x.length; k++) {
      if (!Object.is(x[k], c.x[k])) bad.push({ f: s2.forest.features[k], js: x[k], py: c.x[k] });
    }
    if (bad.length) f4.x.push({ src: c.src, bad: bad.slice(0, 4) });
    const p = predictCore(s2, teams, c.in.blue, c.in.red, c.in.league, d, g2.today);
    n4p++;
    const dp = Math.abs(p - c.p2);
    max4 = Math.max(max4, dp);
    if (!(dp < TOL)) f4.p.push({ src: c.src, js: p, py: c.p2 });
  } catch (e) {
    if (!c.error || e.message !== c.error) f4.p.push({ src: c.src, path: "报错", js: e.message, py: c.error ?? "(不该报错)" });
  }
}

// ── 5. 国际赛 ───────────────────────────────────────────────
// 四大赛区的名单两边各写了一份 (feed.ts 的 MAJORS / feature_store.LEAGUES): 拿导出的 known 的键核对
const f5 = { majors: [], pregame: [], known: [], map: [], reason: [], early: [], warns: [] };
const knownKeys = Object.keys(teams.known).sort();
if (knownKeys.join("|") !== [...MAJORS].sort().join("|")) f5.majors.push({ js: [...MAJORS], py: knownKeys });
for (const c of g5.pregame_league) {
  const js = pregameLeague(teams, c.in[0], c.in[1], c.in[2]);
  if (js !== c.out) f5.pregame.push({ in: c.in, js, py: c.out });
}
for (const c of g5.known_for) {
  const d = diff(knownFor(teams, c.in), c.out);
  if (d) f5.known.push({ in: c.in, ...d });
}
// spec: null = 不校验, [] = 空表 (必须返回 null), "*" = 并集, 赛区名 = 该赛区的表 (store.known_teams(赛区))
const knownOf = (spec) =>
  spec === null || Array.isArray(spec) ? spec : spec === "*" ? teams.known_all : knownTeams(teams, spec);
for (const c of g5.map_team) {
  const js = mapTeam(teams, c.in[0], knownOf(c.in[1]));
  if (js !== c.out) f5.map.push({ in: c.in, js, py: c.out });
}
// 看板的几句说明和开局那一段: 整个返回值逐字段比 (键的有无、句子一字不差)
for (const c of g5.pregame_reason) {
  const d = diff(pregameReason(c.in[0], c.in[1], c.in[2]), c.out);
  if (d) f5.reason.push({ in: c.in, ...d });
}
for (const c of g5.early) {
  const d = diff(earlyPrediction(c.in[0], c.in[1], c.in[2], c.in[3], c.in[4]), c.out);
  if (d) f5.early.push({ in: c.in, ...d });
}
for (const c of g5.board_warnings) {
  const d = diff(boardWarnings(c.in[0], c.in[1], c.in[2]), c.out);
  if (d) f5.warns.push({ in: c.in, ...d });
}

// ── 报告 ────────────────────────────────────────────────────
const fmt = (v) => v.toExponential(2);
const n1 = gi.cases.length;
console.log(`1. 局内特征/概率  ${n1} 条 (真实 ${gi.n_real} / 合成 ${gi.n_synthetic})`);
console.log(`     特征不一致 ${f1.features.length}  警告不一致 ${f1.warnings.length}  切片不一致 ${f1.slice.length}  概率超差 ${f1.prob.length}`);
console.log(`     max |Δraw| ${fmt(maxRaw)}  max |Δcal| ${fmt(maxCal)}  raw 逐位相等 ${exactRaw}/${n1}`);
console.log(`2. 局内整条响应   ${n2} 条   不一致 ${f2.length}`);
console.log(`3. 赛前整条响应   ${gs.cases.length} 条   不一致 ${f3.length}`);
console.log(`4. BP 后          ${g2.cases.length} 局阵容 (算出概率 ${n4p})  选手名 ${g2.names.length} 条`);
console.log(`     名字不一致 ${f4.names.length}  BP 不一致 ${f4.draft.length}  特征不一致 ${f4.x.length}  概率超差 ${f4.p.length}  max |Δp| ${fmt(max4)}`);
console.log(
  `5. 国际赛        pregame_league ${g5.pregame_league.length} 条  known_for ${g5.known_for.length} 条  map_team ${g5.map_team.length} 条`,
);
console.log(
  `     四大名单不一致 ${f5.majors.length}  pregame_league 不一致 ${f5.pregame.length}  known_for 不一致 ${f5.known.length}  map_team 不一致 ${f5.map.length}`,
);
console.log(
  `     看板说明 ${g5.pregame_reason.length} 条 不一致 ${f5.reason.length}  开局 ${g5.early.length} 条 不一致 ${f5.early.length}  局内提醒 ${g5.board_warnings.length} 条 不一致 ${f5.warns.length}`,
);

const show = (title, arr) => {
  if (!arr.length) return;
  console.log(`\n── ${title} (共 ${arr.length}, 前 5 条) ──`);
  for (const f of arr.slice(0, 5)) console.log(JSON.stringify(f));
};
show("1 特征", f1.features);
show("1 警告", f1.warnings);
show("1 切片", f1.slice);
show("1 概率", f1.prob);
show("2 局内响应", f2);
show("3 赛前响应", f3);
show("4 选手名", f4.names);
show("4 BP", f4.draft);
show("4 特征", f4.x);
show("4 概率", f4.p);
show("5 四大名单", f5.majors);
show("5 pregame_league", f5.pregame);
show("5 known_for", f5.known);
show("5 map_team", f5.map);
show("5 看板说明", f5.reason);
show("5 开局", f5.early);
show("5 局内提醒", f5.warns);

const total =
  f1.features.length + f1.warnings.length + f1.slice.length + f1.prob.length + f2.length + f3.length +
  f4.names.length + f4.draft.length + f4.x.length + f4.p.length +
  f5.majors.length + f5.pregame.length + f5.known.length + f5.map.length +
  f5.reason.length + f5.early.length + f5.warns.length;
console.log(total ? "\n✗ 未通过" : "\n✓ 通过 —— 浏览器版和 Python 线上是同一套模型");
process.exit(total ? 1 : 0);
