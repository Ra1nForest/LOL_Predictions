/**
 * 黄金测试: 浏览器版模型 (src/web/) 必须和 Python 线上逐位一致。三组:
 *
 *   1. 局内特征/概率   IngameModel.make_row + predict
 *                      输入是 collected/ 的真实直播帧, 外加专打边界的合成局面
 *   2. 局内整条响应     api._ingame_core —— 看板头条和走势图每个点用的就是它
 *   3. 赛前整条响应     POST /predict —— 赛前页面
 *   4. BP 后            api._board_draft + api._predict_core —— 看板那条"BP 后"参考线。
 *                      名字对齐 (选手去战队前缀、英雄 id 归一化) 和特征向量逐位核对
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
import { draftFromMeta } from "../src/web/board.ts";
import { oePlayerName } from "../src/web/names.ts";

const here = dirname(fileURLToPath(import.meta.url));
const read = (p) => JSON.parse(readFileSync(join(here, p), "utf8"));
const TOL = 1e-6;

let model, s1, s2, teams, ex, gi, gs, g2;
try {
  model = loadIngame(read("../public/web/ingame_live.json"));
  s1 = loadStage1(read("../public/web/stage1_pre.json"));
  s2 = loadStage2(read("../public/web/stage2_post.json"));
  teams = read("../public/web/teams.json");
  ex = read("../public/web/explain.json");
  gi = read("golden/ingame_cases.json");
  gs = read("golden/stage1_cases.json");
  g2 = read("golden/stage2_cases.json");
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
  if (i.blue && i.red) {
    try {
      pre = preContext(s1, teams, i.blue, i.red, i.league, gi.today).preRow;
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

const total =
  f1.features.length + f1.warnings.length + f1.slice.length + f1.prob.length + f2.length + f3.length +
  f4.names.length + f4.draft.length + f4.x.length + f4.p.length;
console.log(total ? "\n✗ 未通过" : "\n✓ 通过 —— 浏览器版和 Python 线上是同一套模型");
process.exit(total ? 1 : 0);
