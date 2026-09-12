/**
 * 核对: 组件里 var(--x) 引用的变量, styles.css 里都定义了吗?
 *
 * 为什么需要这个: **CSS 变量拼错不报错**。SVG 里 stroke="var(--typo)" 只是
 * 静默失效, 页面照常渲染, 颜色悄悄没了 —— TypeScript 看不见, 构建也不报。
 * 重写样式表改了 token 命名那次, 就有两个变量 (--accent / --line-2) 留在
 * 组件里没跟上, 曲线和网格线的颜色直接失效。
 *
 * 这正是这个项目最典型的失败形状: 不抛异常, 只是结果错了。
 *
 * 跑: node scripts/check-css-vars.mjs   (npm run check:css)
 */
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

// 必须走 fileURLToPath: Windows 上 URL.pathname 是 "/C:/..." 带前导斜杠,
// 直接拿去 join 会得到 "C:\C:\..."
const SRC = fileURLToPath(new URL("../src", import.meta.url));

function walk(dir) {
  const out = [];
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) out.push(...walk(p));
    else if (/\.(tsx?|css)$/.test(name)) out.push(p);
  }
  return out;
}

const files = walk(SRC);
const css = files.filter((f) => f.endsWith(".css"));

// 定义: 只认 `--x:` 形式的声明 (排除 var(--x) 的引用)
const defined = new Set();
for (const f of css) {
  const text = readFileSync(f, "utf8");
  for (const m of text.matchAll(/(^|[;{\s])(--[a-z0-9-]+)\s*:/gi)) {
    defined.add(m[2]);
  }
}

const problems = [];
for (const f of files) {
  const text = readFileSync(f, "utf8");
  text.split("\n").forEach((line, i) => {
    for (const m of line.matchAll(/var\(\s*(--[a-z0-9-]+)\s*([,)])/gi)) {
      const name = m[1];
      // var(--x, fallback) 有兜底值, 拼错也还有救, 但仍然值得报
      if (!defined.has(name)) {
        problems.push(`${f.replace(SRC, "src")}:${i + 1}  ${name}`);
      }
    }
  });
}

if (problems.length) {
  console.error(`未定义的 CSS 变量 (${problems.length} 处):`);
  for (const p of problems) console.error("  " + p);
  process.exit(1);
}
console.log(`CSS 变量核对通过 — 定义 ${defined.size} 个, 引用全部有着落`);
