/**
 * 构建前清空 static/assets。
 *
 * vite 的 emptyOutDir 被关掉了 (outDir 是 static/, 那里还放着 legacy.html
 * —— 旧版单文件前端, 留着对照)。代价是带 hash 的产物会一次次堆积:
 * 每次构建都换文件名, 旧的没人删, 传到服务器上就是一堆没人引用的死文件。
 */
import { existsSync, readdirSync, rmSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const dir = fileURLToPath(new URL("../../static/assets", import.meta.url));
if (existsSync(dir)) {
  const n = readdirSync(dir).length;
  rmSync(dir, { recursive: true, force: true });
  console.log(`清掉旧产物 ${n} 个 (${dir})`);
}
