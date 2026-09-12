import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// 后端所有需要代理的前缀。开发时 vite dev server 把它们转给 FastAPI ——
// 服务只绑 127.0.0.1:8000, 本机通过 SSH 隧道就能打到线上那台。
const API_PREFIXES = [
  "/health",
  "/teams",
  "/esports",
  "/predict",
  "/session",
  "/debate",
  "/api",
];

// 两种构建:
//   默认        → ../static, 由 api.py 挂载, 数据全部请求服务器
//   --mode pages → ../dist-pages, GitHub Pages 用的静态版 (.env.pages 里 VITE_STATIC=1),
//                  数据和模型都在浏览器里算, 不需要任何服务器
export default defineConfig(({ mode }) => {
  const pages = mode === "pages";
  return {
    plugins: [react()],
    // Pages 挂在 /<仓库名>/ 下。相对路径让同一份产物放在任何子路径都能用,
    // 不必把仓库名写死进构建配置
    base: pages ? "./" : "/",
    // public/web/ 是静态版要下载的模型文件 (~1.4 MB), 服务器版用不到, 别复制进 static/
    publicDir: pages ? "public" : false,
    server: pages
      ? { port: 5174 }
      : {
          port: 5173,
          proxy: Object.fromEntries(
            API_PREFIXES.map((p) => [
              p,
              { target: "http://127.0.0.1:8000", changeOrigin: true },
            ]),
          ),
        },
    build: pages
      ? { outDir: "../dist-pages", emptyOutDir: true, sourcemap: false }
      : {
          // 直接产到 FastAPI 挂载的目录 (api.py: app.mount("/", StaticFiles(...)))
          // 服务器上没有 Node, 构建只在本地做, 传的是产物。
          outDir: "../static",
          // 不清空 —— static/legacy.html 是旧版单文件前端, 留着随时对照
          emptyOutDir: false,
          sourcemap: true,
        },
  };
});
