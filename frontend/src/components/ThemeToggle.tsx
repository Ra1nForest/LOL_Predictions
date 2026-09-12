import { useEffect, useState } from "react";

type Theme = "auto" | "light" | "dark";
const KEY = "lolpredict.theme";

/**
 * 主题切换。**默认浅色**, 不是跟随系统 —— 视觉是按白底调的
 * (底色比卡片深一档 + 极淡投影撑层次), 暗色是备选。
 * 选过之后记住, 所以 auto 只在从没选过时才轮到。
 */
export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>(() => {
    const v = localStorage.getItem(KEY);
    return v === "light" || v === "dark" || v === "auto" ? v : "light";
  });

  useEffect(() => {
    const root = document.documentElement;
    if (theme === "auto") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", theme);
    localStorage.setItem(KEY, theme);
  }, [theme]);

  const opts: [Theme, string, string][] = [
    ["light", "○", "亮色"],
    ["dark", "●", "暗色"],
    ["auto", "A", "跟随系统"],
  ];
  return (
    <div className="theme-toggle" role="group" aria-label="主题">
      {opts.map(([v, icon, label]) => (
        <button
          key={v}
          className={theme === v ? "on" : ""}
          onClick={() => setTheme(v)}
          title={label}
          aria-label={label}
          aria-pressed={theme === v}
        >
          {icon}
        </button>
      ))}
    </div>
  );
}
