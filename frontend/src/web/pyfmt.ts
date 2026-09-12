/**
 * Python 的数字格式化, 逐位复现 —— 浏览器版要和服务器吐出一模一样的文案和 JSON。
 *
 * 为什么不能直接用 toFixed: Python 的 format(x, ".2f") 和 round(x, 2) 都是对**精确的
 * 二进制值**做四舍六入五成双; JS 的 toFixed 遇到恰好的 .5 会往大的方向进。差别只落在
 * 恰好卡在两档正中间的值上 (比如 0.03125 保留 4 位: Python 给 0.0312, toFixed 给
 * 0.0313)。很少见 —— 但"很少见的错数字"正是这个项目最常见的 bug 形态。
 *
 * 判断"是不是恰好的 .5"靠多取 30 位小数: 在本项目的数值范围里 (|x| 在 1e-3 到 1e20 之间),
 * 离两档正中间最近的非中点 double 也差着 1e-23 以上, 远大于 1e-30, 不会误判。
 */

/** format(x, f".{d}f") */
export function pyFixed(x: number, d: number): string {
  if (Number.isNaN(x)) return "nan";
  if (!Number.isFinite(x)) return x > 0 ? "inf" : "-inf";
  const neg = x < 0 || Object.is(x, -0);
  const a = Math.abs(x);
  // toFixed 在 1e21 以上会退成科学计数法。本项目格式化的数都远小于它
  if (a >= 1e21) return (neg ? "-" : "") + String(a);
  const s = a.toFixed(Math.min(100, d + 30));
  const dot = s.indexOf(".");
  const ip = dot < 0 ? s : s.slice(0, dot);
  const fp = dot < 0 ? "" : s.slice(dot + 1);
  let digits = ip + fp.slice(0, d);
  const rest = fp.slice(d);
  const first = rest.length > 0 ? rest.charCodeAt(0) - 48 : 0;
  let up: boolean;
  if (first !== 5) up = first > 5;
  else if (/[1-9]/.test(rest.slice(1))) up = true;
  else up = (digits.charCodeAt(digits.length - 1) - 48) % 2 === 1; // 恰好 .5: 五成双
  if (up) digits = incDecimal(digits);
  const intLen = digits.length - d;
  const out = d > 0 ? `${digits.slice(0, intLen)}.${digits.slice(intLen)}` : digits;
  return neg ? `-${out}` : out;
}

function incDecimal(s: string): string {
  const a = s.split("");
  for (let i = a.length - 1; i >= 0; i--) {
    if (a[i] === "9") {
      a[i] = "0";
    } else {
      a[i] = String.fromCharCode(a[i]!.charCodeAt(0) + 1);
      return a.join("");
    }
  }
  return "1" + a.join("");
}

/** round(x, d) —— 返回的是离那个十进制结果最近的 double, 和 Python 一样 */
export function pyRound(x: number, d: number): number {
  if (!Number.isFinite(x)) return x;
  return Number(pyFixed(x, d));
}

/** format(x, f",.{d}f") —— 千分位逗号 */
export function pyComma(x: number, d: number): string {
  const s = pyFixed(x, d);
  const neg = s.startsWith("-");
  const body = neg ? s.slice(1) : s;
  const dot = body.indexOf(".");
  const ip = dot < 0 ? body : body.slice(0, dot);
  const tail = dot < 0 ? "" : body.slice(dot);
  return (neg ? "-" : "") + ip.replace(/\B(?=(\d{3})+(?!\d))/g, ",") + tail;
}

/** format(x, f".{d}%") —— CPython 先乘 100 (浮点乘法) 再按 'f' 格式化 */
export function pyPct(x: number, d: number): string {
  return `${pyFixed(x * 100, d)}%`;
}

/** f"{name}" 在 name 为 None 时写 "None" —— 错误信息要和服务器一字不差 */
export function pyStr(s: string | null | undefined): string {
  return s ?? "None";
}
