import { useEffect, useMemo, useRef, useState } from "react";
import type { Reason } from "../api/types";
import { T } from "../i18n";

/**
 * 胜率解析 —— 从"胜率解析"那四个字**原地**冒出来的气泡群。
 *
 * 为什么是气泡而不是列表:
 * 权重是模型内部的相对量, **没有可读单位**。列表要么不表达轻重 (等于丢信息),
 * 要么配一根横杠 —— 而横杠会诱导人去比长度差, 那个比例不成立。
 * 气泡的直径同样是"大约", 读法本来就是模糊的, 正好和这份数据的精度相称:
 * 一眼看出谁大谁小, 但没人会去量两个圆差了几个百分点。
 *
 * 鼠标交互全部在 CSS transform 上做 (translate + scale), 不改布局 ——
 * 于是不会触发重排, 十来个气泡跟手也不会掉帧。
 *
 * 移出即收: 这是个"顺手看一眼"的东西, 不该还要专门去点关闭。
 */

/** 悬停时气泡涨到多大 —— 要放得下一整句依据加一行权重。 */
const HOVER_D = 176;
/** 被挤开时两个球之间留的缝 */
const GAP = 9;
/** 面板内壁到最外那个球的余量 */
const PAD = 14;

/** 极坐标环形排布: 最重的在中心, 其余按权重降序绕圈。
 *  不用真正的圆堆积算法 —— 十来个气泡, 环形已经足够, 而且**布局稳定**:
 *  同一批数据每次渲染位置一样, 不会因为算法随机化而跳来跳去。
 *
 *  环半径**必须跟着实际直径算**。第一版写死 30 + ring*27, 而中心那个球
 *  直径就有 80 —— 圆心才隔 53px, 两个大球直接叠在一起。 */
function layout(dias: number[]) {
  const n = dias.length;
  const pos: { x: number; y: number }[] = [];
  if (n === 0) return pos;
  pos.push({ x: 0, y: 0 });
  let i = 1;
  let ring = 1;
  let inner = dias[0]! / 2; // 已占用的半径
  while (i < n) {
    const cap = Math.min(n - i, 4 + ring * 2);
    const slice = dias.slice(i, i + cap);
    const maxD = Math.max(...slice);
    // 环半径 = 内圈边缘 + 本环半径 + 一点间隙
    const r = inner + maxD / 2 + 7;
    for (let k = 0; k < cap; k++) {
      // 每一环错开半格, 免得和内环连成一条直线
      const a = (k / cap) * Math.PI * 2 + (ring % 2 ? Math.PI / cap : 0);
      // 面板是**正圆**, 所以横纵同比 —— 压扁只会让上下空出一圈月牙
      pos.push({ x: Math.cos(a) * r, y: Math.sin(a) * r });
      i++;
    }
    inner = r + maxD / 2;
    ring++;
  }
  return pos;
}

export function WhyBubbles({
  reasons,
  onClose,
}: {
  reasons: Reason[];
  onClose: () => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<number | null>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);

  const { items, restD, hoverD } = useMemo(() => {
    const sorted = [...reasons].sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight));
    const max = Math.max(...sorted.map((r) => Math.abs(r.weight)), 1e-9);
    const rels = sorted.map((r) => Math.abs(r.weight) / max);
    // 开方压缩直径差: 直接按比例的话第一个会大到第六个的好几倍, 那又变回
    // "用尺寸比诱导人"的老问题。开方之后仍然一眼分得出大小, 但不夸张。
    const dias = rels.map((rel) => 30 + 34 * Math.sqrt(rel));
    // 位置要等直径都算完才能定 —— 环半径是按实际直径推的
    const pos = layout(dias);
    const list = sorted.map((r, i) => ({
      ...r,
      i,
      rel: rels[i]!,
      d: dias[i]!,
      ...pos[i]!,
    }));

    // 面板直径**按内容算**, 而且静止态和悬停态各算一份:
    //   静止 —— 刚好裹住环形排布。依据少时圆就小, 不会杵一个空荡荡的大盘。
    //   悬停 —— 中间那个胀到 HOVER_D, 其余被挤到它外面, 圆得跟着让开。
    // 取两者的较大值当成固定尺寸也行, 但那样"四条依据"和"十条依据"一样大,
    // 等于没有自适应。让它随悬停胀缩, 反而正好把"挤开"这件事表达出来。
    const restR = Math.max(...list.map((b) => Math.hypot(b.x, b.y) + b.d / 2), 1);
    // 被挤开后每个球的外缘 = 大球半径 + 缝 + 自己直径
    const pushR = Math.max(
      HOVER_D / 2,
      ...list.map((b) => HOVER_D / 2 + GAP + b.d),
    );
    return {
      items: list,
      restD: Math.ceil(2 * (restR + PAD)),
      hoverD: Math.ceil(2 * (pushR + PAD)),
    };
  }, [reasons]);

  const panel = hover == null ? restD : hoverD;

  /** 某个球在"有人被悬停"时该待在哪。
   *  被悬停的那个回到圆心; 其余**沿自己的方向被推到大球外面** ——
   *  不是淡出让位, 是真的挤开, 所以中间腾出的空间看得见来路。 */
  const place = (b: (typeof items)[number]) => {
    if (hover == null) return { x: b.x, y: b.y, d: b.d };
    if (hover === b.i) return { x: 0, y: 0, d: HOVER_D };
    const need = HOVER_D / 2 + GAP + b.d / 2;
    let dist = Math.hypot(b.x, b.y);
    let ux: number;
    let uy: number;
    if (dist < 0.5) {
      // 正中那个球本身没有方向 —— 给它一个: 躲向被悬停者的反方向
      const h = items.find((x) => x.i === hover)!;
      const hd = Math.hypot(h.x, h.y) || 1;
      ux = -h.x / hd;
      uy = -h.y / hd;
      dist = 0;
    } else {
      ux = b.x / dist;
      uy = b.y / dist;
    }
    const out = Math.max(dist, need);
    return { x: ux * out, y: uy * out, d: b.d };
  };

  return (
    <div
      className="wb"
      ref={ref}
      role="dialog"
      aria-label={T("胜率解析")}
      // 直径按内容算, 所以写在内联样式里而不是 CSS
      style={{ width: panel, height: panel }}
      onMouseLeave={onClose}
    >
      <div className="wb-stage">
        {items.map((r) => {
          const on = hover === r.i;
          // 悬停的那个**涨大并浮到正中**, 其余被挤到四周 (见 place)。
          // 解释写在球里, 视线不用离开气泡。
          const { x: px, y: py, d } = place(r);
          return (
            // 外层只负责**定位** (React 的内联 transform), 内层负责**呼吸**
            // (CSS 动画)。分开是因为两者都要用 transform, 写在同一个元素上
            // 后者会盖掉前者 —— 气泡会全叠回正中。
            <button
              key={r.i}
              className={`wb-bub${on ? " on" : ""}`}
              style={{
                width: d,
                height: d,
                marginLeft: -d / 2,
                marginTop: -d / 2,
                transform: `translate(${px}px, ${py}px)`,
                // **不再靠淡出让位** —— 它们是被真的挤开的, 位置本身就说明了
                // 让路这件事。再压暗一次反而让人以为那几条不重要了。
                opacity: 0.42 + 0.58 * r.rel,
              }}
              onMouseEnter={() => setHover(r.i)}
              // **移出气泡就复位**, 不必等移出整个面板。
              // 漏了这一条的话, 从大球滑到面板空白处时它会一直撑在那里 ——
              // 而"移开就缩回去"正是这个交互该有的手感。
              onMouseLeave={() => setHover(null)}
              onFocus={() => setHover(r.i)}
              onBlur={() => setHover(null)}
              aria-label={r.text}
            >
              <span
                className="wb-ball"
                style={{
                  // 逐个错开, 免得十个球同频呼吸像在闪
                  animationDelay: `${(r.i % 5) * 0.42}s`,
                  animationDuration: `${3 + (r.i % 3) * 0.6}s`,
                }}
              >
                {on ? (
                  <>
                    <em>{r.text}</em>
                    <i>{T("相对权重 {n}%", { n: Math.round(r.rel * 100) })}</i>
                  </>
                ) : (
                  Math.round(r.rel * 100)
                )}
              </span>
            </button>
          );
        })}
      </div>

    </div>
  );
}
