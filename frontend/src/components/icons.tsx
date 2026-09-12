/**
 * 目标资源图标。
 *
 * 刻意用**内联 SVG** 而不是外链图片 (Community Dragon / ddragon 之类):
 *   - 不依赖网络。对方改一次路径, 页面上就是一排碎图标, 而且不报错。
 *   - 跟着 currentColor 走, 明暗主题和蓝红配色都不用另做一套。
 *   - 尺寸只有几百字节, 比一次 HTTP 往返便宜得多。
 *
 * 造型按 LoL 里的实物简化: 防御塔取"底座+塔身+雉堞+顶端法球"的剪影,
 * 小龙取龙头侧面, 大龙取纳什男爵的多刺长吻, 水晶取抑制器的菱形晶体。
 * 目标是 14px 下还能一眼分清, 不是还原细节。
 */

interface IconProps {
  className?: string;
}

const base = {
  viewBox: "0 0 24 24",
  fill: "currentColor",
  "aria-hidden": true as const,
  focusable: "false" as const,
};

/** 防御塔 */
export function TowerIcon({ className }: IconProps) {
  return (
    <svg {...base} className={className}>
      <path d="M12 1.6 13.5 4h-3L12 1.6Z" opacity=".55" />
      <path d="M8.6 5.2h6.8v2.1H8.6z" />
      <path d="M8.6 5.2V3.6h1.5v1.6h1.1V3.6h1.6v1.6h1.1V3.6h1.5v1.6z" />
      <path d="M9.6 7.9h4.8l-.7 9.4H10.3z" />
      <path d="M6.6 17.6h10.8c.5 0 .9.4.9.9v1c0 .5-.4.9-.9.9H6.6a.9.9 0 0 1-.9-.9v-1c0-.5.4-.9.9-.9Z" />
      <circle cx="12" cy="11.4" r="1.7" opacity=".55" />
    </svg>
  );
}

/** 小龙 */
export function DragonIcon({ className }: IconProps) {
  return (
    <svg {...base} className={className}>
      {/* 翼 */}
      <path d="M4.1 6.2c2.6.3 4.5 1.5 5.7 3.5-1.9.3-3.5-.2-4.7-1.4-.5-.5-.8-1.2-1-2.1Z" opacity=".55" />
      {/* 头颈 */}
      <path d="M14.8 4.3c2.6 0 4.6 1.7 5.1 4.2.3 1.6-.1 3-1.2 4.2l1.4 1.5-2.3.5-.6 2.2-1.7-1.6c-1.3.6-2.6.7-3.9.3l-1.3 2.3-1-2.6-2.7.4 1.3-2.3c-1.5-1.2-2.2-2.7-2.1-4.6 1.5 1.4 3 2 4.6 1.8.9-.1 1.7-.5 2.4-1.2.8-.8 1.4-1.9 1.7-3.3Z" />
      <circle cx="16.4" cy="8.2" r="1.05" opacity=".45" />
    </svg>
  );
}

/** 大龙 (纳什男爵) */
export function BaronIcon({ className }: IconProps) {
  return (
    <svg {...base} className={className}>
      {/* 背刺 */}
      <path d="M6.4 8.1 5 4.6l3 2.1zM10 6.6 9.2 3l2.4 2.8zM14 6.5l.3-3.6 1.7 3.2z" opacity=".55" />
      {/* 长吻 */}
      <path d="M4.6 10.2c1.2-1.5 3-2.3 5.4-2.4 2.6-.1 4.9.5 6.9 1.8l3.4-1.2-1.4 2.6 2 1.7-2.6.5-.4 2.5-2.2-1.5c-1.9 1.4-4 2-6.4 1.7-2.3-.3-4-1.4-5-3.3l-2.6.6z" />
      {/* 齿 */}
      <path d="M8 15.4l.7 2.2.9-1.9zM11.2 16.1l.5 2.6 1.1-2.3zM14.6 15.3l1 2 .5-2.4z" opacity=".55" />
      <circle cx="8.5" cy="11.4" r="1.15" opacity=".4" />
    </svg>
  );
}

/** 水晶 (抑制器) */
export function InhibitorIcon({ className }: IconProps) {
  return (
    <svg {...base} className={className}>
      <path d="M12 2.2 18.4 12 12 21.8 5.6 12z" opacity=".28" />
      <path d="M12 5.4 16.3 12 12 18.6 7.7 12z" />
      <path d="M12 5.4 16.3 12H12zM7.7 12H12v6.6z" opacity=".45" />
    </svg>
  );
}

/** 击杀 (交叉双剑) */
export function KillIcon({ className }: IconProps) {
  return (
    <svg {...base} className={className}>
      <path d="M4.6 3.2 8 3.4l9.4 11.2-1.9 1.6L6.1 5zM19.4 3.2 16 3.4 6.6 14.6l1.9 1.6L17.9 5z" />
      <path d="M15.2 16.9l1.9-1.6 2.6 3.1c.4.5.4 1.2-.1 1.6-.5.4-1.2.4-1.6-.1zM8.8 16.9l-1.9-1.6-2.6 3.1c-.4.5-.4 1.2.1 1.6.5.4 1.2.4 1.6-.1z" opacity=".55" />
    </svg>
  );
}

/* ── 属性图标 ──────────────────────────────────────────────
   为什么自己画: Riot **没有公开一套完整的属性图标**。查过四个来源 ——
   perk-images/statmods 有护甲/魔抗/攻速/韧性但没有攻击力、法强、暴击、吸血;
   champion-details 只有 abilitypower 和 attackspeed 两个;
   floatingtext 只有护甲/魔抗/暴击; itemshop 那几张是图集 (要写死精灵坐标,
   改版就错位)。凑四个来源, art style 会明显打架。
   所以按 icons.tsx 已有的路子自己画一套: 单色、跟 currentColor、几百字节。 */

/** 攻击力 (剑) */
export function AdIcon({ className }: IconProps) {
  return (
    <svg {...base} className={className}>
      <path d="M20.4 2.2 21.8 3.6 11.2 14.2 9.8 12.8z" />
      <path d="M9.1 13.5l1.4 1.4-1.8 1.8-1.4-1.4z" opacity=".55" />
      <path d="M6.6 16l1.4 1.4-2.5 2.5a1 1 0 0 1-1.4-1.4z" />
      <path d="M3.4 17.2 2 18.6l3.4 3.4 1.4-1.4z" opacity=".55" />
    </svg>
  );
}

/** 法术强度 (法球) */
export function ApIcon({ className }: IconProps) {
  return (
    <svg {...base} className={className}>
      <circle cx="12" cy="11" r="6.2" opacity=".28" />
      <circle cx="12" cy="11" r="4" />
      <path d="M12 1.4l.9 2.4 2.4.9-2.4.9-.9 2.4-.9-2.4L8.7 4.7l2.4-.9z" opacity=".7" />
      <path d="M19.6 15l.6 1.6 1.6.6-1.6.6-.6 1.6-.6-1.6-1.6-.6 1.6-.6z" opacity=".5" />
    </svg>
  );
}

/** 护甲 (盾) */
export function ArmorIcon({ className }: IconProps) {
  return (
    <svg {...base} className={className}>
      <path d="M12 2.2 20 5v6.4c0 4.4-3.1 8.4-8 10.4-4.9-2-8-6-8-10.4V5z" opacity=".3" />
      <path d="M12 4.6 17.6 6.5v4.9c0 3.3-2.3 6.3-5.6 7.9-3.3-1.6-5.6-4.6-5.6-7.9V6.5z" />
    </svg>
  );
}

/** 魔抗 (带符文的盾) */
export function MrIcon({ className }: IconProps) {
  return (
    <svg {...base} className={className}>
      <path d="M12 2.2 20 5v6.4c0 4.4-3.1 8.4-8 10.4-4.9-2-8-6-8-10.4V5z" opacity=".3" />
      <path d="M12 4.6 17.6 6.5v4.9c0 3.3-2.3 6.3-5.6 7.9-3.3-1.6-5.6-4.6-5.6-7.9V6.5z" opacity=".55" />
      <path d="M12 7.2l1.1 2.9 2.9 1.1-2.9 1.1L12 15.2l-1.1-2.9-2.9-1.1 2.9-1.1z" />
    </svg>
  );
}

/** 攻速 (速度线 + 箭头) */
export function AsIcon({ className }: IconProps) {
  return (
    <svg {...base} className={className}>
      <path d="M13.2 2.4 6.4 12.8h4.3l-1.5 8.8 7.2-10.8h-4.5z" />
      <path d="M2.6 7.4h4.2v1.8H2.6zM1.4 11.1h3.5v1.8H1.4zM2.6 14.8h4.2v1.8H2.6z" opacity=".45" />
    </svg>
  );
}

/** 暴击 (爆裂) */
export function CritIcon({ className }: IconProps) {
  return (
    <svg {...base} className={className}>
      <path d="M12 1.6l2.3 6.1 6.1-2.3-3.6 5.4 5.4 3.6-6.5.3 1.4 6.4-4.9-4.3-4.9 4.3 1.4-6.4-6.5-.3 5.4-3.6L3.6 5.4l6.1 2.3z" opacity=".3" />
      <path d="M12 6.4l1.4 3.6 3.6 1.4-3.6 1.4L12 16.4l-1.4-3.6L7 11.4l3.6-1.4z" />
    </svg>
  );
}

/** 吸血 (血滴) */
export function LifestealIcon({ className }: IconProps) {
  return (
    <svg {...base} className={className}>
      <path d="M12 2.2c3.8 4.6 6.4 8 6.4 11.2A6.4 6.4 0 0 1 12 19.8a6.4 6.4 0 0 1-6.4-6.4C5.6 10.2 8.2 6.8 12 2.2Z" opacity=".3" />
      <path d="M12 6.6c2.4 3 4 5.2 4 7.1a4 4 0 0 1-8 0c0-1.9 1.6-4.1 4-7.1Z" />
    </svg>
  );
}

/** 韧性 (挣断的锁链) */
export function TenacityIcon({ className }: IconProps) {
  return (
    <svg {...base} className={className}>
      <path d="M8.4 5.6a3.4 3.4 0 0 1 4.8 0l1.6 1.6-1.7 1.7-1.6-1.6a1 1 0 0 0-1.4 0L8.4 8.9a1 1 0 0 0 0 1.4l1.6 1.6L8.3 13.6l-1.6-1.6a3.4 3.4 0 0 1 0-4.8z" />
      <path d="M15.6 18.4a3.4 3.4 0 0 1-4.8 0l-1.6-1.6 1.7-1.7 1.6 1.6a1 1 0 0 0 1.4 0l1.7-1.6a1 1 0 0 0 0-1.4l-1.6-1.6 1.7-1.7 1.6 1.6a3.4 3.4 0 0 1 0 4.8z" opacity=".55" />
      <path d="M18.8 3.4l1.4 1.4-3 3-1.4-1.4zM4.8 17.2l1.4 1.4-3 3-1.4-1.4z" opacity=".4" />
    </svg>
  );
}

/** 经济 (金币) */
export function GoldIcon({ className }: IconProps) {
  return (
    <svg {...base} className={className}>
      <circle cx="12" cy="12" r="9" opacity=".28" />
      <circle cx="12" cy="12" r="6.4" />
      <path d="M12 7.6c1.7 0 2.9.8 3.3 2.2l-1.9.5c-.2-.6-.7-1-1.4-1-.9 0-1.5.5-1.5 1.1 0 .6.4.9 1.6 1.2 1.9.4 3.1 1 3.1 2.6 0 1.6-1.3 2.6-3.2 2.6-1.9 0-3.2-.9-3.6-2.4l1.9-.5c.2.7.8 1.1 1.7 1.1s1.4-.4 1.4-1c0-.6-.5-.9-1.7-1.2-1.8-.4-3-1-3-2.6 0-1.6 1.3-2.6 3.3-2.6Z" opacity=".45" />
    </svg>
  );
}
