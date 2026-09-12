import { useState } from "react";
import {
  TowerIcon,
  DragonIcon,
  BaronIcon,
  InhibitorIcon,
  GoldIcon,
} from "./icons";

/**
 * 目标资源图标 —— 用 Riot 自己的小地图图标, 拿不到就退回内联 SVG。
 *
 * 为什么换成外链: 手绘那套在 14px 下分不清龙和大龙, 而且七种小龙只能用
 * 单个汉字 (火/地/海/风/法/化) 凑合。官方这套 64x64 带 alpha、自带深色描边,
 * 浅底深底都清晰, **而且每种龙有专属图案** —— 这是汉字标签给不了的。
 *
 * 为什么还留着内联 SVG: icons.tsx 的注释说得对 —— 外链的失败方式是
 * "一排碎图标而且不报错"。所以这里不是二选一, 是**外链优先 + 加载失败自动
 * 退回**。CommunityDragon 哪天改路径, 界面只是变回原来的样子, 不会开天窗。
 *
 * 路径用 `latest`: 它跟着版本走, 新龙种 (atakhan/grub 之类) 出现时不用改代码。
 * 代价是 Riot 改名会静默失效 —— 由上面那层兜底接住。
 */
const CDRAGON =
  "https://raw.communitydragon.org/latest/game/assets/ux/minimap/icons";
/** 经济不在小地图图标里 —— 用游戏内飘金币数字时的那个官方币图标。 */
const GOLD_ICON =
  "https://raw.communitydragon.org/latest/game/assets/ux/floatingtext/goldicon.png";

/** 帧里 dragons 数组给的是这些字符串, 直接映射到官方图标文件名。 */
const DRAKE_FILE: Record<string, string> = {
  infernal: "dragon_infernal",
  mountain: "dragon_mountain",
  ocean: "dragon_ocean",
  cloud: "dragon_cloud",
  hextech: "dragon_hextech",
  chemtech: "dragon_chemtech",
  elder: "dragon_elder",
};

export const DRAKE_CN: Record<string, string> = {
  infernal: "火龙",
  mountain: "地龙",
  ocean: "海龙",
  cloud: "风龙",
  hextech: "法龙",
  chemtech: "化龙",
  elder: "远古龙",
};

type Kind = "tower" | "dragon" | "baron" | "inhibitor" | "gold";

const FILE: Record<Exclude<Kind, "gold">, string> = {
  tower: "tower",
  dragon: "dragon",
  baron: "baron",
  inhibitor: "inhibitor",
};

const FALLBACK = {
  tower: TowerIcon,
  dragon: DragonIcon,
  baron: BaronIcon,
  inhibitor: InhibitorIcon,
  gold: GoldIcon,
};

export function ObjectiveIcon({
  kind,
  drake,
  className,
  title,
}: {
  kind: Kind;
  /** 给了就用这一种小龙的专属图标 */
  drake?: string;
  className?: string;
  title?: string;
}) {
  const [broken, setBroken] = useState(false);
  const Fallback = FALLBACK[kind];

  if (broken) return <Fallback className={className} />;

  const src =
    kind === "gold"
      ? GOLD_ICON
      : `${CDRAGON}/${drake ? (DRAKE_FILE[drake] ?? "dragon") : FILE[kind]}.png`;
  return (
    <img
      className={className}
      src={src}
      alt={title ?? kind}
      title={title}
      loading="lazy"
      onError={() => setBroken(true)}
    />
  );
}
