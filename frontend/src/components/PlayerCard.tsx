import { useEffect, useRef } from "react";
import type { Player } from "../api/types";
import {
  AdIcon,
  ApIcon,
  ArmorIcon,
  MrIcon,
  AsIcon,
  CritIcon,
  LifestealIcon,
  TenacityIcon,
} from "./icons";

/**
 * 选手详情浮窗 —— 点记分板上的头像展开。
 *
 * 为什么做成浮窗而不是铺在记分板上: 实时帧里每个选手有二十多个字段
 * (KDA/补刀/经济/血量/装备/参团/伤害占比/视野/八项属性/符文/技能加点)。
 * 十个人全铺开就是两百多个数字, 谁也读不完; 而这些字段**大部分时候不看,
 * 要看的时候必须有**。所以记分板只留一眼要看的, 其余点开。
 *
 * 这里**不做任何计算判断**: 参团率、伤害占比这些都是上游直接给的数,
 * 血条只是把 current/max 画成宽度。UI 不放领域逻辑 (见 CLAUDE.md)。
 */

const ROLE_CN: Record<string, string> = {
  top: "上单", jng: "打野", jungle: "打野", mid: "中单",
  bot: "ADC", bottom: "ADC", sup: "辅助", support: "辅助",
};

const pct = (v: number | null | undefined) =>
  v == null ? "—" : `${Math.round(v * 100)}%`;
const num = (v: number | null | undefined) => (v == null ? "—" : String(v));
const k = (n: number) => (n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n));

/**
 * 属性 —— 图标 + 中文名 + 数值。
 *
 * 图标用 icons.tsx 里自画的那一套, 不是装备图标: Riot 没有公开完整的属性
 * 图标 (见 icons.tsx 里那段说明), 而拿基础装备顶替会让人以为是"出了这件装备"。
 */
const STATS = [
  { k: "attack_damage", cn: "攻击力", Icon: AdIcon },
  { k: "ability_power", cn: "法强", Icon: ApIcon },
  { k: "armor", cn: "护甲", Icon: ArmorIcon },
  { k: "magic_resist", cn: "魔抗", Icon: MrIcon },
  { k: "attack_speed", cn: "攻速", Icon: AsIcon },
  { k: "crit", cn: "暴击", Icon: CritIcon },
  { k: "life_steal", cn: "吸血", Icon: LifestealIcon },
  { k: "tenacity", cn: "韧性", Icon: TenacityIcon },
] as const;

export function PlayerCard({ p, onClose }: { p: Player; onClose: () => void }) {
  const ref = useRef<HTMLDivElement>(null);

  // Esc 关闭 + 点外面关闭。两个都要有 —— 浮窗盖在记分板上, 只留一个
  // 关闭按钮的话, 想看被挡住的那一行得先精确点到那个小叉。
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) onClose();
    };
    document.addEventListener("keydown", onKey);
    // 捕获阶段: 记分板上的头像自己带 onClick, 冒泡阶段会先被它吃掉,
    // 表现为"点另一个头像关不掉当前这个"。
    document.addEventListener("mousedown", onDown, true);
    return () => {
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("mousedown", onDown, true);
    };
  }, [onClose]);

  // 血条和出装**不在这里** —— 它们已经常驻在记分板行内。浮窗只放那些
  // "偶尔才看"的东西, 两处都放等于让人在同一屏里读两遍同样的数字。
  const st = p.stats;
  const runes = p.perks?.perks ?? [];

  return (
    <div className={`pc ${p.side}`} ref={ref} role="dialog" aria-label="选手详情">
      <div className="pc-head">
        {p.champion_icon ? (
          <img className="pc-champ" src={p.champion_icon} alt={p.champion ?? ""} />
        ) : (
          <div className="pc-champ" />
        )}
        <div className="pc-who">
          <div className="pc-name">{p.summoner_name ?? "—"}</div>
          <div className="pc-sub">
            {p.champion ?? "—"} · {ROLE_CN[p.role ?? ""] ?? p.role ?? "—"} · {p.level} 级
          </div>
        </div>
        <button className="pc-x" onClick={onClose} aria-label="关闭">
          ×
        </button>
      </div>

      <div className="pc-grid">
        <Cell label="KDA" value={`${p.kills}/${p.deaths}/${p.assists}`} />
        <Cell label="补刀" value={String(p.cs)} />
        <Cell label="经济" value={k(p.gold)} />
        <Cell label="参团率" value={pct(p.kill_participation)} />
        <Cell label="伤害占比" value={pct(p.damage_share)} />
        <Cell label="视野" value={`${num(p.wards_placed)} / ${num(p.wards_destroyed)}`} />
      </div>

      {st && (
        <Block title="属性">
          {/* 图标 + 数字。中文名放进 title —— 八个标签摊开占的地方比数字还多,
              而图标本身就分得清 (剑/法球/盾/魔法盾/速度/爆裂/血滴/断链)。 */}
          <div className="pc-stats">
            {STATS.map(({ k: key, cn, Icon }) => {
              const raw = st[key as keyof typeof st];
              // 暴击和韧性上游给的是 0-1 的比例, 其余是绝对值
              const v = key === "crit" || key === "tenacity" ? pct(raw) : num(raw);
              return (
                <div className="pc-stat" key={key} title={cn}>
                  <Icon className="pc-stat-ico" />
                  <b>{v}</b>
                </div>
              );
            })}
          </div>
        </Block>
      )}

      {runes.length > 0 && (
        <Block
          title={
            "符文" +
            (p.perks?.style?.name
              ? ` · ${p.perks.style.name}${p.perks.sub_style?.name ? " / " + p.perks.sub_style.name : ""}`
              : "")
          }
        >
          <div className="pc-runes">
            {runes.map((r, i) =>
              r.icon ? (
                <img key={i} src={r.icon} alt={r.name ?? String(r.id)} title={r.name ?? String(r.id)} />
              ) : (
                // 小符文碎片没有独立图标; 名字也解析不出来时退回显示 id,
                // 宁可露出一个数字, 不要凭空编一个名字。
                <span className="pc-shard" key={i} title={r.name ?? String(r.id)}>
                  {r.name ?? r.id}
                </span>
              ),
            )}
          </div>
        </Block>
      )}

      {!!p.abilities?.length && (
        <Block title={`技能加点 (${p.abilities.length} 级)`}>
          <div className="pc-abil">
            {p.abilities.map((a, i) => (
              <span className={`ab ab-${a.toLowerCase()}`} key={i}>
                {a}
              </span>
            ))}
          </div>
        </Block>
      )}
    </div>
  );
}

function Cell({ label, value }: { label: string; value: string }) {
  return (
    <div className="pc-cell">
      <div className="pc-cell-l">{label}</div>
      <div className="pc-cell-v">{value}</div>
    </div>
  );
}

function Block({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="pc-block">
      <div className="pc-block-t">{title}</div>
      {children}
    </div>
  );
}
