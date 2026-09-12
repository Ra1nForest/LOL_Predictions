import { useState } from "react";
import type { Lane, Player, TeamObjectives } from "../api/types";
import { PlayerCard } from "./PlayerCard";
import { ObjectiveIcon, DRAKE_CN } from "./ObjectiveIcon";
import { KillIcon } from "./icons";
import { T } from "../i18n";

// 位置名 (上单/打野…) 不再显示 —— 五行顺序固定就是上野中下辅, 而两边的
// 头像和英雄名已经说明了是谁。中间那一栏因此能窄下来给经济差腾地方。
// 中文名仍然在浮窗里 (PlayerCard 的 ROLE_CN)。
const LANE_ORDER = ["top", "jungle", "mid", "bottom", "support"];
const LANE_ALIAS: Record<string, string> = {
  jng: "jungle",
  bot: "bottom",
  sup: "support",
};
const norm = (r: string | null) => (r ? (LANE_ALIAS[r] ?? r) : "");
const k = (n: number) => (n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n));

interface Props {
  players: Player[];
  lanes: Lane[];
  blue: TeamObjectives;
  red: TeamObjectives;
  blueName: string;
  redName: string;
}

/**
 * 记分板 —— 队伍总览 + 五路对位, 一块面板搞定, 排布照 LoL 观战界面:
 * 蓝方在左、红方在右, 数值向中线镜像, 中间是路名和该路经济差。
 *
 * 为什么合并 (原来是 局势 / 选手数据 两块并排):
 * 同一路的两个人本来就该挨着比, 拆成两张表要左右扫视才能对上号。镜像布局
 * 让"谁在这一路领先"变成一眼可见, 不需要读数。
 *
 * 注意: 推塔/小龙/大龙这些**模型看不到** —— Stage 4 只吃经济/经验/补刀/
 * 人头。已经过闸测过 (research/gate_objectives.py), 对正确率没有效果, 放在
 * 这里纯粹是给人看局势的, 不参与任何预测。
 */
export function Scoreboard({ players, lanes, blue, red, blueName, redName }: Props) {
  const bs = players.filter((p) => p.side === "blue");
  const rs = players.filter((p) => p.side === "red");
  const laneOf = new Map(lanes.map((l) => [l.lane, l]));
  // 展开中的选手。按 participant_id 记而不是记整个对象 —— 轮询每 10 秒换一次
  // players 数组, 记对象的话浮窗里的数字会**冻在点开的那一刻**不再更新。
  const [openPid, setOpenPid] = useState<number | null>(null);
  const open = players.find((p) => p.participant_id === openPid) ?? null;

  return (
    <section className="panel pad sb">
      <h3 className="card-title">
        {T("记分板")}<span className="sb-hint">{T("左右滑动看全部")}</span>
      </h3>

      {/* 手机上放不下时**整块左右滑**, 不改布局: 镜像的三栏、七格装备、战绩都保留。
          早先窄屏会退成上下堆叠, 实测挤成一团、对位关系也没了。
          队伍总览和五路对位放在同一个滚动区里, 中轴才能一起滑、保持对齐;
          选手详情浮窗留在滚动区外面, 否则会被 overflow 裁掉。 */}
      <div className="sb-scroll">
        <div className="sb-body">
          {/* 队伍总览: 资源点分列两边, 中间只留人头比分和经济差。
              中间那一栏是全场唯一的"合并读数" —— 比分和经济差本来就是双方共有的
              一个数, 摊到两边反而要自己做减法。 */}
          <div className="sb-head">
            <TeamSide o={blue} name={blueName} side="blue" />
            <div className="sb-head-mid">
              <div className="sb-score">
                <b className="blue">{blue.kills}</b>
                <KillIcon className="sb-kill-ico" />
                <b className="red">{red.kills}</b>
              </div>
              {/* 队伍总经济差用 "+xxxx", 颜色是领先方的。这里和下面每一路的
                  箭头写法**故意不同**: 这一处只有一个数、位置固定在正中,
                  加号已经够清楚; 而逐路那五行要在一眼之内分辨方向, 箭头比
                  正负号快。 */}
              <div
                className={`sb-golddiff${
                  blue.gold > red.gold ? " blue" : red.gold > blue.gold ? " red" : ""
                }`}
              >
                {blue.gold === red.gold
                  ? T("经济持平")
                  : `+${Math.abs(blue.gold - red.gold).toLocaleString()}`}
              </div>
            </div>
            <TeamSide o={red} name={redName} side="red" />
          </div>

          {/* 五路对位 */}
          <div className="sb-lanes">
            {LANE_ORDER.map((lane) => {
              const b = bs.find((p) => norm(p.role) === lane);
              const r = rs.find((p) => norm(p.role) === lane);
              if (!b && !r) return null;
              const gap = laneOf.get(lane);
              const d = gap?.gold_diff ?? 0;
              return (
                <div className="sb-lane" key={lane}>
                  <PlayerCell p={b} side="blue" onOpen={setOpenPid} openPid={openPid} />
                  {/* 中间只留经济差。位置名 (上单/打野…) 删掉了 —— 五行的顺序
                      本来就是固定的上中野下辅, 而两边头像和英雄名已经说明了是谁。
                      箭头**指向领先的一方**: 蓝方领先就 "‹ 1,751", 红方就
                      "1,751 ›"。方向 + 颜色两重编码同一件事, 扫一眼就知道
                      这一路是谁在压制, 不用先读符号再想正负是谁。 */}
                  <div className="sb-mid">
                    {gap &&
                      (d === 0 ? (
                        <div className="sb-gap">{T("持平")}</div>
                      ) : (
                        <div className={`sb-gap ${d > 0 ? "blue" : "red"}`}>
                          {d > 0 && <i className="arw">‹</i>}
                          {Math.abs(d).toLocaleString()}
                          {d < 0 && <i className="arw">›</i>}
                        </div>
                      ))}
                  </div>
                  <PlayerCell p={r} side="red" onOpen={setOpenPid} openPid={openPid} />
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* 详情**覆盖在计分板上**, 不再往下顶开一块。
          蓝方(左)的卡片贴左边往右展开, 红方(右)的贴右边往左展开 ——
          点谁就从谁那一侧长出来, 视线不用跨过整块面板去找。
          两侧都留 8px, 所以不会超出面板边界。 */}
      {open && (
        <div className={`pc-wrap ${open.side}`}>
          <PlayerCard p={open} onClose={() => setOpenPid(null)} />
        </div>
      )}
    </section>
  );
}

/** 一侧的资源点。领先的项加重 —— 五个数字全一样重就等于没有重点。 */
function TeamSide({
  o,
  name,
  side,
}: {
  o: TeamObjectives;
  name: string;
  side: "blue" | "red";
}) {
  // 上色按**取值本身**, 不按领不领先: 0 保持墨色, 非 0 才染成己方颜色。
  // 领先不再加粗 —— 一排里两三个粗体反而看不出重点, 而"有没有拿到"本身
  // 就是这几项最要紧的信息 (塔龙大龙水晶都是从 0 开始数的)。
  const stat = (
    icon: "gold" | "tower" | "dragon" | "baron" | "inhibitor",
    title: string,
    v: number | string,
    on: boolean,
  ) => (
    <span key={icon} className={`sb-stat${on ? " on" : ""}`} title={title}>
      <ObjectiveIcon kind={icon} className="sb-ico" title={title} />
      {v}
    </span>
  );
  // **镜像**: 从中线往外是 经济 → 塔 → 龙 → 大龙 → 水晶。两侧顺序相反,
  // 于是同一项在左右两栏里离中线一样远, 视线横着扫就能比。
  const items = [
    stat("gold", T("总经济"), k(o.gold), o.gold > 0),
    stat("tower", T("推塔"), o.towers, o.towers > 0),
    stat("dragon", T("小龙"), o.dragons, o.dragons > 0),
    stat("baron", T("大龙"), o.barons, o.barons > 0),
    stat("inhibitor", T("水晶 (抑制器)"), o.inhibitors, o.inhibitors > 0),
  ];
  return (
    <div className={`sb-team ${side}`}>
      <div className={`sb-team-name ${side}`}>{name}</div>
      <div className="sb-team-stats">
        {side === "blue" ? [...items].reverse() : items}
      </div>
      {/* 拿到的每一条龙, 用官方的专属图标 —— 汉字标签认不出是哪条 */}
      {o.dragon_types.length > 0 && (
        <div className={`sb-drakes ${side}`}>
          {o.dragon_types.map((t, i) => (
            <ObjectiveIcon key={i} kind="dragon" drake={t} className="drake-ico"
                           title={DRAKE_CN[t] ? T(DRAKE_CN[t]!) : t} />
          ))}
        </div>
      )}
    </div>
  );
}

/** 一侧的选手格。红方整体镜像 (row-reverse), 于是数值总是贴着中线。 */
function PlayerCell({
  p,
  side,
  onOpen,
  openPid,
}: {
  p: Player | undefined;
  side: "blue" | "red";
  onOpen: (pid: number | null) => void;
  openPid: number | null;
}) {
  if (!p) return <div className={`sb-cell ${side} empty`} />;
  const isOpen = openPid === p.participant_id;
  const hp = p.max_health ? (p.current_health ?? 0) / p.max_health : null;
  const dead = p.current_health === 0;
  return (
    <div className={`sb-cell ${side}`}>
      {/* 头像做成按钮: 点开详情。用 <button> 而不是给 <img> 加 onClick ——
          键盘能 Tab 到, 读屏能念出来, 而且自带焦点样式。
          等级角标压在头像下角, 位置跟着阵营走 (蓝方右下、红方左下), 于是
          它总是落在**靠中线**那一侧, 不会被旁边的名字挤到。 */}
      <button
        className={`champ-btn${isOpen ? " on" : ""}${dead ? " dead" : ""}`}
        onClick={() => onOpen(isOpen ? null : p.participant_id)}
        title={T("{name} — 点开看详情", { name: p.summoner_name ?? "" })}
        aria-expanded={isOpen}
      >
        {p.champion_icon ? (
          <img className="champ-icon" src={p.champion_icon} alt={p.champion ?? ""} />
        ) : (
          <div className="champ-icon" />
        )}
        {/* 阵亡: 头像压一层黑白遮罩 + "阵亡" 两个字, 一直到复活。
            比在血条上写字醒目得多 —— 头像是这一行视觉上最重的元素,
            它一灰整行就"暗"下去了。 */}
        {dead && <span className="champ-dead">{T("阵亡")}</span>}
        <span className="champ-lv">{p.level}</span>
      </button>

      <div className="sb-who">
        <div className="sb-name">{p.summoner_name ?? "—"}</div>
        {/* 血条加厚并把 当前/上限 写在条上 —— 只画长度看不出"还剩多少血
            能不能接团"。死了不在这里写字, 那件事交给头像遮罩。 */}
        {hp != null ? (
          <div className={`sb-hp${dead ? " dead" : ""}`}>
            <i style={{ width: `${Math.max(0, Math.min(1, hp)) * 100}%` }} />
            <b>
              {p.current_health}/{p.max_health}
            </b>
          </div>
        ) : (
          <div className="sb-champ">{p.champion ?? "—"}</div>
        )}
      </div>

      {/* 六格装备 + 一格饰品, 排在血条右边 (红方镜像后是左边)。
          饰品**单独隔开并做成圆的**: 它不占正经装备位, 而且帧里它混在同一个
          数组里、位置还不固定 (见后端 trinket_ids())。
          空格子补齐: 位置固定, 眼睛不用重新定位。 */}
      <div className="sb-items">
        {Array.from({ length: 6 }, (_, i) => {
          const it = p.items[i];
          return it ? (
            <img key={i} src={it.icon} alt="" title={String(it.id)} loading="lazy" />
          ) : (
            <span key={i} className="slot" />
          );
        })}
        {p.trinket ? (
          <img className="trinket" src={p.trinket.icon} alt={T("饰品")}
               title={T("饰品 {id}", { id: p.trinket.id })} loading="lazy" />
        ) : (
          <span className="slot trinket" />
        )}
      </div>

      {/* 战绩 + 补刀, 排在装备右边。经济移进浮窗 —— 中间那栏已经用差值
          表达过了, 这里再摆一遍只是占地方。 */}
      <div className="sb-nums">
        <div className="sb-kda">
          {p.kills}/{p.deaths}/{p.assists}
        </div>
        <div className="sb-cs">{T("{n} 刀", { n: p.cs })}</div>
      </div>
    </div>
  );
}
