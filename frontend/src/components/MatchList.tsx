import type { Match, TeamRef } from "../api/types";
import { T } from "../i18n";

function Crest({ team }: { team: TeamRef }) {
  if (team.image) {
    return <img className="crest" src={team.image} alt="" loading="lazy" />;
  }
  return (
    <div className="crest crest-fallback">{(team.code || team.name || "?").slice(0, 3)}</div>
  );
}

function when(iso: string): string {
  const t = new Date(iso);
  if (Number.isNaN(t.getTime())) return "";
  const now = new Date();
  const sameDay = t.toDateString() === now.toDateString();
  const tomorrow = new Date(now.getTime() + 86400_000).toDateString() === t.toDateString();
  const hm = t.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", hour12: false });
  if (sameDay) return T("今天 {hm}", { hm });
  if (tomorrow) return T("明天 {hm}", { hm });
  return `${t.getMonth() + 1}/${t.getDate()} ${hm}`;
}

function Card({
  match,
  kind,
  onOpen,
}: {
  match: Match;
  /** 这张卡在哪个列表里。**比 match.state 可靠得多** —— 见下方说明。 */
  kind: "live" | "upcoming";
  onOpen: (m: Match) => void;
}) {
  const [a, b] = match.teams;

  // **不看 match.state。** 那是赛程接口的字段, 实测会过期得很离谱:
  // 2026-09-04 LPL LGD vs AL 正打到第 3 局 (1-1 的 BO5, 帧里 in_game),
  // 而 state 已经是 completed —— 于是这张卡躺在"正在进行"列表里, 却被打上
  // "已结束"标签, 连"进行中"都不显示。反方向也见过: 2026-08-16 LPL 当天
  // 三场全被标成 completed, 其中两场还没到开赛时间。
  //
  // 换成两个可信的信号:
  //   · kind —— 服务端已经用帧判过了 (/esports/live 走 live_by_frames,
  //     帧是唯一权威), 卡片没必要再猜一遍
  //   · 比分对 BO 算胜负 —— 和后端 _clinched、看板的停轮询判据同一套
  const need = match.best_of != null ? Math.floor(match.best_of / 2) + 1 : null;
  const wins = match.teams.map((t) => t.game_wins ?? 0);
  const done = need != null && wins.some((w) => w >= need);
  const live = kind === "live" && !done;
  const usable = match.predictable && !!a && !!b;

  return (
    <button
      className="card"
      disabled={!usable}
      onClick={() => usable && onOpen(match)}
      title={usable ? "" : T("缺少历史数据, 无法预测")}
    >
      <div className="side">
        {a && <Crest team={a} />}
        <div style={{ minWidth: 0 }}>
          <div className="team-name">{a?.name ?? "TBD"}</div>
          {a && !a.known && <div className="team-sub">{T("无历史数据")}</div>}
        </div>
      </div>

      <div className="mid">
        {live || done ? (
          <div className="score">
            {a?.game_wins ?? 0}–{b?.game_wins ?? 0}
          </div>
        ) : (
          <div className="when">{when(match.start_time)}</div>
        )}
        <div className="tags">
          {live && <span className="tag live">{T("进行中")}</span>}
          {done && <span className="tag">{T("已结束")}</span>}
          <span className="tag">{match.league}</span>
          {match.best_of != null && <span className="tag">BO{match.best_of}</span>}
        </div>
      </div>

      <div className="side right">
        {b && <Crest team={b} />}
        <div style={{ minWidth: 0 }}>
          <div className="team-name">{b?.name ?? "TBD"}</div>
          {b && !b.known && <div className="team-sub">{T("无历史数据")}</div>}
        </div>
      </div>
    </button>
  );
}

interface Props {
  title: string;
  matches: Match[];
  /** 这一组是直播还是待开赛。卡片靠它判"进行中", 不再猜 match.state。 */
  kind: "live" | "upcoming";
  loading: boolean;
  error: string | null;
  emptyText: string;
  onOpen: (m: Match) => void;
}

/** 列表自己不带刷新按钮 —— App 每 30 秒、以及切回标签页时自动重取 */
export function MatchList({ title, matches, kind, loading, error, emptyText, onOpen }: Props) {
  return (
    <section className="section">
      <div className="section-head">
        <h2>{title}</h2>
        {!!matches.length && <span className="count">{matches.length}</span>}
      </div>
      {error ? (
        <div className="panel pad err">{error}</div>
      ) : matches.length ? (
        <div className="cards">
          {matches.map((m, i) => (
            <div key={m.match_id} className="rise" style={{ animationDelay: `${Math.min(i, 8) * 0.04}s` }}>
              <Card match={m} kind={kind} onOpen={onOpen} />
            </div>
          ))}
        </div>
      ) : (
        <div className="panel pad empty">{loading ? T("加载中…") : emptyText}</div>
      )}
    </section>
  );
}
