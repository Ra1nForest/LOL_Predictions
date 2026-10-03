import type { Match, TeamRef } from "../api/types";
import { T, leagueName } from "../i18n";

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
  // **没有赛前 ≠ 不能预测。** pregame_league 为 null 的是国际赛里跨赛区的对阵 (赛前和 BP 后模型
  // 在那上面没有预测力, 后端不给), 或者有队不在四大赛区的数据里 —— 这两种开局 3 分钟起都有局内
  // 胜率, 卡片要能点开。待定的队 (TBD) 不算: 那时连谁打都不知道。
  // 赛前只可能来自跨赛区模型 C, 而它算不算得出来要看板重放本赛事才知道 (队名映射、赛程取得到) ——
  // 列表不知道, 所以这里只说"赛前只看跨赛区模型": C 给了数、没给数都成立。原来写的"赛前不预测"在 C 上线后
  // 多数时候是错的。
  // 四大赛区的 pregame_league 恒为本赛区, 所以国内比赛的卡片和原来一模一样。
  const tbd = (t: TeamRef | undefined) => !t || !t.name || t.name === "TBD";
  const noPregame = !tbd(a) && !tbd(b) && match.pregame_league === null;
  const usable = !!a && !!b && (match.predictable || noPregame);
  const noPregameText = T("赛前只看跨赛区模型 · 开局后有局内胜率");

  return (
    <button
      className="card"
      disabled={!usable}
      onClick={() => usable && onOpen(match)}
      title={noPregame ? noPregameText : usable ? "" : T("缺少历史数据, 无法预测")}
    >
      <div className="side">
        {a && <Crest team={a} />}
        <div style={{ minWidth: 0 }}>
          <div className="team-name">{a?.name ?? "TBD"}</div>
          {a && !a.known && !noPregame && <div className="team-sub">{T("无历史数据")}</div>}
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
          <span className="tag">{leagueName(match.league)}</span>
          {match.best_of != null && <span className="tag">BO{match.best_of}</span>}
        </div>
        {noPregame && <div className="note">{noPregameText}</div>}
      </div>

      <div className="side right">
        {b && <Crest team={b} />}
        <div style={{ minWidth: 0 }}>
          <div className="team-name">{b?.name ?? "TBD"}</div>
          {b && !b.known && !noPregame && <div className="team-sub">{T("无历史数据")}</div>}
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
