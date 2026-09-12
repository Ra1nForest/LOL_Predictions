import { useEffect, useMemo, useRef, useState } from "react";
import { api, mutations, ApiError, IS_STATIC } from "../api/client";
import type { BoardResponse, DebateRequestBody, Match, PredictResponse } from "../api/types";
import { WinChart } from "./WinChart";
import { GoldChart } from "./GoldChart";
import { ProbabilityScale } from "./ProbabilityScale";
import { Scoreboard } from "./Scoreboard";
import { ThemeToggle } from "./ThemeToggle";
import { DebatePanel } from "./DebatePanel";
import { PreMatch } from "./PreMatch";
import { T } from "../i18n";
import { useFineTimeline, useLivePlayback } from "./useLivePlayback";
import { mergeFine } from "./chartKit";

const STATE_CN: Record<string, string> = {
  in_game: "进行中",
  paused: "暂停中",
  finished: "已结束",
};

// 数据源的粒度是 10 秒: 窗口的 startingTime 必须对齐到整 10 秒 (实测不对齐的一律 400),
// 新数据每 10 秒才出一批。但**什么时候出**和我们的定时器没有对齐 —— 按 10 秒轮询,
// 新窗口出现后平均要白等 5 秒才被取到。改成 5 秒, 白等减半 (约 2.5 秒), 代价是一半的
// 请求拿回的是同一个窗口: 静态版这些请求是访客自己的浏览器直接发给 lolesports 的,
// 服务器版有 3 秒的窗口缓存, 两边都付得起。
// 再往下压没有意义: 上游本身的发布延迟实测约 50 秒 (2026-09-13 LEC, 两个端点都是
// 往回退 60 秒才有数据), 那才是大头, 我们能省的只有这几秒。
// 没开打时没必要盯这么紧。
const POLL_LIVE_MS = 5_000;
const POLL_PRE_MS = 60_000;
// 出错后的重试间隔。给得比 LIVE 长: 失败多半是后端在冷启动 (加载五年 CSV),
// 按 10 秒去撞只是连撞几十次都失败。
const POLL_ERR_MS = 20_000;

interface BoardProps {
  matchId: string;
  /** 列表点进来时已经有队名了, 先拿它渲染, 免得等 board 回来才有标题。
   *  从 URL 直达时是 null, 队名等后端的 match 字段。 */
  initial: Match | null;
  onBack: () => void;
}

export function Board({ matchId, initial, onBack }: BoardProps) {
  const [gameId, setGameId] = useState<string | null>(null);
  const [data, setData] = useState<BoardResponse | null>(null);
  const [pre, setPre] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  // 切回标签页时 +1, 用来把轮询 effect 立刻重跑一遍
  const [nudge, setNudge] = useState(0);
  // 请求序号: 轮询 + 手动切局会并发, 只认最后发出的那个响应
  const seq = useRef(0);

  useEffect(() => {
    let alive = true;
    let timer: number | undefined;

    // 只发一个请求。曾经试过"先要不带曲线的快查、再补曲线"的两段式, 撤掉了:
    // 冷路径的开销**不在曲线上**, 而在 window(precise_minute=True) 的暂停
    // 扫描 —— 那一步不管要不要曲线都得跑, 所以 curves=false 冷调用同样是
    // 11 秒 (实测), 两段式只是白多发一个请求。
    // 首屏延迟改由 App 里的 prefetchBoard 解决: 在用户还在看列表时就预热。
    const tick = async () => {
      const mine = ++seq.current;
      try {
        const d = await api.board(matchId, gameId);
        if (!alive || mine !== seq.current) return;
        setData(d);
        setError(null);

        const lv = d.live;
        const hasLive = !!lv && !!d.prediction?.probability_blue;

        // 停不停轮询, 看的是**整个系列赛**打完没有, 不是当前这一局。
        //
        // 原来写的是"这一局 finished 就停", 于是第一局一结束定时器就不再续期,
        // 页面永远停在第一局 —— 第二局开了也不会跳过去, 只能手动刷新。
        // 服务端本来是会跟的 (current_game() 找不到在打的局时回落到
        // playable[-1], 第二局一开就选中它), 但前端已经不问了。
        //
        // stalled 同理不能作为停止条件: 帧流断了恰恰多半是"这局其实打完了、
        // 上游没推 finished", 那正是下一局要来的时候。
        //
        // 判据用 game_wins 对 best_of 自己算 (和后端 _clinched 同一套), 不用
        // match.state —— 后者实测会把没开赛的比赛标成 completed
        // (2026-08-16 LPL 三场), 信它会让整场比赛一进页面就不再刷新。
        const bo = d.match?.best_of ?? null;
        const wins = (d.match?.teams ?? []).map((t) => t.game_wins ?? 0);
        const need = bo != null ? Math.floor(bo / 2) + 1 : null;
        const seriesDone =
          need != null && wins.length > 0 && wins.some((w) => w >= need);
        // best_of 缺失时退回旧判据, 免得对着一场真的打完的比赛永远轮询
        const keep = need != null ? !seriesDone : d.match?.state !== "completed";
        if (keep) {
          // **系列赛进行中就用快档, 哪怕这一刻没有局内帧。**
          // 两局之间 (上一局刚结束、下一局还没开) 也是 hasLive=false, 但那
          // 恰恰是最该盯紧的时刻 —— 下一局随时开打。原来这里退到 60 秒档,
          // 于是新一局开了要等一分多钟才出现, 表现为"必须手动刷新"。
          //
          // "已经赢过局但还没打完" 也算进行中, 而且这一条**不依赖 match.state**
          // —— 局间休息正是它最容易报错的时候。
          const midSeries = need != null && !seriesDone && wins.some((w) => w > 0);
          const soon = hasLive || midSeries || d.match?.state === "inProgress";
          timer = window.setTimeout(tick, soon ? POLL_LIVE_MS : POLL_PRE_MS);
        }
      } catch (e) {
        if (!alive || mine !== seq.current) return;
        setError(e instanceof ApiError ? e.message : String(e));
        // **失败也要接着排下一轮。** 原来这里不排, 于是**任何一次请求失败都会
        // 永久停掉轮询** —— 服务重启、网络抖一下、后端偶发 502, 页面就再也不
        // 更新了, 只能手动刷新。实测 2026-09-01: 重启 API 那一下
        // (ERR_CONNECTION_REFUSED) 让页面停摆了 17 分钟, 直到窗口重新获得焦点
        // 才靠 nudge 复活。
        //
        // 用更长的间隔重试: 后端多半是在冷启动 (加载五年 CSV 要几十秒), 按
        // 10 秒去撞只会连撞几十次都失败。
        timer = window.setTimeout(tick, POLL_ERR_MS);
      } finally {
        if (alive && mine === seq.current) setLoading(false);
      }
    };

    setLoading(true);
    void tick();
    return () => {
      alive = false;
      if (timer) clearTimeout(timer);
    };
  }, [matchId, gameId, nudge]);

  // 回到这个标签页时立刻取一次, 不等下一个轮询周期。
  //
  // 浏览器会**限制后台标签页的定时器** (Chrome 降到每分钟一次, 挂久了直接
  // 冻结整个页面), 所以切走再切回来时页面上很可能是几分钟前的状态 —— 而这
  // 正是"每次比赛开了都得手动刷新"的另一半原因。
  useEffect(() => {
    const wake = () => {
      if (document.visibilityState === "visible") setNudge((n) => n + 1);
    };
    document.addEventListener("visibilitychange", wake);
    window.addEventListener("focus", wake);
    return () => {
      document.removeEventListener("visibilitychange", wake);
      window.removeEventListener("focus", wake);
    };
  }, []);

  // 比赛进行中 (静态版): 记分板、胜率、走势图末端改用逐帧回放, 见 web/player.ts。
  // 其余情况 shown 就是轮询拿到的那份
  const shown = useLivePlayback(data) ?? data;
  // 走势图逐秒 (静态版): 历史部分后台补成每秒一个点, 旧局和直播一个样。见 useFineTimeline
  const fine = useFineTimeline(data);
  const shownTimeline = shown?.timeline;
  // 回放中 (shown 是回放改写过的那份): 逐秒历史只画到播放头, 见 mergeFine
  const cap = shown !== data ? shownTimeline?.[shownTimeline.length - 1]?.minute : undefined;
  const timeline = useMemo(() => mergeFine(shownTimeline ?? [], fine, cap), [shownTimeline, fine, cap]);

  const teams = data?.match?.teams ?? initial?.teams ?? [];
  const blueName = teams[0]?.model_name ?? teams[0]?.name ?? T("蓝方");
  const redName = teams[1]?.model_name ?? teams[1]?.name ?? T("红方");

  const lv = shown?.live ?? null;
  const pr = shown?.prediction ?? null;
  // **model 缺席时不能走局内视图。** 后端在开局不足 10 分钟时返回 too_early:
  // 它带着 probability_blue (赛前/BP 后的概率), 却**没有 model** —— 于是
  // 老的 hasLive 判定为真, 渲染时读 pr.model.p_min 抛 TypeError,
  // 整个 React 树卸载, 页面变全白。实测 2026-09-01 LCK 第 2 局开局那一刻。
  //
  // 这个坑以前碰不到, 是因为上一局结束后轮询就停了, 页面根本活不到下一局
  // 开局。轮询修好之后每一局的前十分钟都会经过这里。
  //
  // 落到赛前视图正是后端那段 note 的原意: "局内模型最早在第 10 分钟的切片上
  // 训练过, 这里显示的是 BP 后/赛前的概率 —— 它不看局内数据"。
  const card = pr?.model;
  const hasLive = !!lv && pr?.probability_blue != null && !!card;

  // **记分板只要有帧就放出来。**
  //
  // hasLive 管的是"局内**概率**能不能显示", 它要求有 model —— 因为胜率刻度
  // 和走势图都要读 p_min/p_max。但记分板一个模型字段都不用: 阵容、补刀、
  // 经济、装备、血量全是帧里直接给的。
  //
  // 混在一起的代价是开局前十分钟整块局内视图都消失: 后端在第 10 分钟前
  // 返回 too_early (不带 model), 于是有实时数据却什么都不显示, 只能看赛前页。
  // 那十分钟恰恰是 BP 刚结束、大家最想看阵容和对线的时候。
  const hasFrames = !!lv && (lv.players?.length ?? 0) > 0;

  // **Stage 2: 已经有 BP、但局内模型还用不上的那一段。**
  //
  // 后端在开局不足 10 分钟时返回 too_early, 里面**带着 BP 后的概率**
  // (postdraft_probability_blue) —— 那是 Stage 2 算出来的, 已经看过双方阵容,
  // 比纯赛前 (Stage 1, 只看队伍历史) 信息多一层。
  //
  // 原来这一段整个退回赛前视图, 于是页面显示的是 Stage 1 的数, 而更好的
  // Stage 2 结果就摆在响应里没人用。三个阶段现在各归各位:
  //   Stage 1  没开打          -> PreMatch, 赛前概率
  //   Stage 2  有 BP、无局内模型 -> 这里, BP 后概率 + 记分板
  //   Stage 4  局内模型可用      -> hasLive, 局内概率 + 走势图
  const postP = pr?.postdraft_probability_blue ?? null;
  const earlyP = pr?.probability_blue ?? null;
  const hasEarly = !hasLive && earlyP != null;
  // 用的到底是哪一档 —— 后端的 source 字段说了算, 不自己按有没有值去猜
  const earlyIsDraft = pr?.source === "post_draft" || postP != null;

  // 没有局内读数 = 比赛还没开打, 回退到赛前预测 (Stage 1/2)。
  // 队名映射不出来就不发 —— 后端只认模型认识的队名, 拿 API 原名去问必错。
  const league = data?.match?.league ?? initial?.league;
  const bm = teams[0]?.model_name;
  const rm = teams[1]?.model_name;
  useEffect(() => {
    // hasEarly 时也不发 —— 看板自己已经带回了 BP 后概率, 再去问一次
    // /predict 拿的是信息更少的 Stage 1, 白花一次调用还会闪一下。
    if (hasLive || hasEarly || !data || !bm || !rm || !league) return;
    let alive = true;
    mutations
      .predict({ blue_team: bm, red_team: rm, league })
      .then((r) => alive && setPre(r))
      .catch(() => alive && setPre(null));
    return () => {
      alive = false;
    };
  }, [hasLive, data, bm, rm, league]);

  // 局标签按进度**增量**长出来, 不是一进页面就摆满 BO5 的五个空位 ——
  // 那五个里有四个还不存在, 摆出来只是让人以为已经打了五局。
  //
  // 判据是 g.playable (persisted 的 games[].state 属于 inProgress/completed),
  // **不是 match.state**: 后者实测会整片报错 (2026-08-16 LPL 当天三场全被标成
  // completed, 其中两场还没到开赛时间, 见 /esports/upcoming 的注释)。拿它当
  // 准绳, 标签会在一局都没开打时就冒出来。
  //
  // 只有"已开赛但第一局还没推帧"那一小段用得着 match.state: 这时一局都还不
  // playable, 但页面该已经对准第 1 局 (显示它的赛前预测)。这里信错了代价也小
  // —— 多显示一个第 1 局的标签而已, 而漏显示会让人以为页面卡住了。
  const allGames = data?.games ?? [];
  const played = allGames.filter((g) => g.playable);
  const shownGames =
    played.length > 0
      ? played
      : data?.match?.state === "inProgress" && allGames.length > 0
        ? [allGames[0]!]
        : [];
  // BO1 没有可切的对象, 一个孤零零的标签只是噪声
  const showTabs = shownGames.length > 0 && allGames.length > 1;
  const latestGameId = shownGames[shownGames.length - 1]?.game_id ?? null;
  // 服务端一局都没选中时 (还没开打) 高亮最后一个 —— 那就是页面正对着的那局
  const activeGameId = data?.selected_game_id ?? latestGameId;

  const stateLabel = lv
    ? lv.stalled
      ? T("数据中断")
      : T(STATE_CN[lv.game_state] ?? "状态未知")
    : "";

  // 局号/分钟已经由局切换标签和 summary 各自表达了, 这里只留**异常**:
  // 暂停多久、数据断在哪一刻。正常进行时应当是空的 —— 一行没有信息的
  // "第 4 局 · 第 37 分钟 · 已结束" 只是把三处重复的东西再说一遍。
  const meta: string[] = [];
  if (lv && lv.paused_seconds >= 60) {
    meta.push(T("期间暂停约 {n} 分钟", { n: Math.round(lv.paused_seconds / 60) }));
  }
  // 数据有多旧, 在**判死之前**就要说。
  //
  // stalled 是 10 分钟才翻的开关, 而在那之前界面对落后只字不提 —— 实测
  // 2026-09-06 LPL WE vs IG, 上游按真实时间的 15-62% 发帧, 延迟一路涨到
  // 11 分钟; 在越过 600 秒之前, 页面把 9 分钟前的第 16 分钟当作当前战况
  // 显示, 而比赛实际已经打到 23 分钟。数字全对, 只是全是过去的。
  //
  // 阈值 180 秒: 正常直播 stale 约 70-100 秒 (数据源自身就落后 70 秒),
  // 所以它在正常情况下不会出现。暂停时也会触发, 但那时"数据落后 N 分钟"
  // **依然是事实** —— 这一行只陈述数据有多旧, 不判断比赛在不在打, 所以
  // 不存在误报。判死活是 stalled 的事, 两者分开。
  if (lv?.stalled) {
    meta.push(T("数据停留在 {n} 分钟前", { n: Math.round(lv.stale_seconds / 60) }));
  } else if (lv && lv.game_state !== "finished" && lv.stale_seconds >= 180) {
    // 打完的局不说: 最后一帧当然越来越旧, 回看第 2 局时写"数据落后约 75 分钟"没有任何意义
    meta.push(T("数据落后约 {n} 分钟", { n: Math.round(lv.stale_seconds / 60) }));
  }
  // hasEarly 时不写 —— 那一段**已经开打了** (只是局内模型还没启用),
  // 写"尚未开始"是错的, 而"第几分钟"上面那一屏自己会说。
  if (!hasLive && !hasEarly) {
    // 同样**不看 match.state**: 实测 2026-09-04 LPL LGD vs AL 打到第 3 局
    // (1-1 的 BO5) 而 state 已是 completed —— 局间休息时这里就会报"已结束",
    // 而系列赛根本没打完。按比分对 BO 算, 和后端 _clinched 同一套。
    const bo = data?.match?.best_of ?? null;
    const w = (data?.match?.teams ?? []).map((t) => t.game_wins ?? 0);
    const nd = bo != null ? Math.floor(bo / 2) + 1 : null;
    const seriesOver = nd != null && w.some((x) => x >= nd);
    // 三种情况要分开: 打完了 / 打了一半正在换局 / 真的还没开始。
    // 中间那种以前也会被写成"尚未开始", 而那时系列赛已经打了两局。
    meta.push(
      T(seriesOver ? "已结束" : played.length > 0 ? "局间休息" : "尚未开始"),
    );
  }
  // 已经开打但局内模型还没启用: 报第几分钟, 让人知道"在等什么"
  if (hasEarly && pr?.minute != null) {
    meta.push(T("第 {n} 分钟", { n: pr.minute }));
  }

  // 结束/暂停/中断时, 那句 "第 37 分钟 · 势均力敌" 里的局势描述已经没有意义 ——
  // 直接换成状态本身。
  //
  // **不去切后端返回的 summary 字符串**: 那句话的格式是后端的自由, 按 "·"
  // 拆开再换掉最后一段, 哪天它多加一节就悄悄取错。这里改成用 pr.minute
  // (独立字段) 自己拼一句, 完全不依赖文案格式。
  const done = !!lv && (lv.stalled || lv.game_state !== "in_game");
  const summaryText =
    done && pr?.minute != null
      ? `${T("第 {n} 分钟", { n: pr.minute })} · ${stateLabel}`
      : (pr?.summary ?? null);

  // 走势图末端画什么。**"结束"和"暂停"必须分开** —— 暂停时这一局还没分
  // 胜负, 画奖杯就是在宣布一个还没发生的结果。帧流中断同理: 数据断了不等于
  // 比赛完了, 那恰恰是最不该给出结论的时候。
  const endState: "live" | "paused" | "finished" = !lv
    ? "live"
    : lv.game_state === "finished"
      ? "finished"
      : lv.stalled || lv.game_state === "paused"
        ? "paused"
        : "live";

  const footnotes = [
    card?.note,
    card?.range_note,
    card?.trained_through && T("训练数据截至 {date}", { date: card.trained_through }),
    pr?.disclaimer,
  ].filter(Boolean);

  return (
    <div>
      <div className="board-nav rise">
        <button className="btn" onClick={onBack}>
          {T("← 返回")}
        </button>
        {/* 局切换居中放在顶栏。用两侧各一个 spacer 而不是 justify-content,
            这样标签组是相对**整行**居中的, 不会因为左右按钮宽度不同 (中英文
            混排时经常不同) 而偏一点。 */}
        <span className="spacer" />
        {/* 标签**不再挂在 hasLive 下**: 局间休息 (上一局结束、下一局还在 BP)
            和开赛前都没有局内帧, 而那恰恰是最需要知道"打到第几局了"的时候。
            原来这时整排标签会整个消失。 */}
        {showTabs ? (
          <div className="tabs nav-tabs">
            {shownGames.map((g) => (
              <button
                key={g.game_id}
                className={`tab${activeGameId === g.game_id ? " on" : ""}`}
                // 点**最新**那一局 = 跟随直播: gameId 置空后由服务端的
                // current_game() 挑当前局, 于是下一局开打时页面自己跟过去。
                // 点更早的局才是钉住 —— 用户明确要回看那一局, 就不该被
                // 新开的一局拽走。
                onClick={() => setGameId(g.game_id === latestGameId ? null : g.game_id)}
              >
                {T("第 {n} 局", { n: g.number ?? "?" })}
              </button>
            ))}
          </div>
        ) : null}
        <span className="spacer" />
        <ThemeToggle />
      </div>

      <header className="hero rise d1">
        {/* 大比分放在两队之间, 取代原来那个 "vs"。
            数字来自 teams[].game_wins, 而 teams 已经**按本局选边重排过**
            ([0] 永远是蓝方) —— 所以左边这个数就是左边这支队的, 不会串。
            没开打 (game_wins 全为 0 或缺失) 时退回 "vs", 免得一进页面先看到
            一个毫无意义的 0–0。 */}
        <div className="hero-teams">
          <h2>{blueName}</h2>
          {teams[0]?.game_wins != null &&
          teams[1]?.game_wins != null &&
          (teams[0].game_wins > 0 || teams[1].game_wins > 0) ? (
            <span className="series-score">
              <b className={teams[0].game_wins > teams[1].game_wins ? "lead" : ""}>
                {teams[0].game_wins}
              </b>
              <i>–</i>
              <b className={teams[1].game_wins > teams[0].game_wins ? "lead" : ""}>
                {teams[1].game_wins}
              </b>
            </span>
          ) : (
            <span className="vs">vs</span>
          )}
          <h2>{redName}</h2>
        </div>
        {meta.length > 0 && (
          <div className="hero-meta">
            {meta.map((m, i) => (
              <span key={i}>
                {i > 0 && <span className="dot">· </span>}
                {m}
              </span>
            ))}
          </div>
        )}
        {!hasLive && !hasEarly && !meta.length && loading && (
          <div className="hero-meta">{T("加载中…")}</div>
        )}

        {/* Stage 2: BP 后概率。刻度和局内那条**用同一个组件**, 于是从 BP 后
            切到局内时数字是在原地变化, 而不是换了一屏。
            pMin/pMax 给 null —— too_early 没有 model, 没有区间可标, 组件
            本来就允许为空 (刻度不画死区)。 */}
        {hasEarly && (
          <>
            <ProbabilityScale
              blueName={blueName}
              redName={redName}
              probabilityBlue={earlyP!}
              pMin={null}
              pMax={null}
              pregame={pr!.pregame_probability_blue}
              postdraft={postP}
              reasons={pr!.reasons}
            />
            <div className="summary-row">
              <p className="summary">
                {T(earlyIsDraft ? "BP 后" : "赛前")} ·{" "}
                {T("{team} 占优", { team: earlyP! >= 0.5 ? blueName : redName })}
              </p>
              <p className="shift">
                {T(earlyIsDraft ? "已计入双方阵容, 局内模型尚未启用" : "尚未开打")}
              </p>
            </div>
          </>
        )}

        {hasLive && (
          <>
            <ProbabilityScale
              blueName={blueName}
              redName={redName}
              probabilityBlue={pr!.probability_blue!}
              pMin={card!.p_min}
              pMax={card!.p_max}
              pregame={pr!.pregame_probability_blue}
              postdraft={pr!.postdraft_probability_blue}
              reasons={pr!.reasons}
            />
            {/* 局势描述和"相比赛前移动了多少"排在同一行, 左右对齐。
                两句讲的是同一件事的两面 (此刻怎么样 / 相对开局变了多少),
                原来上下堆着中间还留一段空白, 读起来像两个不相干的段落。 */}
            {(summaryText || pr!.shift_note) && (
              <div className="summary-row">
                {summaryText && <p className="summary">{summaryText}</p>}
                {pr!.shift_note && <p className="shift">{pr!.shift_note}</p>}
              </div>
            )}
          </>
        )}
      </header>

      {/* 赛前视图 (Stage 1): **只在真的还没开打时**出现。
          一旦有了 BP 后概率 (hasEarly), 上面那一屏已经把当前概率讲完了,
          再叠一个赛前视图就是同一件事说两遍, 而且两个数还不一样。 */}
      {!hasLive && !hasEarly && pre && (
        <PreMatch data={pre} blueName={blueName} redName={redName} />
      )}

      <div className="stack">
        {/* side_warning **不显示**: 那是"实时帧和赛程接口对本局选边说法不一"
            的提示。页面上所有数字本来就统一按实时帧分组 (见 CLAUDE.md 的
            frames-vs-persisted-sides), 实时帧就是权威源, 两者打架时按帧走
            已经是正确行为 —— 再弹一个黄框只是把一个内部实现细节推给读者。
            代价: 转播画面偶尔会把左右两边显示反, 此时页面不再解释原因。
            后端字段保留, 想恢复只要把它加回这里。

            lv.note 保留 —— 那是**信号本身**出问题 (帧流断了之类),
            影响每个数字对不对得上, 必须显眼。 */}
        {error && <div className="panel pad err">{error}</div>}
        {lv?.note && <div className="warn">{lv.note}</div>}

        {/* hasEarly 时不显示 —— 那句 note 讲的是"局内模型还没启用", 而上面
            的 shift 已经说过同一件事了。 */}
        {!hasLive && !hasEarly && !pre && (
          <div className="panel pad empty">
            {loading ? T("加载中…") : (pr?.note ?? pr?.error ?? T("暂无预测数据"))}
          </div>
        )}

        {/* 记分板放在走势图**上面**: 先看清"现在场上是什么样", 再看
            "它是怎么走到这一步的"。原来摆在最底下, 要一路滚到头才看得到
            阵容和补刀。 */}
        {hasFrames && (
          <div className="rise d2">
            <Scoreboard
              // 换局就整块重挂: 数字补间只该在同一局里滚, 不能从上一局的数滚到这一局
              key={lv!.game_id}
              players={lv!.players}
              lanes={lv!.lanes}
              blue={lv!.teams.blue}
              red={lv!.teams.red}
              blueName={blueName}
              redName={redName}
            />
          </div>
        )}

        {hasLive && shown && timeline.length > 0 && (
          <>
            <section className="panel pad rise d2">
              <h3 className="card-title">{T("胜率走势")}</h3>
              <WinChart
                series={timeline}
                pMin={card!.p_min}
                pMax={card!.p_max}
                blueName={blueName}
                redName={redName}
                pregame={pr!.pregame_probability_blue}
                postdraft={pr!.postdraft_probability_blue}
                endState={endState}
              />
            </section>
            <section className="panel pad rise d3">
              <h3 className="card-title">{T("经济差")}</h3>
              <GoldChart series={timeline} blueName={blueName} redName={redName} />
            </section>
          </>
        )}

        {/* AI 复核。要能跑必须凑齐: 队名映射得上 + 有当前局面。
            champs 从选手表里取本局英雄 —— 后端靠它算阵容强势期。
            静态版 (GitHub Pages) 没有它: 要调 NVIDIA 的 API, key 不能放进网页。 */}
        {(() => {
          if (IS_STATIC || !hasLive || !bm || !rm || !league) return null;
          const champs = (side: "blue" | "red") =>
            lv!.players.filter((p) => p.side === side).map((p) => p.champion).filter(Boolean) as string[];
          const bc = champs("blue");
          const rc = champs("red");
          const body: DebateRequestBody = {
            blue_team: bm,
            red_team: rm,
            league,
            ingame: {
              minute: pr!.minute,
              golddiff: lv!.teams.blue.gold - lv!.teams.red.gold,
              xpdiff: null,
              csdiff: lv!.teams.blue.cs - lv!.teams.red.cs,
              blue_kills: lv!.teams.blue.kills,
              red_kills: lv!.teams.red.kills,
              gold_total: lv!.teams.blue.gold,
            },
            blue_champions: bc.length === 5 ? bc : null,
            red_champions: rc.length === 5 ? rc : null,
            players: lv!.players.map((p) => ({
              side: p.side,
              role: p.role,
              summoner_name: p.summoner_name,
              champion: p.champion,
              level: p.level,
              kills: p.kills,
              deaths: p.deaths,
              assists: p.assists,
              cs: p.cs,
              gold: p.gold,
              damage_share: p.damage_share,
            })),
            lanes: lv!.lanes,
          };
          return <DebatePanel body={body} resetKey={`${matchId}:${lv!.game_id}`} />;
        })()}

        {/* 模型的局限**不删, 只是默认收起**。这个项目的立身之本就是把
            不确定性讲清楚 (见 README), 但界面不该一上来就糊一脸术语。
            想看的人点开就有, 不看的人也不会被误导成"这数字很确定"。 */}
        {hasLive && (footnotes.length > 0 || pr!.warnings.length > 0) && (
          <details className="fineprint">
            <summary>
              {T("准确率约")}{" "}
              {card!.overall_accuracy != null
                ? `${Math.round(card!.overall_accuracy * 100)}%`
                : "—"}{" "}
              · {T("模型说明")}
              {pr!.warnings.length > 0 && T(" 与 {n} 条提醒", { n: pr!.warnings.length })}
            </summary>
            {/* 两类说明合到一处: 上面原本还有一块"N 条模型说明"的折叠框, 和
                这里讲的是同一件事 (模型的局限), 分成两处只是让人多点一次。
                提醒在前 —— 它们是**针对这一次预测**的 (切片外推、缺了某个
                特征), 比通用的准确率说明更该先看到。 */}
            {pr!.warnings.length > 0 && (
              <ul className="fine-warns">
                {pr!.warnings.map((w, i) => (
                  <li key={i}>{w}</li>
                ))}
              </ul>
            )}
            {footnotes.length > 0 && <p className="mono">{footnotes.join(" · ")}</p>}
          </details>
        )}
      </div>
    </div>
  );
}
