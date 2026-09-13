"""
回归测试 —— 每一条都对应一个真实踩过的坑
==========================================

这个项目出过的 bug 几乎都是同一类: **不抛异常, 只是给出看着合理的错数字**。
选边标反、时钟偏 32 秒、大整数被当浮点、曲线和头条用了不同输入 —— 全都是
靠对照独立真值才发现的, 靠看代码看不出来。

所以这里测的不是"函数会不会崩", 而是"函数会不会静默地算错"。每个测试都
注明它守的是哪个坑, 改动时如果红了, 先想清楚那个坑是不是又回来了。

用标准库 unittest, 不引 pytest —— 这个项目连前端都没有构建步骤, 不该为
测试加依赖。

跑: python -m unittest discover -s tests -v
"""
from __future__ import annotations
import sys
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "research"))


class TestClinched(unittest.TestCase):
    """坑: BO3 打到 1-0 时整场比赛从直播列表里消失。

    原因是用「赢过至少一局」判系列赛结束, 而 1-0 的 BO3 恰恰是最常见的
    直播状态。必须按赛制算需要几胜。
    """

    def _match(self, best_of, wins):
        import esports_feed as EF
        return EF.Match(
            match_id="x", league="LPL", start_time="", state="inProgress",
            best_of=best_of,
            teams=[EF.TeamRef(api_name="A", code="A", game_wins=wins[0]),
                   EF.TeamRef(api_name="B", code="B", game_wins=wins[1])])

    def test_bo3_at_1_0_is_not_over(self):
        from api import _clinched
        self.assertFalse(_clinched(self._match(3, (1, 0))),
                         "BO3 打到 1-0 还没结束 —— 这正是最常见的直播状态")

    def test_bo3_at_2_0_is_over(self):
        from api import _clinched
        self.assertTrue(_clinched(self._match(3, (2, 0))))

    def test_bo5_at_2_1_is_not_over(self):
        from api import _clinched
        self.assertFalse(_clinched(self._match(5, (2, 1))))

    def test_bo5_at_3_2_is_over(self):
        from api import _clinched
        self.assertTrue(_clinched(self._match(5, (3, 2))))

    def test_bo1_at_1_0_is_over(self):
        from api import _clinched
        self.assertTrue(_clinched(self._match(1, (1, 0))))


class UpdateStatusRecord(unittest.TestCase):
    """坑: 日更失败是完全静默的, 两次都是用户肉眼发现数据旧了才查出来。

    gate 保住了线上模型 (拉取失败就不往下走), 于是"坏了"和"没事"从外部看
    一模一样。留档就是为了让这件事在 /health 上看得见。
    """

    def setUp(self):
        import tempfile
        import update_status
        self.us = update_status
        self._orig = update_status.STATUS_FILE
        self.tmp = tempfile.TemporaryDirectory()
        update_status.STATUS_FILE = __import__("pathlib").Path(self.tmp.name) / "s.json"

    def tearDown(self):
        self.us.STATUS_FILE = self._orig
        self.tmp.cleanup()

    def test_带BOM的留档也要读得出来(self):
        """PowerShell 的 `Out-File -Encoding utf8` 是**带 BOM** 的。

        用 utf-8 读会让 json.loads 抛异常, 被兜底吞成"没有留档" —— 一个
        专门用来暴露故障的文件, 自己变成了静默故障。这台机器上 PS 的 BOM
        坑已经咬过好几次 (改 .ps1、改 MEMORY.md)。
        """
        import json
        payload = {"last_run": "2026-09-04T05:00:00", "outcome": "failed",
                   "last_fetch_ok": None, "consecutive_failures": 3,
                   "last_failure": {"stage": "拉取数据", "reason": "OAuth 过期"}}
        raw = json.dumps(payload, ensure_ascii=False)
        self.us.STATUS_FILE.write_bytes(b"\xef\xbb\xbf" + raw.encode("utf-8"))
        self.assertEqual(self.us.read().get("consecutive_failures"), 3,
                         "带 BOM 就读不出来 —— 故障会被当成'没有留档'")
        self.assertIn("OAuth", self.us.summary()["message"])

    def test_没有留档时不报错(self):
        """/health 在旧版本或全新机器上必须照常应答。"""
        self.assertEqual(self.us.read(), {})
        s = self.us.summary()
        self.assertEqual(s["level"], "unknown")

    def test_失败不能擦掉上次成功的时间(self):
        """那正是判断流程死活最需要的信息。"""
        self.us.record("ok", fetch_ok=True)
        first = self.us.read()["last_success"]
        self.us.record("failed", stage="拉取数据", reason="invalid_grant")
        s = self.us.read()
        self.assertEqual(s["last_success"], first,
                         "一失败就把上次成功时间擦了, 等于不知道坏了多久")
        self.assertEqual(s["last_fetch_ok"], first)

    def test_连续失败会累加成功后归零(self):
        for _ in range(3):
            self.us.record("failed", stage="拉取数据", reason="x")
        self.assertEqual(self.us.read()["consecutive_failures"], 3)
        self.us.record("ok", fetch_ok=True)
        self.assertEqual(self.us.read()["consecutive_failures"], 0)

    def test_失败原因在成功后仍然留着(self):
        """用来回答"上次是怎么坏的"。"""
        self.us.record("failed", stage="指标过闸", reason="准确率掉太多")
        self.us.record("ok", fetch_ok=True)
        self.assertEqual(self.us.read()["last_failure"]["stage"], "指标过闸")

    def test_数据没变化也算拉取成功(self):
        """**这条是那次 OAuth 故障的教训。**

        "上游没有新数据"和"我们根本没连上上游"结局都是不重训。只记一个布尔量
        的话两者混在一起, 而判级必须看 last_fetch_ok 才分得开。
        """
        self.us.record("ok", fetch_ok=True)      # 数据没变化那条路径
        self.assertIsNotNone(self.us.read()["last_fetch_ok"])
        self.assertEqual(self.us.summary()["level"], "fresh")

    def test_拉取失败不刷新_last_fetch_ok(self):
        self.us.record("ok", fetch_ok=True)
        ok_at = self.us.read()["last_fetch_ok"]
        self.us.record("failed", stage="拉取数据", reason="invalid_grant", fetch_ok=False)
        self.assertEqual(self.us.read()["last_fetch_ok"], ok_at)

    def test_判级按拉取时间而不是整轮成功(self):
        """训练失败但拉取成功时, 数据链路是通的, 不该报成"很久没拉到数据"。"""
        self.us.record("failed", stage="训练 (train.py)", reason="boom", fetch_ok=True)
        s = self.us.summary()
        self.assertEqual(s["level"], "fresh")
        self.assertIn("连续失败", s["message"])
        self.assertIn("训练", s["message"])

    def test_超过阈值判成坏了(self):
        import json
        from datetime import datetime, timedelta
        old = (datetime.now() - timedelta(hours=100)).strftime(self.us._FMT)
        self.us.STATUS_FILE.write_text(json.dumps({
            "last_run": old, "outcome": "failed", "last_success": old,
            "last_fetch_ok": old, "consecutive_failures": 7,
            "last_failure": {"time": old, "stage": "拉取数据",
                             "reason": "Google OAuth token 已过期"},
        }), encoding="utf-8")
        s = self.us.summary()
        self.assertEqual(s["level"], "very_stale")
        self.assertEqual(s["consecutive_failures"], 7)
        self.assertIn("OAuth", s["message"])

    def test_oauth失败会被翻译成能照做的一句话(self):
        """留档里塞三十行 google.auth 的 traceback 等于没记。"""
        import daily_update
        out = ("Traceback (most recent call last):\n  ...\n"
               "google.auth.exceptions.RefreshError: "
               "('invalid_grant: Token has been expired or revoked.')")
        hint = daily_update._oauth_hint(out)
        self.assertIsNotNone(hint)
        self.assertIn("--auth --force", hint)

    def test_pipeline_每个出口都返回四元组(self):
        """有十个 return、五种退出码 —— 漏掉一个, 那种失败就永远不被记录。"""
        import ast
        import inspect
        import textwrap
        import daily_update
        src = inspect.getsource(daily_update._pipeline)
        tree = ast.parse(textwrap.dedent(src))
        rets = [n for n in ast.walk(tree) if isinstance(n, ast.Return)]
        self.assertGreaterEqual(len(rets), 9)
        for r in rets:
            self.assertIsInstance(r.value, ast.Tuple, "return 必须是四元组")
            self.assertEqual(len(r.value.elts), 4,
                             "(退出码, 停在哪一步, 原因, 拉取是否成功)")


class SeriesScoreFreshness(unittest.TestCase):
    """坑: BO3 已经 2-1 打完, 列表里还挂着 1-1。

    系列赛比分只有一个来源 (result.gameWins), 但它在两个端点里都有:
    getSchedule 走 schedule_ttl=300, getEventDetails 走 live_ttl=20。
    一局刚结束时读前者, 旧比分会被焊死五分钟。

    连带的第二个症状: /esports/live 的"局间休息"那一支靠比分判断系列赛
    有没有结束, 读到旧比分就判不出来, 于是**打完的比赛赖在直播列表里**。
    """

    def _feed(self, wins):
        """只桩掉 HTTP 那一层, match_wins 的解析逻辑照常跑。"""
        from esports_feed import EsportsFeed
        payload = {"data": {"event": {"match": {"teams": [
            {"name": "Team A", "code": "TA", "result": {"gameWins": wins[0]}},
            {"name": "Team B", "code": "TB", "result": {"gameWins": wins[1]}},
        ]}}}}
        f = EsportsFeed()
        f._sess = type("S", (), {
            "get": lambda self, url, **kw: type("R", (), {
                "status_code": 200, "json": lambda self: payload})()})()
        return f

    def _match(self, best_of, wins):
        import esports_feed as EF
        return EF.Match(
            match_id="x", league="LEC", start_time="", state="inProgress",
            best_of=best_of,
            teams=[EF.TeamRef(api_name="Team A", code="TA", game_wins=wins[0]),
                   EF.TeamRef(api_name="Team B", code="TB", game_wins=wins[1])])

    def test_比分按队名取得回来(self):
        f = self._feed((2, 1))
        w = f.match_wins("x")
        self.assertEqual(w["teama"], 2)
        self.assertEqual(w["teamb"], 1)
        self.assertEqual(w["ta"], 2, "缩写也要能查 —— 两边偶尔只有一个对得上")

    def test_旧比分会被新的覆盖(self):
        from api import _refresh_wins
        m = self._match(3, (1, 1))          # schedule 给的旧值
        self.assertTrue(_refresh_wins(m, self._feed((2, 1))))
        self.assertEqual([t.game_wins for t in m.teams], [2, 1],
                         "决胜局打完后比分必须跟上, 否则列表停在 1-1")

    def test_取不到比分时不许把已有的抹掉(self):
        """显示旧比分, 好过显示 None —— 更好过显示错的。"""
        from api import _refresh_wins
        from esports_feed import EsportsFeed
        m = self._match(3, (1, 1))
        broken = EsportsFeed()
        broken._sess = type("S", (), {
            "get": lambda self, url, **kw: (_ for _ in ()).throw(RuntimeError("上游挂了"))})()
        self.assertFalse(_refresh_wins(m, broken))
        self.assertEqual([t.game_wins for t in m.teams], [1, 1])

    def test_比分刷新后已结束的系列赛判得出来(self):
        """这是"打完了还赖在直播列表里"的直接成因。"""
        from api import _clinched, _refresh_wins
        m = self._match(3, (1, 1))
        self.assertFalse(_clinched(m), "旧比分 1-1 判不出结束 —— 正是它让比赛赖着")
        _refresh_wins(m, self._feed((2, 1)))
        self.assertTrue(_clinched(m), "刷新后 2-1 必须判成已结束")

    def test_getEventDetails_用的是短缓存(self):
        """两个端点差 15 倍, 这就是整个修复的立足点。"""
        from esports_feed import EsportsFeed
        f = EsportsFeed()
        self.assertEqual(f.live_ttl, 20)
        self.assertEqual(f.schedule_ttl, 300)
        f2 = self._feed((2, 1))
        f2.match_wins("x")
        url = [k for k in f2._cache if "getEventDetails" in k]
        self.assertTrue(url, "match_wins 必须走 getEventDetails")
        expiry, _ = f2._cache[url[0]]
        self.assertLess(expiry - time.time(), 60,
                        "比分走了长缓存就等于没修")


class AdaptiveLag(unittest.TestCase):
    """实时画面落后多少, 基本等于我们减掉的那个 lag。

    窗口实测是以 startingTime 为中心的 (帧覆盖 T-6s..T+3s), 所以 lag=60
    就是画面落后约 60 秒 —— 那 60 不是上游的真实发布延迟, 只是当年测出来
    能稳定命中的保守值。自适应的作用是把它降到**这条转播实际能给到**的档位,
    而且只在真取到帧时才降。
    """

    def _feed(self, publish_lag):
        """桩: 只有 lag >= publish_lag 的窗口才有帧, 更近的一律 204。

        publish_lag 就是"上游提前多久发布"。记下每次问到的档位, 以便断言
        到底发了几个请求。
        """
        from datetime import datetime, timezone
        from esports_feed import EsportsFeed
        f = EsportsFeed(no_frames_ttl=0)
        f.asked = []

        def frame(ts):
            side = {"totalGold": 0, "totalKills": 0, "participants": []}
            return {"rfc460Timestamp": ts, "gameState": "in_game",
                    "blueTeam": dict(side), "redTeam": dict(side)}

        def get(_self, url, **kw):
            if "startingTime=" not in url:
                # 不带 startingTime = 问开局帧, 恒有 (见模块顶部第 3 条)
                ts = "2026-08-30T00:00:00Z"
                return type("R", (), {"status_code": 200,
                                      "json": lambda self: {"frames": [frame(ts)]}})()
            ts = url.split("startingTime=")[1]
            t = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            lag = (datetime.now(timezone.utc) - t).total_seconds()
            f.asked.append(round(lag / 10) * 10)
            ok = lag >= publish_lag
            return type("R", (), {
                "status_code": 200 if ok else 204,
                "json": lambda self: ({"frames": [frame(ts)]} if ok else None)})()

        f._sess = type("S", (), {"get": get})()
        return f

    def test_默认从60起步(self):
        f = self._feed(publish_lag=60)
        self.assertEqual(f._lag.get("g", 60), 60)

    def test_上游给得起就会降档(self):
        """核心行为: 上游其实 20 秒就发布, lag 应该一路降到 20。"""
        from esports_feed import EsportsFeed
        f = self._feed(publish_lag=20)
        for _ in range(8):
            f._lag_probe.clear()          # 免去等 lag_probe_sec
            f.window("g")
        self.assertEqual(f._lag["g"], 20,
                         "上游 20 秒就有数据, 却还在减 60 秒 —— 白白落后 40 秒")

    def test_降到上游给不了的档就停住(self):
        """上游 50 秒才发布时, 不能降到 40 —— 那会取不到帧。"""
        f = self._feed(publish_lag=50)
        for _ in range(8):
            f._lag_probe.clear()
            f.window("g")
        self.assertEqual(f._lag["g"], 50,
                         "降过头了, 会把有帧的档换成空档")

    def test_不试探时每轮只发一个请求(self):
        """试探要额外花一个请求, 所以必须限频, 否则热路径请求数翻倍。"""
        f = self._feed(publish_lag=60)
        f.window("g")                      # 第一轮会试探
        f._cache.clear()
        n0 = len(f.asked)
        f.window("g")                      # 紧接着的一轮不该再试探
        self.assertEqual(len(f.asked) - n0, 1,
                         "限频没生效 —— 每轮都多发一个请求")

    def test_历史查询不参与自适应(self):
        """问过去的某个确定时刻没有"发布延迟"可言, 掺进来只会弄脏状态。"""
        from datetime import datetime, timedelta, timezone
        f = self._feed(publish_lag=20)
        f.window("g", at=datetime.now(timezone.utc) - timedelta(hours=2))
        self.assertNotIn("g", f._lag)

    def test_整条链都空时退回默认档(self):
        """否则这一局会永远卡在一个取不到帧的档上。"""
        f = self._feed(publish_lag=10_000)   # 怎么问都没有
        f._lag["g"] = 30
        self.assertIsNone(f.window("g"))
        self.assertNotIn("g", f._lag)


class TestLagged(unittest.TestCase):
    """坑: startingTime 不对齐 10 秒就返回 204 (空 body, 不是报错)。"""

    def test_floors_to_ten_seconds_then_subtracts(self):
        from esports_feed import EsportsFeed
        at = datetime(2026, 8, 20, 12, 34, 57, 123456, tzinfo=timezone.utc)
        got = EsportsFeed._lagged(60, at)
        # 57 秒向下取整到 50, 再减 60 秒 -> 12:33:50
        self.assertEqual(got, "2026-08-20T12:33:50Z")

    def test_always_on_ten_second_boundary(self):
        from esports_feed import EsportsFeed
        base = datetime(2026, 8, 20, 12, 0, 0, tzinfo=timezone.utc)
        for s in range(0, 60):
            ts = EsportsFeed._lagged(60, base + timedelta(seconds=s))
            self.assertEqual(int(ts[17:19]) % 10, 0,
                             f"秒数必须是 10 的倍数, 得到 {ts}")


class TestTeamMapping(unittest.TestCase):
    """坑: difflib 把 "Beijing JDG Esports" 匹配成了 "LNG Esports"。

    看着合理, 是另一支队。队名错了不报错, 只会拿错队伍的历史算特征。
    """

    def test_known_alias_maps(self):
        from esports_feed import map_team
        self.assertEqual(map_team("Beijing JDG Esports"), "JD Gaming")

    def test_unknown_returns_none_not_a_guess(self):
        from esports_feed import map_team
        known = ["JD Gaming", "LNG Esports", "Top Esports"]
        self.assertIsNone(map_team("Beijing JDG Espor", known),
                          "拼错的队名必须返回 None, 绝不能模糊匹配到别的队")

    def test_tbd_is_none(self):
        from esports_feed import map_team
        self.assertIsNone(map_team("TBD"))


class TestPairedT(unittest.TestCase):
    """坑: 配对 SD 为零那个分支被抄漏过三次, 导致「每折都同向」的稳定效应
    被判成噪声。全项目只能有这一份算术。
    """

    def test_identical_folds_give_zero_t(self):
        from harness import paired_t
        d, sd, t, ratio = paired_t([0.1, 0.2, 0.3], [0.1, 0.2, 0.3])
        self.assertEqual(t, 0.0)

    def test_constant_positive_shift_is_infinite_evidence(self):
        from harness import paired_t
        # 每折都恰好 +0.05 -> 不确定性为零, 证据拉满
        d, sd, t, ratio = paired_t([0.1, 0.2, 0.3], [0.15, 0.25, 0.35])
        self.assertAlmostEqual(d, 0.05, places=9)
        self.assertEqual(t, float("inf"),
                         "逐折完全同向时 t 应当是 +inf, 不是 0")

    def test_sign_follows_direction(self):
        from harness import paired_t
        _, _, t_up, _ = paired_t([0.1, 0.2], [0.3, 0.5])
        _, _, t_dn, _ = paired_t([0.3, 0.5], [0.1, 0.2])
        self.assertGreater(t_up, 0)
        self.assertLess(t_dn, 0)


class TestClockCalibration(unittest.TestCase):
    """坑: 分钟数系统性偏大 32 秒。

    原因是按"小兵出生 1:05 才有金币收入"填了 FIRST_INCOME_SEC=65, 而实测
    (拿 OE 的精确局长当真值) 第一次金币增长在约 33 秒。这个常数改回 65
    会让线上分钟数再次偏大半分钟, 切片选择在边界上就会选错一档。
    """

    def test_first_income_constant_is_calibrated_not_guessed(self):
        from esports_feed import EsportsFeed
        self.assertEqual(EsportsFeed.FIRST_INCOME_SEC, 33,
                         "这个常数是拿 OE 局长标定出来的, 不是按游戏机制推的")
        self.assertEqual(EsportsFeed.START_GOLD, 2500)

    def test_calibration_guard_rejects_absurd_zero_points(self):
        """算出来的零点离 frames[0] 太远时必须放弃, 不能瞎调。"""
        from esports_feed import EsportsFeed
        f = EsportsFeed.__new__(EsportsFeed)      # 不跑 __init__, 只测纯逻辑
        raw = datetime(2026, 8, 20, 12, 0, 0, tzinfo=timezone.utc)
        # 伪造一个"第一次金币增长"发生在 raw 之后 10 分钟的荒谬情况
        f._window = lambda *a, **k: {"frames": [{
            "rfc460Timestamp": (raw + timedelta(minutes=10)).strftime(
                "%Y-%m-%dT%H:%M:%S.000Z"),
            "blueTeam": {"totalGold": 9999}, "redTeam": {"totalGold": 9999}}]}
        f._lagged = EsportsFeed._lagged
        f._ts = EsportsFeed._ts
        # 不设的话 grab 里读 window_ttl 抛 AttributeError, 被 _calibrate_start 的
        # except 吞成 None —— 测试照样绿, 测的却不是合理性闸
        f.window_ttl = 3
        got = f._calibrate_start("g", raw)
        self.assertIsNone(got, "零点算到 frames[0] 之后 9 分钟, 必须拒绝")


class TestGameStartRetry(unittest.TestCase):
    """坑: 第一次看到一局就校准开局零点, 那时要用的窗口还没发布, 校准失败退回
    frames[0] —— 而这个退回值和成功结果一样进了长缓存, 整局分钟数偏 8~20 秒,
    采集快照的分钟标签跟着偏 (2026-09-12: 在跑的服务说第 36 分钟, 全新进程说 35)。
    浏览器版 frontend/src/web/feed.ts 的 gameStart 是同一套逻辑, 改一边要改另一边。
    """

    def _feed(self, raw, result):
        import threading
        from esports_feed import EsportsFeed
        f = EsportsFeed.__new__(EsportsFeed)
        f._starts, f._start_retry, f._obs = {}, {}, {"g": {0: ("旧零点下的观测",)}}
        f._lock = threading.Lock()
        f._window = lambda *a, **k: {"frames": [
            {"rfc460Timestamp": raw.strftime("%Y-%m-%dT%H:%M:%S.000Z")}]}
        f._ts = EsportsFeed._ts
        calls = []

        def cal(gid, r):
            calls.append(r)
            return result[0]
        f._calibrate_start = cal
        return f, calls

    def test_young_game_failed_calibration_is_retried_not_cached(self):
        raw = (datetime.now(timezone.utc) - timedelta(seconds=90)).replace(microsecond=0)
        result = [None]
        f, calls = self._feed(raw, result)
        self.assertEqual(f._game_start("g"), raw, "失败时先用 frames[0] 顶着")
        self.assertNotIn("g", f._starts, "开局才 90 秒, 校准失败只说明窗口还没发布, 不能进长缓存")
        f._game_start("g")
        self.assertEqual(len(calls), 1, "失败后 START_RETRY_SEC 之内不该重算")

        f._start_retry["g"] = 0                    # 快进到可以重试
        result[0] = raw - timedelta(seconds=19)
        self.assertEqual(f._game_start("g"), raw - timedelta(seconds=19))
        self.assertEqual(f._starts["g"], raw - timedelta(seconds=19))
        self.assertNotIn("g", f._obs, "按旧零点分桶的暂停观测必须作废")

    def test_settled_game_failed_calibration_is_cached(self):
        raw = (datetime.now(timezone.utc) - timedelta(hours=1)).replace(microsecond=0)
        f, calls = self._feed(raw, [None])
        self.assertEqual(f._game_start("g"), raw)
        self.assertEqual(f._starts.get("g"), raw, "窗口早已定型还校不了, 就认定这一局校不了")
        f._game_start("g")
        self.assertEqual(len(calls), 1)


class TestChildProcessEncoding(unittest.TestCase):
    """坑: 子进程的 stdout 是管道时, 中文 Windows 上按 GBK 写, daily_update 按 UTF-8 读 ——
    fetch_data 打印的"已更新"读回来是乱码, 于是每次都判成"数据没有变化"。本机日更
    2026-09-04 起一次没重训过, 日志每轮都是成功。
    """

    def test_child_chinese_output_reads_back(self):
        enc = getattr(sys.stdout, "encoding", None)
        import daily_update                         # 导入时会把本进程 stdout 改成 UTF-8
        if enc and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding=enc)    # 改回来, 别弄乱测试输出
        rc, out = daily_update.run([sys.executable, "-c", "print('已更新')"], timeout=60)
        self.assertEqual(rc, 0)
        self.assertIn("已更新", out, "子进程的中文输出读回来必须还是中文")


class TestWebPublishStatus(unittest.TestCase):
    """坑: 网站发布是日更的旁路, 它的结果如果被整轮的 record() 冲掉, 网站停在旧模型上
    就又成了静默故障。"""

    def test_record_keeps_web_publish(self):
        import tempfile
        import update_status
        old = update_status.STATUS_FILE
        with tempfile.TemporaryDirectory() as d:
            update_status.STATUS_FILE = Path(d) / "s.json"
            try:
                update_status.record_web("failed", "对账没过")
                update_status.record("ok", fetch_ok=True, exit_code=0)
                s = update_status.summary()
                self.assertEqual((s.get("web_publish") or {}).get("outcome"), "failed")
                self.assertIn("网站", s.get("message") or "", "网站发布失败必须出现在 /health 的提示里")
            finally:
                update_status.STATUS_FILE = old


class TestLiveNamesMatchTraining(unittest.TestCase):
    """坑: 看板把 lolesports 的名字原样交给 Stage 2 / Stage 4 查表, 而训练只见过 OE 的写法 ——
    选手带战队前缀 (IGTheShy vs TheShy) 一个都查不到, 英雄用 Data Dragon id (LeeSin vs
    Lee Sin) 查不到 13%。不报错, 只是熟练度恒为 0、阵容强势期少算几个英雄。
    浏览器版 frontend/src/web/names.ts 是同一套规则, 由 test:golden 对账。
    """

    def test_champion_key_matches_ddragon_ids_to_oe_names(self):
        from feature_store import champion_key as k
        for ddid, oe in (("LeeSin", "Lee Sin"), ("KSante", "K'Sante"), ("MissFortune", "Miss Fortune"),
                         ("MonkeyKing", "Wukong"), ("Renata", "Renata Glasc"),
                         ("Nunu", "Nunu & Willump"), ("DrMundo", "Dr. Mundo"), ("Kaisa", "Kai'Sa"),
                         ("Leblanc", "LeBlanc"), ("JarvanIV", "Jarvan IV")):
            self.assertEqual(k(ddid), k(oe), ddid)
        self.assertNotEqual(k("Nunu"), k("Nami"))

    def test_store_lookup_accepts_both_spellings(self):
        import pandas as pd
        from feature_store import FeatureStore
        d = pd.Timestamp("2026-01-01")
        s = FeatureStore(champ_records={("Lee Sin", "LPL"): [(d, 1)] * 4 + [(d, 0)]},
                         player_champ={("Wei", "Lee Sin"): [(d, 1)] * 3})
        self.assertEqual(s.champ_winrate("LeeSin", "LPL"), 0.8, "看板传的是 Data Dragon id")
        self.assertEqual(s.player_champ_stats("Wei", "LeeSin"), (1.0, 3))
        self.assertEqual(s.champ_winrate("Lee Sin", "LPL"), 0.8, "训练传的 OE 名字必须照旧")
        import math
        self.assertTrue(math.isnan(s.champ_winrate("Azir", "LPL")), "认不出来的照旧查不到")

    def test_comp_scaling_accepts_ddragon_ids(self):
        from ingame_service import IngameModel
        m = IngameModel.__new__(IngameModel)
        m.scaling = {"Lee Sin": 1.0, "Wukong": 2.0, "K'Sante": 3.0, "Azir": 4.0}
        self.assertEqual(m.comp_scaling(["LeeSin", "MonkeyKing", "KSante", "Azir", "Unknown"]), (2.5, 4))

    def test_player_name_strips_only_the_match_team_codes(self):
        from esports_feed import oe_player_name as f
        self.assertEqual(f("IGTheShy", ["IG", "AL"]), "TheShy")
        self.assertEqual(f("TH Hype", ["TH", "G2"]), "Hype")
        self.assertEqual(f("T1Faker", ["T", "T1"]), "Faker", "长的简称优先")
        self.assertEqual(f("Faker", ["IG", "AL"]), "Faker", "不是前缀就原样返回")
        self.assertEqual(f("IG", ["IG"]), "IG", "名字整个就是简称时不剥成空串")
        self.assertEqual(f(None, ["IG"]), "")

    def test_board_draft_hands_oe_player_names_to_stage2(self):
        import api

        class Feed:
            def game_metadata(self, gid):
                out = {}
                for i, r in enumerate(["top", "jungle", "mid", "bottom", "support"]):
                    out[i + 1] = {"summoner_name": f"IGP{i}", "champion": "LeeSin", "role": r, "side": "blue"}
                    out[i + 6] = {"summoner_name": f"ALQ{i}", "champion": "Azir", "role": r, "side": "red"}
                return out

        class Team:
            def __init__(self, code):
                self.code = code

        class Minfo:
            teams = [Team("IG"), Team("AL")]

        d = api._board_draft(Feed(), "g", Minfo())
        self.assertEqual(d["blue"]["jng"], {"player": "P1", "champion": "LeeSin"})
        self.assertEqual(d["red"]["sup"]["player"], "Q4")


class TestBackfillJoin(unittest.TestCase):
    """坑: 只按蓝方队名 + 局号 + 最近日期配对, 同一天两场比赛会配错,
    标签张冠李戴且完全不报错。必须两队都对上。
    """

    def _oe(self):
        import pandas as pd
        return pd.DataFrame([
            {"gameid": "G1", "league": "LPL", "date": pd.Timestamp("2026-08-01 10:00"),
             "teamname": "A", "red_name": "B", "game": 1, "result": 1},
            {"gameid": "G2", "league": "LPL", "date": pd.Timestamp("2026-08-01 14:00"),
             "teamname": "A", "red_name": "C", "game": 1, "result": 0},
        ])

    def test_matches_when_both_teams_agree(self):
        from backfill_late import find_oe
        when = datetime(2026, 8, 1, 10, 30, tzinfo=timezone.utc)
        row = find_oe(self._oe(), "LPL", when, "A", "B", 1)
        self.assertIsNotNone(row)
        self.assertEqual(row["gameid"], "G1")

    def test_rejects_when_opponent_differs(self):
        from backfill_late import find_oe
        when = datetime(2026, 8, 1, 10, 30, tzinfo=timezone.utc)
        # A 当天打了两场, 但对手是 D 的那场 OE 里没有 -> 必须返回 None,
        # 绝不能退而求其次配到 A vs B 或 A vs C
        self.assertIsNone(find_oe(self._oe(), "LPL", when, "A", "D", 1))

    def test_rejects_when_too_far_in_time(self):
        from backfill_late import find_oe
        when = datetime(2026, 8, 5, 10, 0, tzinfo=timezone.utc)
        self.assertIsNone(find_oe(self._oe(), "LPL", when, "A", "B", 1))


class TestLeagueNameMapping(unittest.TestCase):
    """坑: lolesports 和 OE 的赛区名不一致, 对不上不报错, 只是一局都配不上。"""

    def test_mismatched_names_are_listed_explicitly(self):
        from backfill_late import LEAGUE_TO_OE, ALL_LEAGUES
        self.assertEqual(LEAGUE_TO_OE["LCK Challengers"], "LCKC")
        self.assertEqual(LEAGUE_TO_OE["EMEA Masters"], "EM")
        # 四大赛区名字本来就一致, 不该出现在映射表里
        for lg in ("LPL", "LCK", "LEC", "LCS"):
            self.assertNotIn(lg, LEAGUE_TO_OE)
            self.assertIn(lg, ALL_LEAGUES)


class TestIngameStateBounds(unittest.TestCase):
    """坑: minute=None 时整个响应变成一段 pydantic 报错文本。"""

    def test_none_minute_is_rejected(self):
        from api import IngameState
        with self.assertRaises(Exception):
            IngameState(minute=None, golddiff=0, xpdiff=None, csdiff=0,
                        blue_kills=0, red_kills=0, gold_total=1000)

    def test_lower_bound_is_three(self):
        """下界从 5 降到 3 是有意的 (曲线前几分钟要能动)。
        改回去会让第 3-5 分钟重新变成常量。"""
        from api import IngameState
        s = IngameState(minute=3, golddiff=0, xpdiff=None, csdiff=0,
                        blue_kills=0, red_kills=0, gold_total=1000)
        self.assertEqual(s.minute, 3)
        with self.assertRaises(Exception):
            IngameState(minute=2, golddiff=0, xpdiff=None, csdiff=0,
                        blue_kills=0, red_kills=0, gold_total=1000)

    def test_xpdiff_none_is_allowed(self):
        """帧里没有 XP 字段。传 None 走 live 变体, 绝不能拿 0 顶替 ——
        后者会让 explain 层输出「双方经验持平」, 把缺失讲成实测结论。"""
        from api import IngameState
        s = IngameState(minute=20, golddiff=1000, xpdiff=None, csdiff=0,
                        blue_kills=5, red_kills=3, gold_total=30000)
        self.assertIsNone(s.xpdiff)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class DebateFallback(unittest.TestCase):
    """守的坑: 免费额度降级时, 裁定这一步是单点故障。

    2026-08-20 用户在局中点 AI 复核, 拿到一句「裁定失败: Request timed out」——
    前三轮辩论全跑完了, 只因协调者 google/gemma-4-31b-it 没响应, 整场输出为零。
    当时实测 gemma 从筛选记录的 3.2 秒掉到「3 次挂 2 次、成功那次 26 秒」。

    这几条测的都是"降级时还能不能给出东西", 以及更隐蔽的一条:
    **拿到 JSON 不等于拿到裁定** —— 换上来的模型可能返回一个没有 consensus
    字段的壳, 那种"成功"会让链条停住, 面板照样空白。
    """

    def setUp(self):
        import debate as DB
        self.DB = DB
        DB._FAILS.clear()
        self.d = DB.Debate.__new__(DB.Debate)      # 不联网, 不建 client
        self.d.timeout = 90
        self.d.total_budget = 300
        self.d._deadline = time.monotonic() + 300
        self.d._reserve = 0
        self.calls = []

    def _stub(self, table):
        """table: model -> 返回文本 (None 表示失败)。"""
        def fake(model, system, user, max_tok, temp, tries=2, per_try=None):
            self.calls.append(model)
            out = table.get(model)
            return (out, None) if out else ("", "APITimeoutError")
        self.d._call = fake

    def test_主模型挂了会换模型裁定(self):
        DB = self.DB
        alt = DB.MODELS.get("质疑者", DB._DEFAULTS["质疑者"])
        self._stub({alt: '{"consensus": "蓝方领先成立", "claims": []}'})
        v = self.d._verdict("简报", "记录")
        self.assertEqual(v["consensus"], "蓝方领先成立")
        self.assertEqual(v.get("verdict_model"), alt,
                         "换过模型就必须标出来, 否则用户以为是主模型判的")
        self.assertGreater(len(self.calls), 1, "应当真的试过不止一个模型")

    def test_有claims没consensus不算裁定成功(self):
        """实测 gpt-oss-120b (json_score 0.0) 反复返回这种壳。

        它不能当成"裁定成功"就地返回 —— 面板的主读数就是那句结论。
        但也不能直接丢掉: claims 本身有用。所以要继续找, 找不到才回头用,
        且必须如实说明没有总结句, 绝不替它编一句。
        """
        DB = self.DB
        shell = '{"claims": [{"kind": "reinterpretation", "text": "野区被压"}]}'
        good = '{"consensus": "红方阵容更耐拖", "claims": []}'
        prim = DB.MODELS.get("协调者", DB._DEFAULTS["协调者"])
        alt = DB.MODELS.get("质疑者", DB._DEFAULTS["质疑者"])
        self._stub({prim: shell, alt: good})
        v = self.d._verdict("简报", "记录")
        self.assertEqual(v["consensus"], "红方阵容更耐拖",
                         "有结论句的模型应当胜出, 而不是停在只有 claims 的那个")

    def test_全都只给claims时保底且如实说明(self):
        DB = self.DB
        shell = '{"claims": [{"kind": "reinterpretation", "text": "野区被压"}]}'
        self._stub({m: shell for m in DB.MODELS.values()})
        v = self.d._verdict("简报", "记录")
        self.assertTrue(v["claims"], "claims 有用, 不该丢")
        self.assertIn("没有模型给出总结句", v["consensus"],
                      "缺了结论句必须写明, 不能编一句冒充")

    def test_熔断跳过刚挂过的模型(self):
        DB = self.DB
        dead = "dead/model"
        DB._FAILS[dead] = [DB._TRIP_AFTER, time.monotonic() + 300]
        self.assertTrue(DB._tripped(dead))
        # 熔断后 _call 必须立刻返回, 不占用重试时间
        real = DB.Debate._call
        txt, err = real(self.d, dead, "s", "u", 100, 0.2)
        self.assertEqual(txt, "")
        self.assertIn("熔断", err)

    def test_一次成功会清掉失败计数(self):
        DB = self.DB
        m = "some/model"
        DB._note(m, False)
        self.assertEqual(DB._FAILS[m][0], 1)
        DB._note(m, True)
        self.assertNotIn(m, DB._FAILS,
                         "降级会自己好, 恢复了就要能立刻用回来")


class SideOrdering(unittest.TestCase):
    """守的坑: 本局选边排错, 页面把 A 队的数字写在 B 队名下, 且不报错。

    2026-08-20 BLG vs LGD 第 2 局实测: 头部标 BLG 是蓝色方并显示 86.1% 胜率,
    而实时帧里 participant 1-5 是 LGD 的五个人 —— 领先 +1266 经济、6:1 人头的
    是 LGD。86.1% 其实是 LGD 的数, 却挂在 BLG 名下。喂给模型的更糟: 队名取
    赛程顺序(BLG)、英雄取帧(LGD 的)、经济差取帧(LGD 减 BLG), 三个输入两种朝向。

    两条独立的成因, 分开守:
      1. 队名比对用了 ==, 而 getEventDetails 返回全大写、schedule 返回混合
         大小写。对不上时不重排也不报警, 静默地左右反。
      2. 选边以 persisted 的 games[].teams[].side 为准, 但页面上所有数字都是
         按帧的 blueTeamMetadata 分组算的。两个源约 8% 的比赛会打架。
    """

    def test_队名比对必须大小写无关(self):
        import esports_feed as EF
        # getEventDetails 的写法 vs schedule 的写法, 同一支队
        self.assertEqual(EF._norm("BILIBILI GAMING"), EF._norm("Bilibili Gaming"))
        self.assertEqual(EF._norm("LGD GAMING"), EF._norm("LGD Gaming"))
        # 直接比字符串会漏 —— 这正是当初静默失效的原因
        self.assertNotEqual("BILIBILI GAMING", "Bilibili Gaming")

    def test_不同队伍不能被归一化成同一个(self):
        import esports_feed as EF
        self.assertNotEqual(EF._norm("LGD GAMING"), EF._norm("BILIBILI GAMING"))
        self.assertNotEqual(EF._norm("T1"), EF._norm("T1 Esports Academy"))

    def test_frame_sides_返回字符串id(self):
        """esportsTeamId 是 17-20 位数字。**必须当字符串比**, 否则和
        getEventDetails 里的 id (字符串) 比不上, 又是一次静默失配。"""
        import esports_feed as EF
        f = EF.EsportsFeed.__new__(EF.EsportsFeed)
        f._meta = {}
        f.window_ttl = 5
        md = {"gameMetadata": {
            "blueTeamMetadata": {"esportsTeamId": 99566404846951820},
            "redTeamMetadata": {"esportsTeamId": 99566404853854212}}}
        f._window = lambda gid, at, ttl=0: md
        f._lagged = lambda s: None
        out = f.frame_sides("g")
        self.assertEqual(out["blue"], "99566404846951820")
        self.assertIsInstance(out["blue"], str)
        self.assertIsInstance(out["red"], str)


class DebateBudget(unittest.TestCase):
    """守的坑: 两个限制各管一头, 少一个都不行。

    单次限制防"一个挂死的连接拖住整轮"; 总时限防"每一步都慢但都没超,
    累计跑到十分钟"。2026-08-20 实测一次端到端跑了 173 秒还没出裁定,
    而前端没有客户端超时 —— 只能靠服务端自己收住。

    单次限制要**给宽**: 这些是推理模型, 真在想的时候本来就慢, 限制太紧
    等于把正常输出当故障砍掉。所以宽的那个管单次, 硬的那个管总量。
    """

    def _bare(self, budget=300, elapsed=0.0):
        import debate as DB
        d = DB.Debate.__new__(DB.Debate)
        d.timeout = 90
        d.total_budget = budget
        d._reserve = 0
        d._deadline = time.monotonic() + budget - elapsed
        return d

    def test_预算耗尽时立刻返回超时而不是继续打(self):
        d = self._bare(budget=300, elapsed=300)
        self.assertTrue(d.timed_out)
        called = []
        d.cli = type("C", (), {"chat": None})()
        txt, err = type(d)._call(d, "m", "s", "u", 100, 0.2)
        self.assertEqual(txt, "")
        self.assertIn("总时限", err)
        self.assertFalse(called, "超时后不该再发请求")

    def test_辩论阶段要给裁定留出时间(self):
        """裁定挂掉 = 整场零输出, 是最不该被前两轮吃光时间的一步。"""
        import debate as DB
        d = self._bare(budget=300)
        d._reserve = DB.VERDICT_RESERVE
        # 辩论阶段看到的剩余时间, 必须比真实剩余少一个预留量
        self.assertAlmostEqual(d._left(), 300 - DB.VERDICT_RESERVE, delta=2)
        d._reserve = 0
        self.assertAlmostEqual(d._left(), 300, delta=2)

    def test_单次限制给得够宽(self):
        """限制太紧会把"模型真在推理"误判成故障 —— 实测健康时也有 17-21 秒的。"""
        import debate as DB
        d = DB.Debate.__new__(DB.Debate)
        self.assertGreaterEqual(
            DB.Debate.__init__.__defaults__[1], 60,
            "单次调用超时不该低于 60 秒")

    def test_超时和答不出来要分开报(self):
        d = self._bare(budget=300, elapsed=300)
        d._call = lambda *a, **k: ("", "APITimeoutError")
        v = d._verdict("简报", "记录")
        self.assertIn("复核超时", v["consensus"],
                      "超时要说成超时 —— 重试一次多半就好, 和模型答不出来不是一回事")


class TimelineGaps(unittest.TestCase):
    """坑: 胜率走势卡在开局几分钟不动, 而那一局早就打完了。

    gold_timeline 并发一次性把所有窗口发出去之后, 收集循环里还留着串行版
    那句"取到过数据之后连续三窗空就停"。并发版里请求已经全在手上, 提前停
    省不了任何请求, 只是**把已经取回来的真实数据丢掉**。

    中途暂停 / 转播断流会连着好几分钟取不到帧, 断够三窗后面整局就没了;
    而空窗当时还按一小时缓存, 于是那一小时里每次刷新都卡在同一个点。

    两条都要守: 中间断层之后的数据必须留下, 赛后重复窗口不许拖出长尾。
    """

    def _feed(self, gap_minutes=(), total=20, finished_at=None):
        """造一个假 feed: 第 gap_minutes 分钟的窗口取不到帧。"""
        from esports_feed import EsportsFeed
        f = EsportsFeed.__new__(EsportsFeed)
        f.window_ttl = 3
        # gold_timeline 会调 pause_spans 扣暂停, 那条路要用到这两个
        f._lock = __import__("threading").Lock()
        f._obs = {}
        f.OBS_SETTLE_SEC = EsportsFeed.OBS_SETTLE_SEC
        f._lagged = EsportsFeed._lagged
        f._ts = EsportsFeed._ts
        start = datetime.now(timezone.utc) - timedelta(minutes=total + 5)
        f._game_start = lambda gid: start

        def frame(minute, state="in_game"):
            ts = (start + timedelta(minutes=minute)).strftime(
                "%Y-%m-%dT%H:%M:%S.000Z")
            return {"rfc460Timestamp": ts, "gameState": state,
                    "blueTeam": {"totalGold": 1000 + 100 * minute,
                                 "totalKills": minute, "participants": []},
                    "redTeam": {"totalGold": 1000, "totalKills": 0,
                                "participants": []}}

        def _window(gid, starting, ttl, empty_ttl=None):
            t = datetime.strptime(starting, "%Y-%m-%dT%H:%M:%SZ") \
                        .replace(tzinfo=timezone.utc)
            minute = round((t - start).total_seconds() / 60)
            if minute < 0 or minute in gap_minutes:
                return None                      # 断流: 上游 404 / 204
            if finished_at is not None and minute >= finished_at:
                # 赛后: 每个窗口都返回**同一批帧**, 末帧时间戳恒定
                return {"frames": [frame(finished_at, "finished")]}
            if minute > total:
                return None
            return {"frames": [frame(minute)]}

        f._window = _window
        return f, start

    def test_中间断流之后的数据必须留下(self):
        # 第 5-9 分钟断流 (连续五窗空), 后面还打了十分钟
        feed, start = self._feed(gap_minutes=(5, 6, 7, 8, 9), total=20)
        rows = feed.gold_timeline("g", upto=start + timedelta(minutes=20))
        got = sorted(round(r["minute"]) for r in rows)
        self.assertIn(20, got,
                      "断流之后的十分钟被丢掉了 —— 走势会卡在第 4 分钟不动")
        self.assertEqual(got, [m for m in range(21) if m not in (5, 6, 7, 8, 9)],
                         "除了真正断流的那几分钟, 其余每分钟都该有点")

    def test_赛后重复窗口不许拖出长尾(self):
        # 第 12 分钟打完, 之后每个窗口都返回同一批 finished 帧
        feed, start = self._feed(total=40, finished_at=12)
        rows = feed.gold_timeline("g", upto=start + timedelta(minutes=40))
        fin = [r for r in rows if r["state"] == "finished"]
        self.assertEqual(len(fin), 1,
                         "赛后窗口返回的是同一帧, 按 ts 去重后只该留终局那一个点")
        self.assertLessEqual(max(r["minute"] for r in rows), 12.01,
                             "曲线不该越过终局时刻")

    def test_空结果不进长缓存(self):
        """一次瞬时 404 被按一小时缓存, 等于把抖动焊死成一小时的确定性缺口。"""
        from esports_feed import EsportsFeed
        f = EsportsFeed()
        f._sess = type("S", (), {
            "get": lambda self, url, **kw: type("R", (), {
                "status_code": 204, "json": lambda self: None})()})()
        f._get("http://x/empty", ttl=3600, empty_ttl=3)
        expiry, payload = f._cache["http://x/empty"]
        self.assertIsNone(payload)
        self.assertLess(expiry - time.time(), 10,
                        "空结果用了长 ttl —— 曲线会在一小时里反复缺同一个点")

    def test_有数据时仍然用长缓存(self):
        """历史帧永远不变, 该缓存多久还缓存多久 —— 别把修复扩大化。"""
        from esports_feed import EsportsFeed
        f = EsportsFeed()
        f._sess = type("S", (), {
            "get": lambda self, url, **kw: type("R", (), {
                "status_code": 200, "json": lambda self: {"frames": [1]}})()})()
        f._get("http://x/full", ttl=3600, empty_ttl=3)
        expiry, _ = f._cache["http://x/full"]
        self.assertGreater(expiry - time.time(), 3000)


class PausedClock(unittest.TestCase):
    """坑: 一局打了 41:05, 界面报第 48 分钟。

    帧的 rfc460Timestamp 是**真实世界时刻**, 而暂停期间上游根本不推新帧
    (窗口返回冻结的最后一帧)。拿"最后一帧 - 开局"当局内时间, 就把暂停的
    那几分钟一起算了进去。

    实测 2026-08-23 LEC GX vs TH 第 1 局: 第 3 分钟起暂停 5.9 分钟,
    第 38 分钟又停 1.75 分钟, 墙钟 48.2 分钟对游戏内 41:05。

    分钟数还要拿去选 Stage 4 的切片 (T=10/15/20/25), 所以报大了不只是
    显示难看 —— 会拿一个时间点的局面去套另一个时间点的模型。
    """

    # 双方每分钟合计进账。实测正常一分钟约 +3000 (2026-09-05 LPL 那局)。
    GOLD_PER_MIN = 3000

    def _feed(self, pauses=(), outages=(), total=30):
        """pauses / outages: [(第几分钟开始, 持续几分钟)]。两者窗口表现**一样**
        (都返回冻结帧), 区别只在**比赛有没有在背后继续打**:

          · 暂停 —— 游戏时钟停住, 经济不涨; 恢复后游戏内时间比墙钟少了这一段
          · 断流 —— 只有转播断了, 比赛照打, 恢复那一刻经济跳一大截

        分不清这两者正是 2026-09-05 那个 bug: 16 分钟断流被当成暂停扣掉,
        界面报第 12 分钟而比赛已经打到第 28 分钟。
        """
        from esports_feed import EsportsFeed
        f = EsportsFeed.__new__(EsportsFeed)
        f.window_ttl = 3
        f._lock = __import__("threading").Lock()
        f._obs = {}
        f.OBS_SETTLE_SEC = EsportsFeed.OBS_SETTLE_SEC
        f._lagged = EsportsFeed._lagged
        f._ts = EsportsFeed._ts
        f.FREEZE_SLACK = EsportsFeed.FREEZE_SLACK
        f.PLAY_GOLD_BASE = EsportsFeed.PLAY_GOLD_BASE
        f.PLAY_GOLD_PER_MIN = EsportsFeed.PLAY_GOLD_PER_MIN
        f.pause_seconds = EsportsFeed.pause_seconds.__get__(f)
        start = datetime.now(timezone.utc) - timedelta(minutes=total + 5)
        f._game_start = lambda gid: start

        def frame(at, game_minute, state="in_game"):
            g = self.GOLD_PER_MIN * max(0, game_minute)
            return {"rfc460Timestamp": at.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                    "gameState": state,
                    "blueTeam": {"totalGold": g // 2 + 1000, "totalKills": 1,
                                 "participants": []},
                    "redTeam": {"totalGold": g // 2, "totalKills": 0,
                                "participants": []}}

        def span_at(minute, spans):
            for begin, dur in spans:
                if begin <= minute < begin + dur:
                    return begin
            return None

        def lost_before(minute):
            """这一刻之前, 暂停一共吃掉了多少游戏内时间。断流不吃。"""
            return sum(min(dur, max(0, minute - begin))
                       for begin, dur in pauses)

        def _window(gid, starting, ttl, empty_ttl=None):
            t = datetime.strptime(starting, "%Y-%m-%dT%H:%M:%SZ") \
                        .replace(tzinfo=timezone.utc)
            minute = round((t - start).total_seconds() / 60)
            if minute < 0 or minute > total:
                return None
            pb = span_at(minute, pauses)
            if pb is not None:                     # 暂停: 冻结在暂停开始那一刻
                at = start + timedelta(minutes=pb)
                return {"frames": [frame(at, pb - lost_before(pb), "paused")]}
            ob = span_at(minute, outages)
            if ob is not None:                     # 断流: 同样冻结, 但比赛在走
                at = start + timedelta(minutes=ob)
                return {"frames": [frame(at, ob - lost_before(ob))]}
            at = start + timedelta(minutes=minute)
            return {"frames": [frame(at, minute - lost_before(minute))]}

        f.calls = []
        def counting(gid, starting, ttl, empty_ttl=None):
            f.calls.append(starting)
            return _window(gid, starting, ttl, empty_ttl)
        f._window = counting
        return f, start

    def test_暂停时间要从局内分钟数里扣掉(self):
        # 第 3 分钟停 5 分钟, 第 20 分钟停 2 分钟 -> 墙钟 30, 游戏内 23
        feed, start = self._feed(pauses=((3, 5), (20, 2)), total=30)
        paused = feed.pause_seconds("g", start + timedelta(minutes=30))
        self.assertAlmostEqual(paused / 60, 7, delta=1.5,
                               msg="七分钟的暂停没被识别出来")

    def test_没暂停时不能凭空扣(self):
        feed, start = self._feed(pauses=(), total=30)
        self.assertEqual(feed.pause_seconds("g", start + timedelta(minutes=30)), 0.0)

    def test_断流不能算暂停(self):
        """坑: 2026-09-05 LPL NIP vs JDG 第 2 局。

        转播流断了 16.3 分钟, 帧完全冻住 —— 和暂停在窗口上长得一模一样。
        但恢复的那一刻双方总经济从 25113 跳到 85636 (+60523, 约 3700/分钟),
        补刀 +1214: **比赛一直在打**。当成暂停扣掉的后果是界面报第 12 分钟,
        而实际打到第 28 分钟, 而这个分钟数还要拿去选 Stage 4 的切片。
        """
        feed, start = self._feed(outages=((10, 16),), total=32)
        paused = feed.pause_seconds("g", start + timedelta(minutes=32))
        self.assertEqual(paused, 0.0,
                         "断流被当成暂停了 —— 局内分钟数会凭空少十几分钟")

    def test_暂停和断流同时出现时只扣暂停(self):
        feed, start = self._feed(pauses=((4, 3),), outages=((18, 8),), total=32)
        paused = feed.pause_seconds("g", start + timedelta(minutes=32))
        self.assertAlmostEqual(paused / 60, 3, delta=1.5,
                               msg="该扣的暂停没扣, 或者把断流也一起扣了")

    def test_短断流也不算暂停(self):
        """阈值是"基数 + 每分钟增量", 一分钟的断流同样要能认出来 ——
        否则频繁的小断流会一点点把分钟数啃掉。"""
        feed, start = self._feed(outages=((12, 1),), total=25)
        self.assertEqual(feed.pause_seconds("g", start + timedelta(minutes=25)), 0.0)

    def test_重复扫描只补新增的窗口(self):
        """暂停扫描必须是**增量**的。

        旧实现把算好的暂停区间按 (gameId, 第几分钟) 缓存, 比赛每走一分钟键就
        换一个 —— 于是整局从头重扫。33 个上游窗口 x 约 600ms 往返, 直播中每次
        轮询都要付一遍, 点开一场比赛要等 5-11 秒 (2026-08-30 实测)。

        现在缓存的是**逐窗观测**: 落后现在超过 OBS_SETTLE_SEC 的窗口不会再变,
        只有近端和新增的那几分钟需要重取。
        """
        feed, start = self._feed(pauses=((3, 5),), total=30)
        first = feed.pause_spans("g", start + timedelta(minutes=28))
        self.assertGreater(len(feed.calls), 20, "第一次应当扫全局")

        feed.calls.clear()
        second = feed.pause_spans("g", start + timedelta(minutes=29))
        n2 = len(feed.calls)
        self.assertLess(n2, 8, f"第二次扫了 {n2} 个窗口 —— 增量没生效")

        # 省请求不能省出不同答案
        self.assertEqual(first, second[:len(first)],
                         "增量扫描和全量扫描算出的暂停区间不一致")

    def test_空窗不能被缓存成事实(self):
        """上游的空响应是"不知道", 不是"没暂停" —— 不能写进观测缓存。

        写进去就再也没机会自愈: 上游 404 没有规律 (见 _get 的注释), 一次抖动
        会把那一分钟永久标成缺失, 分钟数就一直偏, 且不报错。
        """
        feed, start = self._feed(pauses=(), total=30)
        real, flaky = feed._window, {"n": 0}

        def sometimes_empty(gid, starting, ttl, empty_ttl=None):
            flaky["n"] += 1
            return None if flaky["n"] % 3 == 0 else real(gid, starting, ttl, empty_ttl)

        feed._window = sometimes_empty
        feed.pause_spans("g", start + timedelta(minutes=28))
        got = len(feed._obs.get("g") or {})

        feed._window = real
        feed.calls.clear()
        feed.pause_spans("g", start + timedelta(minutes=28))
        self.assertGreater(len(feed.calls), 5,
                           "空窗被当成事实缓存了, 后续不再重试 —— 无法自愈")
        self.assertGreater(len(feed._obs.get("g") or {}), got,
                           "重试之后观测数应当变多")

    def test_赛后的冻结帧不算暂停(self):
        """比赛打完之后每个窗口也返回冻结帧, 那是结束不是暂停 —— 一直冻到
        序列末尾都没恢复, 不能计入。否则局内时长会被越扣越短。"""
        feed, start = self._feed(pauses=((25, 99),), total=30)
        self.assertEqual(feed.pause_seconds("g", start + timedelta(minutes=30)), 0.0)


class StalledFrames(unittest.TestCase):
    """坑: 一局早就打完了, 界面还写着"实时跟播"。

    模块顶部第 2 条记的是反向情形 —— 帧已 finished 而 persisted 还说
    inProgress, 那时帧是权威。这次是帧自己卡住: 实测 2026-08-23 LEC
    GX vs TH 第 1 局, persisted 说 completed, 帧却停在 in_game,
    时间戳 44 分钟没变 —— 帧流断了, 从没推送过 finished。

    current_game() 正是靠 live 挑当前局, 局间休息时会一直挑中这一局。
    """

    def _state(self, age_seconds, game_state="in_game"):
        from esports_feed import EsportsFeed
        f = EsportsFeed.__new__(EsportsFeed)
        f.window_ttl = 3
        f._no_frames = {}
        f._lock = __import__("threading").Lock()
        f._obs = {}
        f.OBS_SETTLE_SEC = EsportsFeed.OBS_SETTLE_SEC
        f._lag = {}
        f._lag_probe = {}
        f.min_lag = 20
        f.lag_probe_sec = 20
        f._lagged = EsportsFeed._lagged
        f._ts = EsportsFeed._ts
        f.LIVE_STALE_SEC = EsportsFeed.LIVE_STALE_SEC
        start = datetime.now(timezone.utc) - timedelta(seconds=age_seconds + 1800)
        f._game_start = lambda gid: start
        at = datetime.now(timezone.utc) - timedelta(seconds=age_seconds)
        f._window = lambda gid, starting, ttl, empty_ttl=None: {"frames": [{
            "rfc460Timestamp": at.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "gameState": game_state,
            "blueTeam": {"totalGold": 5000, "totalKills": 1, "participants": []},
            "redTeam": {"totalGold": 4000, "totalKills": 0, "participants": []}}]}
        return EsportsFeed.window(f, "g")

    def test_卡住的帧不算实时(self):
        st = self._state(44 * 60)         # 最后一帧是 44 分钟前的
        self.assertTrue(st.stalled)
        self.assertFalse(st.live,
                         "帧 44 分钟没动还报实时 —— 局间休息时会被当成当前局")

    def test_正常直播仍然算实时(self):
        st = self._state(80)              # 数据源本身就落后约 70 秒
        self.assertFalse(st.stalled)
        self.assertTrue(st.live)

    def test_暂停几分钟不该被当成中断(self):
        """暂停的比赛仍然在进行。阈值卡太紧会把暂停标成"数据流中断"。"""
        st = self._state(5 * 60)
        self.assertFalse(st.stalled)
        self.assertTrue(st.live)


class IsotonicCalibration(unittest.TestCase):
    """坑: 校准层把模型判断对了的东西压扁, 而且不报错。

    实测 >6k 领先的 3009 条快照, 实际胜率 98.0%, XGBoost 原始输出 97.8%
    (几乎分毫不差), Platt 压成 93.3%。Platt 只能画一条固定形状的 S 曲线,
    中间鼓两头塌 —— 是函数形式不对, 不是数据不够。

    换成保序回归后 Brier t=+6.91 / ECE t=+7.12 / 8 折全正。
    **但这只对局内模型成立**: Stage 1/2 的校准集只有 1716 场, 同样的改动
    Brier t=-4.74 (6 折全负)。两处结论相反, 靠的是分别过闸。

    这里守三件事: 保序回归的插值对不对、旧 artifacts 还能不能读、
    以及"声称 isotonic 却没有映射表"时会不会静默地给所有比赛同一个概率。
    """

    def _model(self, cfg):
        from ingame_service import IngameModel
        m = IngameModel.__new__(IngameModel)
        m.a, m.b = cfg.get("coef", 1.0), cfg.get("intercept", 0.0)
        m.calibrator = cfg.get("calibrator", "platt")
        import numpy as _np
        m.iso_x = _np.asarray(cfg.get("iso_x") or [], dtype=float)
        m.iso_y = _np.asarray(cfg.get("iso_y") or [], dtype=float)
        m.clip = float(cfg.get("clip", 0.01))
        if m.calibrator == "isotonic" and len(m.iso_x) < 2:
            m.calibrator = "platt"
        return m

    def test_保序回归按映射表插值(self):
        m = self._model({"calibrator": "isotonic",
                         "iso_x": [0.0, 0.5, 1.0],
                         "iso_y": [0.1, 0.5, 0.9], "clip": 0.01})
        self.assertAlmostEqual(m.calibrate(0.5), 0.5, places=6)
        self.assertAlmostEqual(m.calibrate(0.25), 0.3, places=6)

    def test_两端要夹住不能给绝对确定(self):
        """输出 0 或 1 意味着"绝对不可能/绝对会赢", 一旦错了 Brier 罚满。"""
        m = self._model({"calibrator": "isotonic",
                         "iso_x": [0.0, 1.0], "iso_y": [0.0, 1.0], "clip": 0.01})
        self.assertAlmostEqual(m.calibrate(0.0), 0.01, places=6)
        self.assertAlmostEqual(m.calibrate(1.0), 0.99, places=6)

    def test_超出映射表范围要夹到端点(self):
        """训练时用的是 out_of_bounds="clip", 推理侧必须一致。"""
        m = self._model({"calibrator": "isotonic",
                         "iso_x": [0.2, 0.8], "iso_y": [0.3, 0.7], "clip": 0.01})
        self.assertAlmostEqual(m.calibrate(0.0), 0.3, places=6)
        self.assertAlmostEqual(m.calibrate(1.0), 0.7, places=6)

    def test_旧artifacts没有calibrator字段仍按platt走(self):
        """换代码和换模型文件可以分开做, 少一个字段不该让服务起不来。"""
        m = self._model({"coef": 3.4, "intercept": -1.6})
        self.assertEqual(m.calibrator, "platt")
        import math as _m
        self.assertAlmostEqual(m.calibrate(0.5),
                               1 / (1 + _m.exp(-(3.4 * 0.5 - 1.6))), places=9)

    def test_声称isotonic却没有映射表要退回platt(self):
        """坑: np.interp 拿空表会把所有输入映射成同一个常数 —— 每场比赛
        给同一个概率, 而且完全不报错。这正是本项目最典型的失败形状。"""
        m = self._model({"calibrator": "isotonic", "iso_x": [], "iso_y": [],
                         "coef": 3.4, "intercept": -1.6})
        self.assertEqual(m.calibrator, "platt", "空映射表必须退回 Platt")
        self.assertNotAlmostEqual(m.calibrate(0.2), m.calibrate(0.9), places=3)

    def test_单调性(self):
        m = self._model({"calibrator": "isotonic",
                         "iso_x": [0.0, 0.3, 0.7, 1.0],
                         "iso_y": [0.05, 0.4, 0.6, 0.95], "clip": 0.01})
        vals = [m.calibrate(r) for r in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)]
        self.assertEqual(vals, sorted(vals), "校准必须单调, 否则会改变排序")


class ClipWarning(unittest.TestCase):
    """坑: 换了校准之后, "概率被压扁了"这条警告变成了陈旧断言。

    它原先写在 make_row 里, 按"第20分钟后 且 |经济差|>=5000"硬编码触发,
    说"校准把概率压在区间内, 实测胜率更高"。那在 Platt 时代是对的 (顶格
    0.94, 而 6k+ 领先实测 98%)。换成保序回归后范围 [0.01, 0.99], 领先 8000
    给 95.9%, 离上限还远 —— 这句话就成了假的。

    陈旧的警告比没有警告更糟: 下游 (Stage 3 的 agent、前端) 会把它当事实
    引用。所以判据必须是**实际输出有没有触到边界**, 不是输入长什么样。
    """

    def _m(self, p_min=0.01, p_max=0.99):
        from ingame_service import IngameModel
        m = IngameModel.__new__(IngameModel)
        m.metrics = {"p_min": p_min, "p_max": p_max}
        return m

    def test_没到边界就不该警告(self):
        m = self._m()
        self.assertIsNone(m.clip_warning(0.959),
                          "95.9% 离 99% 的上限还远, 不该说概率被压扁了")
        self.assertIsNone(m.clip_warning(0.5))

    def test_顶到上限要警告(self):
        m = self._m()
        w = m.clip_warning(0.99)
        self.assertIsNotNone(w)
        self.assertIn("上限", w)

    def test_触到下限要警告(self):
        m = self._m()
        w = m.clip_warning(0.01)
        self.assertIsNotNone(w)
        self.assertIn("下限", w)

    def test_大经济差本身不再触发警告(self):
        """守的正是这次的回归: 判据从"输入很极端"改成了"输出触到边界"。"""
        from ingame_service import IngameModel
        import inspect
        src = inspect.getsource(IngameModel.make_row)
        self.assertNotIn("超出模型的有效表达范围", src,
                         "这条警告必须在预测之后按实际概率发, 不能留在 make_row")


class GoldTotalFallback(unittest.TestCase):
    """坑: gold_total 缺失时按 1200*T 估算, 比实测低 25~32%。

    golddiff_norm = golddiff / gold_total 是重要性最高的特征 (0.345),
    分母估小就把领先幅度放大 1.33~1.46 倍 —— 模型以为的优势比实际大三四成,
    而且不报错。自动跟播不受影响 (帧里有真实总经济), 手动填局面每次都踩。

    实测中位数 (五年约 9 万条 OE 记录, research/audit_gold_total.py):
        T=10 15918   T=15 25037   T=20 34597   T=25 43854
    """

    def test_兜底值贴近实测中位数(self):
        from ingame_service import GOLD_TOTAL_MEDIAN
        measured = {10: 15918, 15: 25037, 20: 34597, 25: 43854}
        for T, want in measured.items():
            got = GOLD_TOTAL_MEDIAN[T]
            self.assertLess(abs(got - want) / want, 0.05,
                            f"T={T} 的兜底值偏离实测中位数超过 5%")

    def test_不能退回到那个拍脑袋的公式(self):
        from ingame_service import GOLD_TOTAL_MEDIAN
        for T in (10, 15, 20, 25):
            self.assertNotEqual(GOLD_TOTAL_MEDIAN[T], 1200 * T,
                                f"T={T} 又变回 1200*T 了 —— 那个值低 25~32%")

    def test_随时间单调递增(self):
        from ingame_service import GOLD_TOTAL_MEDIAN
        vals = [GOLD_TOTAL_MEDIAN[T] for T in (10, 15, 20, 25)]
        self.assertEqual(vals, sorted(vals))


class RollingMinPeriods(unittest.TestCase):
    """坑: 离线 rolling(min_periods=3), 在线 tail(10).mean() 没有下限。

    一支只打过 1 场的队伍在线上会拿到 rolling_wr = 0.0 或 1.0, 而模型训练时
    从没见过这种值 —— 离线保证至少 3 场才出数。不报错, 只是喂了个训练分布
    之外的输入。两条路径现在共用 MIN_ROLL_PERIODS。
    """

    def _store(self, n_games):
        import pandas as pd
        from feature_store import FeatureStore, MIN_ROLL_PERIODS
        rows = []
        for i in range(n_games):
            rows.append({"date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
                         "result": 1, "streak": i, "gamelength_min": 30.0,
                         "game_total_kills": 20.0, "kills": 10.0,
                         "league": "LCK", "teamname": "X"})
        t = pd.DataFrame(rows)
        s = FeatureStore.__new__(FeatureStore)
        s.team_history = {"X": t}
        s.roll_cols = ["rolling_wr", "avg_kills", "streak"]
        return s

    def test_场次不足时返回nan而不是极端值(self):
        s = self._store(1)
        snap = s.team_snapshot("X")
        self.assertTrue(snap["rolling_wr"] != snap["rolling_wr"],
                        "只有 1 场时 rolling_wr 必须是 NaN, 不能是 1.0")

    def test_场次够了就出数(self):
        s = self._store(5)
        snap = s.team_snapshot("X")
        self.assertAlmostEqual(snap["rolling_wr"], 1.0)

    def test_两条路径共用同一个常数(self):
        """离线的 min_periods 和在线的下限必须是同一个值, 不能各写各的。"""
        import inspect
        from feature_store import FeatureStore, MIN_ROLL_PERIODS
        self.assertEqual(MIN_ROLL_PERIODS, 3)
        src = inspect.getsource(FeatureStore.from_frame)
        self.assertNotIn("min_periods=3", src,
                         "离线路径又写回字面量 3 了 —— 改一边忘另一边正是这个坑")


class DisclaimerFreshness(unittest.TestCase):
    """坑: 免责声明里的数字和校准方法写死在代码里。

    原文是 "准确率约 73%, 概率经 Platt 校准 (ECE ≈ 0.03)"。换成保序回归之后
    真实是 77% / ECE 0.016 —— 那句话就成了假的, 而它会原样印在界面上、被
    Stage 3 的 agent 当事实引用。和 clip_warning 是同一个坑。

    注意 Stage 1/2 **仍然是 Platt**, agents.py 和 /debate 简报里那两句是对的,
    不要顺手一起改。
    """

    class _Fake:
        def __init__(self, calibrator, acc, ece, t25):
            self.calibrator = calibrator
            self.metrics = {"accuracy": acc, "ece": ece,
                            "per_T": {"25": {"accuracy": t25}}}

    def test_保序回归时不能说platt(self):
        from api import _ingame_disclaimer
        s = _ingame_disclaimer(self._Fake("isotonic", 0.773, 0.016, 0.835))
        self.assertIn("保序回归", s)
        self.assertNotIn("Platt", s)

    def test_数字来自metrics而不是写死(self):
        from api import _ingame_disclaimer
        s = _ingame_disclaimer(self._Fake("isotonic", 0.773, 0.016, 0.835))
        self.assertIn("77%", s)
        self.assertIn("84%", s)          # 0.835 -> 84%
        self.assertIn("0.016", s)
        self.assertNotIn("73%", s)
        self.assertNotIn("0.03。", s)

    def test_旧artifacts走platt时如实说platt(self):
        from api import _ingame_disclaimer
        s = _ingame_disclaimer(self._Fake("platt", 0.737, 0.03, 0.805))
        self.assertIn("Platt", s)
        self.assertNotIn("保序回归", s)


class MarketArithmetic(unittest.TestCase):
    """坑: 拿 1/赔率 直接当市场的概率预测。

    庄家挂出的隐含概率加起来是 1.03-1.08, 多的那块是抽水。不剥就去比
    Brier, 市场会被系统性判成"过度自信", 于是模型白捡一个不存在的胜利
    —— 而且这个错误在任何输出里都看不出来, 只会让结论整体偏向我们。
    """

    def test_剥完抽水概率和为一(self):
        from market import devig
        for odds in ([1.90, 1.90], [1.50, 2.70], [3.10, 1.40], [4.0, 3.5, 2.1]):
            self.assertAlmostEqual(sum(devig(odds)), 1.0, places=12,
                                   msg=f"{odds} 剥完不为 1")

    def test_对称赔率剥完是五五开(self):
        from market import devig
        a, b = devig([1.90, 1.90])
        self.assertAlmostEqual(a, 0.5, places=12)
        self.assertAlmostEqual(b, 0.5, places=12)

    def test_不剥抽水会高估热门(self):
        from market import devig, implied
        odds = [1.50, 2.70]
        naive = implied(odds[0])
        real = devig(odds)[0]
        # 高估的量级 (~2.4pp) 和我们要检测的效应量同一个数量级 ——
        # 这就是为什么这一步不是讲究而是必需
        self.assertGreater(naive - real, 0.02)
        self.assertLess(naive - real, 0.04)

    def test_赔率不大于一要报错(self):
        from market import implied
        for bad in (1.0, 0.98, 0.0, -2.0):
            with self.assertRaises(ValueError):
                implied(bad)

    def test_公平赔率的期望值为零(self):
        from market import ev
        # 真概率 0.4, 公平赔率 2.5 -> EV 恰好 0
        self.assertAlmostEqual(ev(0.4, 2.5), 0.0, places=12)
        self.assertLess(ev(0.4, 2.3), 0.0)
        self.assertGreater(ev(0.4, 2.7), 0.0)

    def test_没有优势时凯利给零而不是负数(self):
        from market import kelly
        self.assertEqual(kelly(0.4, 2.3), 0.0)
        self.assertGreater(kelly(0.4, 2.7), 0.0)

    def test_收盘线价值的符号(self):
        from market import clv
        # 拿到 2.10, 收盘跌到 1.90 -> 我们拿的价格更好, CLV 为正
        self.assertGreater(clv(2.10, 1.90), 0)
        self.assertLess(clv(1.90, 2.10), 0)


class GateVsMarketDirection(unittest.TestCase):
    """坑: Brier 越小越好, 而 gate() 认的是越大越好。

    judge() 里那一步取负如果漏了或者写反, 判决方向会整个颠倒 —— 一个比
    盘口差的模型会被判成"打赢市场", 而且过程中不会有任何异常。这是这套
    东西里后果最严重、症状最少的一个错。
    """

    @staticmethod
    def _rows(p_model_on_truth, n=200):
        """造 n 个盘: 市场一律 1.90/1.90 (剥完 0.5/0.5), 实际结果轮流。
        模型对真实结果给 p_model_on_truth 的概率。"""
        from market import devig
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        out = []
        for k in range(n):
            y = k % 2
            p = [p_model_on_truth, 1 - p_model_on_truth]
            if y == 1:
                p = p[::-1]
            out.append({"t": (base + timedelta(hours=k)).isoformat(),
                        "odds": [1.90, 1.90], "p_model": p, "y": y,
                        "_t": base + timedelta(hours=k),
                        "_q": devig([1.90, 1.90])})
        return out

    def _judge_quiet(self, rows):
        import contextlib, io
        from gate_vs_market import judge
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            return judge(rows, "test", n_folds=4)

    def test_模型更准时判为采纳(self):
        # 每次都给真实结果 0.9 -> 远好于市场的 0.5
        self.assertTrue(self._judge_quiet(self._rows(0.9)),
                        "模型明显更准却没被判赢 —— 取负那一步写反了")

    def test_模型更差时判为不采纳(self):
        # 每次都给真实结果 0.1 -> 远差于市场的 0.5
        self.assertFalse(self._judge_quiet(self._rows(0.1)),
                         "模型明显更差却被判赢 —— 取负那一步写反了")


class JoinedOddsValidation(unittest.TestCase):
    """坑: 一条脏的盘口行悄悄进到 Brier 里。

    这类错误从结果上完全看不出来 —— 它不会抛异常, 只会让某个市场无缘无故
    显得好一点或差一点。所以校验必须在读入时就做掉, 并且说明剔了什么。
    """

    def _write(self, rows):
        import json, tempfile
        f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False,
                                        encoding="utf-8")
        f.write(chr(10).join(json.dumps(r) for r in rows))
        f.close()
        return f.name

    @staticmethod
    def _ok(**kw):
        r = {"t": "2026-01-01T00:00:00+00:00", "game_id": "g", "market": "moneyline",
             "odds": [1.90, 1.90], "p_model": [0.6, 0.4], "y": 0}
        r.update(kw)
        return r

    def test_正常行留下(self):
        from gate_vs_market import load_joined
        rows, drop = load_joined(self._write([self._ok()]))
        self.assertEqual(len(rows), 1)
        self.assertEqual(drop, {})

    def test_概率不为一的行剔掉(self):
        from gate_vs_market import load_joined
        rows, drop = load_joined(self._write([self._ok(p_model=[0.6, 0.6])]))
        self.assertEqual(len(rows), 0)
        self.assertTrue(drop, "模型概率不为 1 应当被剔除并说明原因")

    def test_抽水过高的行剔掉(self):
        from gate_vs_market import load_joined
        # 两边都 1.60 -> 抽水 25%, 远超上限
        rows, _ = load_joined(self._write([self._ok(odds=[1.60, 1.60])]))
        self.assertEqual(len(rows), 0)

    def test_时间对不齐的行剔掉(self):
        from gate_vs_market import load_joined
        # 模型是两小时前算的, 那时的信息集和盘口不是同一个
        rows, _ = load_joined(self._write(
            [self._ok(t_model="2025-12-31T22:00:00+00:00")]))
        self.assertEqual(len(rows), 0,
                         "时间不对齐必须剔除 —— 否则比的是信息集不是模型")

    def test_按时间排序(self):
        from gate_vs_market import load_joined
        late = self._ok(t="2026-01-02T00:00:00+00:00", game_id="late")
        early = self._ok(t="2026-01-01T00:00:00+00:00", game_id="early")
        rows, _ = load_joined(self._write([late, early]))
        self.assertEqual([r["game_id"] for r in rows], ["early", "late"],
                         "折要按时间切, 顺序错了配对就失去意义")


class CRPSArithmetic(unittest.TestCase):
    """坑: CRPS 少了离散度那一项, 就退化成了平均绝对误差。

    退化之后, 一个把方差吹大来"保平安"的分布会拿到更好的分数 —— 而那种
    分布在盘口上是稳定亏钱的。所以第二项必须在。
    """

    def test_与正态闭式解一致(self):
        import numpy as np
        from baseline_dists import crps_sample, _crps_norm
        mu, sd = 30.0, 6.0
        q = np.quantile(np.random.default_rng(0).normal(mu, sd, 200000),
                        np.linspace(0.002, 0.998, 2000))
        for y in (24.0, 30.0, 38.0):
            self.assertAlmostEqual(crps_sample(y, q), _crps_norm(mu, sd, y),
                                   places=1, msg=f"y={y} 处经验与闭式解不符")

    def test_吹大方差要被罚(self):
        import numpy as np
        from baseline_dists import crps_sample
        rng = np.random.default_rng(1)
        tight = rng.normal(30, 6, 20000)
        wide = rng.normal(30, 18, 20000)
        # 真值就在均值上时, 铺得宽的必须更差
        self.assertLess(crps_sample(30.0, tight), crps_sample(30.0, wide),
                        "离散度没被罚 -> CRPS 退化成了 MAE")


class SlowFeedBlanksBoard(unittest.TestCase):
    """坑: 帧流变慢 -> 整个看板变成空白, 页面写"尚未开始"。

    实测 2026-09-06 LPL WE vs IG 第 1 局。上游按真实时间的 15-62% 发布帧
    (连续采样落后 606s -> 635s -> 654s -> 698s), 延迟持续累积。越过
    LIVE_STALE_SEC 之后 stalled=True -> live=False, current_game() 一无所获;
    而 esports_board 的回退用 games[].state, 那场比赛五局全写着 unstarted。
    结果 chosen=None, live/prediction/timeline 全空 —— 比赛打到第 20 分钟,
    页面却说"尚未开始"。

    显示一份**标明了有多旧**的数据, 永远好过什么都不显示。
    """

    @staticmethod
    def _feed(states):
        """states: [(game_state, 帧龄秒) 或 None]  按局号顺序。"""
        import threading
        from esports_feed import EsportsFeed, LiveState
        f = EsportsFeed.__new__(EsportsFeed)
        f._lock = threading.Lock()
        gs = [{"id": f"g{i+1}", "number": i + 1, "state": "unstarted"}
              for i in range(len(states))]
        f.games = lambda mid: gs

        def window(gid, precise_minute=False):
            i = int(gid[1:]) - 1
            spec = states[i]
            if spec is None:
                return None
            state, age = spec
            stalled = state == "in_game" and age > EsportsFeed.LIVE_STALE_SEC
            return LiveState(game_id=gid, game_state=state, minute=20,
                             golddiff=0, csdiff=0, blue_kills=0, red_kills=0,
                             gold_total=0, frame_time="",
                             live=(state == "in_game" and not stalled),
                             stale_seconds=age, stalled=stalled)
        f.window = window
        return f

    def test_帧卡住时仍然交出这一局(self):
        from esports_feed import EsportsFeed
        # 第 1 局在打但帧落后 12 分钟, 其余局没开
        feed = self._feed([("in_game", 760), None, None])
        cur = EsportsFeed.current_game(feed, "m")
        self.assertIsNotNone(
            cur, "帧卡住就返回 None -> 看板全空, 比赛打着却显示尚未开始")
        g, st = cur
        self.assertEqual(g["number"], 1)
        self.assertTrue(st.stalled, "交出去的同时必须标明它是旧的")
        self.assertFalse(st.live)

    def test_正常直播优先于卡住的局(self):
        from esports_feed import EsportsFeed
        # 第 1 局卡住, 第 2 局正常 -> 必须选第 2 局
        feed = self._feed([("in_game", 900), ("in_game", 80), None])
        g, st = EsportsFeed.current_game(feed, "m")
        self.assertEqual(g["number"], 2)
        self.assertTrue(st.live)

    def test_打完的局不算当前局(self):
        from esports_feed import EsportsFeed
        # 全部 finished -> 回退不该命中, 交回 None 走 esports_board 的
        # playable 回退。否则局间休息会被当成"第 3 局正在打"。
        feed = self._feed([("finished", 300), ("finished", 200), None])
        self.assertIsNone(EsportsFeed.current_game(feed, "m"))

    def test_打完的局前面卡住的局不算当前局(self):
        from esports_feed import EsportsFeed
        # 2026-09-12 LEC VIT vs MKOI: 第 1 局帧断在第 1 分钟 (永远 in_game),
        # 第 2 局打完, 第 3 局 BP 还没帧 -> 不能把第 1 局当成正在打的局
        feed = self._feed([("in_game", 7200), ("finished", 600), None])
        self.assertIsNone(
            EsportsFeed.current_game(feed, "m"),
            "越过打完的第 2 局去交第 1 局 -> 页面跳回第 1 局, 第 3 局点不进去")

    def test_一局都没有帧时仍然是None(self):
        from esports_feed import EsportsFeed
        feed = self._feed([None, None, None])
        self.assertIsNone(EsportsFeed.current_game(feed, "m"))

    def test_probe_games交出每一局(self):
        """两个调用方要共用同一批探测结果, 否则 playable 和选局会打架。"""
        from esports_feed import EsportsFeed
        feed = self._feed([("finished", 300), ("in_game", 60), None])
        pairs = EsportsFeed.probe_games(feed, "m")
        self.assertEqual(len(pairs), 3)
        self.assertEqual([g["number"] for g, _ in pairs], [1, 2, 3])
        self.assertIsNone(pairs[2][1], "没开的局应当是 None")
        self.assertEqual(pairs[0][1].game_state, "finished")


class PlayableFromFrames(unittest.TestCase):
    """坑: 哪些局出标签页只看 games[].state, 而那个字段不可信。

    实测 2026-09-06 LPL WE vs IG: 第 1 局已打完 (帧 finished, 第 23 分钟,
    18-7), 五局的 state 全是 unstarted。只信它 -> playable 为空 -> 选局没有
    回退可选、标签页一个不出 -> 看板空白写着"尚未开始", 而系列赛已经 1-0。

    反向也发生过 (2026-08-16 LPL 三场 state=completed 但没开打), 所以两个
    来源取并集, 谁也不单独当权威。
    """

    @staticmethod
    def _games(states):
        return [{"id": f"g{i+1}", "number": i + 1, "state": st}
                for i, st in enumerate(states)]

    def test_state全unstarted但有帧(self):
        from api import _playable_games
        gs = self._games(["unstarted"] * 5)
        out = _playable_games(gs, {"g1": object()})
        self.assertEqual([g["number"] for g in out], [1],
                         "有帧的局必须算 playable, 否则整个看板会空白")

    def test_没帧但state说打过(self):
        from api import _playable_games
        gs = self._games(["completed", "inProgress", "unstarted"])
        out = _playable_games(gs, {})
        self.assertEqual([g["number"] for g in out], [1, 2])

    def test_两个来源取并集(self):
        from api import _playable_games
        gs = self._games(["completed", "unstarted", "unstarted"])
        out = _playable_games(gs, {"g2": object()})
        self.assertEqual([g["number"] for g in out], [1, 2])

    def test_都没有就是空(self):
        from api import _playable_games
        self.assertEqual(_playable_games(self._games(["unstarted"] * 3), {}), [])


class CollectorKeepsLaggedFrames(unittest.TestCase):
    """坑: 采集器把延迟大的帧当无效数据扔掉, 于是整段后期比赛丢失。

    st.live 里含一道判死逻辑 (stale > LIVE_STALE_SEC 就翻 False)。那道逻辑
    是给"挑当前局"用的; 用作采集判据就变成了"上游发得慢 = 这局不存在"。

    实测 2026-09-04 起 LPL 的帧按真实时间约 79% 发布 (区间 31%-164%),
    延迟在一局内累积, 十几分钟就越过 600 秒。后果: 09-04 至 09-06 每一局
    LPL 只采到第 12-17 分钟, 同期 LCK 采到第 30-41 分钟 —— 而 20 分钟以后
    正是 Stage 4 信息量最大的一段。实测修复当天 LGD vs NIP 局4 在
    stale=706s 时采到了第 21 分钟, 旧判据下这一条会被跳过。

    帧带着 frame_time, 有多旧读的人自己看得见; 采集器的职责是别把有效
    数据扔了。
    """

    def _snapshot_guard(self):
        """找到包住 snapshot() 调用的那个 if, 返回它的判据源码。"""
        import ast
        src = (_ROOT / "collect_live.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            calls = [n for n in ast.walk(node) if isinstance(n, ast.Call)]
            if any(isinstance(c.func, ast.Name) and c.func.id == "snapshot"
                   for c in calls):
                return ast.unparse(node.test)
        self.fail("在 collect_live.py 里找不到包住 snapshot() 的 if")

    def test_采集判据不能用live(self):
        test = self._snapshot_guard()
        self.assertNotIn(
            "live", test,
            f"采集判据是 `{test}` —— st.live 含 600 秒判死逻辑, "
            f"用它会把延迟大的后期比赛整段丢掉")

    def test_采集判据用的是game_state(self):
        test = self._snapshot_guard()
        self.assertIn("game_state", test,
                      f"采集判据应当只问'这局在不在打', 实际是 `{test}`")

    def test_卡住的帧仍然是in_game(self):
        """判据换成 game_state 之后, 延迟大的帧确实还会被采。"""
        from esports_feed import EsportsFeed, LiveState
        st = LiveState(game_id="g", game_state="in_game", minute=21,
                       golddiff=0, csdiff=0, blue_kills=0, red_kills=0,
                       gold_total=0, frame_time="",
                       live=False, stale_seconds=706, stalled=True)
        self.assertFalse(st.live, "前提: 这种帧的 live 是 False")
        self.assertEqual(st.game_state, "in_game",
                         "而它的 game_state 仍然是 in_game —— 数据有效, 只是旧")


class BlendPriorContinuity(unittest.TestCase):
    """坑: 看板第 3 分钟从 BP 后概率硬切到局内模型, 胜率一跳 (2026-09-13 实测平均 14 个百分点,
    39% 的局超过 15 —— BP 后说 MKOI 64%, 下一分钟局内给 VIT 83%)。现在 3→15 分钟渐变过渡,
    两端必须严丝合缝, 否则跳变换个地方又回来。research/gate_anchor.py 的 C2。"""

    def test_起点就是BP后(self):
        from api import _blend_prior, BLEND_FROM
        p, w = _blend_prior(0.36, 0.83, BLEND_FROM)
        self.assertEqual(w, 0.0)
        self.assertAlmostEqual(p, 0.36, places=9, msg="第 3 分钟还应该完全是 BP 后的概率")

    def test_终点就是局内模型(self):
        from api import _blend_prior, BLEND_TO
        p, w = _blend_prior(0.36, 0.83, BLEND_TO)
        self.assertEqual((p, w), (0.83, 1.0), "第 15 分钟起应当原样交出局内模型的数")
        p, w = _blend_prior(0.36, 0.83, 40)
        self.assertEqual((p, w), (0.83, 1.0))

    def test_中间落在两者之间且连续(self):
        from api import _blend_prior
        ps = [_blend_prior(0.36, 0.83, m / 10)[0] for m in range(30, 151)]
        self.assertTrue(all(0.36 - 1e-9 <= p <= 0.83 + 1e-9 for p in ps))
        steps = [abs(b - a) for a, b in zip(ps, ps[1:])]
        self.assertLess(max(steps), 0.01, "每 6 秒最多动 1 个百分点, 不应有台阶")
