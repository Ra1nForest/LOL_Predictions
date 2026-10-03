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


class EmptyKnownListIsNotNoValidation(unittest.TestCase):
    """坑: map_team 写的是 `if known:`, 空表是假值, 于是 known=[] 时直接把队名原样放过。

    国际赛按赛事名查已知队伍 (known_teams("Worlds")) 恒为空表 —— 每个国际赛队名都不经校验成了
    model_name, 看板以为能预测, 错误要到 make_row 深处才抛出来又被吞掉。实测
    map_team("GAM Esports", []) 返回 "GAM Esports"。known=None 才是"不校验"。
    """

    def test_空表不是不校验(self):
        from esports_feed import map_team
        self.assertIsNone(map_team("GAM Esports", []), "known=[] 必须返回 None, 不能原样放过")
        self.assertIsNone(map_team("Beijing JDG Esports", []), "别名也一样要落在 known 里")

    def test_别名的目标不在已知队伍里就返回None(self):
        from esports_feed import map_team
        self.assertIsNone(map_team("Beijing JDG Esports", ["T1", "Gen.G"]),
                          "给不在特征库里的队加别名会重新造出'可预测'的假象")
        self.assertEqual(map_team("Beijing JDG Esports", ["JD Gaming"]), "JD Gaming")

    def test_known为None仍然只换别名不校验(self):
        from esports_feed import map_team
        self.assertEqual(map_team("Beijing JDG Esports", None), "JD Gaming")
        self.assertEqual(map_team("Anything Goes", None), "Anything Goes")


class LeagueKeysMatchUpstreamNames(unittest.TestCase):
    """坑: live() 按 m.league.upper() in LEAGUE_IDS 过滤。键写成 "WLDS" / "FIRSTSTAND" 不会报错,
    只会把正在打的国际赛静默丢掉 —— 2026-10-03 DCGI 的 FlyQuest vs LGD 在 getLive 里是 inProgress,
    feed.live() 却返回 [] (那时表里只有四大)。键必须等于 getLeagues 的 league.name.upper()。
    """

    # getLeagues 实测返回的名字 (2026-10-03)
    UPSTREAM = {"Worlds": "98767975604431411", "MSI": "98767991325878492",
                "First Stand": "113464388705111224", "DCGI": "117126995932274206",
                "Esports World Cup": "116838530616006090"}

    def test_每个国际赛的键都等于上游名字的大写(self):
        from esports_feed import LEAGUE_IDS
        for name, lid in self.UPSTREAM.items():
            self.assertEqual(LEAGUE_IDS.get(name.upper()), lid, name)
        for lg in ("LCK", "LPL", "LEC", "LCS"):
            self.assertIn(lg, LEAGUE_IDS, "四大赛区不能丢")

    def test_live保留国际赛而且不改上游的大小写(self):
        from esports_feed import EsportsFeed
        events = [{"league": {"name": n}, "startTime": "2026-10-03T08:00:00Z", "state": "inProgress",
                   "match": {"id": f"M{i}", "strategy": {"count": 3},
                             "teams": [{"name": "T1", "code": "T1"}, {"name": "FlyQuest", "code": "FLY"}]}}
                  for i, n in enumerate(list(self.UPSTREAM) + ["LCK", "LCK Challengers"])]
        f = EsportsFeed()
        f._get = lambda url, ttl, **k: {"data": {"schedule": {"events": events}}}
        got = [m.league for m in f.live()]
        self.assertEqual(got, list(self.UPSTREAM) + ["LCK"],
                         "国际赛要留下、Match.league 保留上游原样; 不在表里的次级联赛照旧过滤掉")

    def test_is_major大小写无关且只认四大(self):
        from esports_feed import is_major
        for lg in ("LCK", "lpl", "Lec", "LCS"):
            self.assertTrue(is_major(lg), lg)
        for lg in ("Worlds", "WORLDS", "MSI", "DCGI", "First Stand", "Esports World Cup", "", None):
            self.assertFalse(is_major(lg), lg)


class _StubStore:
    """只有 pregame_league / 映射队名用得到的两个方法。四队: 两支 LCK、一支 LPL、一支 LCS。"""
    HOME = {"T1": "LCK", "Gen.G": "LCK", "Bilibili Gaming": "LPL", "FlyQuest": "LCS"}

    def home_league(self, team):
        return self.HOME.get(team)

    def known_teams(self, league=None):
        return sorted(t for t, lg in self.HOME.items() if league is None or lg == league)


class PregameLeagueRules(unittest.TestCase):
    """坑: 把国内的赛前/BP 后模型原样搬到国际赛上。

    Stage 1/2 的特征全是赛区内的相对量, 看不见赛区之间的差距, 训练集里也没有一场跨赛区的比赛。
    research/gate_international.py: 523 局历史跨赛区国际赛上准确率约 50%、校准斜率约 0, 却和国内
    一样自信; 同母赛区的国际赛 (314 局) 则和国内一样。所以只有同母赛区才给, 且按母赛区算。
    """

    def test_四大赛区原样返回(self):
        from api import pregame_league
        s = _StubStore()
        self.assertEqual(pregame_league(s, "T1", "Gen.G", "LCK"), "LCK")
        self.assertEqual(pregame_league(s, "T1", "Bilibili Gaming", "LCK"), "LCK",
                         "国内比赛一点都不变 —— 哪怕两队母赛区不同")
        self.assertEqual(pregame_league(s, None, None, "LPL"), "LPL")

    def test_国际赛同母赛区按母赛区算(self):
        from api import pregame_league
        self.assertEqual(pregame_league(_StubStore(), "T1", "Gen.G", "Worlds"), "LCK")
        self.assertEqual(pregame_league(_StubStore(), "Gen.G", "T1", "Esports World Cup"), "LCK")

    def test_跨赛区不给(self):
        from api import pregame_league
        self.assertIsNone(pregame_league(_StubStore(), "T1", "Bilibili Gaming", "Worlds"))
        self.assertIsNone(pregame_league(_StubStore(), "FlyQuest", "T1", "DCGI"))

    def test_有队不认识不给(self):
        from api import pregame_league
        self.assertIsNone(pregame_league(_StubStore(), "T1", None, "MSI"))
        self.assertIsNone(pregame_league(_StubStore(), "GAM Esports", "GAM Esports", "MSI"),
                          "两个 None 母赛区不能被当成'同一个赛区'")
        self.assertIsNone(pregame_league(None, "T1", "Gen.G", "Worlds"), "没有特征库就定不下来")


class IngameIntlLeagueEncoding(unittest.TestCase):
    """坑: 国际赛给局内模型传母赛区 (或别的四大名字) 当 league。

    Stage 4 在全部赛区上训练, 国际赛那几千个快照的 is_lpl/lck/lec/lcs 全是 0。线上若把"同母赛区"
    的 LCK 传进来, 编码就和训练时对不上, 不报错。看板必须给 Stage 4 传赛事本身的名字。
    """

    def test_国际赛的is全为0(self):
        from ingame_service import IngameModel
        m = IngameModel.__new__(IngameModel)
        m.slices, m.scaling, m.features, m.has_xpdiff = [10, 15, 20, 25], {}, [], False
        for lg in ("DCGI", "Worlds", "MSI", "First Stand", "Esports World Cup"):
            row, warns = m.make_row(minute=20, golddiff=1500, xpdiff=None, csdiff=20, blue_kills=5,
                                    red_kills=3, gold_total=35000, league=lg)
            self.assertEqual([row[k] for k in ("is_lpl", "is_lck", "is_lec", "is_lcs")], [0, 0, 0, 0], lg)
            self.assertFalse(any("LPL" in w for w in warns), "LPL 样本偏薄那条警告只给 LPL")


class OeLeagueNameIsCaseInsensitive(unittest.TestCase):
    """坑: 同一个赛区两种写法 —— LEAGUE_IDS 的键是 "WORLDS", 采集 meta 和预测留档存的是上游原样
    "Worlds"。LEAGUE_TO_OE 只收一种, 另一条路径就一局都配不上胜负, 不报错。
    DCGI 故意不映射: OE 的 "DCup" 是 12 月 LPL 系的国内杯赛, 这届全球邀请赛它叫什么还不知道。
    """

    def test_国际赛大小写都换得到OE名字(self):
        from backfill_late import oe_league
        for k, v in (("WORLDS", "WLDs"), ("Worlds", "WLDs"), ("worlds", "WLDs"),
                     ("FIRST STAND", "FST"), ("First Stand", "FST"),
                     ("ESPORTS WORLD CUP", "EWC"), ("Esports World Cup", "EWC"),
                     ("MSI", "MSI"), ("LCK", "LCK"), ("LCK Challengers", "LCKC"), ("EMEA Masters", "EM")):
            self.assertEqual(oe_league(k), v, k)

    def test_DCGI不映射(self):
        from backfill_late import LEAGUE_TO_OE, oe_league
        self.assertNotIn("DCGI", {k.upper() for k in LEAGUE_TO_OE})
        self.assertNotEqual(oe_league("DCGI"), "DCup", "不能猜成国内杯赛的标签")

    def test_国际赛进了回填的扫描范围(self):
        from backfill_late import ALL_LEAGUES
        for k in ("WORLDS", "MSI", "FIRST STAND", "DCGI", "ESPORTS WORLD CUP"):
            self.assertIn(k, ALL_LEAGUES)


class InternationalBoard(unittest.TestCase):
    """坑: 跨赛区的国际赛照旧显示赛前/BP 后、照旧把它渐变进局内头条; 队名不认识的比赛整块看板没有胜率。

    跨赛区上 Stage 1/2 没有预测力 (见 PregameLeagueRules), 渐变 3-15 分钟会把它带进局内曲线。
    这里用假的数据源走一遍 api.esports_board: 跨赛区不能有 p1/p2/渐变; 队名不认识要有局内胜率,
    但不能写留档 (lolesports 原名永远配不上 OE, 是一条永远没有结果的行)。
    """

    def setUp(self):
        import api
        from esports_feed import EsportsFeed, LiveState, Match, TeamRef
        from ingame_service import IngameModel
        try:
            self.ig = IngameModel("_live")
        except Exception as e:                       # artifacts 不在 (比如刚克隆) 就跳过
            self.skipTest(f"局内模型文件不在: {e}")
        self.api = api
        self.logged, self.calls = [], []
        self._old = (dict(api.STATE), api.log_prediction)

        def _log(**k):
            # calls 记每一次调用; logged 和真的 log_prediction 一样, probability_blue 为 None 时不写
            self.calls.append(k)
            if k.get("probability_blue") is not None:
                self.logged.append(k)
        api.log_prediction = _log

        class Stages:                                 # Stage 1/2 一被调用就让测试失败
            def __getitem__(s, k):
                raise AssertionError(f"不该算 {k}")

        api.STATE.clear()
        api.STATE.update(store=_StubStore(), stages=Stages(), ingame=None, ingame_live=self.ig)

        def make_feed(league, names, minute):
            g = {"id": "G1", "number": 1, "state": "inProgress",
                 "teams": [{"id": "1", "side": "blue"}, {"id": "2", "side": "red"}]}
            st = LiveState(game_id="G1", game_state="in_game", minute=minute, golddiff=2500, csdiff=30,
                           blue_kills=6, red_kills=2, gold_total=30000,
                           frame_time="2026-10-03T08:20:00.000Z", live=True)

            class Feed:
                live_ttl = 20
                downsample = staticmethod(EsportsFeed.downsample)

                def live(s):
                    return [Match("M1", league, "2026-10-03T08:00:00Z", "inProgress", 3,
                                  [TeamRef(n, n[:3]) for n in names])]

                def schedule(s, lg, known=None):
                    return []

                def games(s, mid):
                    return [g]

                def probe_games(s, mid):
                    return [(g, st)]

                def current_game(s, mid):
                    return g, st

                def window(s, gid, **k):
                    return st

                def _get(s, url, ttl):
                    return {"data": {"event": {"match": {"teams": [
                        {"id": "1", "name": names[0]}, {"id": "2", "name": names[1]}]}}}}

                def frame_sides(s, gid):
                    return {"blue": "1", "red": "2"}

                def ddragon_version(s):
                    return "16.16.1"

                def players(s, gid):
                    return []

                def game_metadata(s, gid):
                    return {}

                def match_wins(s, mid):
                    return {}

                def gold_timeline(s, gid, upto=None):
                    return [{"minute": float(m), "golddiff": 200 * m, "blue_kills": m // 3, "red_kills": 1,
                             "csdiff": 2 * m, "blue_gold": 1800 * m} for m in range(1, minute + 1)]

            return Feed()

        self.make_feed = make_feed

    def tearDown(self):
        if hasattr(self, "_old"):
            self.api.STATE.clear()
            self.api.STATE.update(self._old[0])
            self.api.log_prediction = self._old[1]

    def _board(self, league, names, minute):
        self.api.STATE["feed"] = self.make_feed(league, names, minute)
        return self.api.esports_board("M1", curves=True, points=60)

    def test_跨赛区没有赛前和BP后也不渐变(self):
        out = self._board("Worlds", ["T1", "BILIBILI GAMING"], 8)
        self.assertEqual([t["model_name"] for t in out["match"]["teams"]], ["T1", "Bilibili Gaming"],
                         "国际赛按四大赛区的并集映射队名, 别名照换")
        self.assertIsNone(out["match"]["pregame_league"])
        pr = out["prediction"]
        self.assertIsNotNone(pr.get("probability_blue"), "局内模型照常给数")
        for k in ("pregame_probability_blue", "postdraft_probability_blue", "shift_from_pregame",
                  "blend_weight", "ingame_probability_blue"):
            self.assertNotIn(k, pr, f"跨赛区不该有 {k}")
        self.assertTrue(pr["warnings"][0].startswith("跨赛区对阵: "), pr["warnings"])
        from ingame_service import NO_PRE_STATS
        self.assertNotIn(NO_PRE_STATS, pr["warnings"],
                         "赛前特征是故意不带的, 不能再说一句'没有可用的赛前队伍统计'")
        self.assertTrue(any(p["probability_blue"] is not None for p in out["timeline"]), "曲线照画")
        self.assertEqual([r["source"] for r in self.logged], ["ingame"], "两队都认识, 局内预测照常留档")
        self.assertEqual(self.logged[0]["league"], "Worlds")

    def test_跨赛区开局前没有任何概率并说明原因(self):
        out = self._board("Worlds", ["T1", "BILIBILI GAMING"], 2)
        pr = out["prediction"]
        self.assertTrue(pr["too_early"])
        self.assertIsNone(pr["probability_blue"])
        self.assertIsNone(pr["source"])
        self.assertIn("跨赛区对阵: ", pr["note"])
        self.assertNotIn("这里显示的是", pr["note"], "不能声称显示了一个并不存在的概率")
        self.assertEqual(self.calls, [], "没有数可记就不调留档 (source 也会和响应里的 null 对不上)")

    def test_队名不认识也有局内胜率但不留档(self):
        out = self._board("DCGI", ["GAM Esports", "T1"], 12)
        self.assertFalse(out["match"]["predictable"])
        self.assertIsNone(out["match"]["pregame_league"])
        pr = out["prediction"]
        self.assertIsNotNone(pr.get("probability_blue"), "队名映射不出来也要有 Stage 4")
        self.assertEqual(pr["blue_team"], "GAM Esports", "原名只当标签用")
        self.assertNotIn("blend_weight", pr)
        self.assertEqual(pr["warnings"][0],
                         "GAM Esports 不在四大赛区的数据里, 没有赛前和 BP 后预测; 开局 3 分钟起直接用局内模型")
        self.assertTrue(any(p["probability_blue"] is not None for p in out["timeline"]))
        from ingame_service import NO_PRE_STATS
        self.assertNotIn(NO_PRE_STATS, pr["warnings"])
        self.assertEqual(self.calls, [], "有队名映射不出来就不能写留档")

    def test_同母赛区的提醒不删没有赛前统计那句(self):
        from ingame_service import NO_PRE_STATS
        w = ["本次没有经验差数据, 改用不含经验差的局内模型。", NO_PRE_STATS]
        self.assertEqual(self.api._board_warnings(w, "cross_region", "X"), ["X", w[0]])
        self.assertEqual(self.api._board_warnings(w, "unmapped", "Y"), ["Y", w[0]])
        self.assertEqual(self.api._board_warnings(w, "intl_home", "Z"), ["Z"] + w,
                         "同母赛区是真的想带赛前特征, 没拿到就该照实说")
        self.assertEqual(self.api._board_warnings(w, None, None), w, "国内看板一个字都不变")

    def test_一个赛区的赛程取不到不连累别的赛区(self):
        # 坑: 九个赛区的赛程包在同一个 try 里, 排在前面的一个国际赛取不到 (FeedError),
        # 四大的直播候选、看板的比赛信息就整批丢掉 —— 上游一次偶发失败, 国内比赛从直播列表消失
        from esports_feed import FeedError
        def broken_worlds(feed, home):
            # 直播列表为空, WORLDS 的赛程取不到, 比赛只在 home 赛区的赛程里
            m = feed.live()[0]

            def schedule(lg, known=None):
                if lg == "WORLDS":
                    raise FeedError("模拟上游失败")
                return [m] if lg == home else []
            feed.live = lambda: []
            feed.schedule = schedule
            feed.live_by_frames = lambda cands: list(cands)
            return feed

        self.api.STATE["feed"] = broken_worlds(self.make_feed("LCK", ["T1", "Gen.G"], 12), "LCK")
        self.assertEqual([x["match_id"] for x in self.api.esports_live()["matches"]], ["M1"])

        # DCGI 在赛区表里排在 WORLDS 后面
        self.api.STATE["feed"] = broken_worlds(self.make_feed("DCGI", ["T1", "BILIBILI GAMING"], 12), "DCGI")
        out = self.api.esports_board("M1", curves=False)
        self.assertIsNotNone(out["match"], "WORLDS 取不到不该让 DCGI 的比赛找不到")
        self.assertEqual(out["match"]["league"], "DCGI")


# ══════════════════════════════════════════════════════════
#  跨赛区模型 C (xregion.py) —— 引擎、导出状态、表的规则
# ══════════════════════════════════════════════════════════

def _oe_raw(games):
    """合成的 OE 原始行 (= xregion._read_raw 的输出: 赛区已改名、日期已解析)。

    games: [(gameid, 时间, 赛区代码, 蓝队, 红队, 蓝方胜, 局序, 蓝方五人 id 或 None, 红方五人 id 或 None)]。
    每局两行 team + 十行选手; 不给选手 id 就用 "队名:位置"。
    """
    import pandas as pd
    from feature_store import ROLES
    rows = []
    for gid, when, lg, blue, red, bw, gno, bp, rp in games:
        ts = pd.Timestamp(when)
        for side, team, pid0, res, ids in (("Blue", blue, 100, int(bw), bp), ("Red", red, 200, 1 - int(bw), rp)):
            base = dict(gameid=gid, datacompleteness="complete", league=lg, playoffs=0, date=ts, game=gno,
                        patch="25.01", side=side, teamname=team, teamid=f"id:{team}", result=res)
            rows.append(dict(base, participantid=pid0, position="team", playername=None, playerid=None,
                             champion=None))
            for k, role in enumerate(ROLES):
                pid = ids[k] if ids else f"{team}:{role}"
                rows.append(dict(base, participantid=(1 if side == "Blue" else 6) + k, position=role,
                                 playername=pid, playerid=pid, champion="Ahri"))
    return pd.DataFrame(rows)


def _xregion_world():
    """一个小世界: 2023 一支日本队 (到 2025 已过期); 2024、2025 两季 LCK / LPL / PCS / VCS 内战和 MSI;
    2025 年中 P1 (PCS)、V1 (VCS) 迁入新赛区 LCP, 另有新队 C3; 2025 WLDs 的第一局是同赛区局, 之后有
    同一时间戳且共用一个赛区的两局、跨 UTC 零点的 BO3、LCP 第一次打跨赛区。WLDs 期间没有国内比赛 ——
    这样"状态 + 重放"必须和整表在线走一遍逐位相同。"""
    import random
    import pandas as pd
    rng = random.Random(7)
    games, n = [], [0]

    def add(when, lg, b, r, gno=1):
        n[0] += 1
        games.append((f"G{n[0]:04d}", when, lg, b, r, rng.random() < 0.55, gno, None, None))

    def season(start, pools):
        h = 0
        for lg, ts in pools.items():
            for a in ts:
                for b in ts:
                    if a < b:
                        for blue, red in ((a, b), (b, a)):
                            h += 1
                            add(pd.Timestamp(start) + pd.Timedelta(hours=7 * h), lg, blue, red)

    for k in range(4):
        add(f"2023-03-0{k + 1} 08:00", "LJL", "J1", "J2")
    big = {"LCK": ["K1", "K2", "K3"], "LPL": ["L1", "L2", "L3"]}
    season("2024-01-01", {**big, "PCS": ["P1", "P2"], "VCS": ["V1", "V2"]})
    for w, b, r in (("2024-05-02 08:00", "K1", "L1"), ("2024-05-02 08:00", "P1", "V1"),
                    ("2024-05-03 08:00", "K1", "P1"), ("2024-05-03 09:00", "L1", "V1"),
                    ("2024-05-04 08:00", "K2", "L2")):
        add(w, "MSI", b, r)
    season("2025-01-01", {**big, "PCS": ["P1", "P2"], "VCS": ["V1", "V2"]})
    add("2025-03-20 06:00", "KeSPA", "K1", "K2")          # 杯赛, 同池: 更新 r; 不是母赛区
    add("2025-03-21 06:00", "KeSPA", "K1", "L1")          # 杯赛, 不同池: 不更新
    for w, b, r in (("2025-05-02 08:00", "K1", "L1"), ("2025-05-02 08:00", "K2", "P1"),
                    ("2025-05-03 08:00", "L2", "V1"), ("2025-05-04 08:00", "K1", "V2")):
        add(w, "MSI", b, r)
    season("2025-06-01", {"LCP": ["P1", "V1", "C3"]})
    season("2025-08-01", big)
    add("2025-10-01 06:00", "WLDs", "K1", "K2")           # 赛事第一局: 同赛区 (切点要算上它)
    add("2025-10-02 08:00", "WLDs", "P1", "K3")           # LCP 第一次跨赛区: 先验 = PCS / VCS 偏移的均值
    add("2025-10-02 08:00", "WLDs", "K2", "L1")           # 同一时间戳、同样有 LCK: 互相看不到
    add("2025-10-02 23:30", "WLDs", "K1", "L2", 1)        # BO3 跨 UTC 零点
    add("2025-10-03 00:20", "WLDs", "L2", "K1", 2)
    add("2025-10-03 01:10", "WLDs", "K1", "L2", 3)
    add("2025-10-04 08:00", "WLDs", "C3", "L3")
    add("2025-10-04 09:00", "WLDs", "V1", "K2")
    return games


class XregionEngine(unittest.TestCase):
    """坑: 看板用"导出的状态 + 赛事内重放"出数, 闸门用"整表按时间走一遍"出数, 两条路只要有一处不同
    (同一时间戳的局先后更新了、衰减多乘了一次、JSON 丢了精度、系数切点放错赛事) 就是一个看着合理的错数字。
    闸门只证明了在线版本能过 —— 看板必须和它逐位相同, 不是"差不多"。"""

    @classmethod
    def setUpClass(cls):
        import pandas as pd
        import xregion as X
        cls.X = X
        cls.table, cls.diag = X.build_table(_oe_raw(_xregion_world()))
        cls.A = X.prepare(cls.table)
        cls.R = X.run_elo(X.P_C, cls.A)
        w = cls.table[cls.table["event"] == "2025 WLDs"]
        cls.wlds = w
        cls.first = pd.Timestamp(w["date"].min())

    def _state(self, **kw):
        import json
        return json.loads(json.dumps(self.X.build_state(self.table, **kw), allow_nan=False))

    def test_状态加重放和整表在线逐位相同(self):
        import json
        import numpy as np
        X, t = self.X, self.table
        # 看板 (from_state 之后) 的 exp 是 portable_exp (浏览器版要逐位复现, 见 xregion.portable_exp); 整表和
        # 导出也拿它走, 两条路才是同一段算术 —— 比的是重放的机制 (分组、衰减的时刻、JSON 精度、系数切点)
        R = X.run_elo(X.P_C, self.A, exp=X.portable_exp)
        st = json.loads(json.dumps(X.build_state(t, asof=self.first, exp=X.portable_exp), allow_nan=False))
        # 期望的系数 = 闸门的 walk-forward: 只用切点前的跨赛区国际赛, c = 切点前的蓝方胜率
        x_o, x_r = (R["ob"] - R["or"]) / X.S, (R["rb"] - R["rr"]) / X.S
        tr = X.target(t).values & (t["date"].values < np.datetime64(self.first))
        beta = X.fit_coef(x_o, x_r, t["blue_win"].values.astype(float), tr, X.const_rate(t, self.first))
        coef = st["coefficients"]["WORLDS"]
        self.assertEqual([coef["a"], coef["b_o"], coef["b_r"]], [float(b) for b in beta],
                         "赛事还没进 OE 时, '全部数据之后'的切点必须等于闸门的'赛事第一局'")
        hist, checked = [], 0
        for i, r in self.wlds.iterrows():
            ti = self.A["t"][i]
            if r["cross_region"]:
                eng = X.Engine.from_state(st)
                eng.play([g for g in hist if g["t"] < ti])
                f = eng.features(r["blue"], r["red"], ti)
                self.assertEqual(f, (R["ob"][i], R["or"][i], R["rb"][i], R["rr"][i]), f"第 {i} 局的 o / r")
                p = X.predict(st, "WORLDS", r["blue"], r["red"], ti, [g for g in hist if g["t"] < ti])
                self.assertEqual(p, X.logistic(coef, R["ob"][i], R["or"][i], R["rb"][i], R["rr"][i]))
                checked += 1
            hist.append({"t": ti, "blue": r["blue"], "red": r["red"], "blue_win": int(r["blue_win"])})
        self.assertEqual(checked, 7, "世界里 7 局跨赛区 WLDs 都要对上")
        self.assertIn("LCP", st["migr"], "LCP 在导出时还没有偏移, 先验靠 migr")
        self.assertFalse(st["leagues"]["LCP"]["has_crossregion_history"])

    def test_同一时间戳的局先全部取赛前值再统一更新(self):
        import math
        X = self.X
        st = self._state(asof=self.first)
        t0 = X.to_t("2025-10-02 08:00")
        a = {"t": t0, "blue": "P1", "red": "K3", "blue_win": 1}
        b = {"t": t0, "blue": "K2", "red": "L1", "blue_win": 0}
        e_batch = X.Engine.from_state(st)
        e_batch.play([a, b])
        # 期望: 两局的赛前值都在任何更新之前取 —— 手算
        e = X.Engine.from_state(st)
        fa, fb = e.features("P1", "K3", t0), e.features("K2", "L1", t0)
        pa = 1 / (1 + X.portable_exp(-(fa[0] + fa[2] - fa[1] - fa[3]) / X.S))     # 看板的引擎用 portable_exp
        pb = 1 / (1 + X.portable_exp(-(fb[0] + fb[2] - fb[1] - fb[3]) / X.S))
        lck = e.o["LCK"] - X.P_C["eta"] * (1 - pa) + X.P_C["eta"] * (0 - pb)
        self.assertEqual(e_batch.o["LCK"], lck)
        e_seq = X.Engine.from_state(st)
        e_seq.play([a, dict(b, t=t0 + 1e-6)])
        self.assertNotEqual(e_seq.o["LCK"], e_batch.o["LCK"], "错开一点时间, 第二局就看得到第一局 —— 结果必须不同")

    def test_重放的局不能乱序(self):
        X = self.X
        e = X.Engine.from_state(self._state(asof=self.first))
        with self.assertRaises(ValueError, msg="递减的 t 直接报错, 不悄悄重排"):
            e.play([{"t": 20000.5, "blue": "K1", "red": "L1", "blue_win": 1},
                    {"t": 20000.4, "blue": "K2", "red": "L2", "blue_win": 1}])

    def test_衰减按半衰期且不往回衰减(self):
        X = self.X
        e = X.Engine()
        e.o["LCS"], e.oprior["LCS"], e.olast["LCS"] = 100.0, 0.0, 1000.0
        self.assertAlmostEqual(e.off("LCS", 1000.0 + 365.0), 50.0, places=10)
        self.assertEqual(e.olast["LCS"], 1365.0)
        v = e.o["LCS"]
        self.assertEqual(e.off("LCS", 1200.0), v, "早于上次衰减的时刻: 不动")
        e.o["LPL"], e.oprior["LPL"], e.olast["LPL"] = -100.0, -600.0, 0.0
        self.assertAlmostEqual(e.off("LPL", 730.0), -600.0 + 500.0 * 0.25, places=9, msg="向先验衰减, 不是向 0")

    def test_第一次出现的赛区取先验(self):
        X = self.X
        e = X.Engine()
        e.o.update({"PCS": -500.0, "VCS": -560.0})
        e.migr["LCP"] += ["PCS", "VCS", "PCS"]
        self.assertEqual(e.off("LCP", 10.0), X.pairwise_mean([-500.0, -560.0, -500.0]),
                         "新赛区 = 迁入队伍原赛区偏移的均值 (按队计)")
        self.assertEqual(e.off("LEC", 10.0), 0.0, "四大的先验是 0")
        self.assertEqual(e.off("TCL", 10.0), -X.P_C["dnm"], "其余赛区的先验是 −600")

    def test_pairwise_mean和np_mean逐位相同(self):
        # 浏览器版照 pairwise_mean 的顺序加; 顺序累加在 8 个以上时会差最后一位
        import random
        import numpy as np
        rng = random.Random(3)
        for n in list(range(1, 140)) + [255, 256, 257, 600]:
            xs = [rng.uniform(-700, 100) for _ in range(n)]
            self.assertEqual(self.X.pairwise_mean(xs), float(np.mean(xs)), n)

    def test_系数切点按赛事(self):
        import numpy as np
        X, t = self.X, self.table
        st = self._state()                            # 全部数据: OE 最后一局在 2025-10, "当年" = 2025
        tg = X.target(t).values
        dates = t["date"].values
        c = st["coefficients"]
        self.assertEqual(c["WORLDS"]["event"], "2025 WLDs")
        self.assertEqual(c["WORLDS"]["cutoff"], X._iso(self.first), "切点是赛事在 OE 的第一局, 同赛区局也算")
        self.assertEqual(c["WORLDS"]["n_train"], int((tg & (dates < np.datetime64(self.first))).sum()))
        msi = t.loc[t["event"] == "2025 MSI", "date"].min()
        self.assertEqual((c["MSI"]["event"], c["MSI"]["cutoff"]), ("2025 MSI", X._iso(msi)))
        for k in ("FIRST STAND", "ESPORTS WORLD CUP", "DCGI"):
            self.assertIsNone(c[k]["cutoff"], f"{k}: 当年的赛事不在 OE → 全部数据之后")
            self.assertEqual(c[k]["n_train"], int(tg.sum()))
        st24 = self._state(asof="2025-01-01")
        self.assertEqual(st24["coefficients"]["MSI"]["event"], "2024 MSI", "'当年'跟着 OE 最后一局走")
        self.assertIsNone(st24["coefficients"]["WORLDS"]["cutoff"])

    def test_母赛区过期或同赛区不给数(self):
        X = self.X
        st = self._state()
        self.assertNotIn("J1", st["teams"], "最后一场常规联赛早于 365 天的队不导出")
        e = X.Engine.from_state(st)
        tw = X.to_t("2025-10-10")
        self.assertIsNone(e.home("J1", tw))
        self.assertIsNone(X.predict(st, "WORLDS", "J1", "K1", tw))
        self.assertIsNone(X.predict(st, "WORLDS", "K1", "K2", tw), "同母赛区: C 不给 (闸门没评过)")
        self.assertIsNone(X.predict(st, "LCK", "K1", "L1", tw), "没有系数的赛区键")
        self.assertIsNotNone(X.predict(st, "worlds", "K1", "L1", tw), "赛区键大小写无关")
        e.last["K1"] = tw - X.STALE_HOME_DAYS - 1
        self.assertIsNone(e.home("K1", tw))

    def test_状态列出OE里已有的国际赛局(self):
        # 重放靠它去重: OE 已经算进状态的局再重放一遍, 评分就被算了两次
        st = self._state()
        ev = st["events"]["2025 WLDs"]
        self.assertEqual(st["game_fields"], ["blue", "red", "game", "date", "blue_win", "status"])
        self.assertEqual(len(ev["games"]), len(self.wlds))
        self.assertEqual(ev["league_key"], "WORLDS")
        self.assertIn(["L2", "K1", 2, "2025-10-03T00:20:00Z", int(self.wlds.iloc[4]["blue_win"]), "cross"], ev["games"])
        self.assertEqual(ev["games"][0][-1], "same")
        self.assertEqual(sum(len(v["games"]) for v in st["events"].values()),
                         int(self.table["is_international"].sum()))

    def test_导出再读回不丢精度(self):
        X = self.X
        st = self._state()
        e1 = self.R["engine"]
        e2 = X.Engine.from_state(st)
        for team in st["teams"]:
            self.assertEqual(e2.r[team], e1.r[team])
        for L in e1.o:
            self.assertEqual((e2.o[L], e2.oprior[L], e2.olast[L]), (e1.o[L], e1.oprior[L], e1.olast[L]))

    def test_to_t和整列算法逐位相同(self):
        import pandas as pd
        X = self.X
        s = pd.Series(pd.to_datetime(["2026-10-02 18:20:44", "2025-01-01 00:00:00", "2023-03-05 23:59:59"])
                      .astype("datetime64[us]"))
        col = ((s - X.EPOCH).dt.total_seconds() / 86400).tolist()
        self.assertEqual([X.to_t(x) for x in s], col)
        self.assertEqual(X.to_t("2026-10-02T18:20:44Z"), col[0])
        self.assertEqual(X.to_t("2026-10-03T02:20:44+08:00"), col[0], "带时区的按 UTC 换算")


class XregionPortableExp(unittest.TestCase):
    """坑: 看板的 C 由 Python 和浏览器两份实现算, 而各平台的 exp 末位互不相同 (本机 math.exp 是 MSVC 运行库的,
    20 万个输入里 1147 个不是正确舍入, 和浏览器 V8 有 7% 的输入差一位)。用平台的 exp, 两边就在约一成的看板上差
    最后一位 —— "两份实现本来就对不上"成了常态, 真正的移植错误就藏在里面。所以从导出状态出发的计算 (重放、衰减、
    logistic) 两边都用同一段只含 IEEE 四则运算的 fdlibm exp; 闸门和每天导出状态 (整表走一遍) 仍用 math.exp,
    闸门的复现一个比特不动。"""

    def test_fdlibm的值(self):
        import math
        import xregion as X
        # 这个输入上 MSVC 的 math.exp 给 0.38356939397440054; fdlibm (也是 V8 的 Math.exp) 给 ...06
        self.assertEqual(X.portable_exp(-0.9582347254583471), 0.3835693939744006)
        self.assertEqual(X.portable_exp(0.0), 1.0)
        self.assertEqual(X.portable_exp(-0.0), 1.0)
        # 纯 fdlibm 的 exp(1) 比 math.e 大一位 (V8 对 x = 1 特判成 Math.E, 这里不跟 —— 两份移植一致就够)
        self.assertEqual(X.portable_exp(1.0), 2.7182818284590455)
        self.assertEqual(X.portable_exp(-746.0), 0.0)
        self.assertEqual(X.portable_exp(710.0), math.inf)
        self.assertTrue(math.isfinite(X.portable_exp(709.7)), "k = 1024 那一档不能溢出 (2.0 ** 1024 要拆两步)")
        self.assertEqual(X.portable_exp(-math.inf), 0.0)
        self.assertTrue(math.isnan(X.portable_exp(math.nan)))

    def test_和math_exp至多差一位(self):
        import math
        import random
        import xregion as X
        rng = random.Random(5)
        for _ in range(20000):
            x = rng.uniform(-40, 40)
            a, b = X.portable_exp(x), math.exp(x)
            self.assertLessEqual(abs(a - b), math.ulp(b), x)

    def test_看板的引擎用它_闸门的引擎不用(self):
        import math
        import xregion as X
        self.assertIs(X.Engine().exp, math.exp, "闸门 / 导出状态: math.exp, 闸门复现不动")
        st = {"params": {**X.P_C, "majors": ["LCK", "LPL", "LEC", "LCS"]}, "teams": {}, "leagues": {}, "migr": {}}
        self.assertIs(X.Engine.from_state(st).exp, X.portable_exp, "从导出状态出发 = 看板, 浏览器版要逐位复现")
        coef = {"a": 0.1, "b_o": 0.7, "b_r": 0.95}
        z = 0.1 + 0.7 * ((15.094420463862296 - -138.9824578406959) / X.S) + 0.95 * ((54.288647534997935 - 20.318060134603556) / X.S)
        self.assertEqual(X.logistic(coef, 15.094420463862296, -138.9824578406959, 54.288647534997935, 20.318060134603556),
                         1 / (1 + X.portable_exp(-z)))


class XregionTableRules(unittest.TestCase):
    """坑: 母赛区 / 系列赛 / 赛区改名错了不报错 —— 一支 LCK 队的母赛区被记成 "KeSPA", 赛区强度就多出一个
    假赛区; 跨 UTC 零点的 BO5 被拆成两个系列赛; 少了 LTA 改名, 2025 一整年北美 / 巴西的比赛消失。"""

    def _table(self, games):
        import xregion as X
        return X.build_table(_oe_raw(games))

    def test_母赛区跳过杯赛(self):
        g, _ = self._table([
            ("A", "2025-09-01 08:00", "LCK", "K1", "K2", 1, 1, None, None),
            ("B", "2025-09-02 08:00", "LPL", "L1", "L2", 1, 1, None, None),
            ("C", "2025-09-20 08:00", "KeSPA", "K1", "K2", 1, 1, None, None),
            ("D", "2025-10-05 08:00", "WLDs", "K1", "L1", 1, 1, None, None)])
        r = g[g["gameid"] == "D"].iloc[0]
        self.assertEqual((r["home_blue"], r["home_red"], r["region_status"]), ("LCK", "LPL", "cross"))

    def test_队名过期365天按阵容认队(self):
        s = [f"s{k}" for k in range(5)]
        g, _ = self._table([
            ("A", "2023-06-01 08:00", "VCS", "Old Secret", "V2", 1, 1, s, None),
            ("B", "2025-08-01 08:00", "LCP", "Whales", "C3", 1, 1, s, None),
            ("C", "2025-08-02 08:00", "LPL", "L1", "L2", 1, 1, None, None),
            ("D", "2025-10-05 08:00", "WLDs", "Old Secret", "L1", 1, 1, s, None),
            ("E", "2025-10-06 08:00", "WLDs", "National", "L1", 1, 1, None, None)])
        d = g[g["gameid"] == "D"].iloc[0]
        self.assertEqual((d["home_name_blue"], d["home_blue"], d["home_source_blue"]), ("VCS", "LCP", "roster"))
        e = g[g["gameid"] == "E"].iloc[0]
        self.assertEqual(e["region_status"], "unknown", "没有任何历史的 (国家队) 不走阵容规则")

    def test_系列赛跨零点不拆开_同日再打另算(self):
        g, _ = self._table([
            ("A", "2025-10-05 23:30", "WLDs", "K1", "L1", 1, 1, None, None),
            ("B", "2025-10-06 00:20", "WLDs", "L1", "K1", 1, 2, None, None),
            ("C", "2025-10-06 08:00", "WLDs", "K1", "L1", 1, 1, None, None)])
        s = g.set_index("gameid")["series"]
        self.assertEqual(s["A"], s["B"])
        self.assertNotEqual(s["B"], s["C"], "局序不增 = 新系列赛")

    def test_读CSV时LTA改名(self):
        import tempfile
        import xregion as X
        raw = _oe_raw([("A", "2025-02-01 08:00", "LTA N", "FlyQuest", "Cloud9", 1, 1, None, None),
                       ("B", "2025-02-01 09:00", "LTA S", "LOUD", "paiN Gaming", 1, 1, None, None)])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "oe.csv"
            raw.to_csv(p, index=False)
            got = X._read_raw([p], verbose=False)
        self.assertEqual(sorted(set(got["league"])), ["CBLOL", "LCS"])

    def _dcgi_like(self, code):
        games = [("A", "2025-05-01 08:00", "LCK", "K1", "K2", 1, 1, None, None),
                 ("B", "2025-05-01 09:00", "LPL", "L1", "L2", 1, 1, None, None),
                 ("C", "2025-05-02 08:00", "LCK", "K3", "K4", 1, 1, None, None),
                 ("D", "2025-05-02 09:00", "LPL", "L3", "L4", 1, 1, None, None),
                 ("E", "2025-05-10 08:00", "MSI", "K1", "L1", 1, 1, None, None)]
        pairs = [("K1", "L1"), ("K2", "L2"), ("K3", "L3"), ("K4", "L4"), ("K1", "L2"), ("K2", "L1")]
        games += [(f"X{k}", "2025-12-01 08:00", code, b, r, 1, 1, None, None) for k, (b, r) in enumerate(pairs)]
        return self._table(games)

    def test_没归类的新赛事代码会被报出来且拒绝导出(self):
        # 全球邀请赛若以一个没登记的新代码进了 OE, 会被当成常规联赛: 参赛队的母赛区和 Elo 池整个挪过去, 不报错
        import xregion as X
        g, diag = self._dcgi_like("DCG")
        self.assertEqual([x["event"] for x in diag["unlisted_lookalikes"]], ["2025 DCG"])
        st = X.build_state(_with_diag(g, diag))
        self.assertEqual(st["teams"]["K1"]["pool"], "DCG", "错就错在这里: K1 的 Elo 池被挪进了一个假赛区")
        self.assertTrue(any("没列入国际赛" in p for p in X.sanity_check(st)))

    def test_已在杯赛表里的代码有跨赛区对打_只提醒不拦(self):
        # 2026-10-04 复核: OE 若把 DCGI 记成 DCup (已在 NON_HOME, 协议排除), 原来的检查照样拒绝导出, 而按提示
        # "加进 NON_HOME" 也清不掉 —— 日更从此每天停在这一步 (Stage 1/2/4 和网站一起)。杯赛不挪池子、不当母赛区,
        # 状态没有错, 只是这些局不计入: 只提醒
        import xregion as X
        g, diag = self._dcgi_like("DCup")
        self.assertEqual(diag["unlisted_lookalikes"], [])
        self.assertEqual([x["event"] for x in diag["nonhome_crossregion"]], ["2025 DCup"])
        self.assertFalse(g.loc[g["league"] == "DCup", "is_international"].any(), "DCup 仍是协议排除的国内杯赛")
        st = X.build_state(_with_diag(g, diag))
        self.assertEqual(st["teams"]["K1"]["pool"], "LCK", "杯赛不挪池子")
        self.assertFalse([p for p in X.sanity_check(st) if "没列入国际赛" in p or "DCup" in p],
                         "(这个小世界的队数不够, 别的检查本来就不过; 只看它不因杯赛被拦)")
        self.assertTrue(any("杯赛 2025 DCup" in x for x in X._summary(st)), "构建时要打出来")
        self.assertNotIn("2025 DCup", st["events"], "不计入 = 也不进去重表, 看板会整个重放它 (不会算两遍)")

    def test_DCGI以登记的代码进OE_算国际赛(self):
        import xregion as X
        g, diag = self._dcgi_like("DCGI")
        self.assertEqual(diag["unlisted_lookalikes"] + diag["nonhome_crossregion"], [])
        self.assertTrue(g.loc[g["league"] == "DCGI", "is_international"].all())
        st = X.build_state(_with_diag(g, diag))
        self.assertEqual(st["teams"]["K1"]["pool"], "LCK")
        self.assertEqual(st["events"]["2025 DCGI"]["n"], 6)
        self.assertEqual(st["coefficients"]["DCGI"]["event"], "2025 DCGI", "切点按闸门: 这个赛事在 OE 里的第一局")
        self.assertEqual(st["coefficients"]["DCGI"]["cutoff"], "2025-12-01T08:00:00Z")


def _with_diag(df, diag):
    df.attrs["diag"] = diag
    return df


def _valid_xregion_state():
    """一份能过 sanity_check 的最小状态 (四大各 8 队, 外加凑够 MIN_TEAMS 的次级联赛队)。"""
    import xregion as X
    teams = {}
    for L in ("LPL", "LCK", "LEC", "LCS"):
        for k in range(8):
            teams[f"{L}-{k}"] = {"pool": L, "home": L, "r": 1.5, "ng": 30, "last_regular": "2026-09-01T00:00:00Z",
                                 "last_regular_t": 20697.0}
    for k in range(X.MIN_TEAMS):
        teams[f"LDL-{k}"] = {"pool": "LDL", "home": "LDL", "r": 0.0, "ng": 3, "last_regular": "2026-09-01T00:00:00Z",
                             "last_regular_t": 20697.0}
    leagues = {L: {"o": -50.0, "prior": 0.0, "last_t": 20650.0, "last": "2026-07-16T00:00:00Z",
                   "has_crossregion_history": True, "crossregion_games": 100, "major": True}
               for L in ("LPL", "LCK", "LEC", "LCS")}
    leagues["LDL"] = {"o": None, "prior": None, "last_t": None, "last": None, "has_crossregion_history": False,
                      "crossregion_games": 0, "major": False}
    coefs = {k: {"a": 0.14, "b_o": 0.7, "b_r": 0.95, "c": 0.53, "event": None, "cutoff": None, "n_train": 979}
             for k in X.LE_EVENT_CODES}
    return {"oe_asof": "2026-10-02T18:20:44Z", "teams": teams, "leagues": leagues, "coefficients": coefs,
            "diag": {"unlisted_lookalikes": []}}


class XregionSanityAndWiring(unittest.TestCase):
    """坑: 状态文件坏了 (系数 NaN、少一年数据、四大缺一个、有个新杯赛代码没归类) 不会让看板报错 ——
    只会让跨赛区的数悄悄变成另一个数。日更必须在换上线前拦住, 而且和其他模型一起换。"""

    def test_好状态通过_坏状态各自被拦(self):
        import copy
        import xregion as X
        good = _valid_xregion_state()
        self.assertEqual(X.sanity_check(good), [])
        cases = []
        s = copy.deepcopy(good); s["coefficients"]["WORLDS"]["b_o"] = float("nan"); cases.append(("NaN 系数", s))
        s = copy.deepcopy(good); del s["coefficients"]["DCGI"]; cases.append(("缺 DCGI 系数", s))
        s = copy.deepcopy(good); s["teams"] = {k: v for k, v in s["teams"].items() if not k.startswith("LEC")}
        cases.append(("LEC 没有队", s))
        s = copy.deepcopy(good); s["teams"] = dict(list(s["teams"].items())[:100]); cases.append(("队太少", s))
        s = copy.deepcopy(good); s["leagues"]["LCS"]["has_crossregion_history"] = False; cases.append(("LCS 没偏移", s))
        s = copy.deepcopy(good); s["diag"]["unlisted_lookalikes"] = [{"event": "2026 DCG"}]; cases.append(("新代码", s))
        s = copy.deepcopy(good)
        s["events"] = {"2026 XYZ": {"code": "XYZ", "year": 2026, "games": []}}
        cases.append(("国际赛代码既没有 lolesports 键也没写明没有", s))
        for name, s in cases:
            self.assertTrue(X.sanity_check(s), name)
        ok = copy.deepcopy(good)
        ok["diag"]["nonhome_crossregion"] = [{"event": "2026 DCup", "n": 6, "cross_first_division": 6}]
        ok["events"] = {"2026 DCGI": {"code": "DCGI", "year": 2026, "games": []},
                        "2026 AC": {"code": "AC", "year": 2026, "games": []},
                        "2025 XYZ": {"code": "XYZ", "year": 2025, "games": []}}      # 往年的不影响切点
        self.assertEqual(X.sanity_check(ok), [], "杯赛里的跨赛区对打只提醒; 有归属的国际赛代码不拦")

    def test_日更里C是旁路_坏状态整份不换_不拦其他模型(self):
        # 2026-10-04 复核: 原来 xregion.py --build 是第三个训练步骤, 它一失败 (比如 OE 收进一个没归类的新代码)
        # 整个日更就停, Stage 1/2/4 和网站天天不更新。现在它是旁路: 坏了只是线上留旧的一份 (整份, 自洽)
        import json
        import tempfile
        enc = getattr(sys.stdout, "encoding", None)
        import daily_update                         # 导入时会把本进程 stdout 改成 UTF-8
        if enc and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding=enc)
        DU = daily_update
        self.assertNotIn("xregion.json", DU.ARTIFACT_FILES, "它不能卡住其他模型的'缺文件'检查")
        self.assertNotIn("xregion.py", [s for s, _a in DU.TRAIN_STEPS])
        self.assertEqual(DU.XREGION_STEP, ("xregion.py", ["--build"]))
        old = (DU.STAGING, DU.run, DU.log)
        try:
            with tempfile.TemporaryDirectory() as d:
                DU.STAGING = Path(d)
                DU.log = lambda *a, **k: None
                p = Path(d) / "xregion.json"
                p.write_text(json.dumps(_valid_xregion_state()), encoding="utf-8")
                self.assertEqual(DU._xregion_problems(p), [])
                DU.run = lambda cmd, env=None, timeout=0: (0, "ok")
                self.assertIsNone(DU._build_xregion({}), "好的: 可以换")
                self.assertTrue(p.exists())
                bad = _valid_xregion_state()
                bad["coefficients"]["MSI"]["a"] = float("inf")
                p.write_text(json.dumps(bad), encoding="utf-8")
                self.assertIn("MSI", DU._build_xregion({}))
                self.assertFalse(p.exists(), "不过检查的那份删掉, 换的时候不会碰线上的旧文件")
                p.write_text('{"teams": {', encoding="utf-8")
                self.assertTrue(DU._xregion_problems(p), "截断的文件")
                DU.run = lambda cmd, env=None, timeout=0: (1, "有没列入国际赛、却有跨赛区一级联赛对打的赛事")
                self.assertIn("rc=1", DU._build_xregion({}))
                self.assertFalse(p.exists())
                self.assertFalse(any("xregion" in b for b in DU.gate({}, {}, None, None)), "缺了它也不拦其他模型")
        finally:
            DU.STAGING, DU.run, DU.log = old

    def test_C的状态没换上_health看得见(self):
        import tempfile
        import update_status as U
        old = U.STATUS_FILE
        try:
            with tempfile.TemporaryDirectory() as d:
                U.STATUS_FILE = Path(d) / "s.json"
                U.record_xregion("ok")
                U.record("ok", fetch_ok=True, exit_code=0)
                first = U.read()["xregion"]["last_success"]
                self.assertIsNotNone(first, "整轮的留档不能把它冲掉")
                U.record_xregion("failed", "xregion.json: 有没列入国际赛的赛事")
                U.record("ok", fetch_ok=True, exit_code=0)
                s = U.summary()
                self.assertEqual(s["xregion"]["outcome"], "failed")
                self.assertEqual(s["xregion"]["last_success"], first, "线上那份是哪天的")
                self.assertIn("跨赛区模型 C 的状态没换上", s["message"])
        finally:
            U.STATUS_FILE = old

    def test_参数是闸门冻结的那一组(self):
        # 改任何一个数都等于换了一个没过闸的模型 (闸门的 DEV 复现也会对不上)
        import math
        import xregion as X
        self.assertEqual(X.P_C, dict(K=2.0, knew=1.0, nnew=10, h=0.0, init=0.0, cup=True, intl_same=False,
                                     eta=24.0, kint=24.0, hl=365.0, dnm=600.0, lag="none"))
        self.assertEqual(X.LAM_C, 30.0)
        self.assertEqual(X.S, 400 / math.log(10))
        self.assertEqual(X.STALE_HOME_DAYS, 365)

    def test_每个非四大的lolesports键都有系数映射(self):
        # LEAGUE_IDS 加了新的国际赛键而这里没加, 看板对它就没有系数 (或者按错的赛事切点)
        import xregion as X
        from esports_feed import LEAGUE_IDS, is_major
        self.assertEqual({k for k in LEAGUE_IDS if not is_major(k)}, set(X.LE_EVENT_CODES))
        self.assertEqual(X.oe_event("Worlds", 2026), "2026 WLDs")
        self.assertEqual(X.oe_event("ESPORTS WORLD CUP", 2026), "2026 EWC")
        self.assertEqual(X.oe_event("DCGI", 2026), "2026 DCGI")
        self.assertNotIn("DCup", X.LE_EVENT_CODES.values(), "DCGI 不能猜成 OE 的国内杯赛 DCup")
        self.assertFalse(X.is_international("DCup", 2026))

    def test_每个国际赛代码要么有lolesports键要么写明没有(self):
        # 加了国际赛代码却忘了 LE_EVENT_CODES: 那个赛区键的切点落到全部数据之后, 把赛事自己的局也拟合进系数
        import xregion as X
        codes = {c for c in X.LE_EVENT_CODES.values() if c}
        self.assertEqual(set(X.INTL_EVENTS), codes | X.NO_LE_KEY)
        self.assertFalse(codes & X.NO_LE_KEY)

    def test_运行时模块不依赖research(self):
        import ast
        src = (_ROOT / "xregion.py").read_text(encoding="utf-8")
        mods = set()
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Import):
                mods |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods.add(node.module.split(".")[0])
        research = {p.stem for p in (_ROOT / "research").glob("*.py")} | {"research"}
        self.assertFalse(mods & research, f"xregion.py import 了研究代码: {mods & research}")


# ══════════════════════════════════════════════════════════
#  跨赛区模型 C 上看板: 赛事内重放、去重、解码、看板接线、留档
# ══════════════════════════════════════════════════════════

_XSTATE_CACHE: dict = {}


def _x_state(asof):
    """合成小世界 (_xregion_world) 在 asof 之前导出的状态, 走一遍 JSON (和看板读到的一样)。"""
    import json
    import xregion as X
    if asof not in _XSTATE_CACHE:
        table, diag = X.build_table(_oe_raw(_xregion_world()))
        st = X.build_state(_with_diag(table, diag), asof=asof)
        _XSTATE_CACHE[asof] = (json.loads(json.dumps(st, allow_nan=False)), table)
    return _XSTATE_CACHE[asof]


def _xs(mid, start, a, b, wa, wb, bo=3, frames=None, ids=("TA", "TB"), oe=True):
    """看板重放的一个系列赛 (xregion.py 看板那一段开头写的形状)。"""
    s = {"match_id": mid, "start": start, "best_of": bo,
         "teams": [{"name": a, "code": a, "oe": a if oe else None, "wins": wa, "id": ids[0]},
                   {"name": b, "code": b, "oe": b, "wins": wb, "id": ids[1]}]}
    if frames is not None:
        s["frames"] = frames
    return s


def _xf(blue_id, red_id, towers, inhib=(0, 0), gold=(60000, 60000)):
    return {"blue_id": blue_id, "red_id": red_id, "towers": list(towers), "inhibitors": list(inhib),
            "gold": list(gold)}


class XregionReplay(unittest.TestCase):
    """坑: 看板把 OE 已经算进状态的局再重放一遍 (同一局算两次)、重放顺序随输入顺序变、解码出来的胜局数和
    比分对不上、缺一局帧就猜一个胜负 —— 全都只会给出另一个看着合理的数。"""

    def test_OE里已有的局不重放_UTC零点两侧都认得(self):
        import xregion as X
        st, table = _x_state(None)                   # 状态里已经有 2025 WLDs 的全部局
        bo3 = table[(table["event"] == "2025 WLDs") & table["blue"].isin(["K1", "L2"])
                    & table["red"].isin(["K1", "L2"])].sort_values("game")
        self.assertEqual(list(bo3["game"]), [1, 2, 3])
        self.assertEqual(sorted({str(d.date()) for d in bo3["date"]}), ["2025-10-02", "2025-10-03"],
                         "这个 BO3 跨 UTC 零点")
        k1 = int(sum((b == "K1") == bool(y) for b, y in zip(bo3["blue"], bo3["blue_win"])))
        frames = {str(g): _xf("TA" if b == "K1" else "TB", "TB" if b == "K1" else "TA",
                              (9, 3) if (b == "K1") == bool(y) else (3, 9))
                  for g, b, y in zip(bo3["game"], bo3["blue"], bo3["blue_win"])}
        cur = _xs("cur", "2025-10-06T08:00:00Z", "K2", "L3", 0, 0)
        ser = [_xs("bo3", "2025-10-02T23:00:00Z", "K1", "L2", k1, 3 - k1, frames=frames), cur]
        res = X.board_predict(st, "WORLDS", "K2", "L3", ser, "cur", 1)
        self.assertEqual((res["deduped"], res["replayed"]), (3, 0))
        self.assertEqual(X.replay_frames_needed(st, ser, "cur", 1), [], "整场都在 OE 里: 一个终局帧都不取")
        t = X.game_t("2025-10-06T08:00:00Z", 1)
        self.assertEqual(res["probability_blue"], X.predict(st, "WORLDS", "K2", "L3", t))
        # 反证: 赛程时间挪出去重窗口, 同样三局就会被再算一次 —— 数变了, 不报错
        ser2 = [dict(ser[0], start="2025-10-04T23:00:00Z"), cur]
        res2 = X.board_predict(st, "WORLDS", "K2", "L3", ser2, "cur", 1)
        self.assertEqual(res2["replayed"], 3)
        self.assertNotEqual(res2["probability_blue"], res["probability_blue"])

    def test_去重窗口的边界(self):
        import xregion as X
        mini = {"teams": {"P": {}, "Q": {}, "R": {}},
                "events": {"e": {"games": [["P", "Q", 1, "2026-10-04T00:20:00Z", 1, "cross"]]}}}
        idx = X.oe_index(mini)
        self.assertTrue(X.in_oe(idx, "Q", "P", 1, "2026-10-03T23:00:00Z"), "蓝红对调也算同一局")
        self.assertTrue(X.in_oe(idx, "P", "Q", 1, "2026-10-04T06:20:00Z"))
        self.assertFalse(X.in_oe(idx, "P", "Q", 1, "2026-10-04T06:20:01Z"))
        self.assertTrue(X.in_oe(idx, "P", "Q", 1, "2026-10-03T06:20:00Z"))
        self.assertFalse(X.in_oe(idx, "P", "Q", 1, "2026-10-03T06:19:59Z"))
        self.assertFalse(X.in_oe(idx, "P", "Q", 2, "2026-10-03T23:00:00Z"), "局号不同")
        self.assertFalse(X.in_oe(idx, "R", "Q", 1, "2026-10-03T23:00:00Z"), "OE 那边是另一支现役队 (P): 另一场比赛")

    def test_OE用过期旧名记的局也认得(self):
        # 2026 EWC: OE 把 Team Secret Whales 记成过期的旧名 "Team Secret" (状态里早被剪掉), lolesports 的
        # "Team Secret" 按别名映射成 "Team Secret Whales"。只认完全相同的队名, 这 4 局就被重放第二遍
        import xregion as X
        mini = {"teams": {"Team Secret Whales": {}, "Karmine Corp": {}, "Sentinels": {}},
                "events": {"2026 EWC": {"games": [
                    ["Karmine Corp", "Team Secret", 1, "2026-07-15T14:52:10Z", 1, "cross"],
                    ["Team Secret", "Karmine Corp", 2, "2026-07-15T15:48:30Z", 0, "cross"]]}}}
        from esports_feed import TEAM_ALIASES
        idx = X.oe_index(mini, TEAM_ALIASES)
        for n in (1, 2):
            self.assertTrue(X.in_oe(idx, "Karmine Corp", "Team Secret Whales", n, "2026-07-15T14:25:00Z"), n)
        self.assertEqual(X.oe_lookup(idx, "Team Secret Whales", "Karmine Corp", 1, "2026-07-15T14:25:00Z"), (1, 1),
                         "OE 的胜负和蓝红跟着换到看板的队序 (KC 蓝方赢)")
        self.assertFalse(X.in_oe(idx, "Sentinels", "Team Secret Whales", 1, "2026-07-15T14:25:00Z"),
                         "两边都对不上 (只有一边是旧名的通配) 不算")
        self.assertFalse(X.in_oe(X.oe_index(mini), "Karmine Corp", "Team Secret Whales", 1, "2026-07-15T14:25:00Z"),
                         "没有别名表就不认旧名")

    def test_旧名只认别名表指向的那一队(self):
        # 2026-10-04 复核: 原来"OE 那边不是现役队"就放行。OE 已收 Sentinels 对旧名 "Team Secret" (09:20 第 1 局),
        # 同一天 13:00 Sentinels 对 FlyQuest 的第 1 局被当成"已在 OE", 再也不重放 —— 少算一局, 不报错
        import xregion as X
        from esports_feed import TEAM_ALIASES
        mini = {"teams": {"Sentinels": {}, "FlyQuest": {}, "Team Secret Whales": {}},
                "events": {"2026 EWC": {"games": [["Sentinels", "Team Secret", 1, "2026-07-15T09:20:00Z", 1, "cross"]]}}}
        idx = X.oe_index(mini, TEAM_ALIASES)
        self.assertFalse(X.in_oe(idx, "Sentinels", "FlyQuest", 1, "2026-07-15T13:00:00Z"))
        self.assertTrue(X.in_oe(idx, "Sentinels", "Team Secret Whales", 1, "2026-07-15T09:00:00Z"))

    def test_重放顺序只看开赛时间和局号_不看输入顺序(self):
        import random
        import xregion as X
        st, _ = _x_state("2025-10-01")
        ser = [_xs("a", "2025-10-02T08:00:00Z", "K1", "L1", 2, 1, frames={
                   "1": _xf("TA", "TB", (9, 2)), "2": _xf("TB", "TA", (8, 3))}),
               _xs("b", "2025-10-02T08:00:00Z", "K2", "P1", 1, 0, bo=1),        # 和 a 同一时刻开打
               _xs("c", "2025-10-02T12:00:00Z", "L2", "V1", 0, 2),
               _xs("cur", "2025-10-03T08:00:00Z", "K3", "L3", 0, 0)]
        base = X.board_predict(st, "WORLDS", "K3", "L3", ser, "cur", 1)
        ts = [g["t"] for g in base["games"]]
        self.assertEqual(ts, sorted(ts))
        self.assertEqual([(g["match_id"], g["number"]) for g in base["games"]],
                         [("a", 1), ("b", 1), ("a", 2), ("a", 3), ("c", 1), ("c", 2)],
                         "同一时刻开打的两个系列赛按局号交错; t 相同的两局一组更新")
        self.assertEqual(base["games"][0]["t"], base["games"][1]["t"])
        rng = random.Random(5)
        for _ in range(5):
            sh = ser[:]
            rng.shuffle(sh)
            self.assertEqual(X.board_predict(st, "WORLDS", "K3", "L3", sh, "cur", 1), base)

    def test_只重放当前局之前的局(self):
        import xregion as X
        st, _ = _x_state("2025-10-01")
        cur = _xs("cur", "2025-10-03T08:00:00Z", "K1", "L1", 2, 1, frames={
            "1": _xf("TA", "TB", (2, 9)), "2": _xf("TB", "TA", (3, 10))})
        later = _xs("later", "2025-10-03T20:00:00Z", "K2", "L2", 1, 0, bo=1)
        ser = [cur, later]
        for k, n in ((1, 0), (2, 1), (3, 2)):
            res = X.board_predict(st, "WORLDS", "K1", "L1", ser, "cur", k)
            self.assertEqual(res["replayed"], n, f"第 {k} 局只能看到前 {n} 局, 看不到之后开赛的系列赛")
        self.assertEqual(X.board_predict(st, "WORLDS", "K1", "L1", ser, "cur", 2)["games"][0]["blue_win"], 0,
                         "2-1 打完、第 3 局归胜者, 第 1 局帧说 B 赢 (约束解码用了整场的比分)")

    def test_解码出的胜局数等于比分_决胜局归胜者(self):
        import random
        import xregion as X
        rng = random.Random(11)
        for _ in range(400):
            bo = rng.choice([3, 5])
            need = bo // 2 + 1
            clinched = rng.random() < 0.6
            if clinched:
                lose = rng.randrange(1, need)
                wa, wb = (need, lose) if rng.random() < 0.5 else (lose, need)
            else:
                wa, wb = rng.randrange(1, need), rng.randrange(1, need)
            frames = {}
            for i in range(1, wa + wb + 1):
                if rng.random() < 0.9:
                    blue_a = rng.random() < 0.5
                    frames[str(i)] = _xf("TA" if blue_a else "TB", "TB" if blue_a else "TA",
                                         (rng.randrange(12), rng.randrange(12)), (rng.randrange(3), rng.randrange(3)),
                                         (rng.randrange(50000, 80000), rng.randrange(50000, 80000)))
            s = _xs("s", "2025-10-02T08:00:00Z", "A", "B", wa, wb, bo=bo, frames=frames)
            dec = X.decode_series(s)
            self.assertEqual([d["number"] for d in dec], list(range(1, wa + wb + 1)))
            self.assertEqual(sum(d["winner"] == 0 for d in dec), wa)
            self.assertEqual(sum(d["winner"] == 1 for d in dec), wb)
            if clinched:
                self.assertEqual(dec[-1]["winner"], 0 if wa > wb else 1, "决胜局归系列赛胜者")
            if any(X.frame_verdict(frames.get(str(i)), ["TA", "TB"]) is None for i in X.frames_needed(s)):
                self.assertTrue(all(d["how"] in ("alternate", "score") for d in dec), "缺一局帧: 整场交替")

    def test_按帧判对的不改_判多了的改把握最小的(self):
        import xregion as X
        s = _xs("s", "2025-10-02T08:00:00Z", "A", "B", 3, 2, bo=5, frames={
            "1": _xf("TA", "TB", (9, 2), (2, 0)), "2": _xf("TB", "TA", (4, 6)),
            "3": _xf("TA", "TB", (6, 6), (1, 0)), "4": _xf("TB", "TA", (2, 11), (0, 3))})
        # 帧说 A 赢了前四局; A 除决胜局外只能再赢 2 局 —— 把握最小的第 3、2 局改判给 B
        self.assertEqual([d["winner"] for d in X.decode_series(s)], [0, 1, 1, 0, 0])

    def test_缺帧退回交替序_领先方先(self):
        import xregion as X
        s = _xs("s", "2025-10-02T08:00:00Z", "A", "B", 2, 3, bo=5, frames={"1": _xf("TA", "TB", (9, 2))})
        dec = X.decode_series(s)
        self.assertEqual([d["winner"] for d in dec], [1, 0, 1, 0, 1])
        self.assertEqual([d["how"] for d in dec], ["alternate"] * 4 + ["score"])
        s = _xs("s", "2025-10-02T08:00:00Z", "A", "B", 1, 1, bo=5)
        self.assertEqual([d["winner"] for d in X.decode_series(s)], [0, 1], "平分时 teams[0] 先")
        self.assertIsNone(X.frame_verdict(_xf("TA", "TX", (9, 2)), ["TA", "TB"]), "帧里的队伍 id 对不上就不猜")

    def test_系数按看板的赛区键取(self):
        import xregion as X
        st, _ = _x_state("2025-10-01")
        ser = [_xs("a", "2025-10-02T08:00:00Z", "K1", "L1", 1, 0, bo=1),
               _xs("cur", "2025-10-03T08:00:00Z", "K2", "L2", 0, 0)]
        t = X.game_t("2025-10-03T08:00:00Z", 1)
        for key in ("WORLDS", "MSI", "DCGI"):
            e = X.Engine.from_state(st)
            e.play(X.board_replay(st, ser, "cur", 1)["games"])
            want = X.logistic(st["coefficients"][key], *e.features("K2", "L2", t))
            self.assertEqual(X.board_predict(st, key.lower(), "K2", "L2", ser, "cur", 1)["probability_blue"], want)
        self.assertIsNone(X.board_predict(st, "LCK", "K2", "L2", ser, "cur", 1)["probability_blue"])

    def test_时间必须带时区_第一局和to_t逐位相同(self):
        import xregion as X
        with self.assertRaises(ValueError):
            X.epoch_s("2026-10-03T08:00:00")         # JS 会按本地时间解析, 两边差几个小时
        self.assertEqual(X.game_t("2026-10-03T08:00:00Z", 1), X.to_t("2026-10-03T08:00:00Z"))
        self.assertEqual(X.game_t("2026-10-03T08:00:00.000Z", 3), (X.epoch_s("2026-10-03T08:00:00Z") + 2) / 86400)

    def test_母赛区没有跨赛区记录要点名(self):
        import xregion as X
        st, _ = _x_state("2025-10-01")              # 此时 LCP 还没打过跨赛区
        self.assertFalse(st["leagues"]["LCP"]["has_crossregion_history"])
        res = X.board_predict(st, "WORLDS", "K1", "C3", [_xs("cur", "2025-10-03T08:00:00Z", "K1", "C3", 0, 0)],
                              "cur", 1)
        self.assertEqual((res["home_red"], res["no_history"]), ("LCP", ["LCP"]))
        self.assertIsNotNone(res["probability_blue"])

    def test_当前系列赛比分没跟上局号就不给(self):
        # 2026-10-04 复核: 第 3 局开打了, 上游的比分 (约晚 6 分钟) 还停在 1-0, 原来悄悄只放第 1 局 —— 数差 7 个
        # 百分点, 不报错, 而开局头 3 分钟那一行赛前数每局只留档一次
        import xregion as X
        st, _ = _x_state("2025-10-01")
        fr = {"1": _xf("TA", "TB", (9, 2)), "2": _xf("TB", "TA", (9, 2))}
        ok = X.board_predict(st, "WORLDS", "K1", "L1", [_xs("cur", "2025-10-03T08:00:00Z", "K1", "L1", 1, 1, bo=5,
                                                             frames=fr)], "cur", 3)
        self.assertEqual((ok["withheld"], ok["replayed"]), (None, 2))
        self.assertIsNotNone(ok["probability_blue"])
        ser = [_xs("cur", "2025-10-03T08:00:00Z", "K1", "L1", 1, 0, bo=5, frames={"1": fr["1"]})]
        lag = X.board_predict(st, "WORLDS", "K1", "L1", ser, "cur", 3)
        self.assertEqual((lag["withheld"], lag["probability_blue"]), ("score_lag", None))
        self.assertEqual(X.replay_frames_needed(st, ser, "cur", 3), [], "反正不给, 一个终局帧都不取")
        self.assertIsNone(X.board_predict(st, "WORLDS", "K1", "L1", ser, "cur", 2)["withheld"],
                          "第 2 局: 比分 1-0 正好跟得上")

    def test_一半在OE的系列赛_剩下的局只在剩下的配额里解码(self):
        # 2026-10-04 复核: OE 收了一个 BO5 的前两局 (K2 两连胜), 比分 K2 2 : 3 L2, 后三局没有终局帧。原来整个
        # 系列赛按交替序解码 (不看 OE), 重放的三局里 K2 又赢一局 —— 状态 + 重放 = K2 3 胜, 系列赛胜负都反了
        import copy
        import xregion as X
        st, _ = _x_state("2025-10-01")
        st = copy.deepcopy(st)
        st["events"]["2025 WLDs"] = {"code": "WLDs", "year": 2025, "games": [
            ["K2", "L2", 1, "2025-10-02T08:10:00Z", 1, "cross"], ["L2", "K2", 2, "2025-10-02T09:00:00Z", 0, "cross"]]}
        other = _xs("s1", "2025-10-02T08:00:00Z", "K2", "L2", 2, 3, bo=5)
        cur = _xs("cur", "2025-10-03T08:00:00Z", "K1", "L1", 0, 0)
        rep_ = X.board_replay(st, [other, cur], "cur", 1)
        self.assertEqual(rep_["deduped"], 2)
        self.assertEqual([(g["number"], g["blue_win"], g["blue"]) for g in rep_["games"]],
                         [(3, 0, "K2"), (4, 0, "K2"), (5, 0, "K2")], "L2 要赢满剩下的三局")
        dec = X.decode_series(other, {1: (0, 0), 2: (0, 1)})
        self.assertEqual([(d["winner"], d["how"]) for d in dec], [(0, "oe"), (0, "oe"), (1, "alternate"),
                                                                   (1, "alternate"), (1, "score")])
        self.assertEqual(X.frames_needed(other, {1: (0, 0), 2: (0, 1)}), [3, 4], "OE 已有的局不取终局帧")
        self.assertEqual(X.replay_frames_needed(st, [other, cur], "cur", 1), [("s1", 3), ("s1", 4)])
        with self.assertRaises(ValueError, msg="OE 说 K2 赢了三局, 比分只有两局: 不是认错了局就是比分错了"):
            X.decode_series(other, {1: (0, 0), 2: (0, 1), 3: (0, 0)})
        with self.assertRaises(ValueError, msg="决胜局之前胜者就赢满了"):
            X.decode_series(_xs("s", "2025-10-02T08:00:00Z", "A", "B", 3, 1, bo=5),
                            {1: (0, 0), 2: (0, 0), 3: (0, 0)})

    def test_还没有帧时给两种选边的平均(self):
        # 2026-10-04 复核: 没开打的局按赛程队序当蓝方, 截距 a 约 +0.14 让数随一个任意的顺序挪 5-7 个百分点,
        # 热门都会换边 (JDG 对 BRO: JDG 蓝 0.557, JDG 红 0.484)
        import xregion as X
        st, _ = _x_state("2025-10-01")
        ser = [_xs("cur", "2025-10-03T08:00:00Z", "K1", "L1", 0, 0)]
        ser_r = [_xs("cur", "2025-10-03T08:00:00Z", "L1", "K1", 0, 0)]
        p_ab = X.board_predict(st, "WORLDS", "K1", "L1", ser, "cur", 1)["probability_blue"]
        p_ba = X.board_predict(st, "WORLDS", "L1", "K1", ser_r, "cur", 1)["probability_blue"]
        self.assertGreater(abs(p_ab - (1 - p_ba)), 0.01, "有选边时两种摆法确实不同 (截距)")
        n_ab = X.board_predict(st, "WORLDS", "K1", "L1", ser, "cur", 1, sides_known=False)
        n_ba = X.board_predict(st, "WORLDS", "L1", "K1", ser_r, "cur", 1, sides_known=False)
        self.assertEqual(n_ab["probability_blue"], (p_ab + (1 - p_ba)) / 2)
        self.assertTrue(n_ab["side_neutral"])
        self.assertAlmostEqual(n_ab["probability_blue"] + n_ba["probability_blue"], 1.0, places=12,
                               msg="不随赛程里的队序变")


class XregionAliases(unittest.TestCase):
    """坑: 给四大之外的队加别名, 重新造出 Stage 1/2 "可预测"的假象; 或者 C 的已知表认不出 lolesports 的赞助名
    (RED Kalunga / MIBR.LOS / Team Secret), 整场跨赛区不给数。"""

    def test_别名只对C的已知表生效(self):
        from esports_feed import map_team
        xknown = ["Deep Cross Gaming", "LØS", "Leviatan", "RED Canids", "T1", "Team Secret Whales"]
        major = ["T1", "Gen.G"]
        for api_name, oe in (("LEVIATÁN", "Leviatan"), ("LOS", "LØS"), ("MIBR.LOS", "LØS"),
                             ("RED Kalunga", "RED Canids"), ("Relove Deep Cross Gaming", "Deep Cross Gaming"),
                             ("Team Secret", "Team Secret Whales")):
            self.assertEqual(map_team(api_name, xknown), oe, api_name)
            self.assertIsNone(map_team(api_name, major), f"{api_name}: 目标不在四大里, Stage 1/2 仍不可预测")
        self.assertIsNone(map_team("Team Secret", ["Team Secret"]),
                          "OE 里同名的旧 Team Secret 已过期; 别名指向现在的 Team Secret Whales")


class _XStore(_StubStore):
    """K1 / L1 在四大的已知队伍里 (predictable), 母赛区不同 —— 国际赛上是跨赛区; C3 不在四大里。

    make_row 给一行固定的赛前特征: 国内看板 (use_pre) 的头条和曲线要走正常带赛前特征的那条路 —— 没有它时
    _pre_context 抛错被吞掉, "国内看板一字不差"比的是两条一样退化了的曲线, 什么都证明不了。
    """
    HOME = {**_StubStore.HOME, "K1": "LCK", "L1": "LPL"}
    PRE = {"diff_avg_kills": 1.5, "diff_avg_deaths": -1.0, "diff_avg_totalgold": 800.0, "diff_avg_towers": 0.5,
           "diff_avg_dragons": 0.3, "diff_avg_golddiffat15": 400.0, "diff_avg_cspm": 2.0, "diff_avg_ckpm": 0.05,
           "diff_rolling_wr": 0.08}

    def make_row(self, blue, red, league, draft=None, playoffs=0):
        return dict(self.PRE), []


class XregionBoard(unittest.TestCase):
    """坑: 跨赛区看板接上 C 之后, 国内 / 同母赛区的看板跟着变了一个字; 赛程取不到时退回"只用 OE"的版本
    (没过闸); 开局渐变的锚、赛前那条线和留档的 source 各说各的。"""

    START = "2025-10-05T08:00:00Z"

    def setUp(self):
        import api
        from esports_feed import EsportsFeed, LiveState, Match, TeamRef
        from ingame_service import IngameModel
        try:
            self.ig = IngameModel("_live")
        except Exception as e:
            self.skipTest(f"局内模型文件不在: {e}")
        self.api = api
        self.st, _ = _x_state("2025-10-01")
        self.logged, self.xcalls, self.stage_calls = [], [], []
        self._old = (dict(api.STATE), api.log_prediction)

        def _log(**k):
            if k.get("probability_blue") is not None:
                self.logged.append(k)
        api.log_prediction = _log
        test0 = self

        class Stage:
            metrics = {}

            def __init__(s, k):
                s.k = k

            def predict(s, row):
                test0.stage_calls.append(s.k)
                return 0.0, 0.57

        class Stages:
            # 国内看板 (use_pre) 要真算出赛前数; C 的看板一次都不该算 (_board 里查)
            def __getitem__(s, k):
                return Stage(k)

        api.STATE.clear()
        api.STATE.update(store=_XStore(), stages=Stages(), ingame=None, ingame_live=self.ig)
        test = self

        def make_feed(league, names, minute, completed=(), wins=(0, 0), number=1, frames=None,
                      no_frames=False, schedule_fail=False, others=None):
            g = {"id": f"G{number}", "number": number, "state": "inProgress",
                 "teams": [{"id": "1", "side": "blue"}, {"id": "2", "side": "red"}]}
            st = LiveState(game_id=g["id"], game_state="in_game", minute=minute, golddiff=2500, csdiff=30,
                           blue_kills=6, red_kills=2, gold_total=30000,
                           frame_time="2025-10-05T08:20:00.000Z", live=True)
            det = {"teams": [{"id": "1", "name": names[0], "code": names[0][:3], "wins": wins[0]},
                             {"id": "2", "name": names[1], "code": names[1][:3], "wins": wins[1]}],
                   "games": [{"number": k, "id": f"G{k}", "state": "completed"} for k in range(1, number + 1)]}

            class Feed:
                live_ttl = 20
                downsample = staticmethod(EsportsFeed.downsample)

                def live(s):
                    return [Match("M1", league, test.START, "inProgress", 5,
                                  [TeamRef(n, n[:3], game_wins=w) for n, w in zip(names, wins)])]

                def schedule(s, lg, known=None):
                    return []

                def games(s, mid):
                    return det["games"][:-1] + [g]

                def probe_games(s, mid):
                    return [] if no_frames else [(g, st)]

                def current_game(s, mid):
                    return None if no_frames else (g, st)

                def window(s, gid, **k):
                    return None if no_frames else st

                def _get(s, url, ttl):
                    return {"data": {"event": {"match": {"teams": [
                        {"id": "1", "name": names[0]}, {"id": "2", "name": names[1]}]}}}}

                def frame_sides(s, gid):
                    return {"blue": "1", "red": "2"}

                def ddragon_version(s):
                    return "16.16.1"

                def players(s, gid):
                    return []

                def game_metadata(s, gid):
                    return {}

                def match_wins(s, mid):
                    return {}

                def gold_timeline(s, gid, upto=None):
                    return [{"minute": float(m), "golddiff": 200 * m, "blue_kills": m // 3, "red_kills": 1,
                             "csdiff": 2 * m, "blue_gold": 1800 * m} for m in range(1, (minute or 0) + 1)]

                # ── C 用的五个 ──
                def event_tournament(s, mid):
                    test.xcalls.append("event_tournament")
                    return "T9"

                def league_tournament(s, lg, start):
                    test.xcalls.append("league_tournament")
                    return "T9"

                def completed_events(s, tid):
                    test.xcalls.append("completed_events")
                    return None if schedule_fail else list(completed)

                def match_detail(s, mid, ttl=None):
                    test.xcalls.append("match_detail")
                    test.assertIsNone(ttl, "共用的 getEventDetails 只能按 live_ttl 读 (见 esports_feed.match_ids)")
                    return det if mid == "M1" else (others or {}).get(mid)

                def match_ids(s, mid):
                    test.xcalls.append("match_ids")
                    test.assertNotEqual(mid, "M1", "当前系列赛的比分要 match_detail 那一份")
                    return (others or {}).get(mid)

                def final_frame(s, gid):
                    test.xcalls.append("final_frame")
                    return (frames or {}).get(gid)

            return Feed()

        self.make_feed = make_feed

    def tearDown(self):
        if hasattr(self, "_old"):
            self.api.STATE.clear()
            self.api.STATE.update(self._old[0])
            self.api.log_prediction = self._old[1]

    def _board(self, feed, c=True):
        if c:
            self.api.STATE["xregion"], self.api.STATE["xregion_known"] = self.st, sorted(self.st["teams"])
        else:
            self.api.STATE.pop("xregion", None)
            self.api.STATE.pop("xregion_known", None)
        self.api.STATE["feed"] = feed
        self.stage_calls.clear()
        out = self.api.esports_board("M1", curves=True, points=60)
        if "xregion" in out:
            self.assertEqual(self.stage_calls, [], "C 的看板不该算 Stage 1/2")
        return out

    @staticmethod
    def _ev(mid, start, a, b, wa, wb, bo=1):
        return {"startTime": start, "match": {"id": mid, "strategy": {"count": bo},
                                              "teams": [{"name": a, "code": a, "result": {"gameWins": wa}},
                                                        {"name": b, "code": b, "result": {"gameWins": wb}}]}}

    def test_跨赛区有C_开局以C为锚渐变_赛前线和说明都换成C(self):
        import xregion as X
        from ingame_service import NO_PRE_STATS
        comp = [self._ev("E1", "2025-10-04T08:00:00Z", "K2", "L2", 1, 0)]
        out = self._board(self.make_feed("Worlds", ["K1", "L1"], 8, completed=comp))
        ser = [_xs("E1", "2025-10-04T08:00:00Z", "K2", "L2", 1, 0, bo=1, ids=(None, None)),
               _xs("M1", self.START, "K1", "L1", 0, 0, bo=5, ids=("1", "2"))]
        p = X.board_predict(self.st, "WORLDS", "K1", "L1", ser, "M1", 1)["probability_blue"]
        xr = out["xregion"]
        self.assertEqual((xr["probability_blue"], xr["replayed"], xr["updated"], xr["game_number"]), (p, 1, 1, 1))
        pr = out["prediction"]
        self.assertEqual(pr["pregame_probability_blue"], round(p, 4))
        self.assertEqual(pr["pregame_source"], "xregion")
        self.assertEqual(pr["blend_weight"], round((8 - 3) / 12, 4))
        z = self.api._blend_prior(p, pr["ingame_probability_blue"], 8)[0]
        self.assertAlmostEqual(pr["probability_blue"], z, places=3, msg="头条 = 局内模型以 C 为锚的渐变")
        self.assertEqual(pr["warnings"][0], self.api._xregion_note([]))
        self.assertNotIn(NO_PRE_STATS, pr["warnings"])
        self.assertNotIn("postdraft_probability_blue", pr, "候选 D 没过闸: 没有 BP 后那条线")
        # 曲线开局那段同一个锚 (和头条同一个调用)
        pt = next(x for x in out["timeline"] if x["minute"] >= 3)
        sd = self.api.IngameState(minute=pt["minute"], golddiff=pt["golddiff"], xpdiff=None,
                                  csdiff=2 * pt["minute"], blue_kills=pt["blue_kills"], red_kills=pt["red_kills"],
                                  gold_total=1800 * pt["minute"])
        ref, _ = self.api._ingame_core("K1", "L1", "Worlds", sd, None, None, False, blend=True, blend_with=p)
        self.assertEqual(pt["probability_blue"], ref["probability_blue"])
        self.assertEqual([r["source"] for r in self.logged], ["ingame"], "两队都在四大里: 局内照常留档")

    def test_开局头3分钟_赛前就是C并按xregion留档(self):
        out = self._board(self.make_feed("Worlds", ["K1", "L1"], 2))
        p = out["xregion"]["probability_blue"]
        pr = out["prediction"]
        self.assertTrue(pr["too_early"])
        self.assertEqual((pr["source"], pr["probability_blue"], pr["pregame_probability_blue"]), ("xregion", p, p))
        self.assertIsNone(pr["postdraft_probability_blue"])
        self.assertEqual(pr["warnings"], [self.api._xregion_note([])])
        self.assertIn("这里显示的是赛前的概率", pr["note"])
        self.assertEqual([(r["source"], r["minute"]) for r in self.logged], [("xregion", None)])

    def test_还没开打_看板自己带回C的赛前数(self):
        import xregion as X
        out = self._board(self.make_feed("Worlds", ["K1", "L1"], None, no_frames=True))
        p = out["xregion"]["probability_blue"]
        self.assertEqual(out["prediction"], self.api._xregion_pregame(p, self.api._xregion_note([])))
        self.assertEqual(self.logged, [], "没有帧的看板从来不留档")
        # 没有帧 = 不知道谁蓝方 (左右只是赛程顺序): 给两种选边的平均
        self.assertTrue(out["xregion"]["side_neutral"])
        ser = [_xs("M1", self.START, "K1", "L1", 0, 0, bo=5, ids=("1", "2"))]
        self.assertEqual(p, X.board_predict(self.st, "WORLDS", "K1", "L1", ser, "M1", 1, sides_known=False)
                         ["probability_blue"])
        with_frames = self._board(self.make_feed("Worlds", ["K1", "L1"], 2))
        self.assertFalse(with_frames["xregion"]["side_neutral"])
        self.assertNotEqual(with_frames["xregion"]["probability_blue"], p, "有帧之后按帧的蓝红")

    def test_比分没跟上局号_不给C_看板和原来一字不差(self):
        # 第 3 局开打 (有帧, 第 2 分钟), 比分还停在 1-0: 不给 C、不留档, 也不取终局帧
        frames = {"G1": _xf("1", "2", (9, 2))}

        def feed():
            return self.make_feed("Worlds", ["K1", "L1"], 2, wins=(1, 0), number=3, frames=frames)
        lag = self._board(feed())
        self.assertNotIn("xregion", lag)
        self.assertNotIn("final_frame", self.xcalls)
        self.assertEqual(self.logged, [], "两段都算不出来: 不留档")
        self.assertEqual(lag, self._board(feed(), c=False), "保持原来的行为和每一句话")

    def test_四大之外的队_C认识也不留档_没有记录的赛区点名(self):
        out = self._board(self.make_feed("DCGI", ["C3", "K1"], 2))
        self.assertFalse(out["match"]["predictable"])
        self.assertEqual(out["xregion"]["no_history"], ["LCP"])
        self.assertEqual(out["prediction"]["warnings"],
                         [self.api._XREGION_NOTE + "; LCP 此前没有跨赛区国际赛记录, 这个数主要靠先验"])
        self.assertEqual(self.logged, [])

    def test_赛程取不到就不给C_看板和原来一字不差(self):
        def feed():
            return self.make_feed("Worlds", ["K1", "L1"], 8, schedule_fail=True)
        with_c = self._board(feed())
        self.assertNotIn("xregion", with_c)
        self.logged.clear()
        self.assertEqual(with_c, self._board(feed(), c=False),
                         "赛程取不到 = 只剩'隔天'版本, 没过闸 —— 不给 C, 保持原来的行为和每一句话")
        self.assertTrue(with_c["prediction"]["warnings"][0].startswith("跨赛区对阵: 赛前和 BP 后模型只在赛区内战上"))

    def test_国内和同母赛区的看板一字不差_也不发C的请求(self):
        for league, names in (("LCK", ["T1", "Gen.G"]), ("LCK", ["K1", "L1"]), ("Worlds", ["T1", "Gen.G"])):
            for minute in (2, 12, None):
                def mk():
                    return self.make_feed(league, names, minute, no_frames=minute is None)
                self.xcalls.clear()
                a = self._board(mk())
                if minute == 12 and names == ["T1", "Gen.G"]:
                    self.assertIn("pre_draft", self.stage_calls, "比的是正常带赛前特征的看板, 不是退化的")
                self.assertEqual(self.xcalls, [], f"{league} {names}: 不该碰 C 的赛程")
                self.assertNotIn("xregion", a)
                self.assertEqual(a, self._board(mk(), c=False), f"{league} {names} 第 {minute} 分钟")

    def test_当前系列赛打到第3局_前两局按终局帧重放(self):
        import xregion as X
        frames = {"G1": _xf("2", "1", (2, 9)), "G2": _xf("1", "2", (3, 10))}    # 两局都是蓝方输
        out = self._board(self.make_feed("Worlds", ["K1", "L1"], 12, wins=(1, 1), number=3, frames=frames))
        xr = out["xregion"]
        self.assertEqual((xr["game_number"], xr["replayed"]), (3, 2))
        ser = [_xs("M1", self.START, "K1", "L1", 1, 1, bo=5, ids=("1", "2"),
                   frames={"1": frames["G1"], "2": frames["G2"]})]
        self.assertEqual(xr["probability_blue"], X.board_predict(self.st, "WORLDS", "K1", "L1", ser, "M1", 3)
                         ["probability_blue"])
        games = X.board_replay(self.st, ser, "M1", 3)["games"]
        self.assertEqual([(g["blue"], g["blue_win"], g["how"]) for g in games],
                         [("L1", 0, "frames"), ("K1", 0, "frames")])

    def test_别的系列赛取不到帧_那个系列赛交替_C照给(self):
        comp = [self._ev("E1", "2025-10-04T08:00:00Z", "K2", "L2", 2, 1, bo=3)]
        out = self._board(self.make_feed("Worlds", ["K1", "L1"], 2, completed=comp, others={}))
        self.assertEqual(out["xregion"]["replayed"], 3, "详情 / 终局帧取不到不让整场的 C 消失")
        self.assertEqual(out["prediction"]["source"], "xregion")
        self.assertIn("match_ids", self.xcalls, "别的系列赛的 id 走 match_ids, 不碰共用缓存的有效期")

    def test_别的系列赛按终局帧解码(self):
        import xregion as X
        comp = [self._ev("E1", "2025-10-04T08:00:00Z", "K2", "L2", 2, 1, bo=3)]
        others = {"E1": {"teams": [{"id": "9", "name": "K2", "code": "K2"}, {"id": "8", "name": "L2", "code": "L2"}],
                         "games": [{"number": k, "id": f"E1G{k}"} for k in (1, 2, 3)]}}
        frames = {"E1G1": _xf("8", "9", (9, 2)), "E1G2": _xf("9", "8", (9, 2))}     # 第 1 局 L2 赢, 第 2 局 K2 赢
        out = self._board(self.make_feed("Worlds", ["K1", "L1"], 2, completed=comp, others=others, frames=frames))
        ser = [_xs("E1", "2025-10-04T08:00:00Z", "K2", "L2", 2, 1, bo=3, ids=("9", "8"),
                   frames={"1": frames["E1G1"], "2": frames["E1G2"]}),
               _xs("M1", self.START, "K1", "L1", 0, 0, bo=5, ids=("1", "2"))]
        self.assertEqual(out["xregion"]["probability_blue"],
                         X.board_predict(self.st, "WORLDS", "K1", "L1", ser, "M1", 1)["probability_blue"])
        self.assertEqual(self.xcalls.count("final_frame"), 2, "决胜局归胜者, 不取")


class XregionMatchIdsCache(unittest.TestCase):
    """坑 (2026-10-04 复核): 重放别的系列赛时用 match_detail(mid, 3600) 取 id, 把共用的 getEventDetails 按一小时
    写进缓存 —— _get 认第一个写入者给的有效期, 那场比赛自己的看板、页面顶上的比分就停在一小时前。"""

    def test_id照常按live_ttl读_齐了另记一份(self):
        from esports_feed import EsportsFeed
        f = EsportsFeed()
        calls = []
        payload = {"data": {"event": {"match": {
            "teams": [{"id": "1", "name": "A", "code": "A", "result": {"gameWins": 2}},
                      {"id": "2", "name": "B", "code": "B", "result": {"gameWins": 1}}],
            "games": [{"number": k, "id": f"G{k}", "state": "completed"} for k in (1, 2, 3)]}}}}

        def fake_get(url, ttl, *a, **k):
            calls.append(ttl)
            return payload
        f._get = fake_get
        ids = f.match_ids("M9")
        self.assertEqual([g["id"] for g in ids["games"]], ["G1", "G2", "G3"])
        self.assertEqual(calls, [f.live_ttl])
        self.assertNotIn("wins", ids["teams"][0], "会变的比分不记")
        f.match_ids("M9")
        self.assertEqual(calls, [f.live_ttl], "齐了就不再请求")
        payload["data"]["event"]["match"]["games"][2]["id"] = None
        f2 = EsportsFeed()
        f2._get = fake_get
        f2.match_ids("M9")
        f2.match_ids("M9")
        self.assertEqual(calls, [f.live_ttl] * 3, "不齐不记, 下次再取")


class PredictionLogXregionSource(unittest.TestCase):
    """坑: 留档的 source 多了 "xregion", 回填和打分不认识它 —— 跨赛区的赛前数混进别的组, 或者整行被丢。"""

    def test_回填和打分按自己的一组(self):
        import contextlib
        import io
        import json
        import tempfile
        import prediction_log as PL
        old = PL.LOG
        try:
            with tempfile.TemporaryDirectory() as d:
                PL.LOG = Path(d) / "log.jsonl"
                rows = []
                for k in range(6):
                    rows.append({"t": "2026-10-05T08:00:00+00:00", "match_id": f"m{k}", "game_id": f"g{k}",
                                 "game_number": 1, "league": "Worlds", "blue": "T1", "red": "Bilibili Gaming",
                                 "minute": None, "minute_key": None, "probability_blue": 0.6, "source": "xregion",
                                 "y": k % 2})
                    rows.append(dict(rows[-1], minute=20.0, minute_key=20, probability_blue=0.8, source="ingame"))
                PL.LOG.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    PL.score()
                txt = buf.getvalue()
        finally:
            PL.LOG = old
        line = next(x for x in txt.splitlines() if x.strip().startswith("xregion"))
        self.assertIn("6 局 /     6 条", line)
        self.assertIn("准确率 0.500", line)
