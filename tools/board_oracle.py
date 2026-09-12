"""
看板的 Python 参照输出 —— 给 frontend/scripts/diff-board.mjs 当"标准答案"。

为什么不直接问正在跑的服务
--------------------------
服务是长期运行的, 它的 EsportsFeed 里攒着**直播时**的缓存 —— 开局零点、暂停观测 ——
比赛结束后不会重算。实测 2026-09-12: LEC GX vs NAVI 第 3 局, 在跑的服务说第 36 分钟,
全新的 Python 进程和浏览器版都说第 35 分钟。差的就是服务在开局头两分钟第一次看到
这局时校准零点失败 (要用的窗口还没发出来), 退回 frames[0] 并永久缓存的那个零点。
拿服务当标准答案, 测的是"服务攒下的历史状态", 不是"代码"。

所以这里每次起一个全新的进程、全新的 EsportsFeed, 走 api.esports_board 同一段代码,
只把全局状态换成干净的。

**log_prediction 被换成空函数。** esports_board 每调一次都会往 predictions/log.jsonl
记一笔; 对着已经打完的比赛去记, 等于往预测留档里塞进"看过结局的预测" —— 那份留档
是拿来给模型打分的 (prediction_log.py --resolve --score)。

    python tools/board_oracle.py MATCH_ID:GAME_ID [...]  [--lists]
    → stdout 一行 JSON: {"MATCH_ID:GAME_ID": board, ..., "__live__": ..., "__upcoming__": ...}
"""
from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    lists = "--lists" in sys.argv
    real_stdout = sys.stdout
    out: dict = {}

    # 加载和 esports_board 自己的 print 全部改道 stderr, stdout 只留最后那一行 JSON
    with contextlib.redirect_stdout(sys.stderr):
        import api
        import esports_feed as EF
        from feature_store import FeatureStore
        from ingame_service import IngameModel

        api.log_prediction = lambda *a, **k: None
        api.STATE["store"] = FeatureStore.from_csv(api.DATA)
        api.STATE["stages"] = {n: api.Stage(n) for n in ("pre_draft", "post_draft")}
        api.STATE["ingame"], api.STATE["ingame_live"] = IngameModel.load_pair()
        api.STATE["feed"] = EF.EsportsFeed()

        for a in args:
            mid, gid = a.split(":")
            try:
                out[a] = api.esports_board(mid, game_id=gid, curves=True, points=60)
            except Exception as e:
                out[a] = {"__error__": f"{type(e).__name__}: {getattr(e, 'detail', e)}"}
        if lists:
            out["__live__"] = api.esports_live()
            out["__upcoming__"] = api.esports_upcoming(limit=24, past_hours=5)

    # allow_nan=False: 服务自己的 JSONResponse 也不许 NaN, 混进来说明哪里坏了
    real_stdout.write(json.dumps(out, ensure_ascii=False, allow_nan=False))
    real_stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
