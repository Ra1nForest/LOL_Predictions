"""
Stage 4 训练器 — 局内模型
==========================
训练 + 校准 + 保存, 供 api.py 的 /predict/ingame 使用。

产出 artifacts/:
    model_ingame.json          完整模型 (手填经济/经验差时用)
    calib_ingame.json
    model_ingame_live.json     live 变体, 不含任何经验差特征
    calib_ingame_live.json

为什么要两个模型
----------------
lolesports 的实时帧里没有 XP 字段 (只有每人的 level, 和训练用的
xpat10/15/20 原始经验值量纲不同)。自动跟播时 xpdiff 拿不到。

给完整模型传一个编造的 xpdiff=0 有两重问题:
  · 模型在吃一个训练时没见过的系统性常量
  · explain 层会把它讲成「双方经验持平」—— 把缺失讲成了实测结论

所以实时路径走一个训练时就不含经验差的模型。代价由
research/gate_ingame_features.py 的配置 F 量出来。

注意 live 变体要排除的是 XP_DERIVED 两项, 不只是 xpdiff ——
x_xp_scaling = xpdiff * scaling_diff (ingame_model.py:270), 留着它
等于经验差信息还在。

英雄强势期表必须一起存 —— 线上推理时不能重算 (要扫全部历史)。

评估模型和服务模型 (闸门 G2, 2026-10-17)
------------------------------------------
每个变体拟**两次**, 两次各管一件事:

  评估模型   按局切 60/20/20 (split_by_game 的默认值): 树拟最早 60% 的局, 保序表拟随后 20%,
             最近 20% 当考卷。metrics 里的数 (accuracy / brier / ece / per_T / per_T_served …)
             **只**出自它; 算完就丢, 不存盘。
  服务模型   split_by_game(mdf, 0.80, 0.20): 树拟最早 80% 的局, 保序表 (和兼容 Platt) 拟最近
             20%, 不留考卷。存成 model_ingame{suffix}.json, 保序表进 calib —— 线上和浏览器版跑的都是它。

为什么不再把评估模型直接上线: 以前就是这么做的, 结果 2026-10 线上的树数据截至 2024-11-16 (一局
LPL 都没有), 保序表截至 2026-01-15 —— 最近 20% 的局 (将近一年) 只用来出成绩, 从没进过线上模型。
research/gate_ingame_refit.py 在 7 个滚动起点上量过"树 80% + 保序最近 20%"对今天的 60/20:
live / 全部赛区 Brier fold 配对 t = +6.54 (7/7 个月更好), 准确率不变, 收益全在刻度上
(校准斜率 0.90 -> 0.96)。特征、超参数、校准方法一律没动, 只换了用哪段数据拟。

为什么 metrics 仍出自评估模型: 服务模型把考卷也拿去拟了, 它没有留出成绩可算; 而 daily_update 的
指标闸拿新旧两份 calib 的 accuracy / ece 相比 —— 旧的 (包括这次改动之前的 artifacts) 都是 60/20/20
的留出成绩, 新的也必须是, 否则闸门比的是两种口径。服务模型比评估模型好多少, 证据是 G2 的滚动起点,
不是这里的数。

唯一的例外是 p_min / p_max: 它们描述**服务**保序表能输出的范围 (clip 之后 iso_y 的最小 / 最大值,
见 reachable_range), 不是评估模型在考卷上碰到的极值。IngameModel.clip_warning 拿它们判断"概率是不是
真的顶到了边界" (CLAUDE.md 的不变量: 警告必须看实际输出) —— 若用评估模型的数, 服务表的顶比它低时,
输出贴着顶警告也永远不响; 比它高时, 没到顶就会说"顶到了上限"。界面的死区和 range_note 也读这两个数,
同样应当描述线上那张表。

calib 里新增的字段 (不影响任何概率; 旧 artifacts 没有它们, 什么都不受影响):
    trees_through    服务树的数据截止日 (最早 80% 的局里最晚的一天)
    iso_window       服务保序表 (和兼容 Platt) 拟合用的局的 [最早, 最晚] 日期
    serving_fit      服务模型的切分比例, 两段各自的日期 / 局数 / 快照数。四个里只有它有人读: 免责声明的措辞
                     (api._ingame_disclaimer / web disclaimer: 成绩出自同一配方的评估版, 线上这份没有留出成绩;
                     导出时原样带进 ingame_live.json), 和研究脚本的 require_unseen (线上文件已经没有留出数据,
                     拿它在 trained_through 之前的局上打分就是样本内)
    metrics_source   metrics 的出处: 评估模型的切分、三段的日期 / 局数 / 快照数, 评估模型在考卷上的
                     概率极值 test_p_range (以前存在 p_min/p_max 里的就是这两个数), 以及例外的键
                     (from_serving_model = p_min, p_max)
    服务树和保序表按局切, 不按天切, 所以 trees_through 和 iso_window 的第一天可能是同一天。
trained_through 的含义不变: 英雄强势期表的日期 = 数据的最后一天 (现在也正好是服务保序表的截止日)。
其余字段 (coef / intercept / iso_x / iso_y / features …) 的名字和含义都没变, 只是换成了服务模型的值,
所以旧版服务代码读新 artifacts、新代码读旧 artifacts 都照常工作。

用法: python ingame_train.py
"""
import numpy as np, pandas as pd, json, os, warnings, sys
from pathlib import Path
warnings.filterwarnings("ignore")
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, brier_score_loss

sys.path.insert(0, str(Path(__file__).parent))
from ingame_model import (SLICES, ROLES, W, PRE_STATS, MIN_CHAMP_GAMES,
                          SHRINK_K, XGB, load, build, features,
                          split_by_game, ece, INTERACTIONS)

YEARS = (2022, 2023, 2024, 2025, 2026)
_ROOT = Path(__file__).parent
# 见 train.py 同处注释: 无人值守流水线用它训到暂存目录
OUT = Path(os.environ.get("LOL_ARTIFACTS") or (_ROOT / "artifacts"))
OUT.mkdir(parents=True, exist_ok=True)

# 经验差及其派生项。实时帧给不了 XP, 这两个必须一起去掉。
XP_DERIVED = {"xpdiff", "x_xp_scaling"}

# 校准用保序回归, 不是 Platt。**这一条只对局内模型成立**, Stage 1/2 不用它 (当时留在
# Platt, G1 之后是 logit_elo, 见 train.py) —— research/gate_calibration{,_pre}.py 分别量过:
#
#   Stage 4  (校准集约 35000 条快照)  Brier t=+6.91  ECE t=+7.12  8 折全正 -> 换
#   Stage 1  (校准集     1716 场)     Brier t=-4.74  6 折全负            -> 不换
#   Stage 2  (校准集     1716 场)     Brier t=-1.97  方向也是变差        -> 不换
#
# README 里"Isotonic 输了, 因为校准数据不够会过拟合"那个结论没有错, 只是它
# 的前提 (数据量) 在局内模型上早已不成立: 那里有六万条校准样本。同一个方法
# 在两处的结论相反, 靠的是分别过闸, 不是推广。
#
# 为什么非换不可: Platt 只能画一条固定形状的 S 曲线, 实测它中间鼓两头塌 ——
# >6k 领先的样本实际胜率 98.0%, XGBoost 原始输出 97.8% (几乎分毫不差),
# 被 Platt 压成 93.3%。**模型判断是对的, 是校准层把它压扁了。**
CLIP = 0.01               # 保序回归会输出 0 和 1; 绝对确定既不真实也无必要

# 两次拟合各自的切分 (树, 保序表), 按局的日期顺序 —— 见模块说明"评估模型和服务模型"。
# EVAL_SPLIT 必须保持 split_by_game 的默认值: metrics 要和以前的 artifacts 同口径, daily_update 的
# 指标闸才是拿同一把尺子在比。SERVE_SPLIT 是 research/gate_ingame_refit.py 过闸的 CANDIDATE,
# 两项加起来必须是 1 (不留考卷); 改它要重新过闸。
EVAL_SPLIT = (0.60, 0.20)     # 剩下 20% 是考卷
SERVE_SPLIT = (0.80, 0.20)

# 线上实际服务的赛区 (OE 的写法): 四大 + 国际赛, 也就是 esports_feed.LEAGUE_IDS 经 backfill_late.oe_league
# 换成的 OE 名字 (DCGI 两边同名)。tests 里有一条核对两者一致 —— LEAGUE_IDS 加了赛区忘了改这里会红。
SERVED_OE_LEAGUES = ("LPL", "LCK", "LEC", "LCS", "WLDs", "MSI", "FST", "EWC", "DCGI")


def served_per_T(tt) -> dict:
    """metrics 里的 per_T_served: 同一个测试段只取线上服务的赛区, 按切片给准确率 / Brier / log loss 和 95% 区间。

    为什么要另存一份: per_T 是**全赛区**的测试段 (2026-10 约 9400 局, 四大 + 国际只占约 22%)。次级联赛更好
    预测, 线上真账 (prediction_log.score) 拿它对照, 线上就显得系统性偏低 —— 当时 per_T 写 71.1/75.1/79.2/82.6%
    (10/15/20/25 分钟), 四大 + 国际只有 68.1/72.4/75.6/80.7%。"线上比留出低 5-7 个点"有一半是口径造成的。
    per_T 原样保留 (界面的置信说明、免责声明、浏览器版都在读它), 这里只是加一个键。

    tt 要有 league / T / y / p 四列, 每局每个切片一行, 所以 ±(1.96·SD/√n, 存的是半宽) 就是按局算的区间。
    准确率按 p > .5 判, 和 per_T 一致。
    """
    out = {}
    s_all = tt[tt["league"].isin(SERVED_OE_LEAGUES)]
    for T in SLICES:
        s = s_all[s_all["T"] == T]
        if len(s) < 20:
            continue
        y = s["y"].astype(int).to_numpy()
        p = s["p"].astype(float).to_numpy()
        hit = ((p > .5).astype(int) == y).astype(float)
        sq = (p - y) ** 2
        pc = np.clip(p, 1e-6, 1 - 1e-6)
        ll = -(y * np.log(pc) + (1 - y) * np.log(1 - pc))
        h = lambda v: float(1.96 * v.std(ddof=1) / np.sqrt(len(v)))
        out[str(T)] = {"n": int(len(s)),
                       "accuracy": float(hit.mean()), "accuracy_ci95": h(hit),
                       "brier": float(sq.mean()), "brier_ci95": h(sq),
                       "log_loss": float(ll.mean()), "log_loss_ci95": h(ll)}
    return out


def fit_calibrated(tr, cal, feats):
    """树拟 tr, 保序表 (和兼容 Platt) 拟树在 cal 上的原始输出。返回 (树, 保序, Platt)。

    评估模型和服务模型走的是同一个函数 —— 两者只许差在喂进来的是哪段数据, 不能在别处悄悄差出
    别的东西 (超参数、校准方法、缺失值的填法)。否则 metrics 量的就不是线上那个配方了。
    """
    Xtr, ytr = tr[feats].astype(float).fillna(-999), tr["y"].astype(int)
    Xc, yc = cal[feats].astype(float).fillna(-999), cal["y"].astype(int)
    m = XGBClassifier(**XGB); m.fit(Xtr, ytr, verbose=False)
    pc = m.predict_proba(Xc)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(pc, yc)
    # Platt 也照旧拟一份存进去。不是留着用, 是留着**能回滚**: 万一服务端
    # 退回旧版本, 旧代码只认 coef/intercept, 字段缺了会直接崩。
    cb = LogisticRegression(C=1e6).fit(pc.reshape(-1, 1), yc)
    return m, iso, cb


def reachable_range(iso_y, clip=CLIP) -> tuple[float, float]:
    """服务保序表经 IngameModel.calibrate 能输出的 [最小, 最大] —— 写进 metrics 的 p_min / p_max。

    保序表单调不减, np.interp 在两端 clamp, 所以输出落在 [iso_y[0], iso_y[-1]] 里且两端都取得到,
    再裁剪到 [clip, 1 - clip]。传进来的要是写进 calib 的那份 iso_y (float64), 这样顶格时 calibrate
    的输出和 p_max 是同一个数, clip_warning 的 cal >= p_max - 1e-6 一定成立。

    不能拿评估模型在考卷上碰到的极值: 那是另一张表、另一批输入 —— 服务表的顶比它低时, 输出贴着顶
    警告也永远不响; 比它高时, 没到顶就说"顶到了上限"。两种都是 CLAUDE.md 里那条"警告必须看实际输出"。
    """
    v = np.clip(np.asarray(iso_y, dtype=float), clip, 1.0 - clip)
    return float(v.min()), float(v.max())


def _span(d) -> dict:
    """一段快照的日期范围和规模 (同一局的几个快照日期相同, 所以就是这段局的日期范围)。"""
    return {"from": str(d["date"].min().date()), "through": str(d["date"].max().date()),
            "games": int(d["gameid"].nunique()), "snapshots": int(len(d))}


def require_unseen(model_or_calib, first_test_day, who: str) -> None:
    """研究脚本拿**线上** Stage 4 文件在历史比赛上打分之前, 先过这一道。

    G2 起存盘的是服务模型: 树拟最早 80% 的局, 保序表拟最近 20% —— serving_fit.iso.through (= trained_through)
    之前的每一局它都见过, 线上文件自己已经没有留出数据。G2 之前写的研究脚本 (gate_anchor / gate_early_minutes /
    gate_calibration / gate_international / gate_dechatter_backfill, 以及 gate_dechatter.mjs / gate_smoothing.mjs
    的同一道检查) 都把"split_by_game 的最后 20%"或"≥ 2026-01-15"当成线上模型没见过的考卷; 对着服务模型重跑,
    就是在样本内打分 —— 不报错, 数偏乐观。旧文件 (没有 serving_fit) 的线上模型就是 60/20/20 的评估模型,
    考卷照旧成立, 这里不拦。

    model_or_calib: calib 的 dict, 或 IngameModel (读它的 serving_fit)。first_test_day: 考卷里最早的一天;
    不晚于服务模型见过的最后一天就退出。要复现 G2 之前的结论: 用 G2 之前的模型文件 (daily_update 留的
    artifacts_prev/, 或 git 历史里的 frontend/public/web/ingame_live.json), 或按 EVAL_SPLIT 用
    fit_calibrated 自己重拟一个评估模型, 再拿它的考卷。
    """
    sf = (model_or_calib.get("serving_fit") if isinstance(model_or_calib, dict)
          else getattr(model_or_calib, "serving_fit", None))
    if not sf:
        return
    seen = sf["iso"]["through"]
    first = str(pd.Timestamp(first_test_day).date())
    if first <= seen:
        raise SystemExit(
            f"{who}: 线上 Stage 4 是 G2 的服务模型, 拟合时见过截至 {seen} 的每一局 (树 {sf['trees']['from']}"
            f"~{sf['trees']['through']}, 保序表 {sf['iso']['from']}~{seen}); 考卷从 {first} 开始, 是样本内打分。"
            f"换用 G2 之前的模型文件, 或用 ingame_train.fit_calibrated 按 EVAL_SPLIT 重拟评估模型 (见 require_unseen)")


def train_variant(mdf, sc, feats, suffix, title):
    """训练 + 校准 + 存盘。suffix 为空是完整模型, "_live" 是实时变体。

    先拟评估模型 (EVAL_SPLIT, 60/20/20) 只为算 metrics, 再拟服务模型 (SERVE_SPLIT, 80/20) 存盘上线 ——
    见模块说明"评估模型和服务模型"。返回评估模型的准确率。
    """
    print(f"\n  ── {title}  ({len(feats)} 特征) ──")

    # ── 评估模型: metrics 只出自它, 不存盘 ──
    tr, cal, te = split_by_game(mdf, *EVAL_SPLIT)
    Xte, yte = te[feats].astype(float).fillna(-999), te["y"].astype(int).values
    m, iso, _ = fit_calibrated(tr, cal, feats)
    raw = m.predict_proba(Xte)[:, 1]
    p = np.clip(iso.predict(raw), CLIP, 1 - CLIP)

    base = max(yte.mean(), 1 - yte.mean())
    acc = accuracy_score(yte, (p > .5).astype(int))
    print(f"    评估模型 (只出 metrics, 不上线): 训练 {len(tr)} / 校准 {len(cal)} / 测试 {len(te)} 快照")
    print(f"    准确率 {acc:.1%} (基线 {base:.1%})   Brier {brier_score_loss(yte,p):.4f}"
          f"   ECE {ece(p, yte):.4f}")
    print(f"    考卷上的概率范围 [{p.min():.2f}, {p.max():.2f}]")

    per_T = {}
    tt = te.copy(); tt["p"] = p
    print(f"    {'T':>6}{'快照':>7}{'准确率':>9}{'高置信':>9}")
    for T in SLICES:
        s = tt[tt["T"] == T]
        if len(s) < 20: continue
        a = accuracy_score(s["y"].astype(int), (s["p"] > .5).astype(int))
        conf = float(((s["p"] > .8) | (s["p"] < .2)).mean())
        per_T[str(T)] = {"accuracy": float(a), "n": int(len(s)),
                         "high_conf_share": conf}
        print(f"    {T:>6}{len(s):>7}{a:>9.1%}{conf:>9.1%}")

    # 线上服务的赛区 (四大 + 国际) 单算一份, 给线上真账对照用 —— 见 served_per_T
    per_T_served = served_per_T(tt)
    print("    线上服务的赛区 (四大 + 国际):")
    for T, v in per_T_served.items():
        print(f"    {T:>6}{v['n']:>7}{v['accuracy']:>9.1%} ± {v['accuracy_ci95']:.1%}"
              f"   Brier {v['brier']:.4f}   LL {v['log_loss']:.4f}")

    # ── 服务模型: 存盘上线, 树 = 最早 80% 的局, 保序表 = 最近 20%, 不留考卷 ──
    s_tr, s_cal, s_rest = split_by_game(mdf, *SERVE_SPLIT)
    # SERVE_SPLIT 两项加起来不是 1 的话, 最近那一段会悄悄不进任何模型 —— 正是这次要修的毛病
    assert s_rest.empty, f"SERVE_SPLIT {SERVE_SPLIT} 留下了 {s_rest['gameid'].nunique()} 局没用上"
    sm, siso, scb = fit_calibrated(s_tr, s_cal, feats)
    iso_x = [float(v) for v in siso.X_thresholds_]
    iso_y = [float(v) for v in siso.y_thresholds_]
    p_lo, p_hi = reachable_range(iso_y)
    span_tr, span_cal = _span(s_tr), _span(s_cal)
    print(f"    服务模型 (上线的是它): 树 {span_tr['games']} 局, 截至 {span_tr['through']}; "
          f"保序 {span_cal['games']} 局, {span_cal['from']} 至 {span_cal['through']}; "
          f"输出可达 [{p_lo:.3f}, {p_hi:.3f}]")
    sm.save_model(str(OUT / f"model_ingame{suffix}.json"))

    # 英雄强势期表 —— 取最新日期的快照
    end = mdf["date"].max()
    scale_tbl = {}
    for ch in sc.tbl:
        v = sc.get(ch, end)
        if not np.isnan(v):
            scale_tbl[ch] = round(float(v), 5)

    cfg = {
        # 线上真正用的校准器。老 artifacts 没有这个字段, ingame_service
        # 读不到时默认回退成 platt —— 换代码和换模型文件可以分开做。
        # 下面的 iso / coef / intercept 都是**服务模型**的 (拟在最近 20% 的局上)
        "calibrator": "isotonic",
        "iso_x": iso_x,
        "iso_y": iso_y,
        "clip": CLIP,
        # 兼容字段, 见 fit_calibrated 里 Platt 那句注释
        "coef": float(scb.coef_[0][0]),
        "intercept": float(scb.intercept_[0]),
        "features": feats,
        "slices": SLICES,
        "scaling_index": scale_tbl,
        "scaling_min_games": MIN_CHAMP_GAMES,
        "scaling_shrink_k": SHRINK_K,
        "interaction_cols": [c for c in INTERACTIONS if c in feats],
        "pre_stats": PRE_STATS,
        "roll_window": W,
        # 数据的最后一天 (强势期表的日期)。含义没变, 现在也正好是服务保序表的截止日
        "trained_through": str(end.date()),
        "variant": "live" if suffix else "full",
        "has_xpdiff": "xpdiff" in feats,
        # ── 下面四个是 G2 新加的说明字段, 服务端和浏览器版都不读 (见模块说明) ──
        # 服务树的数据截止日。trained_through 不是它: 以前两者差了将近两年, 没人看得出来
        "trees_through": span_tr["through"],
        # 服务保序表 (和兼容 Platt) 拟合用的局的日期范围
        "iso_window": [span_cal["from"], span_cal["through"]],
        "serving_fit": {"split": list(SERVE_SPLIT), "trees": span_tr, "iso": span_cal},
        # metrics 出自哪个模型。from_serving_model 列的是例外: 它们描述的是服务模型
        "metrics_source": {
            "model": "evaluation",
            "split": [*EVAL_SPLIT, round(1.0 - sum(EVAL_SPLIT), 6)],
            "trees": _span(tr), "iso": _span(cal), "test": _span(te),
            # 评估模型在考卷上碰到的概率极值 —— 以前 p_min / p_max 存的就是这两个数
            "test_p_range": [float(p.min()), float(p.max())],
            "from_serving_model": ["p_min", "p_max"],
        },
        "metrics": {
            "accuracy": float(acc), "baseline": float(base),
            "lift": float((acc - base) / (1 - base)),
            "brier": float(brier_score_loss(yte, p)),
            "ece": float(ece(p, yte)),
            # 例外: 服务保序表的可达范围, 不是评估模型的考卷极值 —— clip_warning、界面死区、
            # range_note 都拿它说"线上这张表顶到了边界", 见 reachable_range
            "p_min": p_lo, "p_max": p_hi,
            "per_T": per_T,
            # 测试段只取四大 + 国际赛区 (见 served_per_T)。新加的键, 上面的键一个都不动
            "per_T_served": per_T_served,
        },
    }
    with open(OUT / f"calib_ingame{suffix}.json", "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2, ensure_ascii=False)

    print(f"    -> model_ingame{suffix}.json  calib_ingame{suffix}.json")
    return acc


def main():
    print("=" * 70)
    print("  Stage 4 训练器")
    print("=" * 70)

    files = [str(_ROOT / "data" / f"{y}_LoL_esports_match_data_from_OraclesElixir.csv")
             for y in YEARS]
    print("\n[1/3] 加载")
    df = load(files)

    print("\n[2/3] 构建快照")
    mdf, sc = build(df)

    print("\n[3/3] 训练两个变体")
    feats_full = features(mdf, with_interactions=True)
    feats_live = [c for c in feats_full if c not in XP_DERIVED]
    dropped = [c for c in feats_full if c in XP_DERIVED]

    a_full = train_variant(mdf, sc, feats_full, "", "完整模型")
    a_live = train_variant(mdf, sc, feats_live, "_live",
                           f"live 变体 (去掉 {', '.join(dropped)})")

    print(f"\n{'=' * 70}")
    print(f"  完整 {a_full:.1%}   live {a_live:.1%}   "
          f"差 {(a_full - a_live) * 100:+.1f} 个百分点")
    print(f"  (单次留出对比, 未过闸。闸门结论见 research/gate_ingame_features.py)")
    print(f"  上面的准确率都出自评估模型 (60/20/20); 存盘上线的是服务模型 (树 80% + 保序最近 20%),"
          f" 它没有留出成绩, 依据是 research/gate_ingame_refit.py")
    print(f"\n  重启服务后可用 POST /predict/ingame")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
