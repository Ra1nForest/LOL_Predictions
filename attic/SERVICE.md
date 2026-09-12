# 分段式预测服务

三段递进,每段独立可用。

| Stage | 时机 | 输入 | 特征数 |
|---|---|---|---|
| 1 `pre_draft` | BP 之前 | 双方队伍历史 | 79 |
| 2 `post_draft` | BP 之后 | + 10 个英雄选择 | 112 |
| 3 `agent_review` | 可选 | + 三个 LLM 交叉复核 (**advisory-only**) | — |

## 启动

```bash
pip install -r requirements.txt
python train.py                       # 生成 artifacts/
uvicorn api:app --reload --port 8000
```

打开 http://localhost:8000/docs 可交互测试。

Stage 3 需要 NVIDIA NIM 免费 key(https://build.nvidia.com):

```bash
export NVIDIA_API_KEY="nvapi-..."
```

## 调用

```bash
# Stage 1 — BP 之前
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "blue_team": "JD Gaming", "red_team": "Bilibili Gaming", "league": "LPL"
}'

# Stage 2 — 带 BP
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "blue_team": "JD Gaming", "red_team": "Bilibili Gaming", "league": "LPL",
  "draft": {
    "blue": {"top": {"player":"369","champion":"Gnar"},
             "jng": {"player":"Kanavi","champion":"Xin Zhao"},
             "mid": {"player":"Scout","champion":"Ahri"},
             "bot": {"player":"Ruler","champion":"Corki"},
             "sup": {"player":"MISSING","champion":"Leona"}},
    "red":  {"top": {"player":"Bin","champion":"Ambessa"},
             "jng": {"player":"Xun","champion":"Wukong"},
             "mid": {"player":"knight","champion":"Orianna"},
             "bot": {"player":"Elk","champion":"Sivir"},
             "sup": {"player":"ON","champion":"Braum"}}
  }
}'

# Stage 3 — 加 agent 复核
# 同上, 加 "agent_review": true
```

## 响应结构

```jsonc
{
  "stages": {
    "1_pre_draft":  { "probability_blue": 0.326, "evidence": [...], "model": {...} },
    "2_post_draft": { "probability_blue": 0.252, "delta_from_stage1": -0.074, ... },
    "3_agent_review": { "probability_blue": 0.27, "adjustment": +0.018,
                        "confidence": "medium", "reasoning": "...", "reviews": [...] }
  },
  "final": { "probability_blue": 0.252, "implied_fair_odds_blue": 3.963 },
  "warnings": ["..."],
  "disclaimer": "..."
}
```

## Agent 层是 advisory-only

**Stage 3 不修改概率。** 最终概率永远来自 Stage 2。

这不是保守设计,是实测结论。第一版给了协调者 ±0.10 的调整权,结果:

```
Stage 2  0.6305   校准过, 留出测试集验证, ECE 0.046
Stage 3  0.6005   agent 下调 3pp
```

**调整方向是错的。** 该模型的可靠性表在 0.6–0.7 区间显示实际胜率
0.658 对预测 0.643,即模型在此区间略微**低估**。要修正应该上调。

质疑者的论证里有两处事实错误:把 `playoffs=0`(表示"这不是季后赛")
读成"模型未对季后赛做调整";把 ECE(各桶偏差的绝对值加权平均,
不含方向)当成有方向的高估量。

而协调者通过了这个论证 —— 因为它字面上满足了我的约束("指出模型
看不到的因素")。质疑者说的是**一类盲区**("转会会影响结果而模型
看不到"),不是**一个具体事实**("BLG 在 7 月 20 日换了上单")。
prompt 没排除前者。

### 根本原因

三个不联网的 LLM 没有任何模型之外的信息源,所以**永远不存在正当的
调整依据**。给它们调整权,唯一的产出是噪声。收紧 prompt 治不了这个 ——
问题在任务设定,不在措辞。

### 现在 Stage 3 做什么

输出结构化的审查记录,并把每条风险点分类:

| kind | 含义 |
|---|---|
| `reinterpretation` | 对简报内已有证据的重读(如"优势集中在单一特征")。值得看,但模型已经定价了 |
| `external_fact` | 具名、带日期的外部事件。一个**类别**的盲区不算 |
| `error` | agent 自己说错了 |

另外单列 `agent_errors`,记录 agent 对简报的误读。

这样它真正有用的部分保留了 —— 质疑者关于"优势几乎全靠单一特征"和
"25 场样本置信区间宽达 ±9%"的提醒是有价值的,值得读。它只是不该
被允许改数字。

### 三个 agent 仍来自三个不同血统

DeepSeek / Nemotron / MiniMax,协调者用 Gemma。同血统的模型共享
知识盲区,互相审查等于让同一个人检查自己的作业三遍。

选型不看速度也不看参数量,而是按角色实测(见 `finalize_agents.py`):
协调者要求 JSON 合规稳定,**所有** agent 必须通过克制探针 —— 给它
虚构的战队,看它说"无可补充"还是编造转会消息。四组不同虚构对局各
测一次,两个候选模型有约 50% 的编造率,单次测试有一半概率把它们
认证为合格。

## 已知限制

- 队伍历史少于 3 场无法预测,少于 5 场会给警告
- 特征仓库在启动时加载,新比赛需重跑 `train.py` 并重启
- Stage 2 的概率范围约 [0.25, 0.71],一边倒的对局给不出可用判别
- Stage 3 每次约 4 次 LLM 调用,NIM 免费额度约 40 req/min,够单人使用
- 模型 walk-forward 期望 lift ≈ +0.13 ± 0.10,单场预测不确定性大

## 数据新鲜度

`/health` 分赛区报告数据截止日期,`/predict` 顶层返回 `data_quality`:

```jsonc
"data_quality": {
  "league_data_through": "2026-07-17",
  "days_old": 9,
  "level": "stale",        // fresh <=3d | stale 4-9d | very_stale >=10d
  "reliable": false
}
```

滞后超过 10 天时 disclaimer 会加前缀警告。

这个检查存在的原因:LPL 的数据来源和其他赛区不同。LPL 由 TJ Sports
运营而非 Riot,数据要从 `lpl.qq.com` 单独抓取,链路更脆弱,新赛段开始时
常滞后数日至一周。LEC/LCS 走 Riot 官方管道,当日即有。

滚动窗口是 10 场,一周的滞后意味着窗口缺少最近的比赛日;如果期间发生
转会,队伍统计描述的是一支已不存在的阵容,而模型对此毫无感知。

另外 Oracle's Elixir 官方公告:2026 年 BP 规则变更导致部分比赛的英雄
选择记录可能不正确。带 draft 的请求会在 disclaimer 中附上这一条。
