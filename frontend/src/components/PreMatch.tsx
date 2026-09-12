import type { PredictResponse } from "../api/types";
import { ProbabilityScale } from "./ProbabilityScale";
import { T } from "../i18n";

/**
 * 赛前预测视图 —— 比赛还没开打时看到的那一屏。
 *
 * 和看板的区别是**信息集不同**, 不是简化版: 这里没有任何局内读数, 所以
 * 不画时间轴、不显示局势和选手。概率来自 Stage 1/2 (只看队伍历史和 BP),
 * 一旦比赛开打, 外层会自动切到局内模型。
 */
export function PreMatch({
  data,
  blueName,
  redName,
}: {
  data: PredictResponse;
  blueName: string;
  redName: string;
}) {
  const key = data.final.probability_source;
  const stage = data.stages[key];
  const model = stage?.model;
  const p = data.final.probability_blue;

  const footnotes = [
    model?.note,
    model?.range_note,
    data.data_quality?.league_data_through &&
      T("{league} 数据截至 {date}（{days} 天前）", {
        league: data.league,
        date: data.data_quality.league_data_through,
        days: data.data_quality.days_old ?? "?",
      }),
    data.disclaimer,
  ].filter(Boolean);

  return (
    <>
      <ProbabilityScale
        blueName={blueName}
        redName={redName}
        probabilityBlue={p}
        pMin={model?.p_min ?? null}
        pMax={model?.p_max ?? null}
        pregame={null}
        reasons={stage?.reasons}
      />
      {/* 结论和"这是哪一档预测"排同一行左右对齐, 和看板里那一对完全一致。
          .summary / .shift 的 margin-top 都是 0, 间距由 .summary-row 给 ——
          少了这层包裹不只是不同行, 连上边距都会一起没有。 */}
      <div className="summary-row">
        {data.final.summary && <p className="summary">{data.final.summary}</p>}
        <p className="shift">
          {key === "2_post_draft" ? T("已计入 BP 的预测") : T("赛前预测，尚未开打")}
        </p>
      </div>

      <div className="stack" style={{ marginTop: 26 }}>
        {data.warnings?.map((w, i) => (
          <div className="warn" key={i}>
            {w}
          </div>
        ))}

        {/* "主要依据"那一块删了 —— 内容搬进了刻度上"胜率解析"的气泡弹窗,
            和局内那两段用的是同一个入口, 三个阶段行为一致。 */}
        {footnotes.length > 0 && (
          <details className="fineprint">
            <summary>
              {T("准确率约")}{" "}
              {model?.accuracy != null ? `${Math.round(model.accuracy * 100)}%` : "—"} · {T("模型说明")}
            </summary>
            <p className="mono">{footnotes.join(" · ")}</p>
          </details>
        )}
      </div>
    </>
  );
}
