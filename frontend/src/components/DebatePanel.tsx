import { useEffect, useRef, useState } from "react";
import { mutations, ApiError } from "../api/client";
import type { Claim, DebateRequestBody, DebateResponse, Verdict } from "../api/types";

const STATUS_CN: Record<string, string> = {
  survived: "站住了",
  contested: "各执一词",
  conceded: "已让步",
  refuted: "被驳倒",
};
// 站住的排最前 —— 那是最可信的, 让人一眼看到
const ORDER: Record<string, number> = {
  survived: 0,
  contested: 1,
  conceded: 2,
  refuted: 3,
};

/**
 * Stage 3 的复核面板。
 *
 * 三条来自旧版的经验, 迁移时保留:
 *
 * 1. **一次只能跑一个**, 而且结果留在页面上不被轮询刷掉 —— 这是全项目
 *    唯一真花钱的调用 (四次 LLM + 若干次检索)。
 * 2. **只报已用秒数, 不编进度条**。拿不到真实进度, 假的"正在第 2 轮"
 *    比不报还糟。
 * 3. **判断谁掉线要看 text 而不是 error**。后端的内容告警 (如"疑似思维链
 *    泄漏") 也走 error 字段, 那种情况下 agent 其实发言了。只按 error 过滤
 *    会把"答了话但格式有瑕疵"报成"未能参与", 凭空吓人一跳。
 */
export function DebatePanel({ body, resetKey }: { body: DebateRequestBody | null; resetKey: string }) {
  const [data, setData] = useState<DebateResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [secs, setSecs] = useState(0);
  const timer = useRef<number | undefined>(undefined);

  // 换比赛 / 换局: 上一场的结论必须清掉, 否则会挂在新局面下
  useEffect(() => {
    setData(null);
    setError(null);
  }, [resetKey]);

  useEffect(() => () => clearInterval(timer.current), []);

  const run = async () => {
    if (!body || running) return;
    setRunning(true);
    setError(null);
    setSecs(0);
    timer.current = window.setInterval(() => setSecs((s) => s + 1), 1000);
    try {
      setData(await mutations.debate(body));
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      clearInterval(timer.current);
      setRunning(false);
    }
  };

  const v: Verdict | undefined = data?.debate?.verdict ?? data?.debate?.rounds?.["3_verdict"];
  const claims: Claim[] = [...(v?.claims ?? [])].sort(
    (a, b) => (ORDER[a.status] ?? 9) - (ORDER[b.status] ?? 9),
  );
  const r1 = data?.debate?.rounds?.["1_positions"] ?? [];
  const dead = r1.filter((t) => !(t.text || "").trim()).map((t) => t.agent);
  const noisy = r1.filter((t) => t.error && (t.text || "").trim()).map((t) => t.agent);

  return (
    <section className="panel pad">
      <div className="debate-head">
        <h3 className="card-title" style={{ margin: 0 }}>
          AI 复核
        </h3>
        <span className="spacer" />
        <button className="btn" onClick={() => void run()} disabled={!body || running}>
          {running ? `分析中… ${secs}s` : data ? "重新复核" : "开始复核"}
        </button>
      </div>

      {!data && !running && !error && (
        <p className="db-hint">
          让四个模型交叉质证, 总结当前局势。约需一分钟。
        </p>
      )}

      {running && (
        <p className="db-hint">
          分析中, 已用 {secs} 秒, 通常约一分钟。
        </p>
      )}

      {error && <div className="warn" style={{ marginTop: 12 }}>{error}</div>}

      {data && (
        <>
          {data.ingame && (
            <p className="db-hint">
              基于当前局面 {(data.ingame.probability_blue * 100).toFixed(1)}%
              (赛前 {(data.ingame.pregame * 100).toFixed(1)}%, 变动{" "}
              {data.ingame.shift >= 0 ? "+" : ""}
              {(data.ingame.shift * 100).toFixed(1)} 个点)
            </p>
          )}

          {v?.consensus && <div className="db-consensus">{v.consensus}</div>}

          {claims.length > 0 && (
            <div className="db-sec">
              <h4>结论</h4>
              {claims.map((c, i) => (
                <div className="claim" key={i}>
                  <span className={`cst ${c.status}`}>{STATUS_CN[c.status] ?? c.status}</span>
                  <span>
                    {c.claim}
                    {c.note && <span className="cmeta">{c.note}</span>}
                    {c.evidence_url && (
                      <a
                        className="cmeta"
                        href={c.evidence_url}
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        查看来源 ↗
                      </a>
                    )}
                  </span>
                </div>
              ))}
            </div>
          )}

          {!!v?.open_disputes?.length && (
            <div className="db-sec">
              <h4>存在分歧</h4>
              {v.open_disputes.map((d, i) => (
                <div className="claim" key={i}>
                  <span>· {d}</span>
                </div>
              ))}
            </div>
          )}

          {dead.length > 0 && (
            <p className="db-err">
              {dead.length} 个模型未能给出回应 ({dead.join("、")}), 以下结论仅反映其余模型。
            </p>
          )}
          {noisy.length > 0 && (
            <p className="db-hint">{noisy.join("、")} 的输出格式异常, 已清理后采用。</p>
          )}
          {v?.verdict_model && (
            <p className="db-hint">主协调模型无响应, 本次由 {v.verdict_model} 完成汇总。</p>
          )}

          <details className="fineprint" style={{ marginTop: 14 }}>
            <summary>模型说明</summary>
            <p className="mono">
              {data.final.note} {data.disclaimer}
            </p>
          </details>
        </>
      )}
    </section>
  );
}
