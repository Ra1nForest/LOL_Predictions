/**
 * XGBoost (binary:logistic + gbtree) 的纯 JS 推理 —— Stage 1 和 Stage 4 共用这一份。
 *
 * 和 XGBoost 逐位一致的前提, 每一条都由 scripts/golden.mjs 钉住:
 *   1. 输入先转 float32 再和阈值比。阈值、叶子值本来就是 float32 ——
 *      用 float64 比会在阈值附近走错分支, 差一个 ulp 就是另一片叶子
 *   2. 从 base_score 的 logit 起, 按树的顺序逐棵加, 每一步都落回 float32
 *   3. sigmoid 也在 float32 上算: 1 / (expf(min(-x, 88.7)) + 1)
 * 缺失值由调用方先填 -999 —— 两个 Python 模型的 predict() 都是 np.nan_to_num(nan=-999),
 * 所以这里的缺失分支 (default_left) 只是防御, 正常走不到。
 */

/** 和 tools/export_web_model.py 的 FORMAT 必须一致; 改导出格式就改这个号 */
export const FORMAT = "xgb-binary-logistic-v1";

export interface TreeModelFile {
  format: string;
  features: string[];
  base_score: number;
  trees: { l: number[]; r: number[]; f: number[]; c: number[]; d: number[] }[];
  importance: Record<string, number>;
}

interface Tree {
  l: Int32Array;
  r: Int32Array;
  f: Int32Array;
  /** 内部节点是阈值, 叶子节点 (l = -1) 是叶子值 —— 都是 float32 */
  c: Float32Array;
  d: Uint8Array;
}

export interface Forest {
  features: string[];
  featureSet: Set<string>;
  trees: Tree[];
  baseMargin: number;
  /** 用 Map 而不是普通对象: 普通对象上 "constructor" 这种键会查到原型链上 */
  importance: Map<string, number>;
}

export function loadForest(file: TreeModelFile): Forest {
  if (file.format !== FORMAT) {
    throw new Error(`模型文件格式是 ${file.format}, 期望 ${FORMAT} —— 重新导出`);
  }
  // base_score 在概率空间; 起始 margin 是 XGBoost 的 ProbToMargin: -log(1/p - 1), 全程 float
  const p = Math.fround(file.base_score);
  return {
    features: file.features,
    featureSet: new Set(file.features),
    trees: file.trees.map((t) => ({
      l: Int32Array.from(t.l),
      r: Int32Array.from(t.r),
      f: Int32Array.from(t.f),
      c: Float32Array.from(t.c),
      d: Uint8Array.from(t.d),
    })),
    baseMargin: Math.fround(-Math.fround(Math.log(Math.fround(Math.fround(1 / p) - 1)))),
    importance: new Map(Object.entries(file.importance)),
  };
}

/** 按特征顺序组向量, 连同 np.nan_to_num(x, nan=-999.0) 的行为 (±inf → ±最大浮点数) */
export function toVector(features: string[], row: Record<string, number>): Float64Array {
  const x = new Float64Array(features.length);
  features.forEach((f, i) => {
    const v = Object.prototype.hasOwnProperty.call(row, f) ? row[f]! : NaN;
    x[i] = Number.isNaN(v)
      ? -999
      : v === Infinity
        ? Number.MAX_VALUE
        : v === -Infinity
          ? -Number.MAX_VALUE
          : v;
  });
  return x;
}

/** 原始概率, 即 predict_proba(x)[0, 1] —— float32 精度 */
export function predictRaw(forest: Forest, x: Float64Array): number {
  // XGBoost 把输入转成 float32 再比 —— Float32Array 的赋值就是同一个舍入
  const xf = Float32Array.from(x);
  let margin = forest.baseMargin;
  for (const t of forest.trees) {
    let n = 0;
    while (t.l[n]! !== -1) {
      const v = xf[t.f[n]!]!;
      n = Number.isNaN(v) ? (t.d[n] ? t.l[n]! : t.r[n]!) : v < t.c[n]! ? t.l[n]! : t.r[n]!;
    }
    margin = Math.fround(margin + t.c[n]!);
  }
  const e = Math.fround(Math.exp(Math.min(-margin, Math.fround(88.7))));
  return Math.fround(1 / Math.fround(e + 1));
}
