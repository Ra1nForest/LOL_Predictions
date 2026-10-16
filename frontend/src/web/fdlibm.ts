/**
 * 可移植的 exp / log —— fdlibm.py 的逐行移植 (理由见那里的模块文档)。
 *
 * 看板上的数由两份实现算, test:diff 要求逐位相同, 而平台的 exp / log 两两之间都会差一位 (本机 Python 是 MSVC
 * 的, Safari 用系统 libm)。这里只用 IEEE 加减乘除、按位拼装和 2 的整数次幂缩放, 在任何浏览器里都是同一个结果,
 * 和 Python 的 fdlibm.exp / fdlibm.log 逐位相同 (test:golden 的 fdlibm 用例), 也和 V8 的 Math.exp / Math.log
 * 逐位相同 (V8 本来就是 fdlibm; exp 只对 x = 1 特判返回 Math.E —— 这里不跟)。
 *
 * 谁在用: xregion.ts (模型 C 从状态出发的那一段), stage1.ts 的 predictStage1 (Stage 1/2 的校准层)。
 */

const F64 = new Float64Array(1);
const U32 = new Uint32Array(F64.buffer); // 小端: [低位字, 高位字]

// ── exp (e_exp.c) ──
const FD_HALF = [0.5, -0.5];
const FD_LN2HI = [6.9314718036912381649e-1, -6.9314718036912381649e-1]; // 0x3fe62e42 fee00000
const FD_LN2LO = [1.90821492927058770002e-10, -1.90821492927058770002e-10]; // 0x3dea39ef 35793c76
const FD_INVLN2 = 1.442695040888963387;
const FD_P1 = 1.66666666666666019037e-1;
const FD_P2 = -2.77777777770155933842e-3;
const FD_P3 = 6.61375632143793436117e-5;
const FD_P4 = -1.6533902205465251539e-6;
const FD_P5 = 4.13813679705723846039e-8;
const FD_HUGE = 1.0e300;
const FD_TWOM1000 = 9.3326361850321887899e-302; // 2**-1000
const FD_O_THRESHOLD = 7.09782712893383973096e2;
const FD_U_THRESHOLD = -7.4513321910194110842e2;

/** fdlibm 的 __ieee754_exp —— fdlibm.py 的 exp, 两边逐位相同 */
export function exp(x: number): number {
  F64[0] = x;
  let hx = U32[1]! >>> 0;
  const lx = U32[0]! >>> 0;
  const xsb = (hx >>> 31) & 1;
  hx = (hx & 0x7fffffff) >>> 0;
  if (hx >= 0x40862e42) {
    // |x| >= 709.78...
    if (hx >= 0x7ff00000) {
      if (((hx & 0xfffff) | lx) !== 0) return x + x; // NaN
      return xsb === 0 ? x : 0.0; // exp(±inf)
    }
    if (x > FD_O_THRESHOLD) return Infinity;
    if (x < FD_U_THRESHOLD) return 0.0;
  }
  let k = 0;
  let hi = 0;
  let lo = 0;
  if (hx > 0x3fd62e42) {
    // |x| > 0.5 ln2: 约简到 [-0.5 ln2, 0.5 ln2]
    if (hx < 0x3ff0a2b2) {
      hi = x - FD_LN2HI[xsb]!;
      lo = FD_LN2LO[xsb]!;
      k = 1 - xsb - xsb;
    } else {
      k = Math.trunc(FD_INVLN2 * x + FD_HALF[xsb]!); // 向零取整, 同 C 的 (int)
      const t = k;
      hi = x - t * FD_LN2HI[0]!;
      lo = t * FD_LN2LO[0]!;
    }
    x = hi - lo;
  } else if (hx < 0x3e300000) {
    // |x| < 2**-28
    if (FD_HUGE + x > 1) return 1 + x;
  }
  const t = x * x;
  const c = x - t * (FD_P1 + t * (FD_P2 + t * (FD_P3 + t * (FD_P4 + t * FD_P5))));
  if (k === 0) return 1 - ((x * c) / (c - 2.0) - x);
  const y = 1 - (lo - (x * c) / (2.0 - c) - hi);
  // fdlibm 直接往 y 的指数位上加 k —— 乘 2 的整数次幂是精确的, 结果一样; k = 1024 拆两步, 同 Python
  if (k >= -1021) return k < 1024 ? y * 2 ** k : y * 2 * 2 ** 1023;
  return y * 2 ** (k + 1000) * FD_TWOM1000;
}

// ── log (e_log.c) ──
const LN2_HI = 6.9314718036912381649e-1; // 3fe62e42 fee00000
const LN2_LO = 1.90821492927058770002e-10; // 3dea39ef 35793c76
const TWO54 = 1.8014398509481984e16; // 43500000 00000000
const LG1 = 6.66666666666673513e-1; // 3FE55555 55555593
const LG2 = 3.999999999940941908e-1; // 3FD99999 9997FA04
const LG3 = 2.857142874366239149e-1; // 3FD24924 94229359
const LG4 = 2.222219843214978396e-1; // 3FCC71C5 1D8E78AF
const LG5 = 1.818357216161805012e-1; // 3FC74664 96CB03DE
const LG6 = 1.531383769920937332e-1; // 3FC39A09 D078C69F
const LG7 = 1.479819860511658591e-1; // 3FC2F112 DF3E5244

/** fdlibm 的 __ieee754_log —— fdlibm.py 的 log, 两边逐位相同 */
export function log(x: number): number {
  F64[0] = x;
  let hx = U32[1]! | 0; // 高位字按 C 的 int32 (有符号)
  let lx = U32[0]! >>> 0;
  let k = 0;
  if (hx < 0x00100000) {
    // x < 2**-1022 (含负数)
    if (((hx & 0x7fffffff) | lx) === 0) return -Infinity; // log(±0) = -inf
    if (hx < 0) return NaN; // log(负数) = NaN
    k -= 54;
    x *= TWO54; // 次正规数, 先放大
    F64[0] = x;
    hx = U32[1]! | 0;
    lx = U32[0]! >>> 0;
  }
  if (hx >= 0x7ff00000) return x + x; // inf / NaN
  k += (hx >> 20) - 1023;
  hx &= 0x000fffff;
  let i = (hx + 0x95f64) & 0x100000;
  // 把 x 规范到 [sqrt(2)/2, sqrt(2)) 附近: 换高位字, 低位字不动
  U32[0] = lx;
  U32[1] = (hx | (i ^ 0x3ff00000)) >>> 0;
  x = F64[0]!;
  k += i >> 20;
  const f = x - 1.0;
  if ((0x000fffff & (2 + hx)) < 3) {
    // -2**-20 <= f < 2**-20
    if (f === 0) {
      if (k === 0) return 0;
      const dk = k;
      return dk * LN2_HI + dk * LN2_LO;
    }
    const R = f * f * (0.5 - 0.33333333333333333 * f);
    if (k === 0) return f - R;
    const dk = k;
    return dk * LN2_HI - (R - dk * LN2_LO - f);
  }
  const s = f / (2.0 + f);
  const dk = k;
  const z = s * s;
  i = hx - 0x6147a;
  const w = z * z;
  const j = 0x6b851 - hx;
  const t1 = w * (LG2 + w * (LG4 + w * LG6));
  const t2 = z * (LG1 + w * (LG3 + w * (LG5 + w * LG7)));
  i |= j;
  const R = t2 + t1;
  if (i > 0) {
    const hfsq = 0.5 * f * f;
    if (k === 0) return f - (hfsq - s * (hfsq + R));
    return dk * LN2_HI - (hfsq - (s * (hfsq + R) + dk * LN2_LO) - f);
  }
  if (k === 0) return f - s * (f - R);
  return dk * LN2_HI - (s * (f - R) - dk * LN2_LO - f);
}
