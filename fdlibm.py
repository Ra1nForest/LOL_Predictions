"""
可移植的 exp / log (fdlibm 的 e_exp.c / e_log.c, 逐行移植)
==========================================================
为什么不用 math.exp / math.log: 看板上的数由两份实现算 (Python 和浏览器的 frontend/src/web/), test:diff
要求逐位相同, 而**没有哪两个平台的 exp / log 逐位一致**: 本机 math.exp 是 MSVC 运行库的 (实测 20 万个输入里
1147 个不是正确舍入), V8 的 Math.exp / Math.log 是 fdlibm (和本机 math.exp 有 7% 的输入差一位), Safari 用系统
libm, Linux 上的 Python 用 glibc。差一位本身无所谓, 但它会让"同一个看板两边算出不同的数"变成常态, 真正的移植
错误就藏在里面看不见。

这里只用 IEEE 加减乘除、按位拼装和 2 的整数次幂缩放 —— 每一步在任何平台上都是同一个结果。浏览器版
web/fdlibm.ts 是同一段代码的逐行移植, 两边逐位相同 (test:golden 的 fdlibm 用例), 也和 V8 的 Math.exp /
Math.log 逐位相同 (V8 本来就是 fdlibm; exp 只对 x = 1 特判返回 Math.E, 这里不跟, 两份移植一致就够)。

谁在用:
  · xregion (模型 C): 从导出状态出发的重放、衰减、logistic (xregion.portable_exp 就是这里的 exp)
  · Stage 1/2 的校准层 (feature_store.calib_logit / logit_elo_prob、api.Stage 的 Platt): 看板上的 BP 后 /
    赛前概率不取整就输出, 用平台的 exp / log 时约 4.5% 的输入两边差一位, test:diff 用 === 比 (2026-10-17 实测)
训练和闸门 (sklearn / numpy 整表算) 仍用平台的 exp / log: 两种至多差一位, 远在模型的意义之下; 线上对外的
那个数只走这里。
"""
from __future__ import annotations

import math
import struct

# ── exp (e_exp.c) ──────────────────────────────────────────
_HALF = (0.5, -0.5)
_LN2HI = (6.93147180369123816490e-01, -6.93147180369123816490e-01)   # 0x3fe62e42 fee00000
_LN2LO = (1.90821492927058770002e-10, -1.90821492927058770002e-10)   # 0x3dea39ef 35793c76
_INVLN2 = 1.44269504088896338700e+00
_P = (1.66666666666666019037e-01, -2.77777777770155933842e-03, 6.61375632143793436117e-05,
      -1.65339022054652515390e-06, 4.13813679705723846039e-08)
_HUGE = 1.0e+300
_TWOM1000 = 9.33263618503218878990e-302                          # 2**-1000
_O_THRESHOLD = 7.09782712893383973096e+02
_U_THRESHOLD = -7.45133219101941108420e+02


def exp(x) -> float:
    """fdlibm 的 __ieee754_exp, 逐行移植。浏览器版: web/fdlibm.ts 的 exp, 两边逐位相同 (test:golden)。"""
    x = float(x)
    bits = struct.unpack("<Q", struct.pack("<d", x))[0]
    hx, lx = bits >> 32, bits & 0xFFFFFFFF
    xsb = (hx >> 31) & 1
    hx &= 0x7FFFFFFF
    if hx >= 0x40862E42:                            # |x| >= 709.78...
        if hx >= 0x7FF00000:
            if ((hx & 0xFFFFF) | lx) != 0:
                return x + x                        # NaN
            return x if xsb == 0 else 0.0           # exp(±inf)
        if x > _O_THRESHOLD:
            return math.inf
        if x < _U_THRESHOLD:
            return 0.0
    k, hi, lo = 0, 0.0, 0.0
    if hx > 0x3FD62E42:                             # |x| > 0.5 ln2: 约简到 [-0.5 ln2, 0.5 ln2]
        if hx < 0x3FF0A2B2:                         # 且 |x| < 1.5 ln2
            hi, lo, k = x - _LN2HI[xsb], _LN2LO[xsb], 1 - xsb - xsb
        else:
            k = int(_INVLN2 * x + _HALF[xsb])       # 向零取整, 同 C 的 (int)
            t = float(k)
            hi = x - t * _LN2HI[0]
            lo = t * _LN2LO[0]
        x = hi - lo
    elif hx < 0x3E300000:                           # |x| < 2**-28
        if _HUGE + x > 1.0:
            return 1.0 + x
    p1, p2, p3, p4, p5 = _P
    t = x * x
    c = x - t * (p1 + t * (p2 + t * (p3 + t * (p4 + t * p5))))
    if k == 0:
        return 1.0 - ((x * c) / (c - 2.0) - x)
    y = 1.0 - ((lo - (x * c) / (2.0 - c)) - hi)
    # fdlibm 直接往 y 的指数位上加 k —— 乘 2 的整数次幂是精确的, 结果一样。2.0 ** 1024 在 Python 里溢出, 拆两步
    if k >= -1021:
        return y * 2.0 ** k if k < 1024 else y * 2.0 * 2.0 ** 1023
    return y * 2.0 ** (k + 1000) * _TWOM1000


# ── log (e_log.c) ──────────────────────────────────────────
_LOG_LN2_HI = 6.93147180369123816490e-01        # 3fe62e42 fee00000
_LOG_LN2_LO = 1.90821492927058770002e-10        # 3dea39ef 35793c76
_TWO54 = 1.80143985094819840000e+16             # 43500000 00000000
_LG = (6.666666666666735130e-01,                # Lg1 3FE55555 55555593
       3.999999999940941908e-01,                # Lg2 3FD99999 9997FA04
       2.857142874366239149e-01,                # Lg3 3FD24924 94229359
       2.222219843214978396e-01,                # Lg4 3FCC71C5 1D8E78AF
       1.818357216161805012e-01,                # Lg5 3FC74664 96CB03DE
       1.531383769920937332e-01,                # Lg6 3FC39A09 D078C69F
       1.479819860511658591e-01)                # Lg7 3FC2F112 DF3E5244


def _words(x: float) -> tuple[int, int]:
    """(高位字, 低位字), 高位字按 C 的 int32 解释 (有符号)。"""
    b = struct.unpack("<Q", struct.pack("<d", x))[0]
    hi = b >> 32
    return (hi - (1 << 32) if hi >= 0x80000000 else hi), b & 0xFFFFFFFF


def _with_hi(hi: int, lo: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", ((hi & 0xFFFFFFFF) << 32) | lo))[0]


def log(x) -> float:
    """fdlibm 的 __ieee754_log, 逐行移植 (和 V8 的 Math.log 同源)。浏览器版: web/fdlibm.ts 的 log, 两边逐位相同。"""
    x = float(x)
    hx, lx = _words(x)
    k = 0
    if hx < 0x00100000:                             # x < 2**-1022 (含负数)
        if ((hx & 0x7FFFFFFF) | lx) == 0:
            return -math.inf                        # log(±0) = -inf
        if hx < 0:
            return math.nan                         # log(负数) = NaN
        k -= 54
        x *= _TWO54                                 # 次正规数, 先放大
        hx, lx = _words(x)
    if hx >= 0x7FF00000:
        return x + x                                # inf / NaN
    k += (hx >> 20) - 1023
    hx &= 0x000FFFFF
    i = (hx + 0x95F64) & 0x100000
    x = _with_hi(hx | (i ^ 0x3FF00000), lx)         # 把 x 规范到 [sqrt(2)/2, sqrt(2)) 附近
    k += i >> 20
    f = x - 1.0
    if (0x000FFFFF & (2 + hx)) < 3:                 # -2**-20 <= f < 2**-20
        if f == 0.0:
            if k == 0:
                return 0.0
            dk = float(k)
            return dk * _LOG_LN2_HI + dk * _LOG_LN2_LO
        R = f * f * (0.5 - 0.33333333333333333 * f)
        if k == 0:
            return f - R
        dk = float(k)
        return dk * _LOG_LN2_HI - ((R - dk * _LOG_LN2_LO) - f)
    lg1, lg2, lg3, lg4, lg5, lg6, lg7 = _LG
    s = f / (2.0 + f)
    dk = float(k)
    z = s * s
    i = hx - 0x6147A
    w = z * z
    j = 0x6B851 - hx
    t1 = w * (lg2 + w * (lg4 + w * lg6))
    t2 = z * (lg1 + w * (lg3 + w * (lg5 + w * lg7)))
    i |= j
    R = t2 + t1
    if i > 0:
        hfsq = 0.5 * f * f
        if k == 0:
            return f - (hfsq - s * (hfsq + R))
        return dk * _LOG_LN2_HI - ((hfsq - (s * (hfsq + R) + dk * _LOG_LN2_LO)) - f)
    if k == 0:
        return f - s * (f - R)
    return dk * _LOG_LN2_HI - ((s * (f - R) - dk * _LOG_LN2_LO) - f)
