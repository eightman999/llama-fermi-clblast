// CL11 backend: element-wise kernels. One work-item per output element.

// ---------------------------------------------------------------------------------------------
// binary ops with broadcasting of src1: dst = src0 (op) src1, all F32.
// src0 has the shape of dst, src1 is repeated (modulo) along every dimension.
// ---------------------------------------------------------------------------------------------
#define DEF_BIN(NAME, EXPR)                                                                        \
kernel void NAME(global uchar * p0, tdesc t0, global uchar * p1, tdesc t1,                       \
                 global uchar * pd, tdesc td, int n) {                                            \
    const int idx = LGID() * WG_EW + (int)get_local_id(0);                                        \
    if (idx >= n) return;                                                                         \
    int i0, i1, i2, i3;                                                                           \
    SPLIT4(idx, td, i0, i1, i2, i3);                                                              \
    const float a = LD_F32(p0 + OFF4(t0, i0, i1, i2, i3));                                        \
    const float b = LD_F32(p1 + OFF4(t1, FMOD(i0, t1, 0), FMOD(i1, t1, 1), FMOD(i2, t1, 2), FMOD(i3, t1, 3)));    \
    ST_F32(pd + OFF4(td, i0, i1, i2, i3), (EXPR));                                                \
}

DEF_BIN(k_add, a + b)
DEF_BIN(k_sub, a - b)
DEF_BIN(k_mul, a * b)
DEF_BIN(k_div, a / b)

// ---------------------------------------------------------------------------------------------
// unary ops (F32). op codes must match the host:
//   0 silu, 1 relu, 2 neg, 3 abs, 4 sigmoid, 5 tanh, 6 sqr, 7 sqrt, 8 scale (x*p0 + p1), 9 clamp(p0, p1)
// ---------------------------------------------------------------------------------------------
kernel void k_unary(global uchar * ps, tdesc ts, global uchar * pd, tdesc td, int n, int op, float p0, float p1) {
    const int idx = LGID() * WG_EW + (int)get_local_id(0);
    if (idx >= n) return;
    int i0, i1, i2, i3;
    SPLIT4(idx, td, i0, i1, i2, i3);
    const float x = LD_F32(ps + OFF4(ts, i0, i1, i2, i3));
    float y;
    switch (op) {
        case 0:  y = x / (1.0f + exp(-x)); break;
        case 1:  y = fmax(x, 0.0f); break;
        case 2:  y = -x; break;
        case 3:  y = fabs(x); break;
        case 4:  y = 1.0f / (1.0f + exp(-x)); break;
        case 5:  y = tanh(x); break;
        case 6:  y = x * x; break;
        case 7:  y = sqrt(x); break;
        case 8:  y = x * p0 + p1; break;
        default: y = fmin(fmax(x, p0), p1); break;
    }
    ST_F32(pd + OFF4(td, i0, i1, i2, i3), y);
}

// ---------------------------------------------------------------------------------------------
// gated linear units (F32): dst = act(x) * g, act = silu (op 0, SWIGLU) or relu (op 1, REGLU).
// split != 0: x = src0, g = src1. Otherwise x and g are the two halves of the rows of src0
// (the halves are swapped when `swapped` is set).
// nc is the number of columns of dst.
// ---------------------------------------------------------------------------------------------
kernel void k_glu(global uchar * p0, tdesc t0, global uchar * p1, tdesc t1, global uchar * pd, tdesc td,
                  int n, int op, int split, int swapped, int nc) {
    const int idx = LGID() * WG_EW + (int)get_local_id(0);
    if (idx >= n) return;
    int i0, i1, i2, i3;
    SPLIT4(idx, td, i0, i1, i2, i3);
    float x, g;
    if (split) {
        x = LD_F32(p0 + OFF4(t0, i0, i1, i2, i3));
        g = LD_F32(p1 + OFF4(t1, i0, i1, i2, i3));
    } else {
        x = LD_F32(p0 + OFF4(t0, i0 + (swapped ? nc : 0), i1, i2, i3));
        g = LD_F32(p0 + OFF4(t0, i0 + (swapped ? 0 : nc), i1, i2, i3));
    }
    const float a = (op == 0) ? x / (1.0f + exp(-x)) : fmax(x, 0.0f);
    ST_F32(pd + OFF4(td, i0, i1, i2, i3), a * g);
}

// ---------------------------------------------------------------------------------------------
// copy / convert (CPY, DUP, CONT): the elements are paired by linear index, the source and the
// destination may have different shapes (same number of elements) and arbitrary strides.
// ---------------------------------------------------------------------------------------------
#define DEF_CPY(NAME, LD, ST)                                                         \
kernel void NAME(global uchar * ps, tdesc ts, global uchar * pd, tdesc td, int n) {  \
    const int idx = LGID() * WG_EW + (int)get_local_id(0);                            \
    if (idx >= n) return;                                                             \
    int a0, a1, a2, a3, b0, b1, b2, b3;                                               \
    SPLIT4(idx, ts, a0, a1, a2, a3);                                                  \
    SPLIT4(idx, td, b0, b1, b2, b3);                                                  \
    const float v = LD(ps + OFF4(ts, a0, a1, a2, a3));                                \
    ST(pd + OFF4(td, b0, b1, b2, b3), v);                                             \
}

DEF_CPY(k_cpy_f32_f32, LD_F32, ST_F32)
DEF_CPY(k_cpy_f32_f16, LD_F32, ST_F16)
DEF_CPY(k_cpy_f16_f32, LD_F16, ST_F32)
DEF_CPY(k_cpy_f16_f16, LD_F16, ST_F16)

// ---------------------------------------------------------------------------------------------
// transposing copy (e.g. CONT of a transposed view): dst has the shape of src and is contiguous, dim 1 of src is
// contiguous (nb1 == element size) and dim 0 is strided. A work-group of GT*GT work-items copies a 32x32 tile
// (i0, i1) through local memory, so that both the reads and the writes are coalesced.
// bc.tiles / bc.tiles2: number of tiles along dim 0 / dim 1, the group index runs over tiles and (i2, i3).
// ---------------------------------------------------------------------------------------------
#define DEF_CPY_T(NAME, LD, ST)                                                                           \
kernel void NAME(global uchar * ps, tdesc ts, global uchar * pd, tdesc td, bcast bc) {                    \
    local float tile[32 * 33];                                                                            \
    const int lid = (int)get_local_id(0);                                                                 \
    const int tx = lid & 31;                                                                              \
    const int ty = lid >> 5;                                                                              \
    int g = LGID();                                                                                       \
    if (g >= bc.tiles * bc.tiles2 * td.ne2 * td.ne3) return; /* uniform for the work-group */              \
    const int gq  = FDIV_RAW(g, bc.mt, bc.lt);                                                            \
    const int t0  = g - gq * bc.tiles;                                                                    \
    const int gq2 = FDIV_RAW(gq, bc.mt2, bc.lt2);                                                         \
    const int t1  = gq - gq2 * bc.tiles2;                                                                 \
    const int i3 = FDIV(gq2, td, 2);                                                                      \
    const int i2 = gq2 - i3 * td.ne2;                                                                     \
    const int i0b = t0 * 32;                                                                              \
    const int i1b = t1 * 32;                                                                              \
    for (int r = ty; r < 32; r += (GT * GT) / 32) {                                                       \
        const int i0 = i0b + r;                                                                           \
        const int i1 = i1b + tx;                                                                          \
        if (i0 < ts.ne0 && i1 < ts.ne1) {                                                                 \
            tile[r * 33 + tx] = LD(ps + OFF4(ts, i0, i1, i2, i3));                                        \
        }                                                                                                 \
    }                                                                                                     \
    barrier(CLK_LOCAL_MEM_FENCE);                                                                         \
    for (int r = ty; r < 32; r += (GT * GT) / 32) {                                                       \
        const int i1 = i1b + r;                                                                           \
        const int i0 = i0b + tx;                                                                          \
        if (i0 < td.ne0 && i1 < td.ne1) {                                                                 \
            ST(pd + OFF4(td, i0, i1, i2, i3), tile[tx * 33 + r]);                                         \
        }                                                                                                 \
    }                                                                                                     \
}

DEF_CPY_T(k_cpy_t_f32_f32, LD_F32, ST_F32)
DEF_CPY_T(k_cpy_t_f16_f16, LD_F16, ST_F16)

// ---------------------------------------------------------------------------------------------
// memset of buffer ranges (the OpenCL 1.2 clEnqueueFillBuffer must not be used)
// ---------------------------------------------------------------------------------------------
kernel void k_fill_u32(global uchar * p, int off, int n, uint value) {
    const int idx = LGID() * WG_EW + (int)get_local_id(0);
    if (idx >= n) return;
    ((global uint *)(p + off))[idx] = value;
}

kernel void k_fill_u8(global uchar * p, int off, int n, uint value) {
    const int idx = LGID() * WG_EW + (int)get_local_id(0);
    if (idx >= n) return;
    p[off + idx] = (uchar)value;
}
