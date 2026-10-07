// CL11 backend: MUL_MAT with a skinny src1 (mat-vec), one kernel per type of src0.
//
//   dst[i0, i1, i2, i3] = sum_k src0[k, i0, i2 / r2, i3 / r3] * src1[k, i1, i2, i3]
//
// src0: F32/F16/Q4_K/Q6_K (rows contiguous), src1: F32 (rows contiguous), dst: F32. (Q4_0 / Q8_0: mul_mv_r.cl)
// `tpr` threads (a power of 2, chosen by the host) cooperate on one output element, and a work-group
// of WG_MV threads computes WG_MV/tpr outputs. The partial sums are reduced through local memory.
// Quantized rows are processed in units: a unit is a group of 16 (q4_K, q6_K: 16 units per block of 256) values
// handled by one thread.

#define MV_BEGIN                                                                              \
    local float sh[WG_MV];                                                                    \
    const int lid  = (int)get_local_id(0);                                                    \
    const int lane = lid & (tpr - 1);                                                         \
    const int r    = LGID() * (WG_MV >> tprl) + (lid >> tprl);                                \
    const bool valid = r < tc.ne0 * tc.ne1 * tc.ne2 * tc.ne3;                                 \
    global const uchar * arow = pa;                                                           \
    global const uchar * brow = pb;                                                           \
    int i0 = 0, i1 = 0, i2 = 0, i3 = 0;                                                       \
    if (valid) {                                                                              \
        SPLIT4(r, tc, i0, i1, i2, i3);                                                        \
        const int i02 = FDIV_RAW(i2, bc.m2, bc.l2);                                           \
        const int i03 = FDIV_RAW(i3, bc.m3, bc.l3);                                           \
        arow = pa + ta.off + i0 * ta.nb1 + i02 * ta.nb2 + i03 * ta.nb3;                       \
        brow = pb + tb.off + i1 * tb.nb1 + i2 * tb.nb2 + i3 * tb.nb3;                         \
    }                                                                                         \
    const int ne00 = ta.ne0;                                                                  \
    global const float * bf = (global const float *)brow;                                     \
    float acc = 0.0f;

#define MV_END                                                                                \
    sh[lid] = acc;                                                                            \
    barrier(CLK_LOCAL_MEM_FENCE);                                                             \
    for (int s = tpr >> 1; s > 0; s >>= 1) {                                                  \
        if (lane < s) {                                                                       \
            sh[lid] += sh[lid + s];                                                           \
        }                                                                                     \
        barrier(CLK_LOCAL_MEM_FENCE);                                                         \
    }                                                                                         \
    if (valid && lane == 0) {                                                                 \
        ST_F32(pc + tc.off + i0 * tc.nb0 + i1 * tc.nb1 + i2 * tc.nb2 + i3 * tc.nb3, sh[lid]); \
    }

#define MV_ARGS global uchar * pa, tdesc ta, global uchar * pb, tdesc tb, global uchar * pc, tdesc tc, bcast bc, int tpr, int tprl

// ---- F32 ----
// the loads of 4 chunks (of 4 elements per thread) are issued together: the rows are short (K cache) and the kernels are latency bound
kernel void k_mv_f32(MV_ARGS) {
    MV_BEGIN
    if (valid) {
        global const float * af = (global const float *)arow;
        int i = lane * 4;
        for (; i + 12 * tpr + 4 <= ne00; i += 16 * tpr) {
            const float4 a0 = vload4(0, af + i), a1 = vload4(0, af + i + 4 * tpr), a2 = vload4(0, af + i + 8 * tpr), a3 = vload4(0, af + i + 12 * tpr);
            const float4 x0 = vload4(0, bf + i), x1 = vload4(0, bf + i + 4 * tpr), x2 = vload4(0, bf + i + 8 * tpr), x3 = vload4(0, bf + i + 12 * tpr);
            acc += (dot(a0, x0) + dot(a1, x1)) + (dot(a2, x2) + dot(a3, x3));
        }
        for (; i < ne00; i += tpr * 4) {
            if (i + 4 <= ne00) {
                acc += dot(vload4(0, af + i), vload4(0, bf + i));
            } else {
                for (int j = i; j < ne00; ++j) {
                    acc += af[j] * bf[j];
                }
            }
        }
    }
    MV_END
}

// ---- F16 ----
kernel void k_mv_f16(MV_ARGS) {
    MV_BEGIN
    if (valid) {
        global const half * ah = (global const half *)arow;
        int i = lane * 4;
        for (; i + 12 * tpr + 4 <= ne00; i += 16 * tpr) {
            const float4 a0 = vload_half4(0, ah + i), a1 = vload_half4(0, ah + i + 4 * tpr), a2 = vload_half4(0, ah + i + 8 * tpr), a3 = vload_half4(0, ah + i + 12 * tpr);
            const float4 x0 = vload4(0, bf + i), x1 = vload4(0, bf + i + 4 * tpr), x2 = vload4(0, bf + i + 8 * tpr), x3 = vload4(0, bf + i + 12 * tpr);
            acc += (dot(a0, x0) + dot(a1, x1)) + (dot(a2, x2) + dot(a3, x3));
        }
        for (; i < ne00; i += tpr * 4) {
            if (i + 4 <= ne00) {
                acc += dot(vload_half4(0, ah + i), vload4(0, bf + i));
            } else {
                for (int j = i; j < ne00; ++j) {
                    acc += vload_half(j, ah) * bf[j];
                }
            }
        }
    }
    MV_END
}

float4 nib_lo(uint w) {
    return (float4)((float)(w & 0xF), (float)((w >> 8) & 0xF), (float)((w >> 16) & 0xF), (float)((w >> 24) & 0xF));
}
float4 nib_hi(uint w) {
    return (float4)((float)((w >> 4) & 0xF), (float)((w >> 12) & 0xF), (float)((w >> 20) & 0xF), (float)((w >> 28) & 0xF));
}

// ---- Q4_K: block = half d, half dmin, 12 bytes of 6-bit scales/mins, 128 bytes of nibbles ----
// unit s (0..15) of a block: chunk c = s / 4 (64 values), t = s % 4 -> qs bytes [8t, 8t+8) of the chunk
// give the 8 values c*64 + 8t + i (low nibbles) and c*64 + 32 + 8t + i (high nibbles).
kernel void k_mv_q4_k(MV_ARGS) {
    MV_BEGIN
    if (valid) {
        const int nunits = (ne00 >> 8) * 16;
        const float4 ones = (float4)(1.0f);
        for (int u = lane; u < nunits; u += tpr) {
            const int blk = u >> 4;
            const int s   = u & 15;
            const int c   = s >> 2;
            const int t   = s & 3;
            global const uchar * bp = arow + blk * 144;
            const float d    = vload_half(0, (global const half *)bp);
            const float dmin = vload_half(1, (global const half *)bp);
            global const uchar * sc = bp + 4;
            const float sc0 = d * (float)k4_sc(2 * c,     sc);
            const float sc1 = d * (float)k4_sc(2 * c + 1, sc);
            const float m0  = dmin * (float)k4_mn(2 * c,     sc);
            const float m1  = dmin * (float)k4_mn(2 * c + 1, sc);
            global const uint * qp = (global const uint *)(bp + 16 + c * 32 + t * 8);
            const uint w0 = qp[0];
            const uint w1 = qp[1];
            global const float * bx = bf + blk * 256 + c * 64 + t * 8;
            const float4 xl0 = vload4(0, bx);
            const float4 xl1 = vload4(1, bx);
            const float4 xh0 = vload4(0, bx + 32);
            const float4 xh1 = vload4(1, bx + 32);
            const float sl  = dot(nib_lo(w0), xl0) + dot(nib_lo(w1), xl1);
            const float sh_ = dot(nib_hi(w0), xh0) + dot(nib_hi(w1), xh1);
            const float xsl = dot(xl0, ones) + dot(xl1, ones);
            const float xsh = dot(xh0, ones) + dot(xh1, ones);
            acc += sc0 * sl - m0 * xsl + sc1 * sh_ - m1 * xsh;
        }
    }
    MV_END
}

// ---- Q6_K: block = 128 bytes ql (low 4 bits), 64 bytes qh (high 2 bits), 16 int8 scales, half d ----
// unit s (0..15) of a block: half h = s / 8 (128 values), l0 = 4 * (s % 8): it handles the values
// l0..l0+3 of the four quadrants (32 values each) of that half.
kernel void k_mv_q6_k(MV_ARGS) {
    MV_BEGIN
    if (valid) {
        const int nunits = (ne00 >> 8) * 16;
        for (int u = lane; u < nunits; u += tpr) {
            const int blk = u >> 4;
            const int s   = u & 15;
            const int h   = s >> 3;
            const int l0  = (s & 7) * 4;
            global const uchar * bp = arow + blk * 210;
            const int4 qa  = convert_int4(as_uchar4(vload2(0, (global const ushort *)(bp + h * 64 + l0))));
            const int4 qa2 = convert_int4(as_uchar4(vload2(0, (global const ushort *)(bp + h * 64 + 32 + l0))));
            const int4 qb  = convert_int4(as_uchar4(vload2(0, (global const ushort *)(bp + 128 + h * 32 + l0))));
            global const char * scp = (global const char *)(bp + 192 + h * 8 + (l0 >> 4));
            const float d = vload_half(0, (global const half *)(bp + 208));
            const int4 qv0 = ((qa  & 0xF) | ((qb & 3) << 4)) - 32;
            const int4 qv1 = ((qa2 & 0xF) | (((qb >> 2) & 3) << 4)) - 32;
            const int4 qv2 = ((qa  >> 4)  | (((qb >> 4) & 3) << 4)) - 32;
            const int4 qv3 = ((qa2 >> 4)  | (((qb >> 6) & 3) << 4)) - 32;
            global const float * bx = bf + blk * 256 + h * 128 + l0;
            const float s0 = (float)scp[0] * dot(convert_float4(qv0), vload4(0, bx));
            const float s1 = (float)scp[2] * dot(convert_float4(qv1), vload4(0, bx + 32));
            const float s2 = (float)scp[4] * dot(convert_float4(qv2), vload4(0, bx + 64));
            const float s3 = (float)scp[6] * dot(convert_float4(qv3), vload4(0, bx + 96));
            acc += d * (s0 + s1 + s2 + s3);
        }
    }
    MV_END
}

// =============================================================================================
// 4 weights per 32-bit word, converted without the (slow on Fermi) integer to float conversion instructions:
// a byte b (0..255) is the mantissa of the float 2^23 + b, the offset is subtracted afterwards (exact).
#ifdef CL11_NV_ASM
// NVIDIA PTX: prmt picks arbitrary bytes of two 32-bit registers: one instruction builds the float 2^23 + byte k of w
uint cl11_prmt(uint a, uint b, uint sel) {
    uint r;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
    return r;
}
float4 f4_mag(uint w) {
    return (float4)(as_float(cl11_prmt(w, 0x4B000000u, 0x7540u)), as_float(cl11_prmt(w, 0x4B000000u, 0x7541u)),
                    as_float(cl11_prmt(w, 0x4B000000u, 0x7542u)), as_float(cl11_prmt(w, 0x4B000000u, 0x7543u)));
}
#else
float4 f4_mag(uint w) {
    return (float4)(as_float(0x4B000000u | (w & 0xFFu)), as_float(0x4B000000u | ((w >> 8) & 0xFFu)),
                    as_float(0x4B000000u | ((w >> 16) & 0xFFu)), as_float(0x4B000000u | (w >> 24)));
}
#endif
// low / high nibbles of the 4 bytes (0..15)
float4 f4_lo4(uint w)   { return f4_mag(w & 0x0F0F0F0Fu) - (float4)(8388608.0f); }
float4 f4_hi4(uint w)   { return f4_mag((w >> 4) & 0x0F0F0F0Fu) - (float4)(8388608.0f); }
// 6-bit values - 32 (q6_K), the 4 bytes hold values 0..63
float4 f4_q6(uint w)    { return f4_mag(w) - (float4)(8388640.0f); }

// 6-bit scale / min of q4_K from the 3 words (12 bytes) of scales, j in [0, 8)
void k4_scale_min(uint s0, uint s1, uint s2, int j, float * sc, float * mn) {
    if (j < 4) {
        const int sh = 8 * j;
        *sc = (float)((s0 >> sh) & 63);
        *mn = (float)((s1 >> sh) & 63);
    } else {
        const int sh = 8 * (j - 4);
        const uint lo = s2 >> sh;
        *sc = (float)((lo & 0xF)        | (((s0 >> sh) >> 6 & 3) << 4));
        *mn = (float)(((lo >> 4) & 0xF) | (((s1 >> sh) >> 6 & 3) << 4));
    }
}


// ---------------------------------------------------------------------------------------------
// mat-vec with a transposed F16 src0: dst[i0] = sum_k A[k, i0] * x[k], where the elements along i0 are contiguous
// (ta.nb1 == 2) and the elements along k are strided (ta.nb0). It replaces the CONT of the transposed V cache followed by
// the MUL_MAT of the attention (the CONT is not needed).
// A work-group computes 32 consecutive outputs i0 (the work-items of a warp read consecutive halves of the rows of A:
// coalesced) with WG_ROW/32 slices of k per output, the slices are added through local memory. src1 (x) is F32.
// bc.tiles: number of groups of 32 outputs.
// ---------------------------------------------------------------------------------------------
kernel void k_mvt_f16(global uchar * pa, tdesc ta, global uchar * pb, tdesc tb, global uchar * pc, tdesc tc, bcast bc) {
    local float part[WG_ROW];
    const int lid = (int)get_local_id(0);
    const int o  = lid & 31;
    const int ks = lid >> 5;
    const int nks = WG_ROW >> 5;
    int g = LGID();
    if (g >= bc.tiles * tc.ne1 * tc.ne2 * tc.ne3) return; // uniform for the work-group
    int gq = FDIV_RAW(g, bc.mt, bc.lt);
    const int tile = g - gq * bc.tiles;
    g = gq;
    gq = FDIV(g, tc, 1);
    const int i1 = g - gq * tc.ne1;
    g = gq;
    const int i3 = FDIV(g, tc, 2);
    const int i2 = g - i3 * tc.ne2;
    const int i02 = FDIV_RAW(i2, bc.m2, bc.l2);
    const int i03 = FDIV_RAW(i3, bc.m3, bc.l3);

    const int i0 = tile * 32 + o;
    const bool valid = i0 < tc.ne0;
    global const uchar * a = pa + ta.off + (valid ? i0 : 0) * ta.nb1 + i02 * ta.nb2 + i03 * ta.nb3;
    global const uchar * x = pb + tb.off + i1 * tb.nb1 + i2 * tb.nb2 + i3 * tb.nb3;
    const int ne00 = ta.ne0;
    float acc0 = 0.0f, acc1 = 0.0f;
    int k = ks;
    if (valid) {
        for (; k + nks < ne00; k += 2 * nks) {
            acc0 += vload_half(0, (global const half *)(a + k * ta.nb0)) * LD_F32(x + k * tb.nb0);
            acc1 += vload_half(0, (global const half *)(a + (k + nks) * ta.nb0)) * LD_F32(x + (k + nks) * tb.nb0);
        }
        if (k < ne00) {
            acc0 += vload_half(0, (global const half *)(a + k * ta.nb0)) * LD_F32(x + k * tb.nb0);
        }
    }
    part[lid] = acc0 + acc1;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (ks == 0 && valid) {
        float sum = part[o];
        for (int j = 1; j < nks; ++j) {
            sum += part[j * 32 + o];
        }
        ST_F32(pc + tc.off + i0 * tc.nb0 + i1 * tc.nb1 + i2 * tc.nb2 + i3 * tc.nb3, sum);
    }
}
