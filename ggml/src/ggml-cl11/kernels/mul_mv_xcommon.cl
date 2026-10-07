// CL11 backend: shared definitions of the "x" mat-vec kernels (mul_mv_x.cl is included twice, see mvx_defs_*.cl).
//
// Mat-vec kernels for rows with 2-byte alignment (4 for q4_K) and a 16-byte aligned src1, WG_MV a multiple of 32.
// Like the generic kernels (every lane handles a small unit of a block, consecutive lanes handle consecutive units, so the
// reads of the rows are contiguous), but a warp (32 lanes) computes NRW rows, so that src1 is loaded once for the NRW
// rows. Two flavors:
//   k_mvx_*: the chunk of src1 (XCH floats) is copied to local memory once per work-group and shared by NWARP*NRW rows,
//   k_mvw_*: no local memory, the warps are independent and read src1 through the caches.

#ifndef NRW
#define NRW 2
#endif
#define NWARP   (WG_MV / 32)
#define ROWS_WG (NWARP * NRW)

#define MVS_ARGS global uchar * pa, tdesc ta, global uchar * pb, tdesc tb, global uchar * pc, tdesc tc, bcast bc

// prologue: XS_DECL declares the local copy of src1 (k_mvx only); XCHV: elements of src1 per chunk
#define MVX_BEGIN(XCHV)                                                                               \
    XS_DECL(XCHV)                                                                                     \
    local float  red[NRW * WG_MV];                                                                    \
    const int lid  = (int)get_local_id(0);                                                            \
    const int lane = lid & 31;                                                                        \
    const int warp = lid >> 5;                                                                        \
    int g = LGID();                                                                                   \
    if (g >= bc.tiles * tc.ne1 * tc.ne2 * tc.ne3) return; /* uniform for the work-group */            \
    int gq = FDIV_RAW(g, bc.mt, bc.lt);                                                               \
    const int tile = g - gq * bc.tiles;                                                               \
    g = gq;                                                                                           \
    gq = FDIV(g, tc, 1);                                                                              \
    const int i1 = g - gq * tc.ne1;                                                                   \
    g = gq;                                                                                           \
    const int i3 = FDIV(g, tc, 2);                                                                    \
    const int i2 = g - i3 * tc.ne2;                                                                   \
    const int i02 = FDIV_RAW(i2, bc.m2, bc.l2);                                                       \
    const int i03 = FDIV_RAW(i3, bc.m3, bc.l3);                                                       \
    global const uchar  * abase = pa + ta.off + i02 * ta.nb2 + i03 * ta.nb3;                          \
    global const float4 * xrow  = (global const float4 *)(pb + tb.off + i1 * tb.nb1 + i2 * tb.nb2 + i3 * tb.nb3); \
    const int ne00 = ta.ne0;                                                                          \
    int  i0r[NRW];                                                                                    \
    bool vr[NRW];                                                                                     \
    global const uchar * rowp[NRW];                                                                   \
    float acc[NRW];                                                                                   \
    for (int rr = 0; rr < NRW; ++rr) {                                                                \
        i0r[rr]  = tile * ROWS_WG + warp * NRW + rr;                                                  \
        vr[rr]   = i0r[rr] < tc.ne0;                                                                  \
        rowp[rr] = abase + (vr[rr] ? i0r[rr] : 0) * ta.nb1;                                           \
        acc[rr]  = 0.0f;                                                                              \
    }

// reduction of the NRW accumulators over the 32 lanes of the warps and write of the results
#define MVS_END                                                                                       \
    for (int rr = 0; rr < NRW; ++rr) {                                                                \
        red[rr * WG_MV + lid] = acc[rr];                                                              \
    }                                                                                                 \
    barrier(CLK_LOCAL_MEM_FENCE);                                                                     \
    for (int s = 16; s > 0; s >>= 1) {                                                                \
        if (lane < s) {                                                                               \
            for (int rr = 0; rr < NRW; ++rr) {                                                        \
                red[rr * WG_MV + lid] += red[rr * WG_MV + lid + s];                                   \
            }                                                                                         \
        }                                                                                             \
        barrier(CLK_LOCAL_MEM_FENCE);                                                                 \
    }                                                                                                 \
    for (int rr = 0; rr < NRW; ++rr) {                                                                \
        if (lane == 0 && vr[rr]) {                                                                    \
            ST_F32(pc + tc.off + i0r[rr] * tc.nb0 + i1 * tc.nb1 + i2 * tc.nb2 + i3 * tc.nb3, red[rr * WG_MV + lid]); \
        }                                                                                             \
    }

// copy the chunk at element k0 of src1 to local memory (one float4 of padding every 16 float4)
#define MVX_STAGE(XCHV, k0)                                                                           \
    for (int i = lid; i < (XCHV) / 4; i += WG_MV) {                                                   \
        const int gi = ((k0) >> 2) + i;                                                               \
        xs[i + (i >> 4)] = ((gi << 2) < ne00) ? xrow[gi] : (float4)(0.0f);                            \
    }

// 4 bytes of a row at an even byte offset, from two 16-bit loads
uint ld32_u16x2(global const uchar * p) {
    return (uint)(*(global const ushort *)p) | ((uint)(*(global const ushort *)(p + 2)) << 16);
}
