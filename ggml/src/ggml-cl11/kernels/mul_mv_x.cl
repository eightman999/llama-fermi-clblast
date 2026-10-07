// CL11 backend: the mat-vec kernels of mul_mv_xcommon.cl, instantiated by the macros of mvx_defs_staged.cl / mvx_defs_global.cl

// ---- F32: a lane handles the float4 groups lane, lane + 32, ... of a chunk of 1024 elements ----
kernel void KN(f32)(MVS_ARGS) {
    MVX_BEGIN(1024)
    for (int k0 = 0; k0 < ne00; k0 += 1024) {
        XSTAGE(1024, k0)
        XBAR
        for (int t = 0; t < 8; ++t) {
            const int gi = lane + 32 * t;
            if (k0 + 4 * gi < ne00) {
                const float4 x = LDX(gi);
                for (int rr = 0; rr < NRW; ++rr) {
                    acc[rr] += dot(((global const float4 *)rowp[rr])[(k0 >> 2) + gi], x);
                }
            }
        }
        XBAR
    }
    MVS_END
}

// ---- F16 ----
kernel void KN(f16)(MVS_ARGS) {
    MVX_BEGIN(1024)
    for (int k0 = 0; k0 < ne00; k0 += 1024) {
        XSTAGE(1024, k0)
        XBAR
        for (int t = 0; t < 8; ++t) {
            const int gi = lane + 32 * t;
            if (k0 + 4 * gi < ne00) {
                const float4 x = LDX(gi);
                for (int rr = 0; rr < NRW; ++rr) {
                    acc[rr] += dot(vload_half4(0, (global const half *)rowp[rr] + k0 + 4 * gi), x);
                }
            }
        }
        XBAR
    }
    MVS_END
}

// ---- Q4_K: unit = 64 weights (blk, c): 32 bytes of nibbles, 144-byte blocks are always 16-byte aligned: 128-bit loads ----
// a chunk of 2048 elements = 8 blocks = 32 units, one per lane
kernel void KN(q4_k)(MVS_ARGS) {
    MVX_BEGIN(2048)
    const float4 ones = (float4)(1.0f);
    for (int k0 = 0; k0 < ne00; k0 += 2048) {
        XSTAGE(2048, k0)
        XBAR
        if (k0 + (lane >> 2) * 256 < ne00) {
            const int bk = lane >> 2;
            const int c  = lane & 3;
            const int xi = lane * 16;
            const int boff = ((k0 >> 8) + bk) * 144;
            uint4 hd[NRW];     // d, dmin, scales
            uint4 qa[NRW];     // quants 0..15 of the unit
            uint4 qb[NRW];     // quants 16..31
            for (int rr = 0; rr < NRW; ++rr) {
                global const uint4 * bp = (global const uint4 *)(rowp[rr] + boff);
                hd[rr] = bp[0];
                qa[rr] = bp[1 + 2 * c];
                qb[rr] = bp[2 + 2 * c];
            }
            float ql[NRW], qh[NRW];
            float sxl, sxh;
            {
                const float4 x0 = LDX(xi), x1 = LDX(xi + 1), x2 = LDX(xi + 2), x3 = LDX(xi + 3), x4 = LDX(xi + 4), x5 = LDX(xi + 5), x6 = LDX(xi + 6), x7 = LDX(xi + 7);
                sxl = dot((x0 + x1) + (x2 + x3) + (x4 + x5) + (x6 + x7), ones);
                for (int rr = 0; rr < NRW; ++rr) {
                    ql[rr] = (dot(f4_lo4(qa[rr].x), x0) + dot(f4_lo4(qa[rr].y), x1)) + (dot(f4_lo4(qa[rr].z), x2) + dot(f4_lo4(qa[rr].w), x3))
                           + (dot(f4_lo4(qb[rr].x), x4) + dot(f4_lo4(qb[rr].y), x5)) + (dot(f4_lo4(qb[rr].z), x6) + dot(f4_lo4(qb[rr].w), x7));
                }
            }
            {
                const float4 x0 = LDX(xi + 8), x1 = LDX(xi + 9), x2 = LDX(xi + 10), x3 = LDX(xi + 11), x4 = LDX(xi + 12), x5 = LDX(xi + 13), x6 = LDX(xi + 14), x7 = LDX(xi + 15);
                sxh = dot((x0 + x1) + (x2 + x3) + (x4 + x5) + (x6 + x7), ones);
                for (int rr = 0; rr < NRW; ++rr) {
                    qh[rr] = (dot(f4_hi4(qa[rr].x), x0) + dot(f4_hi4(qa[rr].y), x1)) + (dot(f4_hi4(qa[rr].z), x2) + dot(f4_hi4(qa[rr].w), x3))
                           + (dot(f4_hi4(qb[rr].x), x4) + dot(f4_hi4(qb[rr].y), x5)) + (dot(f4_hi4(qb[rr].z), x6) + dot(f4_hi4(qb[rr].w), x7));
                }
            }
            for (int rr = 0; rr < NRW; ++rr) {
                const float d    = vload_half(0, (global const half *)(rowp[rr] + boff));
                const float dmin = vload_half(1, (global const half *)(rowp[rr] + boff));
                float sc0, m0, sc1, m1;
                k4_scale_min(hd[rr].y, hd[rr].z, hd[rr].w, 2 * c,     &sc0, &m0);
                k4_scale_min(hd[rr].y, hd[rr].z, hd[rr].w, 2 * c + 1, &sc1, &m1);
                acc[rr] += d * (sc0 * ql[rr] + sc1 * qh[rr]) - dmin * (m0 * sxl + m1 * sxh);
            }
        }
        XBAR
    }
    MVS_END
}

// ---- Q6_K: unit = 16 weights as in the generic kernel; chunk = 4 blocks = 64 units, a lane handles 2 units ----
kernel void KN(q6_k)(MVS_ARGS) {
    MVX_BEGIN(1024)
    for (int k0 = 0; k0 < ne00; k0 += 1024) {
        XSTAGE(1024, k0)
        XBAR
        for (int t = 0; t < 2; ++t) {
            const int u   = lane + 32 * t;                 // unit in the chunk
            const int bkc = u >> 4;                        // block in the chunk
            if (k0 + bkc * 256 < ne00) {
                const int s16 = u & 15;
                const int h   = s16 >> 3;
                const int l0  = (s16 & 7) * 4;
                const int xi  = bkc * 64 + h * 32 + (l0 >> 2);
                const float4 x0 = LDX(xi), x1 = LDX(xi + 8), x2 = LDX(xi + 16), x3 = LDX(xi + 24);
                const int boff = ((k0 >> 8) + bkc) * 210;
                for (int rr = 0; rr < NRW; ++rr) {
                    global const uchar * bp = rowp[rr] + boff;
                    const uint qa  = ld32_u16x2(bp + h * 64 + l0);
                    const uint qa2 = ld32_u16x2(bp + h * 64 + 32 + l0);
                    const uint qb  = ld32_u16x2(bp + 128 + h * 32 + l0);
                    global const char * scp = (global const char *)(bp + 192 + h * 8 + (l0 >> 4));
                    const float d = vload_half(0, (global const half *)(bp + 208));
                    const uint v0 = (qa & 0x0F0F0F0Fu) | ((qb & 0x03030303u) << 4);
                    const uint v1 = (qa2 & 0x0F0F0F0Fu) | (((qb >> 2) & 0x03030303u) << 4);
                    const uint v2 = ((qa >> 4) & 0x0F0F0F0Fu) | (((qb >> 4) & 0x03030303u) << 4);
                    const uint v3 = ((qa2 >> 4) & 0x0F0F0F0Fu) | (((qb >> 6) & 0x03030303u) << 4);
                    acc[rr] += d * ((float)scp[0] * dot(f4_q6(v0), x0) + (float)scp[2] * dot(f4_q6(v1), x1)
                                  + (float)scp[4] * dot(f4_q6(v2), x2) + (float)scp[6] * dot(f4_q6(v3), x3));
                }
            }
        }
        XBAR
    }
    MVS_END
}
