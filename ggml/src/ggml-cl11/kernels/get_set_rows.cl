// CL11 backend: GET_ROWS and SET_ROWS.

// ---------------------------------------------------------------------------------------------
// GET_ROWS: dst[:, i10, i11, i12] = dequantize(src0[:, idx, i11, i12]), idx = src1[i10, i11, i12] (I32).
// One work-group per row of dst, dst is F32.
// ---------------------------------------------------------------------------------------------
#define DEF_GET_ROWS(NAME, EL)                                                                         \
kernel void NAME(global uchar * p0, tdesc t0, global uchar * p1, tdesc t1, global uchar * pd, tdesc td) { \
    const int lid = (int)get_local_id(0);                                                              \
    const int row = LGID();                                                                            \
    if (row >= t1.ne0 * t1.ne1 * t1.ne2) return;                                                       \
    int i10, i11, i12, i13;                                                                            \
    SPLIT4(row, t1, i10, i11, i12, i13);                                                               \
    const int idx = *(global const int *)(p1 + t1.off + i10 * t1.nb0 + i11 * t1.nb1 + i12 * t1.nb2);   \
    global const uchar * r = p0 + t0.off + i11 * t0.nb2 + i12 * t0.nb3;                                \
    global float * y = (global float *)(pd + td.off + i10 * td.nb1 + i11 * td.nb2 + i12 * td.nb3);     \
    for (int k = lid; k < t0.ne0; k += WG_ROW) {                                                       \
        y[k] = EL(r, idx, k);                                                                          \
    }                                                                                                  \
}

// r: base of the slice (i11, i12) of src0, idx: row, k: element
#define GR_F32(r, idx, k)   dq_f32((r) + (idx) * t0.nb1, k)
#define GR_F16(r, idx, k)   dq_f16((r) + (idx) * t0.nb1, k)
#define GR_Q4_K(r, idx, k)  dq_q4_k((r) + (idx) * t0.nb1, k)
#define GR_Q6_K(r, idx, k)  dq_q6_k((r) + (idx) * t0.nb1, k)
#define GR_Q4_0(r, idx, k)  dqr_q4_0(r, t0.ne1, t0.ne0 >> 5, idx, k)
#define GR_Q8_0(r, idx, k)  dqr_q8_0(r, t0.ne1, t0.ne0 >> 5, idx, k)

DEF_GET_ROWS(k_get_rows_f32,  GR_F32)
DEF_GET_ROWS(k_get_rows_f16,  GR_F16)
DEF_GET_ROWS(k_get_rows_q4_0, GR_Q4_0)
DEF_GET_ROWS(k_get_rows_q8_0, GR_Q8_0)
DEF_GET_ROWS(k_get_rows_q4_k, GR_Q4_K)
DEF_GET_ROWS(k_get_rows_q6_k, GR_Q6_K)

// ---------------------------------------------------------------------------------------------
// SET_ROWS: dst[:, idx, i02, i03] = src0[:, i01, i02, i03] with idx = src1[i01, i02 % ne11, i03 % ne12]
// (I64 when idx64 != 0, I32 otherwise). One work-item per element of src0.
// ---------------------------------------------------------------------------------------------
#define DEF_SET_ROWS(NAME, LD, ST)                                                                      \
kernel void NAME(global uchar * p0, tdesc t0, global uchar * p1, tdesc t1, global uchar * pd, tdesc td, \
                 int n, int idx64) {                                                                    \
    const int g = LGID() * WG_EW + (int)get_local_id(0);                                                \
    if (g >= n) return;                                                                                 \
    int i00, i01, i02, i03;                                                                             \
    SPLIT4(g, t0, i00, i01, i02, i03);                                                                  \
    global const uchar * pi = p1 + t1.off + i01 * t1.nb0 + FMOD(i02, t1, 1) * t1.nb1 + FMOD(i03, t1, 2) * t1.nb2; \
    const int row = idx64 ? (int)(*(global const long *)pi) : *(global const int *)pi;                  \
    const float v = LD(p0 + OFF4(t0, i00, i01, i02, i03));                                              \
    ST(pd + OFF4(td, i00, row, i02, i03), v);                                                           \
}

DEF_SET_ROWS(k_set_rows_f32_f32, LD_F32, ST_F32)
DEF_SET_ROWS(k_set_rows_f32_f16, LD_F32, ST_F16)
DEF_SET_ROWS(k_set_rows_f16_f32, LD_F16, ST_F32)
DEF_SET_ROWS(k_set_rows_f16_f16, LD_F16, ST_F16)
