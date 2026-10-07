// CL11 backend: RMS_NORM, SOFT_MAX and ROPE.

// ---------------------------------------------------------------------------------------------
// RMS_NORM (F32): one work-group per row. The elements of a row are contiguous (nb0 == 4).
// ---------------------------------------------------------------------------------------------
kernel void k_rms_norm(global uchar * ps, tdesc ts, global uchar * pd, tdesc td, float eps) {
    local float sh[WG_ROW];

    const int lid = (int)get_local_id(0);
    const int row = LGID();
    const int nrows = ts.ne1 * ts.ne2 * ts.ne3;
    if (row >= nrows) return; // uniform for the whole group

    int i1, i2, i3;
    { int q = FDIV(row, ts, 1); i1 = row - q * ts.ne1; i3 = FDIV(q, ts, 2); i2 = q - i3 * ts.ne2; }

    global const float * x = (global const float *)(ps + OFF4(ts, 0, i1, i2, i3));
    global       float * y = (global       float *)(pd + OFF4(td, 0, i1, i2, i3));

    float sum = 0.0f;
    for (int i = lid; i < ts.ne0; i += WG_ROW) {
        sum += x[i] * x[i];
    }
    sh[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = WG_ROW / 2; s > 0; s >>= 1) {
        if (lid < s) {
            sh[lid] += sh[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float scale = 1.0f / sqrt(sh[0] / (float)ts.ne0 + eps);
    for (int i = lid; i < ts.ne0; i += WG_ROW) {
        y[i] = x[i] * scale;
    }
}

// ---------------------------------------------------------------------------------------------
// SOFT_MAX (F32 in/out): dst = softmax(src0 * scale + mask), one work-group per row.
// The (optional) mask is F32 or F16, row i1 of the mask is used for row i1 of src0 and it is
// broadcast along dims 2 and 3. ALiBi (max_bias) and attention sinks are not supported.
// ---------------------------------------------------------------------------------------------
kernel void k_soft_max(global uchar * ps, tdesc ts, global uchar * pm, tdesc tm, int has_mask, int mask_f16,
                       global uchar * pd, tdesc td, float scale) {
    local float sh[WG_ROW];

    const int lid = (int)get_local_id(0);
    const int row = LGID();
    const int nrows = ts.ne1 * ts.ne2 * ts.ne3;
    if (row >= nrows) return; // uniform for the whole group

    int i1, i2, i3;
    { int q = FDIV(row, ts, 1); i1 = row - q * ts.ne1; i3 = FDIV(q, ts, 2); i2 = q - i3 * ts.ne2; }

    global const float * x = (global const float *)(ps + OFF4(ts, 0, i1, i2, i3));
    global       float * y = (global       float *)(pd + OFF4(td, 0, i1, i2, i3));
    global const uchar * m = has_mask ? (pm + OFF4(tm, 0, i1, FMOD(i2, tm, 2), FMOD(i3, tm, 3))) : pm;

    // max
    float mx = -INFINITY;
    for (int i = lid; i < ts.ne0; i += WG_ROW) {
        float v = x[i] * scale;
        if (has_mask) {
            v += mask_f16 ? vload_half(i, (global const half *)m) : ((global const float *)m)[i];
        }
        mx = fmax(mx, v);
    }
    sh[lid] = mx;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = WG_ROW / 2; s > 0; s >>= 1) {
        if (lid < s) {
            sh[lid] = fmax(sh[lid], sh[lid + s]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    mx = sh[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // exp and sum (the exponentials are stored in dst, dst may alias src0)
    float sum = 0.0f;
    for (int i = lid; i < ts.ne0; i += WG_ROW) {
        float v = x[i] * scale;
        if (has_mask) {
            v += mask_f16 ? vload_half(i, (global const half *)m) : ((global const float *)m)[i];
        }
        const float e = exp(v - mx);
        y[i] = e;
        sum += e;
    }
    sh[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = WG_ROW / 2; s > 0; s >>= 1) {
        if (lid < s) {
            sh[lid] += sh[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float inv = 1.0f / sh[0];
    for (int i = lid; i < ts.ne0; i += WG_ROW) {
        y[i] *= inv;
    }
}

// ---------------------------------------------------------------------------------------------
// ROPE (normal and NEOX, yarn, freq_factors, n_offs). Same arithmetic as the CPU backend.
// One work-item per pair of elements of a row: the first n_dims/2 items rotate a pair, the others
// copy a pair of elements outside of the rotated window [n_offs, n_offs + n_dims).
// ---------------------------------------------------------------------------------------------
float rope_ramp(float low, float high, int i0) {
    const float y = ((float)(i0 / 2) - low) / fmax(0.001f, high - low);
    return 1.0f - fmin(1.0f, fmax(0.0f, y));
}

#define DEF_ROPE(NAME, LD, ST, ESZ)                                                                         \
kernel void NAME(global uchar * ps, tdesc ts, global uchar * pp, tdesc tp, global uchar * pf, tdesc tf, int has_ff, \
                 global uchar * pd, tdesc td, int n_dims, int neox, int n_offs,                             \
                 float theta_scale, float freq_scale, float ext_factor, float attn_factor,                 \
                 float corr0, float corr1) {                                                               \
    const int hp  = td.ne0 / 2;                                                                            \
    const int row = LGID();                                                                                \
    if (row >= td.ne1 * td.ne2 * td.ne3) return;                                                           \
    int i1, i2, i3;                                                                                        \
    { int q = FDIV(row, td, 1); i1 = row - q * td.ne1; i3 = FDIV(q, td, 2); i2 = q - i3 * td.ne2; }        \
    global const uchar * x = ps + ts.off + i1 * ts.nb1 + i2 * ts.nb2 + i3 * ts.nb3;                        \
    global       uchar * y = pd + td.off + i1 * td.nb1 + i2 * td.nb2 + i3 * td.nb3;                        \
    const float pos = (float)(*(global const int *)(pp + tp.off + i2 * tp.nb0));                           \
    for (int j = (int)get_local_id(0); j < hp; j += WG_ROW) {                                              \
    if (j < n_dims / 2) {                                                                                  \
        const int p = j;                                                                                   \
        float theta = pos;                                                                                 \
        for (int k = 0; k < p; ++k) {                                                                      \
            theta *= theta_scale;                                                                          \
        }                                                                                                  \
        const float ff = has_ff ? *(global const float *)(pf + tf.off + p * tf.nb0) : 1.0f;                \
        const float theta_extrap = theta / ff;                                                             \
        const float theta_interp = freq_scale * theta_extrap;                                              \
        float th = theta_interp;                                                                           \
        float mscale = attn_factor;                                                                        \
        if (ext_factor != 0.0f) {                                                                          \
            const float ramp_mix = rope_ramp(corr0, corr1, 2 * p) * ext_factor;                            \
            th = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;                               \
            mscale *= 1.0f + 0.1f * log(1.0f / freq_scale);                                                \
        }                                                                                                  \
        const float cs = cos(th) * mscale;                                                                 \
        const float sn = sin(th) * mscale;                                                                 \
        const int e0 = neox ? (n_offs + p)              : (n_offs + 2 * p);                                \
        const int e1 = neox ? (n_offs + p + n_dims / 2) : (n_offs + 2 * p + 1);                            \
        const float x0 = LD(x + e0 * ESZ);                                                                 \
        const float x1 = LD(x + e1 * ESZ);                                                                 \
        ST(y + e0 * ESZ, x0 * cs - x1 * sn);                                                               \
        ST(y + e1 * ESZ, x0 * sn + x1 * cs);                                                               \
    } else {                                                                                               \
        int e = 2 * (j - n_dims / 2);                                                                      \
        if (e >= n_offs) e += n_dims;                                                                      \
        ST(y + e * ESZ,       LD(x + e * ESZ));                                                            \
        ST(y + (e + 1) * ESZ, LD(x + (e + 1) * ESZ));                                                      \
    }                                                                                                      \
    }                                                                                                      \
}

DEF_ROPE(k_rope_f32, LD_F32, ST_F32, 4)
DEF_ROPE(k_rope_f16, LD_F16, ST_F16, 2)
