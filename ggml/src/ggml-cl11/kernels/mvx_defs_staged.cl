// mode of mul_mv_x.cl: src1 in local memory (k_mvx_*)
#undef KN
#undef XS_DECL
#undef XSTAGE
#undef XBAR
#undef LDX
#define KN(n) k_mvx_##n
#define XS_DECL(XCHV) local float4 xs[(XCHV) / 4 + (XCHV) / 64];
#define XSTAGE(XCHV, k0) MVX_STAGE(XCHV, k0)
#define XBAR barrier(CLK_LOCAL_MEM_FENCE);
#define LDX(i) xs[(i) + ((i) >> 4)]
