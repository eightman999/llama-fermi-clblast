// mode of mul_mv_x.cl: src1 read through the caches (k_mvw_*)
#undef KN
#undef XS_DECL
#undef XSTAGE
#undef XBAR
#undef LDX
#define KN(n) k_mvw_##n
#define XS_DECL(XCHV)
#define XSTAGE(XCHV, k0)
#define XBAR
#define LDX(i) xrow[((k0) >> 2) + (i)]
