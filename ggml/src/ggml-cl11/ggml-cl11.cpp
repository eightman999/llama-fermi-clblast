// CL11: ggml backend that only needs OpenCL 1.1 (host API and OpenCL C 1.1).
//
// Target: old GPUs such as the NVIDIA GeForce GT 430 (Fermi, "OpenCL 1.1 CUDA"):
//   - no sub-groups, no cl_khr_fp16, no doubles in the kernels,
//   - clCreateBuffer is limited to CL_DEVICE_MAX_MEM_ALLOC_SIZE (1/4 of the memory),
//   - PCIe transfers are slow: the whole graph is enqueued on one in-order queue and the host only
//     synchronizes in synchronize() and get_tensor().
//
// Only the OpenCL 1.1 host API is used (CL_TARGET_OPENCL_VERSION 110): no clEnqueueFillBuffer,
// no clEnqueueMarkerWithWaitList, no clCreateProgramWithBuiltInKernels, no clCreateImage, no SVM.

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 110
#endif

#ifdef __APPLE__
#  ifndef CL_SILENCE_DEPRECATION
#    define CL_SILENCE_DEPRECATION
#  endif
#  include <OpenCL/opencl.h>
#else
#  include <CL/cl.h>
#endif

#include "ggml-cl11.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-cl11-kernels.h" // generated: ggml_cl11_sources[] (the .cl files)

#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <mutex>
#include <string>
#include <vector>

#define CL11_NUM_SOURCES (sizeof(ggml_cl11_sources) / sizeof(ggml_cl11_sources[0]))

// ggml-alloc asks for the base address of a buffer: the tensors of a CL11 buffer are addressed as
// offsets from this dummy (non-null, page aligned) pointer.
#define CL11_BASE ((void *) 0x1000)

// largest cl_mem (and buffer) size: the kernels address a buffer with 32-bit signed byte offsets
#define CL11_MAX_BUFFER_SIZE ((size_t) 0x7ffff000)

// the tensors have to be aligned for the vector loads of the kernels (and for CL11_BASE)
#define CL11_MIN_ALIGNMENT 256

//
// device information (queried without creating a context)
//

struct cl11_device_info {
    cl_platform_id platform = nullptr;
    cl_device_id   device   = nullptr;

    std::string name;
    std::string platform_name;
    std::string device_version; // "OpenCL 1.1 CUDA"
    std::string c_version;      // "OpenCL C 1.1 "

    size_t max_alloc   = 0; // CL_DEVICE_MAX_MEM_ALLOC_SIZE
    size_t global_mem  = 0; // CL_DEVICE_GLOBAL_MEM_SIZE
    size_t local_mem   = 0; // CL_DEVICE_LOCAL_MEM_SIZE
    size_t max_wg      = 0; // CL_DEVICE_MAX_WORK_GROUP_SIZE
    size_t max_item0   = 0; // CL_DEVICE_MAX_WORK_ITEM_SIZES[0]
    size_t align_bytes = 0; // CL_DEVICE_MEM_BASE_ADDR_ALIGN in bytes

    // NVIDIA only (cl_nv_device_attribute_query), 0 when unknown
    int nv_cc_major = 0;
    int nv_cc_minor = 0;
};

// work-group configuration, adapted to the limits of the device and of the compiled kernels
struct cl11_config {
    int wg_ew  = 128; // element-wise kernels
    int wg_row = 128; // rms_norm, soft_max, get_rows
    int wg_mv  = 128; // mat-vec
    int gt     = 16;  // mat-mat: gt*gt work-items, (4*gt)^2 dst tile
    int mvs_nrw = 2;  // x mat-vec: rows per warp
    int wg_mvr  = 64;  // mat-vec of the repacked Q4_0 / Q8_0 weights: work-items per group (32 * warps; the warps split the blocks of the rows)
    int mvr_nrr = 2;   // ... groups of 32 rows per lane (rows per lane)
    int mm_bn   = 8;   // mat-mat: the dst tile is (4 * gt) rows of src0 x (mm_bn * gt) rows of src1 (4 or 8), (gt / 2) x (mm_bn * gt / 4) work-items
};

// Per-GPU tuning. Everything that depends on the architecture is chosen here (and only here), from the compute
// capability (NVIDIA: cl_nv_device_attribute_query) or the device name; the environment variables of the docs override
// the values. Only the GeForce GT 430 (Fermi, compute capability 2.x) has been measured, the other architectures
// get the same values until they are tuned.
struct cl11_tuning {
    cl11_config cfg;                 // work-group sizes, rows per warp / lane (reduced later to what the compiled kernels accept)
    int         mvs_min_k   = 256;   // shortest row that uses the k_mvx / k_mvw mat-vec kernels (K-quants, F16, F32)
    int         mvs_family  = 0;     // 0: automatic (k_mvx for Q4_K, k_mvw for the others), 1: k_mvx, 2: k_mvw
    bool        mvr_const_x = false; // k_mvrc_*: src1 of a single column through a constant-memory sub-buffer (no load / store unit)
    const char * name       = "default";
};


enum cl11_kernel_class {
    CL11_CLASS_EW,
    CL11_CLASS_ROW,
    CL11_CLASS_MV,
    CL11_CLASS_MM,
    CL11_CLASS_MVR,
};

// the 6 types of src0 of GET_ROWS / MUL_MAT
enum cl11_qtype {
    CL11_QT_F32,
    CL11_QT_F16,
    CL11_QT_Q4_0,
    CL11_QT_Q8_0,
    CL11_QT_Q4_K,
    CL11_QT_Q6_K,
    CL11_QT_COUNT,
};

enum cl11_kernel_id {
    K_ADD, K_SUB, K_MUL, K_DIV,
    K_UNARY,
    K_GLU,
    K_CPY_F32_F32, K_CPY_F32_F16, K_CPY_F16_F32, K_CPY_F16_F16,
    K_CPY_T_F32, K_CPY_T_F16,
    K_FILL_U32, K_FILL_U8,
    K_RMS_NORM,
    K_SOFT_MAX,
    K_ROPE_F32, K_ROPE_F16,
    K_GET_ROWS,                                   // + cl11_qtype
    K_SET_ROWS = K_GET_ROWS + CL11_QT_COUNT,      // + 2*src_f16 + dst_f16
    K_MV       = K_SET_ROWS + 4,                  // + cl11_qtype
    K_MM       = K_MV + CL11_QT_COUNT,            // + cl11_qtype
    K_MVX      = K_MM + CL11_QT_COUNT,            // + cl11_qtype: mat-vec, src1 in local memory
    K_MVW      = K_MVX + CL11_QT_COUNT,           // + cl11_qtype: mat-vec, src1 through the caches
    K_MVT_F16  = K_MVW + CL11_QT_COUNT,           // mat-vec with a transposed F16 src0
    K_MVR_Q4_0,                                   // mat-vec with the repacked Q4_0 / Q8_0 weights
    K_MVR_Q8_0,
    K_MVRC_Q4_0,                                  // ... with src1 (one column) in constant memory
    K_MVRC_Q8_0,
    K_COUNT,
};

struct cl11_kernel_desc {
    const char * name;
    cl11_kernel_class cls;
};

// must be in the order of enum cl11_kernel_id
static const cl11_kernel_desc cl11_kernel_table[] = {
    { "k_add",              CL11_CLASS_EW  },
    { "k_sub",              CL11_CLASS_EW  },
    { "k_mul",              CL11_CLASS_EW  },
    { "k_div",              CL11_CLASS_EW  },
    { "k_unary",            CL11_CLASS_EW  },
    { "k_glu",              CL11_CLASS_EW  },
    { "k_cpy_f32_f32",      CL11_CLASS_EW  },
    { "k_cpy_f32_f16",      CL11_CLASS_EW  },
    { "k_cpy_f16_f32",      CL11_CLASS_EW  },
    { "k_cpy_f16_f16",      CL11_CLASS_EW  },
    { "k_cpy_t_f32_f32",    CL11_CLASS_MM  },
    { "k_cpy_t_f16_f16",    CL11_CLASS_MM  },
    { "k_fill_u32",         CL11_CLASS_EW  },
    { "k_fill_u8",          CL11_CLASS_EW  },
    { "k_rms_norm",         CL11_CLASS_ROW },
    { "k_soft_max",         CL11_CLASS_ROW },
    { "k_rope_f32",         CL11_CLASS_ROW },
    { "k_rope_f16",         CL11_CLASS_ROW },
    // K_GET_ROWS + cl11_qtype
    { "k_get_rows_f32",     CL11_CLASS_ROW },
    { "k_get_rows_f16",     CL11_CLASS_ROW },
    { "k_get_rows_q4_0",    CL11_CLASS_ROW },
    { "k_get_rows_q8_0",    CL11_CLASS_ROW },
    { "k_get_rows_q4_k",    CL11_CLASS_ROW },
    { "k_get_rows_q6_k",    CL11_CLASS_ROW },
    // K_SET_ROWS + 2*(src is F16) + (dst is F16)
    { "k_set_rows_f32_f32", CL11_CLASS_EW  },
    { "k_set_rows_f32_f16", CL11_CLASS_EW  },
    { "k_set_rows_f16_f32", CL11_CLASS_EW  },
    { "k_set_rows_f16_f16", CL11_CLASS_EW  },
    // K_MV + cl11_qtype
    { "k_mv_f32",           CL11_CLASS_MV  },
    { "k_mv_f16",           CL11_CLASS_MV  },
    { nullptr,             CL11_CLASS_MV  }, // repacked type: k_mvr_*
    { nullptr,             CL11_CLASS_MV  }, // repacked type: k_mvr_*
    { "k_mv_q4_k",          CL11_CLASS_MV  },
    { "k_mv_q6_k",          CL11_CLASS_MV  },
    // K_MM + cl11_qtype
    { "k_mm_f32",           CL11_CLASS_MM  },
    { "k_mm_f16",           CL11_CLASS_MM  },
    { "k_mm_q4_0",          CL11_CLASS_MM  },
    { "k_mm_q8_0",          CL11_CLASS_MM  },
    { "k_mm_q4_k",          CL11_CLASS_MM  },
    { "k_mm_q6_k",          CL11_CLASS_MM  },
    // K_MVX + cl11_qtype
    { "k_mvx_f32",          CL11_CLASS_MV  },
    { "k_mvx_f16",          CL11_CLASS_MV  },
    { nullptr,             CL11_CLASS_MV  }, // repacked type: k_mvr_*
    { nullptr,             CL11_CLASS_MV  }, // repacked type: k_mvr_*
    { "k_mvx_q4_k",         CL11_CLASS_MV  },
    { "k_mvx_q6_k",         CL11_CLASS_MV  },
    // K_MVW + cl11_qtype
    { "k_mvw_f32",          CL11_CLASS_MV  },
    { "k_mvw_f16",          CL11_CLASS_MV  },
    { nullptr,             CL11_CLASS_MV  }, // repacked type: k_mvr_*
    { nullptr,             CL11_CLASS_MV  }, // repacked type: k_mvr_*
    { "k_mvw_q4_k",         CL11_CLASS_MV  },
    { "k_mvw_q6_k",         CL11_CLASS_MV  },
    { "k_mvt_f16",          CL11_CLASS_ROW },
    { "k_mvr_q4_0",         CL11_CLASS_MVR },
    { "k_mvr_q8_0",         CL11_CLASS_MVR },
    { "k_mvrc_q4_0",        CL11_CLASS_MVR },
    { "k_mvrc_q8_0",        CL11_CLASS_MVR },
};
static_assert(sizeof(cl11_kernel_table) / sizeof(cl11_kernel_table[0]) == K_COUNT, "cl11_kernel_table does not match cl11_kernel_id");

// tensor descriptor passed to the kernels by value, must match `tdesc` in kernels/common.cl
struct cl11_td {
    int32_t  ne[4];
    int32_t  nb[4];
    int32_t  off;
    uint32_t fm[4]; // fast division by ne[i]
    uint32_t fl[4];
};
static_assert(sizeof(cl11_td) == 17 * sizeof(int32_t), "cl11_td layout");

// magic numbers m, l such that n / d == (mulhi(n, m) + n) >> l for 0 <= n < 2^31
static void cl11_fastdiv_init(uint32_t d, uint32_t & m, uint32_t & l) {
    m = 0;
    l = 0;
    if (d == 0) {
        return;
    }
    while (l < 32 && ((uint64_t) 1 << l) < d) {
        ++l;
    }
    m = (uint32_t) ((((uint64_t) 1 << 32) * (((uint64_t) 1 << l) - d)) / d + 1);
}

// must match `bcast` in kernels/common.cl
struct cl11_bcast {
    int32_t  r2, r3;
    uint32_t m2, l2, m3, l3;
    int32_t  tiles;
    uint32_t mt, lt;
    int32_t  tiles2;
    uint32_t mt2, lt2;
};
static_assert(sizeof(cl11_bcast) == 12 * sizeof(int32_t), "cl11_bcast layout");

static cl11_bcast cl11_make_bcast(int64_t r2, int64_t r3, int64_t tiles, int64_t tiles2) {
    cl11_bcast b = {};
    b.r2 = (int32_t) r2;
    b.r3 = (int32_t) r3;
    cl11_fastdiv_init((uint32_t) r2, b.m2, b.l2);
    cl11_fastdiv_init((uint32_t) r3, b.m3, b.l3);
    b.tiles = (int32_t) tiles;
    cl11_fastdiv_init((uint32_t) tiles, b.mt, b.lt);
    b.tiles2 = (int32_t) tiles2;
    cl11_fastdiv_init((uint32_t) tiles2, b.mt2, b.lt2);
    return b;
}

struct cl11_device_context {
    cl11_device_info info;
    int              index = 0;
    std::string      name; // "CL11_<index>"

    // protects the lazily created state and the kernel objects (clSetKernelArg is not thread safe)
    std::mutex mutex;

    // created on first use
    bool             context_ready = false;
    bool             program_ready = false;
    bool             failed        = false;
    cl_context       context = nullptr;
    cl_command_queue queue   = nullptr;
    cl_program       program = nullptr;
    cl_kernel        kernels[K_COUNT] = {};
    cl11_config      cfg;

    // mat-vec is used for src1 with at most this many columns (GGML_CL11_MV_MAX_COLS)
    int mv_max_cols = 1;
    // tuning / debugging (environment variables)
    int  mv_tpr     = 0;      // GGML_CL11_MV_TPR: threads per row of the generic mat-vec kernels (0 = automatic)
    bool mvs_enabled = true;  // GGML_CL11_MV_STAGED=0: use the generic mat-vec kernels only
    int  mvs_min_k   = 256;   // minimum row length for the x mat-vec kernels
    bool use_nv_asm  = false; // NVIDIA: inline PTX in the kernels (GGML_CL11_NO_ASM=1 disables it)
    bool no_cpy_t    = false; // GGML_CL11_NO_CPY_T=1: no transposing copy kernel
    bool no_mvt      = false; // GGML_CL11_NO_MVT=1: do not fuse CONT + MUL_MAT
    int  mvs_family  = 0;     // 0: automatic, 1: k_mvx (src1 in local memory), 2: k_mvw (src1 through the caches)
    bool mvr_const_x = false; // GGML_CL11_MVR_CONST_X=0/1
    size_t max_const = 0;     // CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE

    // sub-buffers of src1 for the constant memory (key: parent, offset, size); released with the parent buffer
    struct const_sub {
        cl_mem   parent;
        size_t   offset;
        size_t   size;
        cl_mem   sub;
    };
    std::vector<const_sub> const_subs;

    // GGML_CL11_PROFILE=1: time every kernel with OpenCL events, the summary is printed when the backend is freed
    struct prof_stat {
        std::string name;
        double      ns    = 0;
        size_t      count = 0;
        double      bytes = 0; // bytes of the weights (MUL_MAT)
    };
    struct prof_pending {
        cl_event event;
        int      key;
    };
    bool                      profile = false;
    int                       prof_key = -1; // key of the node that is being enqueued
    std::vector<prof_stat>    prof_stats;
    std::vector<prof_pending> prof_pending_events;

    ggml_backend_buffer_type buft = {};
    ggml_backend_device      dev  = {};
};

static std::vector<std::unique_ptr<cl11_device_context>> g_cl11_devices;

static bool cl11_debug() {
    static const bool debug = getenv("GGML_CL11_DEBUG") != nullptr;
    return debug;
}

static const char * cl11_err_str(cl_int err) {
    switch (err) {
        case CL_SUCCESS:                         return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND:                return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE:            return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE:          return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:   return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES:                return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY:              return "CL_OUT_OF_HOST_MEMORY";
        case CL_BUILD_PROGRAM_FAILURE:           return "CL_BUILD_PROGRAM_FAILURE";
        case CL_INVALID_VALUE:                   return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE:                  return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT:                 return "CL_INVALID_CONTEXT";
        case CL_INVALID_COMMAND_QUEUE:           return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_MEM_OBJECT:              return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_KERNEL_ARGS:             return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_GROUP_SIZE:         return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE:          return "CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_WORK_SIZE:        return "CL_INVALID_GLOBAL_WORK_SIZE";
        case CL_INVALID_BUFFER_SIZE:             return "CL_INVALID_BUFFER_SIZE";
        case CL_INVALID_OPERATION:               return "CL_INVALID_OPERATION";
        case CL_INVALID_ARG_SIZE:                return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_ARG_VALUE:               return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_INDEX:               return "CL_INVALID_ARG_INDEX";
        default:                                 return "unknown OpenCL error";
    }
}

#define CL11_CHECK(expr)                                                                   \
    do {                                                                                   \
        const cl_int _err = (expr);                                                        \
        if (_err != CL_SUCCESS) {                                                          \
            GGML_LOG_ERROR("%s: %s failed: %s (%d)\n", __func__, #expr, cl11_err_str(_err), (int) _err); \
            return false;                                                                  \
        }                                                                                  \
    } while (0)

static int cl11_pow2_floor(size_t x) {
    int p = 1;
    while ((size_t) p * 2 <= x) {
        p *= 2;
    }
    return p;
}

//
// device enumeration
//

static std::string cl11_device_str(cl_device_id dev, cl_device_info param) {
    size_t size = 0;
    if (clGetDeviceInfo(dev, param, 0, nullptr, &size) != CL_SUCCESS || size == 0) {
        return "";
    }
    std::string s(size, '\0');
    if (clGetDeviceInfo(dev, param, size, &s[0], nullptr) != CL_SUCCESS) {
        return "";
    }
    while (!s.empty() && s.back() == '\0') {
        s.pop_back();
    }
    return s;
}

template <typename T>
static T cl11_device_val(cl_device_id dev, cl_device_info param, T def = T()) {
    T v = def;
    if (clGetDeviceInfo(dev, param, sizeof(T), &v, nullptr) != CL_SUCCESS) {
        return def;
    }
    return v;
}

static std::string cl11_lower(std::string s) {
    for (auto & c : s) {
        c = (char) std::tolower((unsigned char) c);
    }
    return s;
}

static cl11_tuning cl11_tuning_for(const cl11_device_info & info) {
    cl11_tuning t;
    const std::string lname = cl11_lower(info.name);
    const bool fermi = info.nv_cc_major == 2 || (info.nv_cc_major == 0 && lname.find("gt 430") != std::string::npos);
    if (fermi) {
        // GeForce GT 430 (GF108: 2 SMs, 96 cores, 1.4 GHz, ~22 GB/s, 32768 registers per SM)
        t.name = "Fermi (sm_2x)";
        t.cfg.wg_mv    = 128;
        t.cfg.mvs_nrw  = 2;
        t.cfg.wg_mvr   = 64;   // 2 warps share the rows of a work-group (tg of Qwen3-0.6B Q8_0 / navspec Q4_0: 128: 20.4 / 28.6, 64: 21.2 / 29.0, 256: 18.3 / 25.2)
        t.cfg.mm_bn    = 8;    // 64 x 128 tiles: src0 is read once for the 128 columns of a prompt batch (fewer re-reads of src1 than 64 x 64)
        t.cfg.mvr_nrr  = 2;    // 2 rows per lane: halves the loads of src1 (4 rows per lane: fewer warps, slower)
        t.mvs_min_k    = 256;
        t.mvs_family   = 0;
        t.mvr_const_x  = true;   // Fermi: +15% (Q8_0) .. +30% (Q4_0), the x loads of 2 rows compete with the weights for the load units
    }
    // Kepler (GT 710 / GT 730, sm_3x) and everything else: not tuned yet, same values
    return t;
}

// GGML_CL11_DEVICES: comma separated list of substrings of the device names (case insensitive)
static bool cl11_device_selected(const std::string & name) {
    const char * env = getenv("GGML_CL11_DEVICES");
    if (env == nullptr || *env == '\0') {
        return true;
    }
    const std::string lname = cl11_lower(name);
    std::string item;
    std::string list = std::string(env) + ",";
    for (char c : list) {
        if (c == ',') {
            // trim
            size_t b = item.find_first_not_of(" \t");
            size_t e = item.find_last_not_of(" \t");
            if (b != std::string::npos) {
                const std::string pat = cl11_lower(item.substr(b, e - b + 1));
                if (lname.find(pat) != std::string::npos) {
                    return true;
                }
            }
            item.clear();
        } else {
            item += c;
        }
    }
    return false;
}

static void cl11_probe_devices() {
    cl_uint n_platforms = 0;
    cl_int err = clGetPlatformIDs(0, nullptr, &n_platforms);
    if (err != CL_SUCCESS || n_platforms == 0) {
        GGML_LOG_DEBUG("%s: no OpenCL platform (%s)\n", __func__, cl11_err_str(err));
        return;
    }
    std::vector<cl_platform_id> platforms(n_platforms);
    if (clGetPlatformIDs(n_platforms, platforms.data(), nullptr) != CL_SUCCESS) {
        return;
    }

    for (cl_platform_id platform : platforms) {
        cl_uint n_devices = 0;
        if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &n_devices) != CL_SUCCESS || n_devices == 0) {
            continue;
        }
        std::vector<cl_device_id> devices(n_devices);
        if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, n_devices, devices.data(), nullptr) != CL_SUCCESS) {
            continue;
        }

        char pname[256] = {};
        clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(pname) - 1, pname, nullptr);

        for (cl_device_id device : devices) {
            cl11_device_info info;
            info.platform      = platform;
            info.device        = device;
            info.name          = cl11_device_str(device, CL_DEVICE_NAME);
            info.platform_name = pname;
            info.device_version = cl11_device_str(device, CL_DEVICE_VERSION);
            info.c_version     = cl11_device_str(device, CL_DEVICE_OPENCL_C_VERSION);
            info.max_alloc     = (size_t) cl11_device_val<cl_ulong>(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE);
            info.global_mem    = (size_t) cl11_device_val<cl_ulong>(device, CL_DEVICE_GLOBAL_MEM_SIZE);
            info.local_mem     = (size_t) cl11_device_val<cl_ulong>(device, CL_DEVICE_LOCAL_MEM_SIZE);
            info.max_wg        = cl11_device_val<size_t>(device, CL_DEVICE_MAX_WORK_GROUP_SIZE);
            info.align_bytes   = cl11_device_val<cl_uint>(device, CL_DEVICE_MEM_BASE_ADDR_ALIGN) / 8;
            if (cl11_device_str(device, CL_DEVICE_EXTENSIONS).find("cl_nv_device_attribute_query") != std::string::npos) {
                // CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV / MINOR_NV
                info.nv_cc_major = (int) cl11_device_val<cl_uint>(device, (cl_device_info) 0x4000);
                info.nv_cc_minor = (int) cl11_device_val<cl_uint>(device, (cl_device_info) 0x4001);
            }

            if (const char * env = getenv("GGML_CL11_FAKE_MAX_ALLOC")) {
                // debugging: pretend that CL_DEVICE_MAX_MEM_ALLOC_SIZE is this many bytes (e.g. 252706816 for a GT 430)
                info.max_alloc = std::min(info.max_alloc, (size_t) strtoull(env, nullptr, 10));
            }

            size_t item_sizes[3] = {0, 0, 0};
            clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(item_sizes), item_sizes, nullptr);
            info.max_item0 = item_sizes[0];

            const bool available = cl11_device_val<cl_bool>(device, CL_DEVICE_AVAILABLE, CL_FALSE);
            const bool compiler  = cl11_device_val<cl_bool>(device, CL_DEVICE_COMPILER_AVAILABLE, CL_FALSE);
            const bool little    = cl11_device_val<cl_bool>(device, CL_DEVICE_ENDIAN_LITTLE, CL_FALSE);

            // the smallest configuration needs 64 work-items per group and 16 KiB of local memory
            if (!available || !compiler || !little || info.local_mem < 16 * 1024 ||
                info.max_wg < 64 || info.max_item0 < 64 || info.max_alloc < 1024 * 1024) {
                GGML_LOG_DEBUG("%s: skipping device '%s' (available=%d compiler=%d little=%d local_mem=%zu max_wg=%zu max_alloc=%zu)\n",
                        __func__, info.name.c_str(), available, compiler, little, info.local_mem, info.max_wg, info.max_alloc);
                continue;
            }
            if (!cl11_device_selected(info.name)) {
                GGML_LOG_DEBUG("%s: device '%s' not selected by GGML_CL11_DEVICES\n", __func__, info.name.c_str());
                continue;
            }

            auto ctx = std::make_unique<cl11_device_context>();
            ctx->info  = info;
            ctx->index = (int) g_cl11_devices.size();
            ctx->name  = "CL11_" + std::to_string(ctx->index);
            g_cl11_devices.push_back(std::move(ctx));
        }
    }
}

//
// lazy context / program creation
//

static bool cl11_build_program(cl11_device_context * c, const cl11_config & cfg, std::string & log) {
    char opts[512];
    snprintf(opts, sizeof(opts), "-cl-std=CL1.1 -DWG_EW=%d -DWG_ROW=%d -DWG_MV=%d -DGT=%d -DNRW=%d -DWG_MVR=%d -DNRR=%d -DMMBN=%d%s %s",
            cfg.wg_ew, cfg.wg_row, cfg.wg_mv, cfg.gt, cfg.mvs_nrw, cfg.wg_mvr, cfg.mvr_nrr, cfg.mm_bn, c->use_nv_asm ? " -DCL11_NV_ASM" : "", getenv("GGML_CL11_BUILD_OPTS") ? getenv("GGML_CL11_BUILD_OPTS") : "");

    // GGML_CL11_KERNEL_DIR: read the .cl files from this directory instead of the embedded copies (development)
    std::vector<std::string> file_sources;
    std::vector<const char *> sources(ggml_cl11_sources, ggml_cl11_sources + CL11_NUM_SOURCES);
    if (const char * dir = getenv("GGML_CL11_KERNEL_DIR")) {
        file_sources.resize(CL11_NUM_SOURCES);
        for (size_t i = 0; i < CL11_NUM_SOURCES; ++i) {
            const std::string path = std::string(dir) + "/" + ggml_cl11_source_names[i];
            std::ifstream f(path, std::ios::binary);
            if (!f) {
                GGML_LOG_ERROR("%s: cannot read %s\n", __func__, path.c_str());
                return false;
            }
            std::stringstream ss;
            ss << f.rdbuf();
            file_sources[i] = ss.str();
            sources[i] = file_sources[i].c_str();
        }
        GGML_LOG_INFO("%s: using the kernel sources of %s\n", __func__, dir);
    }

    cl_int err = CL_SUCCESS;
    c->program = clCreateProgramWithSource(c->context, (cl_uint) CL11_NUM_SOURCES, sources.data(), nullptr, &err);
    if (err != CL_SUCCESS) {
        GGML_LOG_ERROR("%s: clCreateProgramWithSource failed: %s\n", __func__, cl11_err_str(err));
        return false;
    }

    err = clBuildProgram(c->program, 1, &c->info.device, opts, nullptr, nullptr);

    size_t log_size = 0;
    clGetProgramBuildInfo(c->program, c->info.device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
    log.assign(log_size, '\0');
    if (log_size > 0) {
        clGetProgramBuildInfo(c->program, c->info.device, CL_PROGRAM_BUILD_LOG, log_size, &log[0], nullptr);
    }

    if (err != CL_SUCCESS) {
        GGML_LOG_ERROR("%s: failed to build the kernels for %s (%s) with options '%s': %s\n%s\n", __func__,
                c->name.c_str(), c->info.name.c_str(), opts, cl11_err_str(err), log.c_str());
        clReleaseProgram(c->program);
        c->program = nullptr;
        return false;
    }
    if (const char * path = getenv("GGML_CL11_DUMP_BINARY")) {
        // development: write the program binary (PTX for NVIDIA) to this file
        size_t bin_size = 0;
        if (clGetProgramInfo(c->program, CL_PROGRAM_BINARY_SIZES, sizeof(bin_size), &bin_size, nullptr) == CL_SUCCESS && bin_size > 0) {
            std::vector<unsigned char> bin(bin_size);
            unsigned char * ptr = bin.data();
            if (clGetProgramInfo(c->program, CL_PROGRAM_BINARIES, sizeof(ptr), &ptr, nullptr) == CL_SUCCESS) {
                std::ofstream f(path, std::ios::binary);
                f.write((const char *) bin.data(), (std::streamsize) bin_size);
            }
        }
    }
    if (cl11_debug() && log.size() > 1) {
        GGML_LOG_INFO("%s: build log for %s:\n%s\n", __func__, c->name.c_str(), log.c_str());
    }
    return true;
}

static void cl11_release_kernels(cl11_device_context * c) {
    for (int i = 0; i < K_COUNT; ++i) {
        if (c->kernels[i]) {
            clReleaseKernel(c->kernels[i]);
            c->kernels[i] = nullptr;
        }
    }
    if (c->program) {
        clReleaseProgram(c->program);
        c->program = nullptr;
    }
}

// creates cl_context and the command queue, the caller holds c->mutex
static bool cl11_ensure_context(cl11_device_context * c) {
    if (c->context_ready) {
        return true;
    }
    if (c->failed) {
        return false;
    }

    cl_int err = CL_SUCCESS;
    const cl_context_properties props[] = { CL_CONTEXT_PLATFORM, (cl_context_properties) c->info.platform, 0 };
    c->context = clCreateContext(props, 1, &c->info.device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        GGML_LOG_ERROR("%s: clCreateContext failed for %s: %s\n", __func__, c->name.c_str(), cl11_err_str(err));
        c->failed = true;
        return false;
    }
    c->profile = getenv("GGML_CL11_PROFILE") != nullptr;
    c->queue = clCreateCommandQueue(c->context, c->info.device, c->profile ? CL_QUEUE_PROFILING_ENABLE : 0, &err);
    if (err != CL_SUCCESS) {
        GGML_LOG_ERROR("%s: clCreateCommandQueue failed for %s: %s\n", __func__, c->name.c_str(), cl11_err_str(err));
        clReleaseContext(c->context);
        c->context = nullptr;
        c->failed = true;
        return false;
    }

    if (const char * env = getenv("GGML_CL11_MV_MAX_COLS")) {
        c->mv_max_cols = std::max(0, atoi(env));
    }
    if (const char * env = getenv("GGML_CL11_MV_TPR")) {
        c->mv_tpr = std::max(0, atoi(env));
    }
    if (const char * env = getenv("GGML_CL11_MV_STAGED")) {
        c->mvs_enabled = atoi(env) != 0;
    }
    c->no_cpy_t = getenv("GGML_CL11_NO_CPY_T") != nullptr;
    c->no_mvt   = getenv("GGML_CL11_NO_MVT") != nullptr;
    {
        char vendor[256] = {};
        clGetPlatformInfo(c->info.platform, CL_PLATFORM_VENDOR, sizeof(vendor) - 1, vendor, nullptr);
        c->use_nv_asm = strstr(vendor, "NVIDIA") != nullptr && getenv("GGML_CL11_NO_ASM") == nullptr;
    }
    {
        const cl11_tuning t = cl11_tuning_for(c->info);
        c->mvs_min_k  = t.mvs_min_k;
        c->mvs_family = t.mvs_family;
        c->mvr_const_x = t.mvr_const_x;
        c->max_const  = (size_t) cl11_device_val<cl_ulong>(c->info.device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE);
    }
    if (const char * env = getenv("GGML_CL11_MVR_CONST_X")) {
        c->mvr_const_x = atoi(env) != 0;
    }
    if (const char * env = getenv("GGML_CL11_MVS_FAMILY")) {
        c->mvs_family = atoi(env);
    }
    if (const char * env = getenv("GGML_CL11_MVS_MIN_K")) {
        c->mvs_min_k = std::max(0, atoi(env));
    }

    c->context_ready = true;
    return true;
}

// builds the program and creates the kernels, the caller holds c->mutex.
// The work-group sizes are not hard coded: they start from what the device allows and are reduced
// (and the program is rebuilt) until every kernel can be launched with them.
static bool cl11_ensure_program(cl11_device_context * c) {
    if (c->program_ready) {
        return true;
    }
    if (!cl11_ensure_context(c)) {
        return false;
    }

    const cl11_tuning tuning = cl11_tuning_for(c->info);
    cl11_config cfg = tuning.cfg;
    {
        size_t limit_wg = std::min(c->info.max_wg, c->info.max_item0);
        if (const char * env = getenv("GGML_CL11_MAX_WG")) {
            // debugging / tuning: cap the work-group size
            limit_wg = std::min<size_t>(limit_wg, (size_t) std::max(1, atoi(env)));
        }
        const int cap = cl11_pow2_floor(limit_wg);
        cfg.wg_ew  = std::min(cfg.wg_ew,  cap);
        cfg.wg_row = std::min(cfg.wg_row, cap);
        if (const char * env = getenv("GGML_CL11_MVS_NRW")) {
            cfg.mvs_nrw = std::max(1, std::min(8, atoi(env)));
        }
        if (const char * env = getenv("GGML_CL11_WG_MV")) {
            cfg.wg_mv = cl11_pow2_floor((size_t) std::max(1, atoi(env)));
        }
        cfg.wg_mv  = std::min(cfg.wg_mv,  cap);
        if (const char * env = getenv("GGML_CL11_WG_MVR")) {
            cfg.wg_mvr = cl11_pow2_floor((size_t) std::max(1, atoi(env)));
        }
        if (const char * env = getenv("GGML_CL11_MVR_NRR")) {
            cfg.mvr_nrr = std::max(1, std::min(8, atoi(env)));
        }
        cfg.wg_mvr = std::min(cfg.wg_mvr, cap);
        if (const char * env = getenv("GGML_CL11_MM_BN")) {
            cfg.mm_bn = atoi(env) >= 8 ? 8 : 4;
        }
        while (cfg.gt * cfg.gt > cap && cfg.gt > 4) {
            cfg.gt /= 2;
        }
    }

    for (int attempt = 0; attempt < 8; ++attempt) {
        std::string log;
        if (!cl11_build_program(c, cfg, log)) {
            c->failed = true;
            return false;
        }

        bool ok = true;
        for (int i = 0; i < K_COUNT && ok; ++i) {
            if (cl11_kernel_table[i].name == nullptr) {
                continue;
            }
            cl_int err = CL_SUCCESS;
            c->kernels[i] = clCreateKernel(c->program, cl11_kernel_table[i].name, &err);
            if (err != CL_SUCCESS) {
                GGML_LOG_ERROR("%s: clCreateKernel(%s) failed: %s\n", __func__, cl11_kernel_table[i].name, cl11_err_str(err));
                cl11_release_kernels(c);
                c->failed = true;
                return false;
            }
        }

        // the work-group size limit of each kernel depends on its register usage
        int limit[5] = { INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX };
        for (int i = 0; i < K_COUNT; ++i) {
            if (c->kernels[i] == nullptr) {
                continue;
            }
            size_t wgs = 0;
            if (clGetKernelWorkGroupInfo(c->kernels[i], c->info.device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(wgs), &wgs, nullptr) != CL_SUCCESS || wgs == 0) {
                wgs = 1;
            }
            const int cls = cl11_kernel_table[i].cls;
            limit[cls] = std::min(limit[cls], (int) std::min<size_t>(wgs, INT_MAX));
            if (cl11_debug()) {
                GGML_LOG_INFO("%s: kernel %-20s max work-group size %zu\n", __func__, cl11_kernel_table[i].name, wgs);
            }
        }

        cl11_config next = cfg;
        if (limit[CL11_CLASS_EW]  < cfg.wg_ew)  { next.wg_ew  = cl11_pow2_floor(limit[CL11_CLASS_EW]);  }
        if (limit[CL11_CLASS_ROW] < cfg.wg_row) { next.wg_row = cl11_pow2_floor(limit[CL11_CLASS_ROW]); }
        if (limit[CL11_CLASS_MV]  < cfg.wg_mv)  { next.wg_mv  = cl11_pow2_floor(limit[CL11_CLASS_MV]);  }
        if (limit[CL11_CLASS_MVR] < cfg.wg_mvr) { next.wg_mvr = cl11_pow2_floor(limit[CL11_CLASS_MVR]); }
        if (limit[CL11_CLASS_MM]  < cfg.gt * cfg.gt) {
            while (next.gt > 4 && next.gt * next.gt > limit[CL11_CLASS_MM]) {
                next.gt /= 2;
            }
        }

        if (next.wg_ew == cfg.wg_ew && next.wg_row == cfg.wg_row && next.wg_mv == cfg.wg_mv && next.wg_mvr == cfg.wg_mvr && next.gt == cfg.gt) {
            if (limit[CL11_CLASS_EW] < cfg.wg_ew || limit[CL11_CLASS_ROW] < cfg.wg_row ||
                limit[CL11_CLASS_MV] < cfg.wg_mv || limit[CL11_CLASS_MVR] < cfg.wg_mvr || limit[CL11_CLASS_MM]  < cfg.gt * cfg.gt) {
                GGML_LOG_ERROR("%s: the kernels cannot be launched on %s (work-group size limits %d/%d/%d/%d)\n", __func__,
                        c->name.c_str(), limit[0], limit[1], limit[2], limit[3]);
                cl11_release_kernels(c);
                c->failed = true;
                return false;
            }
            // all kernels accept the configuration
            c->cfg = cfg;
            c->program_ready = true;
            GGML_LOG_INFO("%s: %s: %s, %s, kernels built (work-group sizes: element-wise %d, row %d, mat-vec %d, mat-mat %dx%d)\n",
                    __func__, c->name.c_str(), c->info.name.c_str(), c->info.device_version.c_str(),
                    cfg.wg_ew, cfg.wg_row, cfg.wg_mv, cfg.gt, cfg.gt);
            return true;
        }

        GGML_LOG_DEBUG("%s: rebuilding the kernels with smaller work-groups\n", __func__);
        cl11_release_kernels(c);
        cfg = next;
    }

    GGML_LOG_ERROR("%s: could not find a working work-group configuration for %s\n", __func__, c->name.c_str());
    c->failed = true;
    return false;
}

//
// kernel launch helpers
//

struct cl11_args {
    cl_kernel kernel;
    cl_uint   idx = 0;
    bool      ok  = true;

    explicit cl11_args(cl_kernel k) : kernel(k) {}

    template <typename T>
    cl11_args & set(const T & v) {
        const cl_int err = clSetKernelArg(kernel, idx++, sizeof(T), &v);
        if (err != CL_SUCCESS) {
            GGML_LOG_ERROR("%s: clSetKernelArg(%u) failed: %s\n", __func__, idx - 1, cl11_err_str(err));
            ok = false;
        }
        return *this;
    }
};

// collect the timings of the finished kernels (the queue has been finished by the caller)
static void cl11_prof_collect(cl11_device_context * c) {
    for (const auto & p : c->prof_pending_events) {
        cl_ulong t0 = 0, t1 = 0;
        clGetEventProfilingInfo(p.event, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr);
        clGetEventProfilingInfo(p.event, CL_PROFILING_COMMAND_END,   sizeof(t1), &t1, nullptr);
        if (p.key >= 0 && (size_t) p.key < c->prof_stats.size()) {
            c->prof_stats[p.key].ns += (double) (t1 - t0);
        }
        clReleaseEvent(p.event);
    }
    c->prof_pending_events.clear();
}

static int cl11_prof_key(cl11_device_context * c, const std::string & name) {
    for (size_t i = 0; i < c->prof_stats.size(); ++i) {
        if (c->prof_stats[i].name == name) {
            return (int) i;
        }
    }
    c->prof_stats.emplace_back();
    c->prof_stats.back().name = name;
    return (int) c->prof_stats.size() - 1;
}

static void cl11_prof_print(cl11_device_context * c) {
    if (c->prof_stats.empty()) {
        return;
    }
    std::vector<const cl11_device_context::prof_stat *> v;
    double total = 0;
    for (const auto & s : c->prof_stats) {
        v.push_back(&s);
        total += s.ns;
    }
    std::sort(v.begin(), v.end(), [](auto * a, auto * b) { return a->ns > b->ns; });
    fprintf(stderr, "cl11 profile of %s: total kernel time %.3f ms\n", c->name.c_str(), total / 1e6);
    for (auto * s : v) {
        char gbs[64] = "";
        if (s->bytes > 0 && s->ns > 0) {
            snprintf(gbs, sizeof(gbs), "  weights %.2f GB/s", s->bytes / s->ns);
        }
        fprintf(stderr, "  %-44s %6zu calls %10.3f ms %8.1f us/call %5.1f%%%s\n", s->name.c_str(), s->count,
                s->ns / 1e6, s->ns / 1e3 / std::max<size_t>(1, s->count), 100.0 * s->ns / std::max(1.0, total), gbs);
    }
    c->prof_stats.clear();
}

// enqueue `n_groups` work-groups of `local` work-items. The launch is 2-D so that no dimension
// exceeds 32768 groups; the kernels linearize the group index themselves (LGID()).
static bool cl11_launch(cl11_device_context * c, const cl11_args & args, size_t local, size_t n_groups) {
    if (!args.ok) {
        return false;
    }
    if (n_groups == 0) {
        return true;
    }
    const size_t gx = std::min<size_t>(n_groups, 32768);
    const size_t gy = (n_groups + gx - 1) / gx;
    const size_t global_size[2] = { gx * local, gy };
    const size_t local_size[2]  = { local, 1 };
    cl_event event = nullptr;
    const cl_int err = clEnqueueNDRangeKernel(c->queue, args.kernel, 2, nullptr, global_size, local_size, 0, nullptr, c->profile ? &event : nullptr);
    if (event != nullptr) {
        c->prof_pending_events.push_back({ event, c->prof_key });
    }
    if (err != CL_SUCCESS) {
        GGML_LOG_ERROR("%s: clEnqueueNDRangeKernel failed: %s (groups %zu x %zu, local %zu)\n", __func__, cl11_err_str(err), gx, gy, local);
        return false;
    }
    return true;
}

static size_t cl11_div_up(size_t a, size_t b) {
    return (a + b - 1) / b;
}

//
// buffers
//

struct cl11_buffer_context {
    cl11_device_context * dctx = nullptr;
    cl_mem                mem  = nullptr;
    size_t                size = 0; // allocated size
};

static ggml_backend_buffer_t cl11_tensor_buffer(const ggml_tensor * t) {
    return t->view_src ? t->view_src->buffer : t->buffer;
}

// byte offset of the tensor data inside of its cl_mem
static size_t cl11_tensor_offset(const ggml_tensor * t) {
    return (size_t) ((const char *) t->data - (const char *) CL11_BASE);
}

struct cl11_tref {
    cl_mem  mem;
    cl11_td td;
};

static cl11_tref cl11_ref(const ggml_tensor * t) {
    cl11_tref r;
    auto * bctx = (cl11_buffer_context *) cl11_tensor_buffer(t)->context;
    r.mem = bctx->mem;
    for (int i = 0; i < 4; ++i) {
        r.td.ne[i] = (int32_t) t->ne[i];
        r.td.nb[i] = (int32_t) t->nb[i];
    }
    r.td.off = (int32_t) cl11_tensor_offset(t);
    for (int i = 0; i < 4; ++i) {
        cl11_fastdiv_init((uint32_t) t->ne[i], r.td.fm[i], r.td.fl[i]);
    }
    return r;
}

//
// Repacked layout "R32" of the Q4_0 / Q8_0 tensors
//
// The blocks of Q8_0 (34 bytes) and Q4_0 (18 bytes) are only 2-byte aligned: the natural layout needs 16-bit loads and
// a lot of address arithmetic on a Fermi GPU. set_tensor therefore stores these tensors (when cl11_repacked() is true,
// that is a property of the tensor alone, so supports_op / the kernels / get_tensor agree) in a layout made for coalesced
// 128-bit loads, get_tensor converts back. The size is the same as ggml_nbytes(). For every slice of the tensor
// (ne2 * ne3 slices of M = ne1 rows, nb = ne0 / 32 blocks per row), the rows are grouped by 32 (rs = number of rows of the
// group, 32 except for the last group), QB = 32 (Q8_0) or 16 (Q4_0) bytes of quants per block:
//   quants, at g * 32 * nb * QB, 16-byte unit u of the group:
//     Q8_0: u = (b * 2 + h) * rs + lane   bytes 16 * h .. 16 * h + 15 of the quants of block b of the row 32 * g + lane, xor 0x80
//     Q4_0: u = b * rs + lane             the 16 bytes of the nibbles of block b of the row
//   scales (half), at M * nb * QB + 2 * (g * 32 * nb + b * rs + lane)
// (see kernels/mul_mv_r.cl, dqr_q8_0 / dqr_q4_0 in kernels/common.cl)
//

static bool cl11_rp_type(ggml_type type) {
    return type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q8_0;
}

static bool cl11_repacked(const ggml_tensor * t) {
    if (t == nullptr || !cl11_rp_type(t->type) || t->view_src != nullptr || !ggml_is_contiguous(t)) {
        return false;
    }
    // the slices must keep the 16-byte alignment of the 128-bit loads
    const int64_t nb = t->ne[0] / 32;
    return t->ne[2] * t->ne[3] == 1 || (nb * t->ne[1]) % 8 == 0;
}

// a view of a repacked tensor cannot be addressed (the kernels and the transfers know the layout of the whole tensor only)
static void cl11_rp_check_view(const ggml_tensor * t) {
    if (t->view_src != nullptr && cl11_repacked(t->view_src)) {
        GGML_ABORT("CL11: the tensor %s is a view of a repacked Q4_0 / Q8_0 tensor (%s), which is not supported\n", t->name, t->view_src->name);
    }
}

struct cl11_rp_geom {
    bool    q8;
    int     nb;      // blocks per row
    int     bs;      // bytes of a block in the natural layout
    int     qb;      // bytes of quants per block
    int64_t M;       // rows per slice
    size_t  rowb;    // bytes of a row in the natural layout
};

static cl11_rp_geom cl11_rp_geometry(const ggml_tensor * t) {
    cl11_rp_geom g;
    g.q8   = t->type == GGML_TYPE_Q8_0;
    g.nb   = (int) (t->ne[0] / 32);
    g.bs   = g.q8 ? 34 : 18;
    g.qb   = g.q8 ? 32 : 16;
    g.M    = t->ne[1];
    g.rowb = (size_t) g.nb * g.bs;
    return g;
}

// natural -> repacked: `nrows` consecutive rows (a multiple of 32, except at the end of a slice) -> quants q, scales d
static void cl11_rp_pack(const cl11_rp_geom & G, const uint8_t * nat, int64_t nrows, uint8_t * q, uint8_t * d) {
    for (int64_t g0 = 0; g0 < nrows; g0 += 32) {
        const int rs = (int) std::min<int64_t>(32, nrows - g0);
        uint8_t * qg = q + (size_t) (g0 / 32) * 32 * G.nb * G.qb;
        uint8_t * dg = d + (size_t) (g0 / 32) * 32 * G.nb * 2;
        for (int lane = 0; lane < rs; ++lane) {
            const uint8_t * row = nat + (size_t) (g0 + lane) * G.rowb;
            for (int b = 0; b < G.nb; ++b) {
                const uint8_t * p = row + (size_t) b * G.bs;
                memcpy(dg + ((size_t) b * rs + lane) * 2, p, 2);
                if (G.q8) {
                    for (int h = 0; h < 2; ++h) {
                        uint8_t * o = qg + (((size_t) b * 2 + h) * rs + lane) * 16;
                        const uint8_t * s = p + 2 + 16 * h;
                        for (int i = 0; i < 16; ++i) {
                            o[i] = s[i] ^ 0x80;
                        }
                    }
                } else {
                    memcpy(qg + ((size_t) b * rs + lane) * 16, p + 2, 16);
                }
            }
        }
    }
}

// repacked -> natural
static void cl11_rp_unpack(const cl11_rp_geom & G, const uint8_t * q, const uint8_t * d, int64_t nrows, uint8_t * nat) {
    for (int64_t g0 = 0; g0 < nrows; g0 += 32) {
        const int rs = (int) std::min<int64_t>(32, nrows - g0);
        const uint8_t * qg = q + (size_t) (g0 / 32) * 32 * G.nb * G.qb;
        const uint8_t * dg = d + (size_t) (g0 / 32) * 32 * G.nb * 2;
        for (int lane = 0; lane < rs; ++lane) {
            uint8_t * row = nat + (size_t) (g0 + lane) * G.rowb;
            for (int b = 0; b < G.nb; ++b) {
                uint8_t * p = row + (size_t) b * G.bs;
                memcpy(p, dg + ((size_t) b * rs + lane) * 2, 2);
                if (G.q8) {
                    for (int h = 0; h < 2; ++h) {
                        const uint8_t * s = qg + (((size_t) b * 2 + h) * rs + lane) * 16;
                        uint8_t * o = p + 2 + 16 * h;
                        for (int i = 0; i < 16; ++i) {
                            o[i] = s[i] ^ 0x80;
                        }
                    }
                } else {
                    memcpy(p + 2, qg + ((size_t) b * rs + lane) * 16, 16);
                }
            }
        }
    }
}

// copy `size` bytes at byte `offset` of the (natural layout of the) tensor from / to the device. The conversion is done
// per chunk of whole groups of rows of a slice; a chunk that is only partly written is read, patched and written back.
static bool cl11_rp_transfer(cl11_device_context * c, cl_mem mem, const ggml_tensor * t, uint8_t * host, size_t offset, size_t size, bool to_dev) {
    const cl11_rp_geom G = cl11_rp_geometry(t);
    const size_t slice_nat = (size_t) G.M * G.rowb;
    const size_t end = offset + size;
    const size_t base = cl11_tensor_offset(t);
    const int64_t chunk_groups = std::max<int64_t>(1, (int64_t) ((size_t) 24 * 1024 * 1024 / (32 * G.rowb)));

    std::vector<uint8_t> qbuf, dbuf, nbuf;
    for (size_t s = offset / slice_nat; s <= (end - 1) / slice_nat; ++s) {
        const size_t slice_lo = s * slice_nat;
        const size_t lo = std::max(offset, slice_lo) - slice_lo;
        const size_t hi = std::min(end, slice_lo + slice_nat) - slice_lo;
        const int64_t g_first = (int64_t) (lo / G.rowb) / 32;
        const int64_t g_last  = (int64_t) ((hi - 1) / G.rowb) / 32;
        const size_t q_slice = base + slice_lo;
        const size_t d_slice = q_slice + (size_t) G.M * G.nb * G.qb;

        for (int64_t gc = g_first; gc <= g_last; gc += chunk_groups) {
            const int64_t rA = gc * 32;
            const int64_t rB = std::min<int64_t>(G.M, std::min(g_last + 1, gc + chunk_groups) * 32);
            const int64_t nrows = rB - rA;
            const size_t nat_lo = (size_t) rA * G.rowb, nat_hi = (size_t) rB * G.rowb;
            const size_t ov_lo = std::max(lo, nat_lo), ov_hi = std::min(hi, nat_hi);
            const bool full = ov_lo == nat_lo && ov_hi == nat_hi;
            const size_t qn = (size_t) nrows * G.nb * G.qb, dn = (size_t) nrows * G.nb * 2;
            const size_t q_off = q_slice + (size_t) rA * G.nb * G.qb;
            const size_t d_off = d_slice + (size_t) rA * G.nb * 2;
            uint8_t * hp = full ? host + (slice_lo + nat_lo - offset) : nullptr; // the chunk lies inside of the host range

            qbuf.resize(qn);
            dbuf.resize(dn);
            if (to_dev) {
                if (full) {
                    cl11_rp_pack(G, hp, nrows, qbuf.data(), dbuf.data());
                } else {
                    nbuf.resize((size_t) nrows * G.rowb);
                    CL11_CHECK(clEnqueueReadBuffer(c->queue, mem, CL_TRUE, q_off, qn, qbuf.data(), 0, nullptr, nullptr));
                    CL11_CHECK(clEnqueueReadBuffer(c->queue, mem, CL_TRUE, d_off, dn, dbuf.data(), 0, nullptr, nullptr));
                    cl11_rp_unpack(G, qbuf.data(), dbuf.data(), nrows, nbuf.data());
                    memcpy(nbuf.data() + (ov_lo - nat_lo), host + (slice_lo + ov_lo - offset), ov_hi - ov_lo);
                    cl11_rp_pack(G, nbuf.data(), nrows, qbuf.data(), dbuf.data());
                }
                CL11_CHECK(clEnqueueWriteBuffer(c->queue, mem, CL_TRUE, q_off, qn, qbuf.data(), 0, nullptr, nullptr));
                CL11_CHECK(clEnqueueWriteBuffer(c->queue, mem, CL_TRUE, d_off, dn, dbuf.data(), 0, nullptr, nullptr));
            } else {
                CL11_CHECK(clEnqueueReadBuffer(c->queue, mem, CL_TRUE, q_off, qn, qbuf.data(), 0, nullptr, nullptr));
                CL11_CHECK(clEnqueueReadBuffer(c->queue, mem, CL_TRUE, d_off, dn, dbuf.data(), 0, nullptr, nullptr));
                if (full) {
                    cl11_rp_unpack(G, qbuf.data(), dbuf.data(), nrows, hp);
                } else {
                    nbuf.resize((size_t) nrows * G.rowb);
                    cl11_rp_unpack(G, qbuf.data(), dbuf.data(), nrows, nbuf.data());
                    memcpy(host + (slice_lo + ov_lo - offset), nbuf.data() + (ov_lo - nat_lo), ov_hi - ov_lo);
                }
            }
        }
    }
    return true;
}

// memset of a range of a buffer with a kernel (OpenCL 1.2 clEnqueueFillBuffer is not allowed)
static bool cl11_fill(cl11_device_context * c, cl_mem mem, size_t offset, size_t size, uint8_t value) {
    if (size == 0) {
        return true;
    }
    if (!cl11_ensure_program(c)) {
        return false;
    }

    const uint32_t pattern = 0x01010101u * (uint32_t) value;
    const size_t head = std::min(size, (4 - offset % 4) % 4);
    const size_t body = (size - head) / 4;
    const size_t tail = size - head - body * 4;

    auto fill_bytes = [&](size_t off, size_t n) -> bool {
        if (n == 0) {
            return true;
        }
        cl11_args a(c->kernels[K_FILL_U8]);
        a.set(mem).set((int32_t) off).set((int32_t) n).set((cl_uint) value);
        return cl11_launch(c, a, c->cfg.wg_ew, cl11_div_up(n, c->cfg.wg_ew));
    };

    if (!fill_bytes(offset, head)) {
        return false;
    }
    if (body > 0) {
        // the work-items address the buffer with int offsets: split very large fills
        const size_t chunk = 1u << 28; // words
        for (size_t done = 0; done < body; done += chunk) {
            const size_t n = std::min(chunk, body - done);
            cl11_args b(c->kernels[K_FILL_U32]);
            b.set(mem).set((int32_t) (offset + head + done * 4)).set((int32_t) n).set((cl_uint) pattern);
            if (!cl11_launch(c, b, c->cfg.wg_ew, cl11_div_up(n, c->cfg.wg_ew))) {
                return false;
            }
        }
    }
    return fill_bytes(offset + head + body * 4, tail);
}

static void ggml_backend_cl11_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto * bctx = (cl11_buffer_context *) buffer->context;
    {
        std::lock_guard<std::mutex> lock(bctx->dctx->mutex);
        auto & subs = bctx->dctx->const_subs;
        for (size_t i = 0; i < subs.size();) {
            if (subs[i].parent == bctx->mem) {
                clReleaseMemObject(subs[i].sub);
                subs.erase(subs.begin() + i);
            } else {
                ++i;
            }
        }
    }
    if (bctx->mem) {
        clReleaseMemObject(bctx->mem);
    }
    delete bctx;
}

static void * ggml_backend_cl11_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
    return CL11_BASE;
}

static void ggml_backend_cl11_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    auto * bctx = (cl11_buffer_context *) buffer->context;
    std::lock_guard<std::mutex> lock(bctx->dctx->mutex);
    if (!cl11_fill(bctx->dctx, bctx->mem, cl11_tensor_offset(tensor) + offset, size, value)) {
        GGML_ABORT("%s: memset failed\n", __func__);
    }
}

static void ggml_backend_cl11_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    auto * bctx = (cl11_buffer_context *) buffer->context;
    if (size == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(bctx->dctx->mutex);
    cl11_rp_check_view(tensor);
    if (cl11_repacked(tensor)) {
        if (!cl11_rp_transfer(bctx->dctx, bctx->mem, tensor, (uint8_t *) const_cast<void *>(data), offset, size, true)) {
            GGML_ABORT("%s: transfer of the repacked tensor %s failed\n", __func__, tensor->name);
        }
        return;
    }
    // blocking write: it also waits for the kernels that are already enqueued
    const cl_int err = clEnqueueWriteBuffer(bctx->dctx->queue, bctx->mem, CL_TRUE, cl11_tensor_offset(tensor) + offset, size, data, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        GGML_ABORT("%s: clEnqueueWriteBuffer failed: %s\n", __func__, cl11_err_str(err));
    }
}

static void ggml_backend_cl11_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    auto * bctx = (cl11_buffer_context *) buffer->context;
    if (size == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(bctx->dctx->mutex);
    cl11_rp_check_view(tensor);
    if (cl11_repacked(tensor)) {
        if (!cl11_rp_transfer(bctx->dctx, bctx->mem, tensor, (uint8_t *) data, offset, size, false)) {
            GGML_ABORT("%s: transfer of the repacked tensor %s failed\n", __func__, tensor->name);
        }
        return;
    }
    const cl_int err = clEnqueueReadBuffer(bctx->dctx->queue, bctx->mem, CL_TRUE, cl11_tensor_offset(tensor) + offset, size, data, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        GGML_ABORT("%s: clEnqueueReadBuffer failed: %s\n", __func__, cl11_err_str(err));
    }
}

static bool ggml_backend_cl11_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    auto * dctx_buf = (cl11_buffer_context *) buffer->context;
    cl11_device_context * c = dctx_buf->dctx;
    const size_t size = ggml_nbytes(src);
    if (size == 0) {
        return true;
    }

    ggml_backend_buffer_t src_buf = cl11_tensor_buffer(src);
    if (src_buf == nullptr) {
        return false;
    }

    if (ggml_backend_buffer_is_host(src_buf)) {
        // host -> device
        std::lock_guard<std::mutex> lock(c->mutex);
        if (cl11_repacked(dst)) {
            return cl11_rp_transfer(c, dctx_buf->mem, dst, (uint8_t *) src->data, 0, size, true);
        }
        return clEnqueueWriteBuffer(c->queue, dctx_buf->mem, CL_TRUE, cl11_tensor_offset(dst), size, src->data, 0, nullptr, nullptr) == CL_SUCCESS;
    }

    if (src_buf->iface.free_buffer == ggml_backend_cl11_buffer_free_buffer) {
        if (cl11_repacked(src) != cl11_repacked(dst)) {
            return false; // different layouts: let ggml copy through the host
        }
        auto * sctx = (cl11_buffer_context *) src_buf->context;
        if (sctx->dctx != c) {
            return false; // another cl_context: let ggml copy through the host
        }
        const size_t so = cl11_tensor_offset(src);
        const size_t d_o = cl11_tensor_offset(dst);
        if (sctx->mem == dctx_buf->mem && so < d_o + size && d_o < so + size) {
            return false; // overlapping regions are not allowed in clEnqueueCopyBuffer
        }
        std::lock_guard<std::mutex> lock(c->mutex);
        return clEnqueueCopyBuffer(c->queue, sctx->mem, dctx_buf->mem, so, d_o, size, 0, nullptr, nullptr) == CL_SUCCESS;
    }

    return false;
}

static void ggml_backend_cl11_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * bctx = (cl11_buffer_context *) buffer->context;
    std::lock_guard<std::mutex> lock(bctx->dctx->mutex);
    if (!cl11_fill(bctx->dctx, bctx->mem, 0, bctx->size, value)) {
        GGML_ABORT("%s: clear failed\n", __func__);
    }
}

static const ggml_backend_buffer_i ggml_backend_cl11_buffer_i = {
    /* .free_buffer   = */ ggml_backend_cl11_buffer_free_buffer,
    /* .get_base      = */ ggml_backend_cl11_buffer_get_base,
    /* .init_tensor   = */ nullptr,
    /* .memset_tensor = */ ggml_backend_cl11_buffer_memset_tensor,
    /* .set_tensor    = */ ggml_backend_cl11_buffer_set_tensor,
    /* .get_tensor    = */ ggml_backend_cl11_buffer_get_tensor,
    /* .set_tensor_2d = */ nullptr,
    /* .get_tensor_2d = */ nullptr,
    /* .cpy_tensor    = */ ggml_backend_cl11_buffer_cpy_tensor,
    /* .clear         = */ ggml_backend_cl11_buffer_clear,
    /* .reset         = */ nullptr,
};

//
// buffer type
//

// nominal limit of a cl_mem (what get_max_size reports, ggml-alloc splits the buffers with it)
static size_t cl11_max_buffer_size(const cl11_device_context * c) {
    return std::min(c->info.max_alloc, CL11_MAX_BUFFER_SIZE);
}

// GGML_CL11_STRICT_ALLOC=1: never allocate more than CL_DEVICE_MAX_MEM_ALLOC_SIZE in one buffer
static bool cl11_strict_alloc() {
    static const bool strict = [] {
        const char * env = getenv("GGML_CL11_STRICT_ALLOC");
        return env != nullptr && atoi(env) != 0;
    }();
    return strict;
}

// limit of alloc_buffer: some drivers (NVIDIA 390) accept buffers larger than CL_DEVICE_MAX_MEM_ALLOC_SIZE,
// so a single tensor that is larger than the nominal limit (e.g. the logits) can still be allocated.
// Never more than the global memory, and never more than what the 32-bit offsets of the kernels can address.
static size_t cl11_max_alloc_size(const cl11_device_context * c) {
    if (cl11_strict_alloc()) {
        return cl11_max_buffer_size(c);
    }
    return std::min(std::max(c->info.global_mem, c->info.max_alloc), CL11_MAX_BUFFER_SIZE);
}

static const char * ggml_backend_cl11_buft_get_name(ggml_backend_buffer_type_t buft) {
    auto * c = (cl11_device_context *) buft->context;
    return c->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_cl11_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    auto * c = (cl11_device_context *) buft->context;

    // keep a few spare bytes: the size is never zero and always a multiple of 16
    const size_t alloc_size = std::max<size_t>(16, (size + 15) & ~(size_t) 15);
    if (alloc_size > cl11_max_alloc_size(c)) {
        GGML_LOG_ERROR("%s: cannot allocate %.2f MiB on %s: the limit of a single allocation is %.2f MiB%s\n", __func__,
                size / 1024.0 / 1024.0, c->name.c_str(), cl11_max_alloc_size(c) / 1024.0 / 1024.0,
                cl11_strict_alloc() ? " (GGML_CL11_STRICT_ALLOC)" : "");
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(c->mutex);
    if (!cl11_ensure_context(c)) {
        return nullptr;
    }

    cl_int err = CL_SUCCESS;
    cl_mem mem = clCreateBuffer(c->context, CL_MEM_READ_WRITE, alloc_size, nullptr, &err);
    if (err != CL_SUCCESS || mem == nullptr) {
        GGML_LOG_ERROR("%s: clCreateBuffer(%.2f MiB) failed on %s: %s\n", __func__,
                alloc_size / 1024.0 / 1024.0, c->name.c_str(), cl11_err_str(err));
        return nullptr;
    }

    if (alloc_size > c->info.max_alloc) {
        // larger than the nominal CL_DEVICE_MAX_MEM_ALLOC_SIZE: the driver may have accepted the size
        // without allocating anything (lazy allocation), touch the end of the buffer to detect a failure now
        static std::once_flag warned;
        std::call_once(warned, [&]() {
            GGML_LOG_WARN("%s: allocating %.2f MiB on %s, more than CL_DEVICE_MAX_MEM_ALLOC_SIZE (%.2f MiB); "
                          "set GGML_CL11_STRICT_ALLOC=1 to refuse such allocations\n", "ggml_backend_cl11_buft_alloc_buffer",
                    alloc_size / 1024.0 / 1024.0, c->name.c_str(), c->info.max_alloc / 1024.0 / 1024.0);
        });
        const uint8_t zeros[16] = {};
        err = clEnqueueWriteBuffer(c->queue, mem, CL_TRUE, alloc_size - sizeof(zeros), sizeof(zeros), zeros, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            GGML_LOG_ERROR("%s: the %.2f MiB allocation on %s is not usable: %s\n", __func__,
                    alloc_size / 1024.0 / 1024.0, c->name.c_str(), cl11_err_str(err));
            clReleaseMemObject(mem);
            return nullptr;
        }
    }

    auto * bctx = new cl11_buffer_context{ c, mem, alloc_size };
    return ggml_backend_buffer_init(buft, ggml_backend_cl11_buffer_i, bctx, size);
}

static size_t ggml_backend_cl11_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    auto * c = (cl11_device_context *) buft->context;
    size_t align = std::max<size_t>(CL11_MIN_ALIGNMENT, c->info.align_bytes);
    // CL11_BASE is 4096-aligned
    align = std::min<size_t>(align, 4096);
    // power of 2
    return (size_t) cl11_pow2_floor(align);
}

static size_t ggml_backend_cl11_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    auto * c = (cl11_device_context *) buft->context;
    return cl11_max_buffer_size(c);
}

static bool ggml_backend_cl11_buft_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return false;
}

static const ggml_backend_buffer_type_i ggml_backend_cl11_buft_i = {
    /* .get_name         = */ ggml_backend_cl11_buft_get_name,
    /* .alloc_buffer     = */ ggml_backend_cl11_buft_alloc_buffer,
    /* .alloc_buffer_n   = */ nullptr,
    /* .get_alignment    = */ ggml_backend_cl11_buft_get_alignment,
    /* .get_max_size     = */ ggml_backend_cl11_buft_get_max_size,
    /* .get_alloc_size   = */ nullptr,
    /* .get_alloc_size_n = */ nullptr,
    /* .is_host          = */ ggml_backend_cl11_buft_is_host,
};

//
// supported operations
//

static int cl11_qtype_index(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:  return CL11_QT_F32;
        case GGML_TYPE_F16:  return CL11_QT_F16;
        case GGML_TYPE_Q4_0: return CL11_QT_Q4_0;
        case GGML_TYPE_Q8_0: return CL11_QT_Q8_0;
        case GGML_TYPE_Q4_K: return CL11_QT_Q4_K;
        case GGML_TYPE_Q6_K: return CL11_QT_Q6_K;
        default:             return -1;
    }
}

static bool cl11_is_float(ggml_type type) {
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16;
}

// the offsets and strides are passed as 32-bit integers
static bool cl11_fits_int(const ggml_tensor * t) {
    if (t == nullptr) {
        return true;
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (t->ne[i] > INT_MAX || t->nb[i] > INT_MAX) {
            return false;
        }
    }
    // (keep a margin: the kernels compute group_index * work_group_size + local_id in 32 bits)
    return ggml_nelements(t) <= INT_MAX - 65536 && ggml_nbytes(t) <= (size_t) INT_MAX - 65536;
}

static bool cl11_supports_op_impl(const cl11_device_context * c, const ggml_tensor * op) {
    GGML_UNUSED(c);

    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    const ggml_tensor * src2 = op->src[2];

    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        default:
            break;
    }

    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        if (!cl11_fits_int(op->src[i])) {
            return false;
        }
    }
    if (!cl11_fits_int(op)) {
        return false;
    }

    switch (op->op) {
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            return src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                   ggml_are_same_shape(src0, op);
        case GGML_OP_SCALE:
        case GGML_OP_CLAMP:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
            return src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_TANH:
                    return src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
                default:
                    return false;
            }
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_REGLU:
                    if (src0->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
                        return false;
                    }
                    if (src1 != nullptr) {
                        return src1->type == GGML_TYPE_F32 && ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, op);
                    }
                    return src0->ne[0] == 2 * op->ne[0];
                default:
                    return false;
            }
        case GGML_OP_RMS_NORM:
            return src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                   src0->nb[0] == sizeof(float) && op->nb[0] == sizeof(float);
        case GGML_OP_SOFT_MAX: {
            if (src0->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32 || src2 != nullptr) {
                return false; // attention sinks are not supported
            }
            float max_bias = 0.0f;
            memcpy(&max_bias, (const int32_t *) op->op_params + 1, sizeof(float));
            if (max_bias != 0.0f) {
                return false; // ALiBi is not supported
            }
            if (src0->nb[0] != sizeof(float) || op->nb[0] != sizeof(float)) {
                return false;
            }
            if (src1 != nullptr) {
                if (!cl11_is_float(src1->type) || src1->nb[0] != ggml_type_size(src1->type) ||
                    src1->ne[0] != src0->ne[0] || src1->ne[1] < src0->ne[1]) {
                    return false;
                }
            }
            return true;
        }
        case GGML_OP_ROPE: {
            const int n_dims = ((const int32_t *) op->op_params)[1];
            const int mode   = ((const int32_t *) op->op_params)[2];
            const int n_offs = ((const int32_t *) op->op_params)[15];
            if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX) {
                return false; // no mrope / vision
            }
            if (!cl11_is_float(src0->type) || op->type != src0->type || src1 == nullptr || src1->type != GGML_TYPE_I32) {
                return false;
            }
            if (src2 != nullptr && (src2->type != GGML_TYPE_F32 || src2->ne[0] < n_dims / 2 || src2->nb[0] != sizeof(float))) {
                return false;
            }
            return n_dims > 0 && n_dims % 2 == 0 && op->ne[0] % 2 == 0 && n_offs >= 0 && n_offs % 2 == 0 &&
                   n_offs + n_dims <= op->ne[0] &&
                   src0->nb[0] == ggml_type_size(src0->type) && op->nb[0] == ggml_type_size(op->type) &&
                   src1->ne[0] >= op->ne[2];
        }
        case GGML_OP_GET_ROWS:
            return cl11_qtype_index(src0->type) >= 0 && (!cl11_rp_type(src0->type) || cl11_repacked(src0)) &&
                   src1->type == GGML_TYPE_I32 && op->type == GGML_TYPE_F32 &&
                   src0->nb[0] == ggml_type_size(src0->type) &&
                   src0->ne[0] == op->ne[0];
        case GGML_OP_SET_ROWS:
            return cl11_is_float(src0->type) && cl11_is_float(op->type) &&
                   (src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32) &&
                   src0->nb[0] == ggml_type_size(src0->type) && op->nb[0] == ggml_type_size(op->type) &&
                   src0->ne[0] == op->ne[0] && src0->ne[2] % src1->ne[1] == 0 && src0->ne[3] % src1->ne[2] == 0;
        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_CONT:
            return cl11_is_float(src0->type) && cl11_is_float(op->type) && ggml_nelements(src0) == ggml_nelements(op);
        case GGML_OP_MUL_MAT:
            return cl11_qtype_index(src0->type) >= 0 && (!cl11_rp_type(src0->type) || cl11_repacked(src0)) &&
                   src1->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                   src0->nb[0] == ggml_type_size(src0->type) &&
                   src1->nb[0] == sizeof(float) && op->nb[0] == sizeof(float) &&
                   src0->ne[0] == src1->ne[0] &&
                   src1->ne[2] % src0->ne[2] == 0 && src1->ne[3] % src0->ne[3] == 0;
        default:
            return false;
    }
}

//
// graph compute
//

static bool cl11_op_binary(cl11_device_context * c, const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    cl11_kernel_id kid;
    switch (dst->op) {
        case GGML_OP_ADD: kid = K_ADD; break;
        case GGML_OP_SUB: kid = K_SUB; break;
        case GGML_OP_MUL: kid = K_MUL; break;
        default:          kid = K_DIV; break;
    }
    const cl11_tref s0 = cl11_ref(src0), s1 = cl11_ref(src1), d = cl11_ref(dst);
    const int32_t n = (int32_t) ggml_nelements(dst);

    cl11_args a(c->kernels[kid]);
    a.set(s0.mem).set(s0.td).set(s1.mem).set(s1.td).set(d.mem).set(d.td).set(n);
    return cl11_launch(c, a, c->cfg.wg_ew, cl11_div_up(n, c->cfg.wg_ew));
}

// UNARY, SCALE, CLAMP, SQR, SQRT: op codes of k_unary
static bool cl11_op_unary(cl11_device_context * c, const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    int32_t code = 0;
    float p0 = 0.0f, p1 = 0.0f;
    switch (dst->op) {
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(dst)) {
                case GGML_UNARY_OP_SILU:    code = 0; break;
                case GGML_UNARY_OP_RELU:    code = 1; break;
                case GGML_UNARY_OP_NEG:     code = 2; break;
                case GGML_UNARY_OP_ABS:     code = 3; break;
                case GGML_UNARY_OP_SIGMOID: code = 4; break;
                case GGML_UNARY_OP_TANH:    code = 5; break;
                default: return false;
            }
            break;
        case GGML_OP_SQR:   code = 6; break;
        case GGML_OP_SQRT:  code = 7; break;
        case GGML_OP_SCALE:
            code = 8;
            memcpy(&p0, (const int32_t *) dst->op_params + 0, sizeof(float));
            memcpy(&p1, (const int32_t *) dst->op_params + 1, sizeof(float));
            break;
        case GGML_OP_CLAMP:
            code = 9;
            memcpy(&p0, (const int32_t *) dst->op_params + 0, sizeof(float));
            memcpy(&p1, (const int32_t *) dst->op_params + 1, sizeof(float));
            break;
        default:
            return false;
    }

    const cl11_tref s = cl11_ref(src0), d = cl11_ref(dst);
    const int32_t n = (int32_t) ggml_nelements(dst);

    cl11_args a(c->kernels[K_UNARY]);
    a.set(s.mem).set(s.td).set(d.mem).set(d.td).set(n).set(code).set(p0).set(p1);
    return cl11_launch(c, a, c->cfg.wg_ew, cl11_div_up(n, c->cfg.wg_ew));
}

static bool cl11_op_glu(cl11_device_context * c, const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    const int32_t code    = ggml_get_glu_op(dst) == GGML_GLU_OP_SWIGLU ? 0 : 1;
    const int32_t split   = src1 != nullptr ? 1 : 0;
    const int32_t swapped = ggml_get_op_params_i32(dst, 1);
    const int32_t nc      = (int32_t) dst->ne[0];

    const cl11_tref s0 = cl11_ref(src0), s1 = cl11_ref(src1 ? src1 : src0), d = cl11_ref(dst);
    const int32_t n = (int32_t) ggml_nelements(dst);

    cl11_args a(c->kernels[K_GLU]);
    a.set(s0.mem).set(s0.td).set(s1.mem).set(s1.td).set(d.mem).set(d.td).set(n).set(code).set(split).set(swapped).set(nc);
    return cl11_launch(c, a, c->cfg.wg_ew, cl11_div_up(n, c->cfg.wg_ew));
}

static bool cl11_op_rms_norm(cl11_device_context * c, const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    float eps = 0.0f;
    memcpy(&eps, (const int32_t *) dst->op_params + 0, sizeof(float));

    const cl11_tref s = cl11_ref(src0), d = cl11_ref(dst);
    cl11_args a(c->kernels[K_RMS_NORM]);
    a.set(s.mem).set(s.td).set(d.mem).set(d.td).set(eps);
    return cl11_launch(c, a, c->cfg.wg_row, (size_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]));
}

static bool cl11_op_soft_max(cl11_device_context * c, const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1]; // mask

    float scale = 1.0f;
    memcpy(&scale, (const int32_t *) dst->op_params + 0, sizeof(float));

    const cl11_tref s = cl11_ref(src0), d = cl11_ref(dst);
    const cl11_tref m = src1 ? cl11_ref(src1) : s;
    cl11_tref mm = m;
    if (!src1) {
        for (int i = 0; i < 4; ++i) {
            mm.td.ne[i] = 1;
            mm.td.nb[i] = 0;
            cl11_fastdiv_init(1, mm.td.fm[i], mm.td.fl[i]);
        }
        mm.td.off = 0;
    }
    const int32_t has_mask = src1 ? 1 : 0;
    const int32_t mask_f16 = (src1 && src1->type == GGML_TYPE_F16) ? 1 : 0;

    cl11_args a(c->kernels[K_SOFT_MAX]);
    a.set(s.mem).set(s.td).set(mm.mem).set(mm.td).set(has_mask).set(mask_f16).set(d.mem).set(d.td).set(scale);
    return cl11_launch(c, a, c->cfg.wg_row, (size_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]));
}

static bool cl11_op_rope(cl11_device_context * c, const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1]; // positions
    const ggml_tensor * src2 = dst->src[2]; // freq factors

    const int32_t * op = (const int32_t *) dst->op_params;
    const int n_dims     = op[1];
    const int mode       = op[2];
    const int n_ctx_orig = op[4];
    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    memcpy(&freq_base,   op +  5, sizeof(float));
    memcpy(&freq_scale,  op +  6, sizeof(float));
    memcpy(&ext_factor,  op +  7, sizeof(float));
    memcpy(&attn_factor, op +  8, sizeof(float));
    memcpy(&beta_fast,   op +  9, sizeof(float));
    memcpy(&beta_slow,   op + 10, sizeof(float));
    const int n_offs = op[15];

    const float theta_scale = powf(freq_base, -2.0f / n_dims);
    float corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    const cl11_tref s = cl11_ref(src0), p = cl11_ref(src1), d = cl11_ref(dst);
    const cl11_tref f = src2 ? cl11_ref(src2) : p;
    const int32_t has_ff = src2 ? 1 : 0;
    const int32_t neox   = mode == GGML_ROPE_TYPE_NEOX ? 1 : 0;
    const int32_t nd = n_dims, no = n_offs;

    cl11_args a(c->kernels[src0->type == GGML_TYPE_F16 ? K_ROPE_F16 : K_ROPE_F32]);
    a.set(s.mem).set(s.td).set(p.mem).set(p.td).set(f.mem).set(f.td).set(has_ff).set(d.mem).set(d.td)
     .set(nd).set(neox).set(no)
     .set(theta_scale).set(freq_scale).set(ext_factor).set(attn_factor).set(corr_dims[0]).set(corr_dims[1]);

    // one work-group per row
    return cl11_launch(c, a, c->cfg.wg_row, (size_t) (dst->ne[1] * dst->ne[2] * dst->ne[3]));
}

static bool cl11_op_get_rows(cl11_device_context * c, const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    const cl11_tref s0 = cl11_ref(src0), s1 = cl11_ref(src1), d = cl11_ref(dst);
    cl11_args a(c->kernels[K_GET_ROWS + cl11_qtype_index(src0->type)]);
    a.set(s0.mem).set(s0.td).set(s1.mem).set(s1.td).set(d.mem).set(d.td);
    return cl11_launch(c, a, c->cfg.wg_row, (size_t) (src1->ne[0] * src1->ne[1] * src1->ne[2]));
}

static bool cl11_op_set_rows(cl11_device_context * c, const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    const int kid = K_SET_ROWS + (src0->type == GGML_TYPE_F16 ? 2 : 0) + (dst->type == GGML_TYPE_F16 ? 1 : 0);
    const cl11_tref s0 = cl11_ref(src0), s1 = cl11_ref(src1), d = cl11_ref(dst);
    const int32_t n = (int32_t) ggml_nelements(src0);
    const int32_t idx64 = src1->type == GGML_TYPE_I64 ? 1 : 0;

    cl11_args a(c->kernels[kid]);
    a.set(s0.mem).set(s0.td).set(s1.mem).set(s1.td).set(d.mem).set(d.td).set(n).set(idx64);
    return cl11_launch(c, a, c->cfg.wg_ew, cl11_div_up(n, c->cfg.wg_ew));
}

static bool cl11_op_cpy(cl11_device_context * c, const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    const cl11_tref s = cl11_ref(src0), d = cl11_ref(dst);

    // transposing copy of a tensor whose dim 1 is contiguous into a contiguous tensor of the same type and shape
    if (src0->type == dst->type && ggml_are_same_shape(src0, dst) && ggml_is_contiguous(dst) &&
        src0->nb[1] == ggml_type_size(src0->type) && src0->nb[0] > src0->nb[1] && src0->ne[0] > 1 && src0->ne[1] > 1 &&
        c->cfg.gt >= 8 && !c->no_cpy_t) {
        const int64_t tiles0 = (dst->ne[0] + 31) / 32;
        const int64_t tiles1 = (dst->ne[1] + 31) / 32;
        const cl11_bcast bc = cl11_make_bcast(1, 1, tiles0, tiles1);
        cl11_args a(c->kernels[src0->type == GGML_TYPE_F16 ? K_CPY_T_F16 : K_CPY_T_F32]);
        a.set(s.mem).set(s.td).set(d.mem).set(d.td).set(bc);
        return cl11_launch(c, a, (size_t) c->cfg.gt * c->cfg.gt, (size_t) (tiles0 * tiles1 * dst->ne[2] * dst->ne[3]));
    }

    const int kid = K_CPY_F32_F32 + (src0->type == GGML_TYPE_F16 ? 2 : 0) + (dst->type == GGML_TYPE_F16 ? 1 : 0);
    const int32_t n = (int32_t) ggml_nelements(src0);

    cl11_args a(c->kernels[kid]);
    a.set(s.mem).set(s.td).set(d.mem).set(d.td).set(n);
    return cl11_launch(c, a, c->cfg.wg_ew, cl11_div_up(n, c->cfg.wg_ew));
}

// sub-buffer [offset, offset + size) of `parent` for a constant-memory kernel argument (cached), nullptr if it cannot be created
// (the origin of a sub-buffer must be aligned to CL_DEVICE_MEM_BASE_ADDR_ALIGN)
static cl_mem cl11_const_sub(cl11_device_context * c, cl_mem parent, size_t offset, size_t size) {
    if (c->info.align_bytes == 0 || offset % c->info.align_bytes != 0) {
        return nullptr;
    }
    for (const auto & e : c->const_subs) {
        if (e.parent == parent && e.offset == offset && e.size == size) {
            return e.sub;
        }
    }
    cl_buffer_region region = { offset, size };
    cl_int err = CL_SUCCESS;
    cl_mem sub = clCreateSubBuffer(parent, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
    if (err != CL_SUCCESS || sub == nullptr) {
        return nullptr;
    }
    c->const_subs.push_back({ parent, offset, size, sub });
    return sub;
}

static int cl11_log2_exact(int v) {
    int l = 0;
    while ((1 << l) < v) {
        ++l;
    }
    return l;
}

static bool cl11_op_mul_mat(cl11_device_context * c, const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    const int qt = cl11_qtype_index(src0->type);
    const cl11_tref s0 = cl11_ref(src0), s1 = cl11_ref(src1), d = cl11_ref(dst);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    const bool rp = cl11_rp_type(src0->type); // repacked weights (R32 layout): the mat-vec kernel is k_mvr_*
    if (rp && ne11 <= c->mv_max_cols && c->cfg.wg_mvr % 32 == 0) {
        const cl11_td & ta = s0.td;
        const cl11_td & tb = s1.td;
        const bool a16 = ta.off % 16 == 0 && ta.nb[2] % 16 == 0 && ta.nb[3] % 16 == 0;
        const bool b16 = tb.off % 16 == 0 && tb.nb[1] % 16 == 0 && tb.nb[2] % 16 == 0 && tb.nb[3] % 16 == 0;
        if (a16 && b16) {
            const int64_t rows_wg = 32 * c->cfg.mvr_nrr;
            const int64_t tiles = (ne01 + rows_wg - 1) / rows_wg;
            const cl11_bcast bc = cl11_make_bcast(r2, r3, tiles, 1);
            const bool q8 = src0->type == GGML_TYPE_Q8_0;
            const size_t x_bytes = (size_t) ne00 * sizeof(float);
            cl_mem xsub = nullptr;
            if (c->mvr_const_x && ne11 == 1 && ne12 == 1 && ne13 == 1 && x_bytes <= std::min<size_t>(c->max_const, 65536)) {
                xsub = cl11_const_sub(c, s1.mem, (size_t) tb.off, x_bytes);
            }
            if (xsub != nullptr) {
                cl11_args a(c->kernels[q8 ? K_MVRC_Q8_0 : K_MVRC_Q4_0]);
                a.set(s0.mem).set(s0.td).set(xsub).set(d.mem).set(d.td).set(bc);
                return cl11_launch(c, a, c->cfg.wg_mvr, (size_t) tiles);
            }
            cl11_args a(c->kernels[q8 ? K_MVR_Q8_0 : K_MVR_Q4_0]);
            a.set(s0.mem).set(s0.td).set(s1.mem).set(s1.td).set(d.mem).set(d.td).set(bc);
            return cl11_launch(c, a, c->cfg.wg_mvr, (size_t) tiles * (size_t) (ne11 * ne12 * ne13));
        }
    }

    if (ne11 <= c->mv_max_cols && !rp) {
        const cl11_td & ta = s0.td;
        const cl11_td & tb = s1.td;

        // staged kernels (src1 is copied to local memory, several rows per work-group):
        // long rows, 4-byte aligned rows of src0 and a 16-byte aligned src1
        const bool a4  = ta.off % 4 == 0 && ta.nb[1] % 4 == 0 && ta.nb[2] % 4 == 0 && ta.nb[3] % 4 == 0;
        const bool a16 = ta.off % 16 == 0 && ta.nb[1] % 16 == 0 && ta.nb[2] % 16 == 0 && ta.nb[3] % 16 == 0;
        const bool b16 = tb.off % 16 == 0 && tb.nb[1] % 16 == 0 && tb.nb[2] % 16 == 0 && tb.nb[3] % 16 == 0;
        bool staged = c->mvs_enabled && c->cfg.wg_mv % 32 == 0 && b16 && ne00 >= c->mvs_min_k;
        switch (src0->type) {
            case GGML_TYPE_Q6_K: staged = staged && ta.off % 2 == 0 && ta.nb[1] % 2 == 0 && ta.nb[2] % 2 == 0 && ta.nb[3] % 2 == 0; break;
            case GGML_TYPE_Q4_K: staged = staged && a4;                    break;
            case GGML_TYPE_F16:  staged = staged && a4 && ne00 % 8 == 0;   break;
            default:             staged = staged && a16 && ne00 % 8 == 0;  break;
        }

        if (staged) {
            const int rows_wg = c->cfg.wg_mv / 32 * c->cfg.mvs_nrw;
            const int64_t tiles = (ne01 + rows_wg - 1) / rows_wg;
            const cl11_bcast bc = cl11_make_bcast(r2, r3, tiles, 1);
            // measured on a GT 430: the x copy in local memory only pays off for q4_K (16 float4 of src1 per lane)
            const int family = c->mvs_family != 0 ? c->mvs_family : (src0->type == GGML_TYPE_Q4_K ? 1 : 2);
            cl11_args a(c->kernels[(family == 2 ? K_MVW : K_MVX) + qt]);
            a.set(s0.mem).set(s0.td).set(s1.mem).set(s1.td).set(d.mem).set(d.td).set(bc);
            return cl11_launch(c, a, c->cfg.wg_mv, (size_t) tiles * (size_t) (ne11 * ne12 * ne13));
        }

        // generic mat-vec: `tpr` threads per output, the number of units of work per row bounds it
        int64_t units;
        switch (src0->type) {
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q6_K: units = (ne00 / 256) * 16; break;
            default:             units = (ne00 + 3) / 4;    break;
        }
        // short rows (K cache, V cache): several units per thread, fewer threads (and reduction steps) per row
        int tpr = cl11_pow2_floor((size_t) std::max<int64_t>(units, 1));
        if (units <= 128) {
            tpr = std::min(tpr, std::max(4, cl11_pow2_floor((size_t) std::max<int64_t>(units / 4, 1))));
        }
        if (c->mv_tpr > 0) {
            tpr = std::min(tpr, cl11_pow2_floor((size_t) c->mv_tpr));
        }
        tpr = std::min(tpr, c->cfg.wg_mv);
        const size_t total = (size_t) (ne01 * ne11 * ne12 * ne13);
        const size_t rows_per_group = (size_t) (c->cfg.wg_mv / tpr);
        const cl11_bcast bc = cl11_make_bcast(r2, r3, 1, 1);

        cl11_args a(c->kernels[K_MV + qt]);
        a.set(s0.mem).set(s0.td).set(s1.mem).set(s1.td).set(d.mem).set(d.td).set(bc).set((int32_t) tpr).set((int32_t) cl11_log2_exact(tpr));
        return cl11_launch(c, a, c->cfg.wg_mv, cl11_div_up(total, rows_per_group));
    }

    // mat-mat: (4*gt) x (mm_bn*gt) tiles (src0 rows x src1 rows)
    const int64_t tile_m = 4 * c->cfg.gt;
    const int64_t tile_n = (int64_t) c->cfg.mm_bn * c->cfg.gt;
    const int64_t tiles_m = (ne01 + tile_m - 1) / tile_m;
    const int64_t tiles_n = (ne11 + tile_n - 1) / tile_n;
    const size_t groups = (size_t) (tiles_m * tiles_n * ne12 * ne13);
    const cl11_bcast bc = cl11_make_bcast(r2, r3, tiles_m, tiles_n);

    cl11_args a(c->kernels[K_MM + qt]);
    a.set(s0.mem).set(s0.td).set(s1.mem).set(s1.td).set(d.mem).set(d.td).set(bc);
    return cl11_launch(c, a, (size_t) (c->cfg.gt / 2) * (c->cfg.mm_bn * c->cfg.gt / 4), groups);
}

// MUL_MAT(CONT(transposed F16 view), x) with a single column x: the CONT is skipped, see k_mvt_f16
static bool cl11_op_mul_mat_t(cl11_device_context * c, const ggml_tensor * dst, const ggml_tensor * a) {
    const ggml_tensor * src1 = dst->src[1];

    const cl11_tref s0 = cl11_ref(a), s1 = cl11_ref(src1), d = cl11_ref(dst);
    const int64_t tiles = (a->ne[1] + 31) / 32;
    const cl11_bcast bc = cl11_make_bcast(src1->ne[2] / a->ne[2], src1->ne[3] / a->ne[3], tiles, 1);

    cl11_args args(c->kernels[K_MVT_F16]);
    args.set(s0.mem).set(s0.td).set(s1.mem).set(s1.td).set(d.mem).set(d.td).set(bc);
    return cl11_launch(c, args, c->cfg.wg_row, (size_t) tiles * (size_t) (src1->ne[1] * src1->ne[2] * src1->ne[3]));
}

// can the CONT node i be skipped because the (only) MUL_MAT that uses it reads the transposed view directly?
static bool cl11_can_skip_cont(const cl11_device_context * c, const ggml_cgraph * cgraph, int i) {
    if (c->no_mvt || c->cfg.wg_row % 32 != 0) {
        return false;
    }
    const ggml_tensor * cont = cgraph->nodes[i];
    if (cont->op != GGML_OP_CONT || !ggml_node_has_n_uses(cgraph, i, 1)) {
        return false;
    }
    const ggml_tensor * a = cont->src[0];
    if (a == nullptr || a->type != GGML_TYPE_F16 || cont->type != GGML_TYPE_F16 || !ggml_are_same_shape(a, cont) ||
        a->nb[1] != ggml_type_size(GGML_TYPE_F16) || a->nb[0] <= a->nb[1] || !cl11_fits_int(a)) {
        return false;
    }
    // the consumer
    for (int j = i + 1; j < cgraph->n_nodes; ++j) {
        const ggml_tensor * mm = cgraph->nodes[j];
        if (mm->src[1] == cont) {
            return false; // used as src1
        }
        if (mm->src[0] != cont) {
            continue;
        }
        const ggml_tensor * src1 = mm->src[1];
        return mm->op == GGML_OP_MUL_MAT && (mm->flags & GGML_TENSOR_FLAG_COMPUTE) != 0 &&
               src1->type == GGML_TYPE_F32 && src1->ne[1] == 1 && src1->nb[0] == sizeof(float) &&
               mm->type == GGML_TYPE_F32 && mm->nb[0] == sizeof(float) &&
               a->ne[0] == src1->ne[0] && src1->ne[2] % a->ne[2] == 0 && src1->ne[3] % a->ne[3] == 0;
    }
    return false;
}

static bool cl11_compute_node(cl11_device_context * c, const ggml_tensor * node) {
    switch (node->op) {
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:       return cl11_op_binary(c, node);
        case GGML_OP_UNARY:
        case GGML_OP_SCALE:
        case GGML_OP_CLAMP:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:      return cl11_op_unary(c, node);
        case GGML_OP_GLU:       return cl11_op_glu(c, node);
        case GGML_OP_RMS_NORM:  return cl11_op_rms_norm(c, node);
        case GGML_OP_SOFT_MAX:  return cl11_op_soft_max(c, node);
        case GGML_OP_ROPE:      return cl11_op_rope(c, node);
        case GGML_OP_GET_ROWS:  return cl11_op_get_rows(c, node);
        case GGML_OP_SET_ROWS:  return cl11_op_set_rows(c, node);
        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_CONT:      return cl11_op_cpy(c, node);
        case GGML_OP_MUL_MAT:   return cl11_op_mul_mat(c, node);
        default:
            GGML_LOG_ERROR("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
            return false;
    }
}

//
// backend
//

struct cl11_backend_context {
    cl11_device_context * dctx;
    std::string           name;
};

static const char * ggml_backend_cl11_get_name(ggml_backend_t backend) {
    auto * bctx = (cl11_backend_context *) backend->context;
    return bctx->name.c_str();
}

static void ggml_backend_cl11_free(ggml_backend_t backend) {
    auto * bctx = (cl11_backend_context *) backend->context;
    if (bctx->dctx->profile) {
        std::lock_guard<std::mutex> lock(bctx->dctx->mutex);
        if (bctx->dctx->queue) {
            clFinish(bctx->dctx->queue);
            cl11_prof_collect(bctx->dctx);
        }
        cl11_prof_print(bctx->dctx);
    }
    delete bctx;
    delete backend;
}

static void ggml_backend_cl11_synchronize(ggml_backend_t backend) {
    auto * bctx = (cl11_backend_context *) backend->context;
    cl11_device_context * c = bctx->dctx;
    std::lock_guard<std::mutex> lock(c->mutex);
    if (c->queue) {
        const cl_int err = clFinish(c->queue);
        if (err != CL_SUCCESS) {
            GGML_LOG_ERROR("%s: clFinish failed: %s\n", __func__, cl11_err_str(err));
        }
    }
}

static enum ggml_status ggml_backend_cl11_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    auto * bctx = (cl11_backend_context *) backend->context;
    cl11_device_context * c = bctx->dctx;

    std::lock_guard<std::mutex> lock(c->mutex);
    if (!cl11_ensure_program(c)) {
        return GGML_STATUS_FAILED;
    }

    std::vector<const ggml_tensor *> skipped_conts;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        ggml_tensor * node = cgraph->nodes[i];

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        if (ggml_is_empty(node) || ggml_op_is_empty(node->op)) {
            continue;
        }

        // CONT of a transposed view that only feeds a mat-vec: skipped, the MUL_MAT reads the view (k_mvt_f16)
        if (node->op == GGML_OP_CONT && cl11_can_skip_cont(c, cgraph, i)) {
            skipped_conts.push_back(node);
            continue;
        }
        if (node->op == GGML_OP_MUL_MAT && std::find(skipped_conts.begin(), skipped_conts.end(), node->src[0]) != skipped_conts.end()) {
            if (c->profile) {
                c->prof_key = cl11_prof_key(c, "MUL_MAT f16 transposed (CONT skipped)");
                c->prof_stats[c->prof_key].count++;
            }
            if (!cl11_op_mul_mat_t(c, node, node->src[0]->src[0])) {
                GGML_LOG_ERROR("%s: failed to compute node %d (%s, %s)\n", __func__, i, ggml_op_desc(node), node->name);
                return GGML_STATUS_FAILED;
            }
            continue;
        }

        if (c->profile) {
            std::string key = ggml_op_desc(node);
            if (node->op == GGML_OP_MUL_MAT) {
                key += std::string(" ") + ggml_type_name(node->src[0]->type) + (node->src[1]->ne[1] <= c->mv_max_cols ? " mv" : " mm");
                key += " n=" + std::to_string(node->src[1]->ne[1] <= c->mv_max_cols ? 1 : node->src[1]->ne[1]);
            } else if (node->op == GGML_OP_GET_ROWS || node->op == GGML_OP_CPY || node->op == GGML_OP_SET_ROWS) {
                key += std::string(" ") + ggml_type_name(node->src[0]->type) + "->" + ggml_type_name(node->type);
            }
            if (getenv("GGML_CL11_PROFILE_SHAPES")) {
                const ggml_tensor * s0 = node->src[0];
                char buf[160];
                snprintf(buf, sizeof(buf), " src0[%lld,%lld,%lld,%lld] nb[%zu,%zu,%zu,%zu]", s0 ? (long long) s0->ne[0] : 0, s0 ? (long long) s0->ne[1] : 0,
                        s0 ? (long long) s0->ne[2] : 0, s0 ? (long long) s0->ne[3] : 0, s0 ? s0->nb[0] : 0, s0 ? s0->nb[1] : 0, s0 ? s0->nb[2] : 0, s0 ? s0->nb[3] : 0);
                key += buf;
            }
            c->prof_key = cl11_prof_key(c, key);
            c->prof_stats[c->prof_key].count++;
            if (node->op == GGML_OP_MUL_MAT) {
                c->prof_stats[c->prof_key].bytes += (double) ggml_nbytes(node->src[0]);
            }
        }

        if (!cl11_compute_node(c, node)) {
            GGML_LOG_ERROR("%s: failed to compute node %d (%s, %s)\n", __func__, i, ggml_op_desc(node), node->name);
            return GGML_STATUS_FAILED;
        }
    }

    // submit the commands, but do not wait for them
    clFlush(c->queue);
    if (c->profile) {
        clFinish(c->queue);
        cl11_prof_collect(c);
    }
    return GGML_STATUS_SUCCESS;
}

static const ggml_backend_i ggml_backend_cl11_i = {
    /* .get_name                = */ ggml_backend_cl11_get_name,
    /* .free                    = */ ggml_backend_cl11_free,
    /* .set_tensor_async        = */ nullptr,
    /* .get_tensor_async        = */ nullptr,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ ggml_backend_cl11_synchronize,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ ggml_backend_cl11_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ nullptr,
};

static ggml_guid_t ggml_backend_cl11_guid() {
    static ggml_guid guid = { 0x43, 0x4c, 0x31, 0x31, 0x6f, 0x70, 0x65, 0x6e, 0x63, 0x6c, 0x31, 0x2e, 0x31, 0x67, 0x67, 0x6d };
    return &guid;
}

//
// device
//

static cl11_device_context * cl11_dev_ctx(ggml_backend_dev_t dev) {
    return (cl11_device_context *) dev->context;
}

static const char * ggml_backend_cl11_device_get_name(ggml_backend_dev_t dev) {
    return cl11_dev_ctx(dev)->name.c_str();
}

static const char * ggml_backend_cl11_device_get_description(ggml_backend_dev_t dev) {
    return cl11_dev_ctx(dev)->info.name.c_str();
}

static void ggml_backend_cl11_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    const cl11_device_context * c = cl11_dev_ctx(dev);
    // OpenCL 1.1 cannot report the free memory: assume that the system (display, ...) keeps some of it
    *total = c->info.global_mem;
    const size_t reserve = std::min<size_t>(128u * 1024 * 1024, *total / 4);
    *free = *total - reserve;
}

static enum ggml_backend_dev_type ggml_backend_cl11_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_cl11_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_cl11_device_get_name(dev);
    props->description = ggml_backend_cl11_device_get_description(dev);
    props->type        = ggml_backend_cl11_device_get_type(dev);
    props->device_id   = nullptr;
    ggml_backend_cl11_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                = */ false,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ false,
        /* .events               = */ false,
        /* .mmap_support         = */ false,
    };
}

static ggml_backend_t ggml_backend_cl11_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    cl11_device_context * c = cl11_dev_ctx(dev);

    {
        std::lock_guard<std::mutex> lock(c->mutex);
        if (!cl11_ensure_context(c)) {
            return nullptr;
        }
    }

    auto * bctx = new cl11_backend_context{ c, c->name };
    return new ggml_backend {
        /* .guid    = */ ggml_backend_cl11_guid(),
        /* .iface   = */ ggml_backend_cl11_i,
        /* .device  = */ dev,
        /* .context = */ bctx,
    };
}

static ggml_backend_buffer_type_t ggml_backend_cl11_device_get_buffer_type(ggml_backend_dev_t dev) {
    return &cl11_dev_ctx(dev)->buft;
}

static bool ggml_backend_cl11_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    return cl11_supports_op_impl(cl11_dev_ctx(dev), op);
}

static bool ggml_backend_cl11_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    // only the buffers of the same device: a cl_mem cannot be used with another cl_context
    return buft->iface.get_name == ggml_backend_cl11_buft_get_name && buft->context == dev->context;
}

static const ggml_backend_device_i ggml_backend_cl11_device_i = {
    /* .get_name             = */ ggml_backend_cl11_device_get_name,
    /* .get_description      = */ ggml_backend_cl11_device_get_description,
    /* .get_memory           = */ ggml_backend_cl11_device_get_memory,
    /* .get_type             = */ ggml_backend_cl11_device_get_type,
    /* .get_props            = */ ggml_backend_cl11_device_get_props,
    /* .init_backend         = */ ggml_backend_cl11_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_cl11_device_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_cl11_device_supports_op,
    /* .supports_buft        = */ ggml_backend_cl11_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

//
// registry
//

static const char * ggml_backend_cl11_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_CL11_NAME;
}

static size_t ggml_backend_cl11_reg_get_device_count(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return g_cl11_devices.size();
}

static ggml_backend_dev_t ggml_backend_cl11_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index < g_cl11_devices.size());
    GGML_UNUSED(reg);
    return &g_cl11_devices[index]->dev;
}

static const ggml_backend_reg_i ggml_backend_cl11_reg_i = {
    /* .get_name         = */ ggml_backend_cl11_reg_get_name,
    /* .get_device_count = */ ggml_backend_cl11_reg_get_device_count,
    /* .get_device       = */ ggml_backend_cl11_reg_get_device,
    /* .get_proc_address = */ nullptr,
};

ggml_backend_reg_t ggml_backend_cl11_reg(void) {
    static std::mutex mutex;
    static ggml_backend_reg reg;
    static bool initialized = false;

    std::lock_guard<std::mutex> lock(mutex);
    if (initialized) {
        return &reg;
    }
    initialized = true;

    reg = ggml_backend_reg {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_cl11_reg_i,
        /* .context     = */ nullptr,
    };

    cl11_probe_devices();

    for (auto & c : g_cl11_devices) {
        c->buft = ggml_backend_buffer_type {
            /* .iface   = */ ggml_backend_cl11_buft_i,
            /* .device  = */ &c->dev,
            /* .context = */ c.get(),
        };
        c->dev = ggml_backend_device {
            /* .iface   = */ ggml_backend_cl11_device_i,
            /* .reg     = */ &reg,
            /* .context = */ c.get(),
        };
    }

    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_cl11_reg)

//
// public API
//

int ggml_backend_cl11_get_device_count(void) {
    return (int) ggml_backend_reg_dev_count(ggml_backend_cl11_reg());
}

void ggml_backend_cl11_get_device_description(int device, char * description, size_t description_size) {
    ggml_backend_reg_t reg = ggml_backend_cl11_reg();
    if (device < 0 || (size_t) device >= ggml_backend_reg_dev_count(reg) || description_size == 0) {
        if (description_size > 0) {
            description[0] = '\0';
        }
        return;
    }
    snprintf(description, description_size, "%s", ggml_backend_dev_description(ggml_backend_reg_dev_get(reg, (size_t) device)));
}

ggml_backend_t ggml_backend_cl11_init(int device) {
    ggml_backend_reg_t reg = ggml_backend_cl11_reg();
    if (device < 0 || (size_t) device >= ggml_backend_reg_dev_count(reg)) {
        GGML_LOG_ERROR("%s: invalid device %d\n", __func__, device);
        return nullptr;
    }
    return ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, (size_t) device), nullptr);
}

bool ggml_backend_is_cl11(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, ggml_backend_cl11_guid());
}

ggml_backend_buffer_type_t ggml_backend_cl11_buffer_type(int device) {
    ggml_backend_reg_t reg = ggml_backend_cl11_reg();
    if (device < 0 || (size_t) device >= ggml_backend_reg_dev_count(reg)) {
        return nullptr;
    }
    return ggml_backend_dev_buffer_type(ggml_backend_reg_dev_get(reg, (size_t) device));
}
