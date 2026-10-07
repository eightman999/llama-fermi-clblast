#ifndef GGML_CL11_H
#define GGML_CL11_H

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

// CL11: a ggml backend that only needs OpenCL 1.1 (host API and OpenCL C 1.1),
// intended for old GPUs such as NVIDIA Fermi (e.g. GeForce GT 430).
//
// Devices are named "CL11_0", "CL11_1", ... and can be filtered with the
// environment variable GGML_CL11_DEVICES (comma separated substrings of the OpenCL device name).

#define GGML_CL11_NAME "CL11"

// initialize the backend on the device with the given index (see ggml_backend_cl11_device_count)
GGML_BACKEND_API ggml_backend_t ggml_backend_cl11_init(int device);
GGML_BACKEND_API bool           ggml_backend_is_cl11(ggml_backend_t backend);

GGML_BACKEND_API int            ggml_backend_cl11_get_device_count(void);
GGML_BACKEND_API void           ggml_backend_cl11_get_device_description(int device, char * description, size_t description_size);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cl11_buffer_type(int device);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cl11_reg(void);

#ifdef  __cplusplus
}
#endif

#endif // GGML_CL11_H
