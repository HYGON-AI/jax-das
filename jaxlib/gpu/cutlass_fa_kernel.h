#ifndef JAXLIB_GPU_CUTLASS_FA_KERNEL_H_
#define JAXLIB_GPU_CUTLASS_FA_KERNEL_H_
#include <cmath>
#include <cstddef>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <cstdlib>
#include "absl/status/statusor.h"
#include "xla/service/custom_call_status.h"
#include "jaxlib/gpu/vendor.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#define HIP_CHECK_RETURN(func)                                           \
    do {                                                                 \
        auto __result = (func);                                          \
        if (__result != hipSuccess) {                                    \
            return JAX_AS_STATUS(__result);                              \
        }                                                                \
    } while (0)
namespace jax {
namespace JAX_GPU_NAMESPACE {
namespace CutLassFa {
    struct CutlassFaDescriptor{
        int batch;
        int seq;
        int head;
        int dims;
        int data_type;
        int layout;
        int workspace_size;
    };
    // Return (workspace size, reserve space size).
    absl::StatusOr<size_t> CutlassComputeWorkspaceReserveSpaceSizes(int batch, int head, int seq, int dims);

    void cutlassFaFwd(gpuStream_t stream, void** buffers, const char* opaque,
                  size_t opaque_len, XlaCustomCallStatus* status);
        // void miOpen_fa_bwd_kernel();
}
}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
#endif
