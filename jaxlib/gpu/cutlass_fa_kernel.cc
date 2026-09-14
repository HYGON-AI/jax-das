#include "jaxlib/gpu/cutlass_fa_kernel.h"
#include "jaxlib/gpu/gpu_kernel_helpers.h"
#include "jaxlib/kernel_helpers.h"
#include "xla/service/custom_call_status.h"
#include "jaxlib/gpu/vendor.h"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>
#include "absl/status/status.h"
#include <dlfcn.h>
#include <filesystem>
#include <fnmatch.h>
#include "absl/strings/str_format.h"
namespace fs = std::filesystem;
namespace jax {
namespace JAX_GPU_NAMESPACE {
    template <typename T>
    std::string ErrorString(T status, const char* file, std::int64_t line,
                            const char* expr) {
    return absl::StrFormat("%s:%d: operation %s failed: %s", file, line, expr,
                            ErrorString(status));
    }
}
}

static std::string find_matching_libraries(const fs::path& directory, const std::string& pattern) {
    if (!fs::exists(directory)) {
        return "";
    }
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file() || entry.is_symlink()) {
            std::string file_name = entry.path().filename().string();
            if (fnmatch(pattern.c_str(), file_name.c_str(), 0) == 0) {
                return entry.path().string();
            }
        }
    }
    return "";
}

static std::string print_library_path() {
    Dl_info info;
    std::string path;
    if (dladdr(reinterpret_cast<void*>(&print_library_path), &info)) {
        path = std::string(info.dli_fname);
    }
    return path;
}

static std::string get_so_path() {
  auto* e_fa_so = std::getenv("FA_SO_PATH");
  if (e_fa_so) {
    fs::path fa_path(e_fa_so);
    if (fs::is_directory(fa_path)) {
      return (fa_path / "libflash_atten_c.so").string();
    }
    return fa_path.string();
  }
  auto c_so_path = print_library_path();
  fs::path c_so_fs(c_so_path);
  for (fs::path dir = c_so_fs.parent_path(); !dir.empty(); dir = dir.parent_path()) {
    auto lib = find_matching_libraries(dir, "libflash_atten_c.so*");
    if (!lib.empty()) {
      return lib;
    }
    if (dir == dir.root_path()) {
      break;
    }
  }
  return "";
}

typedef size_t (*mha_masked_fwd_bias_workspace_func)(int batch, int seq_q, int seq_k, int num_head, int head, int head_size, int dims, bool is_bf16);
typedef void (*mha_masked_fwd_bias_func)(void* cu_q,                                   //* batch_size x seqlen_q x num_heads x head_size
        void* cu_k,                                               //* batch_size x seqlen_k x num_heads_k x head_size
        void* cu_v,                                               //* batch_size x seqlen_k x num_heads_k x head_size
        void* cu_out_,                                           //* batch_size x seqlen_q x num_heads x head_size
        void* cu_alibi_slopes_,                                   //* num_heads or batch_size x num_heads
        void* cu_bias_,                                           //* batch_size x num_head x seqlen_q x seqlen_k
        uint8_t* cu_mask_,                                        //* batch_size x num_head x seqlen_q x seqlen_k
        uint64_t* rng_state,                                       //* size=2 seed, offset
        void* workspace_ptr,
        int batch_size_, int seqlen_q_, int seq_len_kv_,
        int num_heads_q_, int num_heads_kv_,
        int head_size_q_, int head_size_kv_,
        size_t q_stride_b, size_t q_stride_s, size_t q_stride_h, size_t q_stride_d, //* q
        size_t k_stride_b, size_t k_stride_s, size_t k_stride_h, size_t k_stride_d, //* k
        size_t v_stride_b, size_t v_stride_s, size_t v_stride_h, size_t v_stride_d, //* v
        size_t o_stride_b, size_t o_stride_s, size_t o_stride_h, size_t o_stride_d, //* out
        size_t a_stride_b, size_t a_stride_h,                                       //* alibi
        size_t b_stride_b, size_t b_stride_h, size_t b_stride_q, size_t b_stride_k, //* bias
        size_t m_stride_b, size_t m_stride_h, size_t m_stride_q, size_t m_stride_k, //* mask
        const float p_dropout,
        const float softmax_scale,
        bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        const bool return_softmax,
        int seed,
        bool is_bf16,
        bool is_bhsd,
        gpuStream_t stream);

namespace jax {
namespace JAX_GPU_NAMESPACE {
namespace CutLassFa{
    static absl::StatusOr<size_t>
    DoCutlassComputeWorkspaceReserveSpaceSizes(int batch, int seq, int head, int dims) {
        auto fileName = get_so_path();
        if (fileName.empty()) {
            return absl::NotFoundError("libflash_atten_c.so was not found; set FA_SO_PATH to the DTK flash attention library");
        }
        void* handle = dlopen(fileName.c_str(), RTLD_LAZY);
        if(!handle){
            return absl::InternalError(dlerror());
        }
       mha_masked_fwd_bias_workspace_func mha_masked_fwd_bias_workspace = (mha_masked_fwd_bias_workspace_func)dlsym(handle, "mha_masked_fwd_bias_workspace");
       const char* dlsym_error = dlerror();
       if (dlsym_error) {
          dlclose(handle);
          return absl::InternalError(dlsym_error);
       }
        size_t workspace_size = mha_masked_fwd_bias_workspace(batch, seq, seq, head, head, dims, dims, false);
        dlclose(handle);
        return workspace_size;
    }

    absl::StatusOr<size_t> CutlassComputeWorkspaceReserveSpaceSizes(int batch, int seq, int head, int dims) {
        return DoCutlassComputeWorkspaceReserveSpaceSizes(batch, seq, head, dims);
    }

    static absl::Status cutlassFaFwd_(gpuStream_t stream, void** buffers, const char* opaque, size_t opaque_len) {
        auto s = UnpackDescriptor<CutlassFaDescriptor> (opaque, opaque_len);
        JAX_RETURN_IF_ERROR(s.status());
        const CutlassFaDescriptor& d = **s;
        size_t s_b = d.seq * d.head * d.dims;
        size_t s_s = d.head * d.dims;
        size_t s_h = d.dims;
        size_t s_d = 1;

        size_t bias_s_b = d.head * d.seq * d.seq;
        size_t bias_s_h = d.seq * d.seq;
        size_t bias_s_q = d.seq;
        size_t bias_s_k = 1;

        size_t mask_s_b = d.head * d.seq * d.seq;
        size_t mask_s_h = d.seq * d.seq;
        size_t mask_s_q = d.seq;
        size_t mask_s_k = 1;
        auto fileName = get_so_path();
        if (fileName.empty()) {
            return absl::NotFoundError("libflash_atten_c.so was not found; set FA_SO_PATH to the DTK flash attention library");
        }
        void* handle = dlopen(fileName.c_str(), RTLD_LAZY);
        if(!handle){
            return absl::InternalError(dlerror());
        }
        mha_masked_fwd_bias_func mha_masked_fwd_bias = (mha_masked_fwd_bias_func)dlsym(handle, "mha_masked_fwd_bias");
        const char* dlsym_error = dlerror();
        if (dlsym_error) {
            dlclose(handle);
            return absl::InternalError(dlsym_error);
        }
        mha_masked_fwd_bias(buffers[0], buffers[1], buffers[2], buffers[5],
            nullptr,                                    //* alibi
            buffers[3],                                      //* bias
            (uint8_t*)buffers[4],                                      //* mask
            nullptr,                                    //* rng_state
            buffers[6],                              //* workspace
            d.batch, d.seq, d.seq,             //* sizes
            d.head, d.head,
            d.dims, d.dims,
            s_b, s_s, s_h, s_d,                         //* q strides
            s_b, s_s, s_h, s_d,                         //* k strides
            s_b, s_s, s_h, s_d,                         //* v strides
            s_b, s_s, s_h, s_d,                         //* o strides
            1, 1,                                       //* alibi strides
            bias_s_b, bias_s_h, bias_s_q, bias_s_k,     //* bias strides
            mask_s_b, mask_s_h, mask_s_q, mask_s_k,     //* mask strides
            0.0f,                                  //* p_dropout
            1.f / std::sqrt(float(d.dims)),          //* softmax_scale
            false,                                      //* is_causal
            -1, -1,                                     //* window sizes
            0.0f,                                       //* softcap
            false,                                      //* return_softmax
            0,                                          //* seed
            true,   //* is_bf16
            false,                                    //* is_bhsd
            stream
        );
        dlclose(handle);
        return absl::OkStatus();
    }

    void cutlassFaFwd(gpuStream_t stream, void** buffers, const char* opaque, size_t opaque_len, XlaCustomCallStatus* status) {
        auto s = cutlassFaFwd_(stream, buffers, opaque, opaque_len);
        if (!s.ok()) {
            XlaCustomCallStatusSetFailure(status, std::string(s.message()).c_str(),
                                  s.message().length());
        }
    }
}
}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
