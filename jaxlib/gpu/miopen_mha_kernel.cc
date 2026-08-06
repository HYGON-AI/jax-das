#include "jaxlib/gpu/miopen_mha_kernel.h"
#include "jaxlib/gpu/gpu_kernel_helpers.h"
#include "jaxlib/kernel_helpers.h"
#include "xla/service/custom_call_status.h"
#include "jaxlib/gpu/vendor.h"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>
#include "jaxlib/gpu/handle_pool.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
namespace jax {
namespace JAX_GPU_NAMESPACE {
    std::string ErrorString(miopenStatus_t status) {
    return miopenGetErrorString(status);
    }

    template <typename T>
    std::string ErrorString(T status, const char* file, std::int64_t line,
                            const char* expr) {
    return absl::StrFormat("%s:%d: operation %s failed: %s", file, line, expr,
                            ErrorString(status));
    }

    absl::Status AsStatus(miopenStatus_t status, const char* file,
                        std::int64_t line, const char* expr) {
    if (status != miopenStatusSuccess)
        return absl::InternalError(ErrorString(status, file, line, expr));
    return absl::OkStatus();
    }
}  // namespace JAX_GPU_NAMESPACE

using MiopenHandlePool = HandlePool<miopenHandle_t, gpuStream_t>;

template <>
/*static*/ absl::StatusOr<MiopenHandlePool::Handle> MiopenHandlePool::Borrow(gpuStream_t stream) {
  MiopenHandlePool* pool = Instance();
  absl::MutexLock lock(&pool->mu_);
  miopenHandle_t handle;
  if (pool->handles_[stream].empty()) {
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(miopenCreate(&handle)));
  } else {
    handle = pool->handles_[stream].back();
    pool->handles_[stream].pop_back();
  }
  if (stream) {
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(miopenSetStream(handle, stream)));
  }
  return Handle(pool, handle, stream);
}


namespace JAX_GPU_NAMESPACE {
namespace MiopenMha{
    static absl::StatusOr<int>
    DoMiopenComputeWorkspaceReserveSpaceSizes(int batch, int seq, int head, int dims) {
        auto h = MiopenHandlePool::Borrow(/*stream=*/nullptr);
        JAX_RETURN_IF_ERROR(h.status());
        auto& handle = *h;
        miopenMhaDescriptor_t mha_desc;
        miopenProblem_t problem;
        miopenTensorDescriptor_t q_desc;
        miopenTensorDescriptor_t k_desc;
        miopenTensorDescriptor_t v_desc;
        miopenTensorDescriptor_t paddingmask_desc;
        miopenTensorDescriptor_t bias_desc;
        miopenTensorDescriptor_t o_desc;
        MIOPEN_CHECK_RETURN(miopenCreateMhaDescriptor(&mha_desc));
        MIOPEN_CHECK_RETURN(miopenSetMhaDescriptor(mha_desc, 1.0f / std::sqrt(static_cast<float>(dims))));
        MIOPEN_CHECK_RETURN(miopenCreateMhaProblem(&problem, mha_desc, miopenProblemDirectionForward));

        /**-----------------------------------------------------------build input tensor----------------------------------------------------*/
        std::vector<std::size_t> q_dims({(std::size_t)(batch), (std::size_t)(head), (std::size_t)(seq), (std::size_t)(dims)});
        std::vector<std::size_t> q_strides({(std::size_t)(head * seq * dims), (std::size_t)(dims), (std::size_t)(head * dims), (std::size_t)(1)});
        std::vector<std::size_t> k_dims({(std::size_t)(batch), (std::size_t)(head), (std::size_t)(seq), (std::size_t)(dims)});
        std::vector<std::size_t> k_strides({(std::size_t)(head * seq * dims), (std::size_t)(dims), (std::size_t)(head * dims), (std::size_t)(1)});
        std::vector<std::size_t> v_dims({(std::size_t)(batch), (std::size_t)(head), (std::size_t)(seq), (std::size_t)(dims)});
        std::vector<std::size_t> v_strides({(std::size_t)(head * seq * dims), (std::size_t)(dims), (std::size_t)(head * dims), (std::size_t)(1)});
        std::vector<std::size_t> bias_dims({(std::size_t)(batch), (std::size_t)(head), (std::size_t)(seq), (std::size_t)(seq)});
        std::vector<std::size_t> bias_strides({(std::size_t)(head * seq * seq), (std::size_t)(seq * seq), (std::size_t)(seq), (std::size_t)(1)});
        std::vector<std::size_t> paddingmask_dims({(std::size_t)(batch), (std::size_t)(head), (std::size_t)(seq), (std::size_t)(seq)});
        std::vector<std::size_t> paddingmask_strides({(std::size_t)(head * seq * seq), (std::size_t)(seq * seq), (std::size_t)(seq), (std::size_t)(1)});
        MIOPEN_CHECK_RETURN(miopenCreateTensorDescriptor(&q_desc));
        MIOPEN_CHECK_RETURN(miopenCreateTensorDescriptor(&k_desc));
        MIOPEN_CHECK_RETURN(miopenCreateTensorDescriptor(&v_desc));
        MIOPEN_CHECK_RETURN(miopenCreateTensorDescriptor(&paddingmask_desc));
        MIOPEN_CHECK_RETURN(miopenSetTensorDescriptorV2(paddingmask_desc,                 // tensor
                                                        miopenInt8,                       // tensor datatype mask must be int8
                                                        paddingmask_dims.size(),          // tensor dim
                                                        paddingmask_dims.data(),          // dim length  [b, h, s, d]
                                                        paddingmask_strides.data()));     // dim stride  [b, h, s, d]
        MIOPEN_CHECK_RETURN(miopenSetProblemTensorDescriptor(problem, miopenTensorMhaPaddingMask, paddingmask_desc));

        MIOPEN_CHECK_RETURN(miopenCreateTensorDescriptor(&bias_desc));
        MIOPEN_CHECK_RETURN(miopenSetTensorDescriptorV2(bias_desc,                        // tensor
                                                        miopen_TensorDataType,            // tensor datatype
                                                        bias_dims.size(),                 // tensor dim
                                                        bias_dims.data(),                 // dim length  [b, h, s, d]
                                                        bias_strides.data()));            // dim stride  [b, h, s, d]
        MIOPEN_CHECK_RETURN(miopenSetProblemTensorDescriptor(problem, miopenTensorMhaBias, bias_desc));

        MIOPEN_CHECK_RETURN(miopenSetTensorDescriptorV2(q_desc,                           // tensor
                                                        miopen_TensorDataType,            // tensor datatype
                                                        q_dims.size(),                    // tensor dim
                                                        q_dims.data(),                    // dim length  [b, h, s, d]
                                                        q_strides.data()));               // dim stride  [b, h, s, d]
        MIOPEN_CHECK_RETURN(miopenSetTensorDescriptorV2(k_desc,                           // tensor
                                                    miopen_TensorDataType,            // tensor datatype
                                                    k_dims.size(),                    // tensor dim
                                                    k_dims.data(),                    // dim length  [b, h, s, d]
                                                    k_strides.data()));               // dim stride  [b, h, s, d]
        MIOPEN_CHECK_RETURN(miopenSetTensorDescriptorV2(v_desc,                           // tensor
                                                        miopen_TensorDataType,            // tensor datatype
                                                        v_dims.size(),                    // tensor dim
                                                        v_dims.data(),                    // dim length  [b, h, s, d]
                                                        v_strides.data()));               // dim stride  [b, h, s, d]

        MIOPEN_CHECK_RETURN(miopenSetProblemTensorDescriptor(problem, miopenTensorMhaQ, q_desc));
        MIOPEN_CHECK_RETURN(miopenSetProblemTensorDescriptor(problem, miopenTensorMhaK, k_desc));
        MIOPEN_CHECK_RETURN(miopenSetProblemTensorDescriptor(problem, miopenTensorMhaV, v_desc));
        /**--------------------------------------------------------------set output tensor result-------------------------------------------*/
        std::vector<std::size_t> o_dims({(std::size_t)(batch), (std::size_t)(head), (std::size_t)(seq), (std::size_t)(dims)});
        std::vector<std::size_t> o_strides({(std::size_t)(head * seq * dims), (std::size_t)(dims), (std::size_t)(head * dims), (std::size_t)(1)});
        MIOPEN_CHECK_RETURN(miopenCreateTensorDescriptor(&o_desc));
        MIOPEN_CHECK_RETURN(miopenSetTensorDescriptorV2(o_desc,                   // tensor
                                                            miopen_TensorDataType,    // tensor datatype
                                                            o_dims.size(),            // tensor dim
                                                            o_dims.data(),            // dim length  [b, h, s, d]
                                                            o_strides.data()));       // dim stride  [b, h, s, d]
        MIOPEN_CHECK_RETURN(miopenSetProblemTensorDescriptor(problem, miopenTensorMhaO, o_desc));
        /*----------------------------------------------------------tensor args ------------------------------------------------------------*/
        std::vector<miopenTensorArgument_t> tensor_args;
        tensor_args.emplace_back(miopenTensorArgument_t{miopenTensorMhaQ, &q_desc, (void*)123});
        tensor_args.emplace_back(miopenTensorArgument_t{miopenTensorMhaK, &k_desc, (void*)123});
        tensor_args.emplace_back(miopenTensorArgument_t{miopenTensorMhaV, &v_desc, (void*)123});
        tensor_args.emplace_back(miopenTensorArgument_t{miopenTensorMhaBias, &bias_desc, (void*)123});
        tensor_args.emplace_back(miopenTensorArgument_t{miopenTensorMhaPaddingMask, &paddingmask_desc, (void*)123});
        tensor_args.emplace_back(miopenTensorArgument_t{miopenTensorMhaO, &o_desc, (void*)123});
        /*-------------------------------------------------------find 2 solution solution---------------------------------------------------*/
        std::size_t found;
        std::vector<miopenSolution_t> solutions(1);
        MIOPEN_CHECK_RETURN(miopenFindSolutions(handle.get(), problem, nullptr, solutions.data(), &found, solutions.size()));
        if (found == 0) {
            return absl::InternalError("MIOpen MHA did not find a solution.");
        }
        solutions.resize(found);
        size_t workspace_size = 0;
        auto solution = solutions[0]; // solution
        MIOPEN_CHECK_RETURN(miopenGetSolutionWorkspaceSize(solution, &workspace_size));
        // Round up to nearest multiples of 4 so we can return them as f32 arrays.
        workspace_size += (4 - workspace_size % 4) % 4;
        MIOPEN_CHECK_RETURN(miopenDestroyTensorDescriptor(q_desc));
        MIOPEN_CHECK_RETURN(miopenDestroyTensorDescriptor(k_desc));
        MIOPEN_CHECK_RETURN(miopenDestroyTensorDescriptor(v_desc));
        MIOPEN_CHECK_RETURN(miopenDestroyTensorDescriptor(o_desc));
        MIOPEN_CHECK_RETURN(miopenDestroyTensorDescriptor(paddingmask_desc));
        MIOPEN_CHECK_RETURN(miopenDestroyTensorDescriptor(bias_desc));
        MIOPEN_CHECK_RETURN(miopenDestroyProblem(problem));
        return workspace_size;
    }

    absl::StatusOr<int> MiopenComputeWorkspaceReserveSpaceSizes(int batch, int seq, int head, int dims) {
        return DoMiopenComputeWorkspaceReserveSpaceSizes(batch, seq, head, dims);
    }

    static absl::Status miOpenMhaFwd_(gpuStream_t stream, void** buffers, const char* opaque, size_t opaque_len) {
        auto s = UnpackDescriptor<MiopenMhaDescriptor> (opaque, opaque_len);
        JAX_RETURN_IF_ERROR(s.status());
        miopenMhaDescriptor_t mha_desc;
        miopenProblem_t problem;
        miopenTensorDescriptor_t q_desc;
        miopenTensorDescriptor_t k_desc;
        miopenTensorDescriptor_t v_desc;
        miopenTensorDescriptor_t paddingmask_desc;
        miopenTensorDescriptor_t bias_desc;
        miopenTensorDescriptor_t o_desc;
        const MiopenMhaDescriptor& d = **s;
        auto h = MiopenHandlePool::Borrow(stream);
        JAX_RETURN_IF_ERROR(h.status());
        auto& handle = *h;
        MIOPEN_CHECK_RETURN(miopenCreateMhaDescriptor(&mha_desc));
        MIOPEN_CHECK_RETURN(miopenSetMhaDescriptor(mha_desc, 1.0f / std::sqrt(static_cast<float>(d.dims))));
        MIOPEN_CHECK_RETURN(miopenCreateMhaProblem(&problem, mha_desc, miopenProblemDirectionForward));

        /**-----------------------------------------------------------build input tensor----------------------------------------------------*/
        std::vector<std::size_t> q_dims({(std::size_t)(d.batch), (std::size_t)(d.head), (std::size_t)(d.seq), (std::size_t)(d.dims)});
        std::vector<std::size_t> q_strides({(std::size_t)(d.head * d.seq * d.dims), (std::size_t)(d.dims), (std::size_t)(d.head * d.dims), (std::size_t)(1)});
        std::vector<std::size_t> k_dims({(std::size_t)(d.batch), (std::size_t)(d.head), (std::size_t)(d.seq), (std::size_t)(d.dims)});
        std::vector<std::size_t> k_strides({(std::size_t)(d.head * d.seq * d.dims), (std::size_t)(d.dims), (std::size_t)(d.head * d.dims), (std::size_t)(1)});
        std::vector<std::size_t> v_dims({(std::size_t)(d.batch), (std::size_t)(d.head), (std::size_t)(d.seq), (std::size_t)(d.dims)});
        std::vector<std::size_t> v_strides({(std::size_t)(d.head * d.seq * d.dims), (std::size_t)(d.dims), (std::size_t)(d.head * d.dims), (std::size_t)(1)});
        std::vector<std::size_t> bias_dims({(std::size_t)(d.batch), (std::size_t)(d.head), (std::size_t)(d.seq), (std::size_t)(d.seq)});
        std::vector<std::size_t> bias_strides({(std::size_t)(d.head * d.seq * d.seq), (std::size_t)(d.seq * d.seq), (std::size_t)(d.seq), (std::size_t)(1)});
        std::vector<std::size_t> paddingmask_dims({(std::size_t)(d.batch), (std::size_t)(d.head), (std::size_t)(d.seq), (std::size_t)(d.seq)});
        std::vector<std::size_t> paddingmask_strides({(std::size_t)(d.head * d.seq * d.seq), (std::size_t)(d.seq * d.seq), (std::size_t)(d.seq), (std::size_t)(1)});
        MIOPEN_CHECK_RETURN(miopenCreateTensorDescriptor(&q_desc));
        MIOPEN_CHECK_RETURN(miopenCreateTensorDescriptor(&k_desc));
        MIOPEN_CHECK_RETURN(miopenCreateTensorDescriptor(&v_desc));
        MIOPEN_CHECK_RETURN(miopenCreateTensorDescriptor(&paddingmask_desc));
        MIOPEN_CHECK_RETURN(miopenSetTensorDescriptorV2(paddingmask_desc,                 // tensor
                                                    miopenInt8,                       // tensor datatype mask must be int8
                                                    paddingmask_dims.size(),          // tensor dim
                                                    paddingmask_dims.data(),          // dim length  [b, h, s, d]
                                                    paddingmask_strides.data()));     // dim stride  [b, h, s, d]
        MIOPEN_CHECK_RETURN(miopenSetProblemTensorDescriptor(problem, miopenTensorMhaPaddingMask, paddingmask_desc));

        MIOPEN_CHECK_RETURN(miopenCreateTensorDescriptor(&bias_desc));
        MIOPEN_CHECK_RETURN(miopenSetTensorDescriptorV2(bias_desc,                        // tensor
                                                    miopen_TensorDataType,            // tensor datatype
                                                    bias_dims.size(),                 // tensor dim
                                                    bias_dims.data(),                 // dim length  [b, h, s, d]
                                                    bias_strides.data()));            // dim stride  [b, h, s, d]
        MIOPEN_CHECK_RETURN(miopenSetProblemTensorDescriptor(problem, miopenTensorMhaBias, bias_desc));

        MIOPEN_CHECK_RETURN(miopenSetTensorDescriptorV2(q_desc,                           // tensor
                                                    miopen_TensorDataType,            // tensor datatype
                                                    q_dims.size(),                    // tensor dim
                                                    q_dims.data(),                    // dim length  [b, h, s, d]
                                                    q_strides.data()));               // dim stride  [b, h, s, d]
        MIOPEN_CHECK_RETURN(miopenSetTensorDescriptorV2(k_desc,                           // tensor
                                                   miopen_TensorDataType,            // tensor datatype
                                                   k_dims.size(),                    // tensor dim
                                                   k_dims.data(),                    // dim length  [b, h, s, d]
                                                   k_strides.data()));               // dim stride  [b, h, s, d]
        MIOPEN_CHECK_RETURN(miopenSetTensorDescriptorV2(v_desc,                           // tensor
                                                    miopen_TensorDataType,            // tensor datatype
                                                    v_dims.size(),                    // tensor dim
                                                    v_dims.data(),                    // dim length  [b, h, s, d]
                                                    v_strides.data()));               // dim stride  [b, h, s, d]

        MIOPEN_CHECK_RETURN(miopenSetProblemTensorDescriptor(problem, miopenTensorMhaQ, q_desc));
        MIOPEN_CHECK_RETURN(miopenSetProblemTensorDescriptor(problem, miopenTensorMhaK, k_desc));
        MIOPEN_CHECK_RETURN(miopenSetProblemTensorDescriptor(problem, miopenTensorMhaV, v_desc));

        /**--------------------------------------------------------------set output tensor result-------------------------------------------*/
        std::vector<std::size_t> o_dims({(std::size_t)(d.batch), (std::size_t)(d.head), (std::size_t)(d.seq), (std::size_t)(d.dims)});
        std::vector<std::size_t> o_strides({(std::size_t)(d.head * d.seq * d.dims), (std::size_t)(d.dims), (std::size_t)(d.head * d.dims), (std::size_t)(1)});
        MIOPEN_CHECK_RETURN(miopenCreateTensorDescriptor(&o_desc));
        MIOPEN_CHECK_RETURN(miopenSetTensorDescriptorV2(o_desc,                   // tensor
                                                        miopen_TensorDataType,    // tensor datatype
                                                        o_dims.size(),            // tensor dim
                                                        o_dims.data(),            // dim length  [b, h, s, d]
                                                        o_strides.data()));       // dim stride  [b, h, s, d]
        MIOPEN_CHECK_RETURN(miopenSetProblemTensorDescriptor(problem, miopenTensorMhaO, o_desc));
        /*----------------------------------------------------------tensor args ------------------------------------------------------------*/
        std::vector<miopenTensorArgument_t> tensor_args;
        tensor_args.emplace_back(miopenTensorArgument_t{miopenTensorMhaQ, &q_desc, buffers[0]});
        tensor_args.emplace_back(miopenTensorArgument_t{miopenTensorMhaK, &k_desc, buffers[1]});
        tensor_args.emplace_back(miopenTensorArgument_t{miopenTensorMhaV, &v_desc, buffers[2]});
        tensor_args.emplace_back(miopenTensorArgument_t{miopenTensorMhaBias, &bias_desc, buffers[3]});
        tensor_args.emplace_back(miopenTensorArgument_t{miopenTensorMhaPaddingMask, &paddingmask_desc, buffers[4]});
        tensor_args.emplace_back(miopenTensorArgument_t{miopenTensorMhaO, &o_desc, buffers[5]});
        /*-------------------------------------------------------find 2 solution solution---------------------------------------------------*/
        std::size_t found;
        std::vector<miopenSolution_t> solutions(1);
        MIOPEN_CHECK_RETURN(miopenFindSolutions(handle.get(), problem, nullptr, solutions.data(), &found, solutions.size()));
        if (found == 0) {
            return absl::InternalError("MIOpen MHA did not find a solution.");
        }
        solutions.resize(found);
        auto solution = solutions[0]; // solution
        MIOPEN_CHECK_RETURN(miopenRunSolution(handle.get(), solution, tensor_args.size(), tensor_args.data(), buffers[6], d.workspace_size));
        MIOPEN_CHECK_RETURN(miopenDestroyTensorDescriptor(q_desc));
        MIOPEN_CHECK_RETURN(miopenDestroyTensorDescriptor(k_desc));
        MIOPEN_CHECK_RETURN(miopenDestroyTensorDescriptor(v_desc));
        MIOPEN_CHECK_RETURN(miopenDestroyTensorDescriptor(o_desc));
        MIOPEN_CHECK_RETURN(miopenDestroyTensorDescriptor(paddingmask_desc));
        MIOPEN_CHECK_RETURN(miopenDestroyTensorDescriptor(bias_desc));
        MIOPEN_CHECK_RETURN(miopenDestroyProblem(problem));
        return absl::OkStatus();
    }

    void miOpenMhaFwd(gpuStream_t stream, void** buffers, const char* opaque, size_t opaque_len, XlaCustomCallStatus* status) {
        auto s = miOpenMhaFwd_(stream, buffers, opaque, opaque_len);
        if (!s.ok()) {
            XlaCustomCallStatusSetFailure(status, std::string(s.message()).c_str(),
                                  s.message().length());
        }
    }
}
}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
