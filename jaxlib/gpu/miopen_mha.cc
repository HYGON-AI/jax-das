#include <cstddef>
#include <stdexcept>
#include <utility>
#include "nanobind/stl/pair.h"
#include "nanobind/nanobind.h"
#include "jaxlib/gpu/miopen_mha_kernel.h"
#include "jaxlib/gpu/vendor.h"
#include "jaxlib/kernel_nanobind_helpers.h"
#include "jaxlib/absl_status_casters.h"
namespace jax {
namespace JAX_GPU_NAMESPACE {
namespace {
namespace nb = nanobind;
nb::bytes BuildMiopenMhaDescriptor(int batch, int seq, int head, int dims, int data_type, int layout, int workspace_size) {
  return PackDescriptor(MiopenMha::MiopenMhaDescriptor{
      batch, seq, head, dims, data_type, layout, workspace_size
  });
}

nb::dict Registrations() {
  nb::dict dict;
  dict["miOpen_mha_fwd_kernel"] = EncapsulateFunction(MiopenMha::miOpenMhaFwd);
  return dict;
}

NB_MODULE(miopen_mha, m) {
  m.def("registrations", &Registrations);
  m.def("build_miopen_mha_descriptor", &BuildMiopenMhaDescriptor);
  m.def("compute_miopen_workspace_reserve_space_sizes",
        ValueOrThrowWrapper(MiopenMha::MiopenComputeWorkspaceReserveSpaceSizes));
}
}  // namespace
}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
