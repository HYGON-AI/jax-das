// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <stdexcept>
#include <utility>
#include "nanobind/stl/pair.h"
#include "nanobind/nanobind.h"
#include "jaxlib/gpu/cutlass_fa_kernel.h"
#include "jaxlib/gpu/vendor.h"
#include "jaxlib/kernel_nanobind_helpers.h"
#include "jaxlib/absl_status_casters.h"
namespace jax {
namespace JAX_GPU_NAMESPACE {
namespace {
namespace nb = nanobind;
nb::bytes BuildCutlassFaDescriptor(int batch, int seq, int head, int dims, int data_type, int layout, int workspace_size) {
  return PackDescriptor(CutLassFa::CutlassFaDescriptor{
      batch, seq, head, dims, data_type, layout, workspace_size
  });
}

nb::dict Registrations() {
  nb::dict dict;
  dict["cutlass_fa_fwd_kernel"] = EncapsulateFunction(CutLassFa::cutlassFaFwd);
  return dict;
}

NB_MODULE(cutlass_fa, m) {
  m.def("registrations", &Registrations);
  m.def("build_cutlass_fa_descriptor", &BuildCutlassFaDescriptor);
  m.def("compute_cutlass_workspace_reserve_space_sizes",
        ValueOrThrowWrapper(CutLassFa::CutlassComputeWorkspaceReserveSpaceSizes));
}
}  // namespace
}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
