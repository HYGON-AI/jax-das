#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JAX_DIR="${JAX_DIR:-${ROOT_DIR}}"
# The DAS XLA fork is pinned in third_party/xla/revision.bzl and fetched by Bazel
# as an external repository, so building only needs the jax-das checkout.
# Set XLA_DIR to a local XLA tree to override the pinned revision (development).
DTK_DIR="${DTK_DIR:-/opt/dtk}"
AILLVM_DIR="${AILLVM_DIR:-${DTK_DIR}/aillvm}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/dist}"
DTK_VERSION="${DTK_VERSION:-26.04}"
DTK_WHEEL_VERSION_SUFFIX="${DTK_WHEEL_VERSION_SUFFIX:-+das.opt1.dtk$(printf '%s' "${DTK_VERSION}" | tr -d '.')}"
# DTK HIP/DCC 25.10 accepts these targets for precompiled plugin kernels.
# gfx92a is still allowed in XLA runtime codegen, but hipcc rejects it
# as a build target in this DTK release.
TARGETS="${TARGETS:-gfx906,gfx926,gfx928,gfx936,gfx938}"
ROCM_CODEGEN_CONFIG="hcu"

while (($#)); do
  case "$1" in
    -h|--help)
      echo "Usage: $0 [--hcu]"
      echo "  --hcu   Build with HCU ROCm codegen backend (default)."
      echo
      echo "Environment overrides:"
      echo "  JAX_DIR     JAX source tree. Defaults to this script directory."
      echo "  XLA_DIR     Local XLA source tree, overriding the pinned revision"
      echo "              in third_party/xla/revision.bzl. Optional."
      echo "  DTK_DIR     DTK installation. Defaults to /opt/dtk."
      echo "  AILLVM_DIR  HCU LLVM installation. Defaults to \${DTK_DIR}/aillvm."
      echo "  DTK_WHEEL_VERSION_SUFFIX"
      echo "              Wheel local version suffix. Defaults to +das.opt1.dtk2604."
      exit 0
      ;;
    --hcu)
      ROCM_CODEGEN_CONFIG="hcu"
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 [--hcu]" >&2
      exit 2
      ;;
  esac
  shift
done

if [[ ! -f "${DTK_DIR}/env.sh" ]]; then
  echo "DTK env not found: ${DTK_DIR}/env.sh" >&2
  exit 1
fi

if [[ "${JAX_DIR}" != /* ]]; then
  JAX_DIR="${ROOT_DIR}/${JAX_DIR}"
fi
JAX_DIR="$(cd "${JAX_DIR}" && pwd)"

if [[ -n "${XLA_DIR:-}" && "${XLA_DIR}" != /* ]]; then
  XLA_DIR="${JAX_DIR}/${XLA_DIR}"
fi
if [[ -n "${XLA_DIR:-}" ]]; then
  if [[ ! -d "${XLA_DIR}" ]]; then
    echo "XLA source not found: ${XLA_DIR}" >&2
    echo "Unset XLA_DIR to build against the pinned revision in" >&2
    echo "third_party/xla/revision.bzl." >&2
    exit 1
  fi
  XLA_DIR="$(cd "${XLA_DIR}" && pwd)"
fi

if [[ ! -x "${AILLVM_DIR}/bin/clang" ]]; then
  echo "HCU LLVM clang not found: ${AILLVM_DIR}/bin/clang" >&2
  echo "Set AILLVM_DIR=/path/to/aillvm." >&2
  exit 1
fi

set +u
source "${DTK_DIR}/env.sh"
set -u
unset PYTHONPATH

mkdir -p "${OUT_DIR}"
cd "${JAX_DIR}"

echo "ROCm codegen config: ${ROCM_CODEGEN_CONFIG}"
echo "JAX source: ${JAX_DIR}"
if [[ -n "${XLA_DIR:-}" ]]; then
  echo "XLA source: ${XLA_DIR} (override)"
else
  echo "XLA source: pinned revision from third_party/xla/revision.bzl"
fi
echo "LLVM toolchain: ${AILLVM_DIR}"
echo "DTK version: ${DTK_VERSION}"
echo "Wheel version suffix: ${DTK_WHEEL_VERSION_SUFFIX}"
# Bazel may reuse cached Triton from previous builds.
# Remove stale Triton cache to make sure xla-das Triton patches are applied.
echo "Cleaning stale Bazel Triton cache..."
triton_cache_dirs=("${HOME:-/root}"/.cache/bazel/_bazel_"$(id -un)"/*/external/triton*)
[[ -e "${triton_cache_dirs[0]}" ]] && rm -rf "${triton_cache_dirs[@]}" || echo "No stale Triton cache found."

# Only override the pinned XLA revision when a local tree was requested.
XLA_ARGS=()
if [[ -n "${XLA_DIR:-}" ]]; then
  XLA_ARGS+=("--local_xla_path=${XLA_DIR}")
fi

"${PYTHON_BIN}" build/build.py build \
  --wheels=jax,jaxlib,jax-rocm-plugin,jax-rocm-pjrt \
  --python_version=3.11 \
  --rocm_path="${DTK_DIR}" \
  --rocm_version=60 \
  --rocm_amdgpu_targets="${TARGETS}" \
  ${XLA_ARGS[@]+"${XLA_ARGS[@]}"} \
  --clang_path="${AILLVM_DIR}/bin/clang" \
  --output_path="${OUT_DIR}" \
  --bazel_options="--repo_env=ML_WHEEL_TYPE=release" \
  --bazel_options="--repo_env=ML_WHEEL_VERSION_SUFFIX=${DTK_WHEEL_VERSION_SUFFIX}" \
  --bazel_options="--//jaxlib/tools:jaxlib_git_hash=$(git rev-parse HEAD)" \
  --bazel_options="--config=${ROCM_CODEGEN_CONFIG}" \
  --verbose

echo "Wheels written to: ${OUT_DIR}"
