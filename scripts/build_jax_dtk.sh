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
# Externally set PYTHON_VERSION / PYTHON_BIN take precedence; otherwise
# PYTHON_BIN defaults to `python`, and its version is detected from it.
PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ -z "${PYTHON_VERSION:-}" ]]; then
  PYTHON_VERSION="$("${PYTHON_BIN}" --version | awk '{split($2,v,"."); print v[1]"."v[2]}')"
fi
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/dist}"
# Plain linux_<arch> wheels built against the current image's glibc.
LINUX_DIR="${OUT_DIR}/linux"
BUILD_MANYLINUX="${BUILD_MANYLINUX:-1}"
REPAIR_ONLY=0
ARCH="$(uname -m)"
DTK_VERSION="$(cat "${DTK_DIR}/.info/rocm_version")"
rocm_version=$(cat "${DTK_DIR}/.info/rocm_version" | cut -d "-" -f 1 | tr -d '.')
DTK_WHEEL_VERSION_SUFFIX="${DTK_WHEEL_VERSION_SUFFIX:-+das.opt1.dtk${rocm_version}}"
# DTK HIP/DCC 25.10 accepts these targets for precompiled plugin kernels.
# gfx92a is still allowed in XLA runtime codegen, but hipcc rejects it
# as a build target in this DTK release.
TARGETS="${TARGETS:-gfx906,gfx926,gfx928,gfx936,gfx938}"
ROCM_CODEGEN_CONFIG="hcu"

while (($#)); do
  case "$1" in
    -h|--help)
      echo "Usage: $0 [--hcu] [--no-manylinux | --repair-only]"
      echo "  --hcu           Build with HCU ROCm codegen backend (default)."
      echo "  --no-manylinux  Only build linux_${ARCH} wheels into \${OUT_DIR}/linux."
      echo "  --repair-only   Skip the build and regenerate the manylinux wheels from"
      echo "                  the wheels already in \${OUT_DIR}/linux. Wheels left in"
      echo "                  \${OUT_DIR} by older builds of this script are moved there."
      echo "                  Run it in the image that built the wheels."
      echo
      echo "Outputs:"
      echo "  \${OUT_DIR}/linux  linux_${ARCH} wheels for the build image's glibc."
      echo "  \${OUT_DIR}        jax and manylinux wheels repaired by auditwheel,"
      echo "                    with DTK libraries left to the installed DTK."
      echo
      echo "Environment overrides:"
      echo "  JAX_DIR     JAX source tree. Defaults to this script directory."
      echo "  XLA_DIR     Local XLA source tree, overriding the pinned revision"
      echo "              in third_party/xla/revision.bzl. Optional."
      echo "  DTK_DIR     DTK installation. Defaults to /opt/dtk."
      echo "  AILLVM_DIR  HCU LLVM installation. Defaults to \${DTK_DIR}/aillvm."
      echo "  PYTHON_VERSION"
      echo "              Python version such as 3.11. Defaults to the version of"
      echo "              \${PYTHON_BIN}."
      echo "  PYTHON_BIN  Python interpreter. Defaults to 'python'."
      echo "  DTK_WHEEL_VERSION_SUFFIX"
      echo "              Wheel local version suffix. Defaults to +das.opt1.dtk2604."
      echo "  OUT_DIR     Wheel output directory. Defaults to \${JAX_DIR}/dist."
      exit 0
      ;;
    --hcu)
      ROCM_CODEGEN_CONFIG="hcu"
      ;;
    --no-manylinux)
      BUILD_MANYLINUX=0
      ;;
    --repair-only)
      REPAIR_ONLY=1
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 [--hcu] [--no-manylinux | --repair-only]" >&2
      exit 2
      ;;
  esac
  shift
done

if [[ "${REPAIR_ONLY}" == 1 && "${BUILD_MANYLINUX}" != 1 ]]; then
  echo "--repair-only cannot be combined with --no-manylinux or BUILD_MANYLINUX=0." >&2
  exit 2
fi

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

if [[ "${REPAIR_ONLY}" != 1 && ! -x "${AILLVM_DIR}/bin/clang" ]]; then
  echo "HCU LLVM clang not found: ${AILLVM_DIR}/bin/clang" >&2
  echo "Set AILLVM_DIR=/path/to/aillvm." >&2
  exit 1
fi

set +u
source "${DTK_DIR}/env.sh"
set -u
unset PYTHONPATH

# Check the packaging tools up front instead of failing after the long build.
if ! "${PYTHON_BIN}" -m wheel version >/dev/null 2>&1; then
  echo "Python package 'wheel' not found for ${PYTHON_BIN}." >&2
  echo "Install it with: ${PYTHON_BIN} -m pip install wheel" >&2
  exit 1
fi
if [[ "${BUILD_MANYLINUX}" == 1 ]]; then
  if ! "${PYTHON_BIN}" -m auditwheel --version >/dev/null 2>&1 \
      || ! command -v patchelf >/dev/null 2>&1; then
    echo "auditwheel and patchelf are required for the manylinux wheels." >&2
    echo "Install them with: ${PYTHON_BIN} -m pip install auditwheel patchelf" >&2
    echo "or pass --no-manylinux to only build linux_${ARCH} wheels." >&2
    exit 1
  fi
fi

mkdir -p "${OUT_DIR}" "${LINUX_DIR}"
cd "${JAX_DIR}"

build_wheels() {
  # Only repair the wheels produced by this build.
  rm -f "${LINUX_DIR}"/*.whl

  echo "ROCm codegen config: ${ROCM_CODEGEN_CONFIG}"
  echo "JAX source: ${JAX_DIR}"
  if [[ -n "${XLA_DIR:-}" ]]; then
    echo "XLA source: ${XLA_DIR} (override)"
  else
    echo "XLA source: pinned revision from third_party/xla/revision.bzl"
  fi
  echo "LLVM toolchain: ${AILLVM_DIR}"
  echo "Python: ${PYTHON_BIN} ($("${PYTHON_BIN}" -V 2>&1))"
  echo "DTK version: ${DTK_VERSION}"
  echo "Wheel version suffix: ${DTK_WHEEL_VERSION_SUFFIX}"
  # Bazel may reuse cached Triton from previous builds.
  # Remove stale Triton cache to make sure xla-das Triton patches are applied.
  echo "Cleaning stale Bazel Triton cache..."
  shopt -s nullglob
  local triton_cache_entries=(
    "${HOME:-/root}"/.cache/bazel/_bazel_"$(id -un)"/*/external/triton*
    "${HOME:-/root}"/.cache/bazel/_bazel_"$(id -un)"/*/external/@triton.marker
  )
  shopt -u nullglob
  if ((${#triton_cache_entries[@]})); then
    rm -rf "${triton_cache_entries[@]}"
  else
    echo "No stale Triton cache found."
  fi
  # Only override the pinned XLA revision when a local tree was requested.
  local XLA_ARGS=()
  if [[ -n "${XLA_DIR:-}" ]]; then
    XLA_ARGS+=("--local_xla_path=${XLA_DIR}")
  fi

  # build/rocm/rocm.bazelrc turns on --offload-compress, but the Rocky and Ubuntu
  # DTK builds do not decompress each other's kernels (Z_DATA_ERROR at load
  # time), so keep kernels uncompressed to make the wheels portable. build.py
  # appends --bazel_options after --config=rocm, so this empty value wins.
  "${PYTHON_BIN}" build/build.py build \
    --wheels=jax,jaxlib,jax-rocm-plugin,jax-rocm-pjrt \
    --python_version="${PYTHON_VERSION}" \
    --rocm_path="${DTK_DIR}" \
    --rocm_version=60 \
    --rocm_amdgpu_targets="${TARGETS}" \
    ${XLA_ARGS[@]+"${XLA_ARGS[@]}"} \
    --clang_path="${AILLVM_DIR}/bin/clang" \
    --output_path="${LINUX_DIR}" \
    --bazel_options="--action_env=HIPCC_COMPILE_FLAGS_APPEND=" \
    --bazel_options="--repo_env=ML_WHEEL_TYPE=release" \
    --bazel_options="--repo_env=ML_WHEEL_VERSION_SUFFIX=${DTK_WHEEL_VERSION_SUFFIX}" \
    --bazel_options="--//jaxlib/tools:jaxlib_git_hash=$(git rev-parse HEAD)" \
    --bazel_options="--config=${ROCM_CODEGEN_CONFIG}" \
    --verbose
}

if [[ "${REPAIR_ONLY}" != 1 ]]; then
  build_wheels
fi

shopt -s nullglob

if [[ "${REPAIR_ONLY}" == 1 ]]; then
  echo "========== Skipping build (--repair-only) =========="
  # Older builds of this script wrote wheels straight to ${OUT_DIR} with the
  # tag hardcoded by jaxlib/jax.bzl; move them over so they get retagged.
  legacy_wheels=("${OUT_DIR}"/*-manylinux_2_27_"${ARCH}".whl)
  if (( ${#legacy_wheels[@]} > 0 )); then
    echo "Moving ${#legacy_wheels[@]} wheel(s) from ${OUT_DIR} to ${LINUX_DIR}"
    mv -f "${legacy_wheels[@]}" "${LINUX_DIR}/"
  fi
fi

# jaxlib/jax.bzl hardcodes a manylinux_2_27 platform tag, but these wheels are
# built with the DTK toolchain against the build image's glibc. Retag them as
# plain linux_<arch> wheels; auditwheel derives the real manylinux tag below.
echo "========== Retagging wheels as linux_${ARCH} =========="
for wheel in "${LINUX_DIR}"/*manylinux*.whl; do
  "${PYTHON_BIN}" -m wheel tags --platform-tag "linux_${ARCH}" --remove "${wheel}"
done

linux_wheels=("${LINUX_DIR}"/*-linux_"${ARCH}".whl)
pure_wheels=("${LINUX_DIR}"/*-none-any.whl)
if (( ${#linux_wheels[@]} == 0 )); then
  echo "ERROR: no linux_${ARCH} wheels found in ${LINUX_DIR}." >&2
  if [[ "${REPAIR_ONLY}" == 1 ]]; then
    echo "Run this script without --repair-only to build them first." >&2
  fi
  exit 1
fi
echo "linux_${ARCH} wheels written to: ${LINUX_DIR}"

if [[ "${BUILD_MANYLINUX}" != 1 ]]; then
  exit 0
fi

# Generate the manylinux wheels.
echo "========== Generating manylinux wheels =========="
glibc_version_info="$(getconf GNU_LIBC_VERSION 2>/dev/null || true)"
if [[ ! "${glibc_version_info}" =~ ([0-9]+\.[0-9]+) ]]; then
  glibc_version_info="$(ldd --version 2>&1 || true)"
fi
if [[ "${glibc_version_info}" =~ ([0-9]+\.[0-9]+) ]]; then
  glibc_version="${BASH_REMATCH[1]}"
else
  echo "ERROR: unable to determine glibc version for the manylinux tag." >&2
  exit 1
fi
PLATFORM="manylinux_${glibc_version/./_}_${ARCH}"

# Libraries provided by the installed DTK (and the system libraries it pulls
# in) stay external so the wheels use the DTK and driver of the target host.
# One wildcard set covers DTK 26.04 and 26.10.
selected_auditwheel_excludes=(
  'libgalaxyhip.so*'
  'libamdhip64.so*'
  'libhiprtc.so*'
  'libhsa-runtime64.so*'
  'libhsakmt.so*'
  'libamd_comgr.so*'
  'libMIOpen.so*'
  'librccl.so*'
  'libhipblas.so*'
  'libhipblaslt.so*'
  'librocblas.so*'
  'libhipfft.so*'
  'librocfft.so*'
  'librocfft-device-*.so*'
  'libhiprand.so*'
  'librocrand.so*'
  'libhipsolver.so*'
  'librocsolver.so*'
  'libhipsparse.so*'
  'librocsparse.so*'
  'libhipnn.so*'
  'libhipprof*'
  'librocm_smi64.so*'
  'libroctracer64.so*'
  'libroctx64.so*'
  'librocprofiler*'
  'libomp.so*'
  'libnuma.so*'
  'libibverbs*'
  'libnl*'
  'libmpi*'
  'libpimx*'
)

auditwheel_excludes=()
for library in "${selected_auditwheel_excludes[@]}"; do
  auditwheel_excludes+=(--exclude "${library}")
done

for wheel in "${linux_wheels[@]}"; do
  echo "Repairing $(basename "${wheel}") for ${PLATFORM}"
  "${PYTHON_BIN}" -m auditwheel show "${wheel}" || true
  # --only-plat keeps a single ${PLATFORM} tag; without it auditwheel also
  # adds any lower manylinux tag a wheel happens to satisfy.
  "${PYTHON_BIN}" -m auditwheel repair \
    --plat "${PLATFORM}" \
    --only-plat \
    "${auditwheel_excludes[@]}" \
    --wheel-dir "${OUT_DIR}" \
    "${wheel}"
done

# The pure-Python jax wheel needs no repair.
if (( ${#pure_wheels[@]} > 0 )); then
  cp -f "${pure_wheels[@]}" "${OUT_DIR}/"
fi

echo "========== Build complete =========="
echo "manylinux wheels written to: ${OUT_DIR}"
ls -1 "${OUT_DIR}"/*.whl
