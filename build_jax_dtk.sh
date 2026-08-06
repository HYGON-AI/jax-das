#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JAX_DIR="${JAX_DIR:-${ROOT_DIR}}"
# Keep JAX and XLA as sibling source trees by default:
#   /path/to/work/jax
#   /path/to/work/xla
if [[ -z "${XLA_DIR:-}" ]]; then
  XLA_DIR="../xla"
  XLA_DIR_FROM_DEFAULT=1
else
  XLA_DIR_FROM_DEFAULT=0
fi
DTK_DIR="${DTK_DIR:-/opt/dtk}"
AILLVM_DIR="${AILLVM_DIR:-${DTK_DIR}/aillvm}"
TRITON_DIR="${TRITON_DIR:-../triton}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/dist}"
DTK_WHEEL_VERSION_SUFFIX="${DTK_WHEEL_VERSION_SUFFIX:-+das.opt1.dtk2604}"
# DTK HIP/DCC 25.10 accepts these targets for precompiled plugin kernels.
# gfx92a is still allowed in XLA runtime codegen, but hipcc rejects it
# as a build target in this DTK release.
TARGETS="${TARGETS:-gfx906,gfx926,gfx928,gfx936,gfx938}"
ROCM_CODEGEN_CONFIG="hcu"

while (($#)); do
  case "$1" in
    -h|--help)
      echo "Usage: $0 [--hcu|--gcvm]"
      echo "  --hcu   Build with DTK HCU ROCm codegen backend (default)."
      echo "  --gcvm  Build with DTK GCVM ROCm codegen backend."
      echo
      echo "Environment overrides:"
      echo "  JAX_DIR     JAX source tree. Defaults to this script directory."
      echo "  XLA_DIR     XLA source tree. Defaults to ../xla relative to JAX_DIR."
      echo "  DTK_DIR     DTK installation. Defaults to /opt/dtk."
      echo "  AILLVM_DIR  HCU LLVM installation. Defaults to \${DTK_DIR}/aillvm."
      echo "  TRITON_DIR  HCU Triton source with Bazel metadata. Defaults to ../triton."
      echo "  DTK_WHEEL_VERSION_SUFFIX"
      echo "              Wheel local version suffix. Defaults to +das.opt1.dtk2604."
      exit 0
      ;;
    --gcvm)
      ROCM_CODEGEN_CONFIG="gcvm"
      ;;
    --hcu)
      ROCM_CODEGEN_CONFIG="hcu"
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 [--hcu|--gcvm]" >&2
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

if [[ "${XLA_DIR}" != /* ]]; then
  XLA_DIR="${JAX_DIR}/${XLA_DIR}"
fi
if [[ ! -d "${XLA_DIR}" && "${XLA_DIR_FROM_DEFAULT}" == 1 ]]; then
  LEGACY_XLA_DIR="${JAX_DIR}/../xla-jax-0.10.0"
  if [[ -d "${LEGACY_XLA_DIR}" ]]; then
    XLA_DIR="${LEGACY_XLA_DIR}"
  fi
fi
if [[ ! -d "${XLA_DIR}" ]]; then
  echo "XLA source not found: ${XLA_DIR}" >&2
  echo "Set XLA_DIR=/path/to/xla if your XLA checkout is elsewhere." >&2
  exit 1
fi
XLA_DIR="$(cd "${XLA_DIR}" && pwd)"

if [[ "${TRITON_DIR}" != /* ]]; then
  TRITON_DIR="${JAX_DIR}/${TRITON_DIR}"
fi
if [[ ! -f "${TRITON_DIR}/BUILD" ]]; then
  echo "Bazel-enabled Triton source not found: ${TRITON_DIR}" >&2
  echo "Set TRITON_DIR=/path/to/triton." >&2
  exit 1
fi
TRITON_DIR="$(cd "${TRITON_DIR}" && pwd)"

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
echo "XLA source: ${XLA_DIR}"
echo "Triton source: ${TRITON_DIR}"
echo "LLVM toolchain: ${AILLVM_DIR}"
echo "Wheel version suffix: ${DTK_WHEEL_VERSION_SUFFIX}"

"${PYTHON_BIN}" build/build.py build \
  --wheels=jax,jaxlib,jax-rocm-plugin,jax-rocm-pjrt \
  --python_version=3.11 \
  --rocm_path="${DTK_DIR}" \
  --rocm_version=60 \
  --rocm_amdgpu_targets="${TARGETS}" \
  --local_xla_path="${XLA_DIR}" \
  --clang_path="${AILLVM_DIR}/bin/clang" \
  --output_path="${OUT_DIR}" \
  --bazel_options="--override_repository=triton=${TRITON_DIR}" \
  --bazel_options="--repo_env=ML_WHEEL_TYPE=release" \
  --bazel_options="--repo_env=ML_WHEEL_VERSION_SUFFIX=${DTK_WHEEL_VERSION_SUFFIX}" \
  --bazel_options="--//jaxlib/tools:jaxlib_git_hash=$(git rev-parse HEAD)" \
  --bazel_options="--config=${ROCM_CODEGEN_CONFIG}" \
  --verbose

echo "Wheels written to: ${OUT_DIR}"
