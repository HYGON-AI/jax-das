#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DTK_DIR="${DTK_DIR:-/opt/dtk}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
WHEEL_DIR="${WHEEL_DIR:-${ROOT_DIR}/dist}"
TEST_ROOT="${TEST_ROOT:-${ROOT_DIR}/tests}"
TEST_PATTERN="${TEST_PATTERN:-*_test.py}"
RUN_NAME="${RUN_NAME:-dtk_tests_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/test_logs/${RUN_NAME}}"
INSTALL_WHEELS="${INSTALL_WHEELS:-1}"
INSTALL_TEST_DEPS="${INSTALL_TEST_DEPS:-1}"
TEST_TIMEOUT="${TEST_TIMEOUT:-1800}"
EXIT_NONZERO_ON_FAILURE="${EXIT_NONZERO_ON_FAILURE:-1}"
TEST_REQUIREMENTS="${TEST_REQUIREMENTS:-${ROOT_DIR}/build/test-requirements.txt}"

usage() {
  cat <<EOF
Usage:
  ./run_dtk_tests.sh [test_file_or_dir ...]

Runs JAX pytest files one by one from /tmp, keeps going after failures, and
writes logs under test_logs/<run_name>.

Common environment overrides:
  PYTHON_BIN=python3.11
  DTK_DIR=/opt/dtk
  WHEEL_DIR=${ROOT_DIR}/dist
  TEST_TIMEOUT=1800
  PYTEST_ARGS="-k dot_product_attention"
  INSTALL_WHEELS=0
  INSTALL_TEST_DEPS=0
  EXIT_NONZERO_ON_FAILURE=0

Examples:
  ./run_dtk_tests.sh
  ./run_dtk_tests.sh tests/nn_test.py tests/lax_numpy_test.py
  PYTEST_ARGS="-k dot_product_attention" ./run_dtk_tests.sh tests/nn_test.py
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

mkdir -p "${LOG_ROOT}"
SUMMARY_FILE="${LOG_ROOT}/summary.tsv"
FAILED_FILE="${LOG_ROOT}/failed_tests.txt"
ENV_LOG="${LOG_ROOT}/environment.log"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

abspath() {
  local path="$1"
  local dir
  local base
  if [[ "${path}" = /* ]]; then
    printf '%s\n' "${path}"
    return 0
  fi
  dir="$(dirname "${path}")"
  base="$(basename "${path}")"
  printf '%s/%s\n' "$(cd "${dir}" && pwd)" "${base}"
}

pick_wheel() {
  local pattern="$1"
  local description="$2"
  local selected
  if [[ ! -d "${WHEEL_DIR}" ]]; then
    log "Wheel directory not found: ${WHEEL_DIR}" >&2
    return 1
  fi
  selected="$(find "${WHEEL_DIR}" -maxdepth 1 -type f -name "${pattern}" -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | awk 'NR == 1 {print $2}')"
  if [[ -z "${selected}" ]]; then
    log "Missing ${description} wheel matching ${WHEEL_DIR}/${pattern}" >&2
    return 1
  fi
  printf '%s\n' "${selected}"
}

if [[ ! -f "${DTK_DIR}/env.sh" ]]; then
  log "DTK env not found: ${DTK_DIR}/env.sh" >&2
  exit 2
fi

set +u
source "${DTK_DIR}/env.sh"
set -u
unset PYTHONPATH

export PY_COLORS="${PY_COLORS:-1}"
export JAX_SKIP_SLOW_TESTS="${JAX_SKIP_SLOW_TESTS:-true}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-0}"
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_force_compilation_parallelism=1 --xla_gpu_enable_nccl_comm_splitting=false --xla_gpu_enable_command_buffer=}"

if [[ "${INSTALL_WHEELS}" == "1" ]]; then
  jax_wheel="$(pick_wheel 'jax-*-py3-none-any.whl' 'jax')"
  jaxlib_wheel="$(pick_wheel 'jaxlib-*.whl' 'jaxlib')"
  rocm_plugin_wheel="$(pick_wheel 'jax_rocm*_plugin-*.whl' 'ROCm plugin')"
  rocm_pjrt_wheel="$(pick_wheel 'jax_rocm*_pjrt-*.whl' 'ROCm PJRT')"

  log "Installing built wheels from ${WHEEL_DIR}"
  "${PYTHON_BIN}" -m pip install --force-reinstall --no-deps \
    "${jax_wheel}" \
    "${jaxlib_wheel}" \
    "${rocm_plugin_wheel}" \
    "${rocm_pjrt_wheel}"
else
  log "Skipping wheel installation because INSTALL_WHEELS=${INSTALL_WHEELS}"
fi

if [[ "${INSTALL_TEST_DEPS}" == "1" ]]; then
  log "Installing test dependencies from ${TEST_REQUIREMENTS}"
  "${PYTHON_BIN}" -m pip install -r "${TEST_REQUIREMENTS}"
elif ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import absl  # noqa: F401
import hypothesis  # noqa: F401
import pytest  # noqa: F401
PY
then
  log "Some test dependencies are missing. Re-run with INSTALL_TEST_DEPS=1 or install them manually." >&2
  exit 2
fi

{
  echo "date: $(date -Is)"
  echo "root: ${ROOT_DIR}"
  echo "wheel_dir: ${WHEEL_DIR}"
  echo "test_root: ${TEST_ROOT}"
  echo "log_root: ${LOG_ROOT}"
  echo "python: $("${PYTHON_BIN}" -c 'import sys; print(sys.executable)')"
  "${PYTHON_BIN}" - <<'PY'
import importlib.metadata as md
import os
import jax
import jaxlib

print("jax_file:", jax.__file__)
print("jax:", md.version("jax"))
print("jaxlib:", jaxlib.__version__)
for dist in ("jax-rocm6-plugin", "jax-rocm6-pjrt"):
  try:
    print(f"{dist}:", md.version(dist))
  except md.PackageNotFoundError:
    print(f"{dist}: not installed")
print("HIP_VISIBLE_DEVICES:", os.getenv("HIP_VISIBLE_DEVICES"))
print("ROCR_VISIBLE_DEVICES:", os.getenv("ROCR_VISIBLE_DEVICES"))
print("XLA_FLAGS:", os.getenv("XLA_FLAGS"))
try:
  print("backend:", jax.default_backend())
  print("devices:", jax.devices())
except Exception as exc:
  print("device probe failed:", type(exc).__name__, exc)
PY
  "${PYTHON_BIN}" -m pip freeze
} >"${ENV_LOG}" 2>&1

declare -a tests
if (($# > 0)); then
  for arg in "$@"; do
    candidate="${arg}"
    if [[ ! -e "${candidate}" && -e "${ROOT_DIR}/${candidate}" ]]; then
      candidate="${ROOT_DIR}/${candidate}"
    fi
    if [[ -e "${candidate}" ]]; then
      candidate="$(abspath "${candidate}")"
    fi
    if [[ -d "${candidate}" ]]; then
      while IFS= read -r file; do
        tests+=("${file}")
      done < <(find "${candidate}" -type f -name "${TEST_PATTERN}" | sort)
    else
      tests+=("${candidate}")
    fi
  done
else
  while IFS= read -r file; do
    tests+=("${file}")
  done < <(find "${TEST_ROOT}" -type f -name "${TEST_PATTERN}" | sort)
fi

if ((${#tests[@]} == 0)); then
  log "No tests found."
  exit 2
fi

printf 'status\texit_code\tseconds\ttest\tlog\n' >"${SUMMARY_FILE}"
: >"${FAILED_FILE}"
mkdir -p "${ROOT_DIR}/test_logs"
ln -sfn "${LOG_ROOT}" "${ROOT_DIR}/test_logs/latest"

extra_pytest_args=()
if [[ -n "${PYTEST_ARGS:-}" ]]; then
  read -r -a extra_pytest_args <<<"${PYTEST_ARGS}"
fi
pass_count=0
fail_count=0
timeout_count=0
skip_count=0

log "Running ${#tests[@]} test files. Logs: ${LOG_ROOT}"

for test_file in "${tests[@]}"; do
  if [[ ! -f "${test_file}" ]]; then
    log "Skipping missing test path: ${test_file}"
    ((skip_count += 1))
    continue
  fi

  rel="${test_file#${ROOT_DIR}/}"
  safe_name="${rel//\//__}"
  log_file="${LOG_ROOT}/${safe_name%.py}.log"
  start_ts="$(date +%s)"

  log "START ${rel}"
  set +e
  if [[ "${TEST_TIMEOUT}" == "0" ]]; then
    (cd /tmp && "${PYTHON_BIN}" -m pytest -c "${ROOT_DIR}/pyproject.toml" --tb=short "${extra_pytest_args[@]}" "${test_file}") \
      > >(tee "${log_file}") 2>&1
    exit_code=${PIPESTATUS[0]}
  else
    (cd /tmp && timeout "${TEST_TIMEOUT}" "${PYTHON_BIN}" -m pytest -c "${ROOT_DIR}/pyproject.toml" --tb=short "${extra_pytest_args[@]}" "${test_file}") \
      > >(tee "${log_file}") 2>&1
    exit_code=${PIPESTATUS[0]}
  fi
  set -e

  end_ts="$(date +%s)"
  elapsed=$((end_ts - start_ts))

  status="FAIL"
  if [[ ${exit_code} -eq 0 ]]; then
    status="PASS"
    ((pass_count += 1))
  elif [[ ${exit_code} -eq 124 ]]; then
    status="TIMEOUT"
    ((timeout_count += 1))
    ((fail_count += 1))
    printf '%s\n' "${rel}" >>"${FAILED_FILE}"
  else
    ((fail_count += 1))
    printf '%s\n' "${rel}" >>"${FAILED_FILE}"
  fi

  printf '%s\t%s\t%s\t%s\t%s\n' "${status}" "${exit_code}" "${elapsed}" "${rel}" "${log_file}" >>"${SUMMARY_FILE}"
  log "DONE ${rel}: ${status} exit=${exit_code} seconds=${elapsed}"
done

log "Finished. pass=${pass_count} fail=${fail_count} timeout=${timeout_count} skipped_missing=${skip_count}"
log "Summary: ${SUMMARY_FILE}"
log "Failures: ${FAILED_FILE}"

if [[ "${EXIT_NONZERO_ON_FAILURE}" == "1" && ${fail_count} -ne 0 ]]; then
  exit 1
fi
