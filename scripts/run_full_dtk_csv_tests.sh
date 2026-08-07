#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-${SCRIPT_DIR}/..}"
DTK_DIR="${DTK_DIR:-/opt/dtk}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
TEST_ROOT="${TEST_ROOT:-${ROOT_DIR}/tests}"
TEST_PATTERN="${TEST_PATTERN:-*_test.py}"
RUN_NAME="${RUN_NAME:-dtk_full_csv_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/test_logs/${RUN_NAME}}"
TEST_TIMEOUT="${TEST_TIMEOUT:-1800}"

mkdir -p "${LOG_ROOT}/collect" "${LOG_ROOT}/logs" "${LOG_ROOT}/xml"

SUMMARY_FILE="${LOG_ROOT}/summary.tsv"
PROGRESS_FILE="${LOG_ROOT}/progress.txt"
CSV_FILE="${LOG_ROOT}/subcase_results.csv"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${PROGRESS_FILE}"
}

if [[ ! -f "${DTK_DIR}/env.sh" ]]; then
  log "DTK env not found: ${DTK_DIR}/env.sh"
  exit 2
fi

set +u
source "${DTK_DIR}/env.sh"
set -u
unset PYTHONPATH
export PYTHONPATH="${ROOT_DIR}"

export PY_COLORS="${PY_COLORS:-1}"
export JAX_SKIP_SLOW_TESTS="${JAX_SKIP_SLOW_TESTS:-true}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-0}"
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_force_compilation_parallelism=1 --xla_gpu_enable_nccl_comm_splitting=false --xla_gpu_enable_command_buffer=}"

{
  echo "date: $(date -Is)"
  echo "root: ${ROOT_DIR}"
  echo "log_root: ${LOG_ROOT}"
  echo -n "python: "
  "${PYTHON_BIN}" -c 'import sys; print(sys.executable)' || true
  echo "test_timeout: ${TEST_TIMEOUT}"
  "${PYTHON_BIN}" - <<'PY' || true
import importlib.metadata as md
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
PY
  true
} >"${LOG_ROOT}/environment.log" 2>&1

declare -a tests
if (($# > 0)); then
  for test_path in "$@"; do
    if [[ -d "${test_path}" ]]; then
      while IFS= read -r file; do
        tests+=("${file}")
      done < <(find "${test_path}" -type f -name "${TEST_PATTERN}" | sort)
    elif [[ -f "${test_path}" ]]; then
      tests+=("$(cd "$(dirname "${test_path}")" && pwd)/$(basename "${test_path}")")
    elif [[ -f "${ROOT_DIR}/${test_path}" ]]; then
      tests+=("${ROOT_DIR}/${test_path}")
    else
      log "Test path not found: ${test_path}"
      exit 2
    fi
  done
else
  mapfile -t tests < <(find "${TEST_ROOT}" -type f -name "${TEST_PATTERN}" | sort)
fi
printf '%s\n' "${tests[@]}" >"${LOG_ROOT}/test_files.txt"
printf 'status\texit_code\tseconds\ttest\tlog\txml\n' >"${SUMMARY_FILE}"

total="${#tests[@]}"
log "Found ${total} test files"

index=0
for test_file in "${tests[@]}"; do
  index=$((index + 1))
  rel="${test_file#${ROOT_DIR}/}"
  safe="${rel//\//__}"
  safe="${safe%.py}"
  collect_file="${LOG_ROOT}/collect/${safe}.txt"
  collect_log="${LOG_ROOT}/collect/${safe}.log"
  log_file="${LOG_ROOT}/logs/${safe}.log"
  xml_file="${LOG_ROOT}/xml/${safe}.xml"

  log "COLLECT ${index}/${total} ${rel}"
  set +e
  (cd /tmp && "${PYTHON_BIN}" -m pytest -c "${ROOT_DIR}/pyproject.toml" --collect-only -q "${test_file}") \
    >"${collect_file}" 2>"${collect_log}"
  collect_code=$?
  set -e
  if [[ "${collect_code}" != "0" ]]; then
    log "COLLECT_FAIL ${rel} exit=${collect_code}"
  fi

  start_ts="$(date +%s)"
  log "START ${index}/${total} ${rel}"
  set +e
  if [[ "${TEST_TIMEOUT}" == "0" ]]; then
    (cd /tmp && "${PYTHON_BIN}" -m pytest -c "${ROOT_DIR}/pyproject.toml" --tb=short -q \
      --junitxml="${xml_file}" "${test_file}") >"${log_file}" 2>&1
    exit_code=$?
  else
    (cd /tmp && timeout "${TEST_TIMEOUT}" "${PYTHON_BIN}" -m pytest -c "${ROOT_DIR}/pyproject.toml" --tb=short -q \
      --junitxml="${xml_file}" "${test_file}") >"${log_file}" 2>&1
    exit_code=$?
  fi
  set -e
  elapsed=$(($(date +%s) - start_ts))

  status="PASS"
  if [[ "${exit_code}" != "0" ]]; then
    status="FAIL"
  fi
  if [[ "${exit_code}" == "124" ]]; then
    status="TIMEOUT"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${status}" "${exit_code}" "${elapsed}" "${rel}" "${log_file}" "${xml_file}" >>"${SUMMARY_FILE}"
  log "DONE ${rel}: ${status} exit=${exit_code} seconds=${elapsed}"
done

log "Generating CSV ${CSV_FILE}"
"${PYTHON_BIN}" - "${LOG_ROOT}" "${CSV_FILE}" <<'PY'
from __future__ import annotations

import csv
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

run_dir = Path(sys.argv[1])
out_csv = Path(sys.argv[2])
ansi_re = re.compile(r"\x1b\[[0-9;]*m")
ws_re = re.compile(r"\s+")

def clean_reason(text: str | None) -> str:
  if not text:
    return ""
  text = ansi_re.sub("", text)
  text = ws_re.sub(" ", text).strip()
  if len(text) > 500:
    text = text[:497].rstrip() + "..."
  return text

def rel_from_safe(stem: str) -> str:
  return stem.replace("__", "/") + ".py"

def xml_nodeid(rel: str, classname: str, name: str) -> str:
  module = Path(rel).stem
  cls = classname.split(".")[-1] if classname else ""
  if cls and cls != module:
    return f"{rel}::{cls}::{name}"
  return f"{rel}::{name}"

def subcase_from_nodeid(nodeid: str, rel: str) -> str:
  prefix = f"{rel}::"
  return nodeid[len(prefix):] if nodeid.startswith(prefix) else nodeid

def segfault_reason(text: str) -> str | None:
  marker = "Fatal Python error: Segmentation fault"
  if marker not in text:
    return None

  stack = text.split(marker, 1)[1]
  test_frame = re.search(
      r'File "([^"]*/tests/[^"]+)", line (\d+) in ([^\n]+)', stack)
  if test_frame:
    test_path, line, function = test_frame.groups()
    trigger = f"{Path(test_path).name}:{line}::{function.strip()}"
  else:
    trigger = "backend_compile_and_load（日志中未提取到具体测试函数）"

  ignored_frames = {
      "backend_compile_and_load", "wrapper", "_compile_and_write_cache",
      "compile_or_get_cached", "_cached_compilation", "from_hlo", "compile",
      "_pjit_call_impl_python", "_run_python_pjit", "cache_miss",
      "reraise_with_filtered_traceback", "apply_primitive", "process_primitive",
      "bind_with_trace", "bind", "call_wrapped",
  }
  operation = ""
  for path, function in re.findall(
      r'File "([^"]*/site-packages/jax/[^"]+)", line \d+ in ([^\n]+)', stack):
    function = function.strip()
    if function not in ignored_frames:
      operation = f"，相关 JAX 算子栈：{Path(path).name}::{function}"
      break

  return clean_reason(
      "error: Segmentation fault；文件级崩溃触发点：XLA "
      f"backend_compile_and_load 编译 {trigger}{operation} 时发生 native SIGSEGV；"
      "推测根因：gfx936 GPU lowering 与 Triton/AILLVM 适配尚不完整，"
      "需通过 core/gdb native backtrace 最终确认；说明：pytest 若未写出 JUnit XML，"
      "文件内无结果的子测例会批量继承本错误，并非每条子测例都单独发生 SIGSEGV")

def reason_from_log(log_path: Path, fallback: str) -> str:
  if not log_path.exists():
    return fallback
  text = ansi_re.sub("", log_path.read_text(errors="replace"))
  segfault = segfault_reason(text)
  if segfault:
    return segfault
  patterns = [
      r"Fatal Python error: Aborted.*",
      r"LLVM ERROR:.*",
      r"jax\.errors\.JaxRuntimeError:.*",
      r"E\s+jax\.errors\.JaxRuntimeError:.*",
      r"INTERNAL:.*",
      r"No FFI handler registered.*",
      r"HCU clang.*",
      r"Autotuner failed.*",
      r"error:.*",
  ]
  for pattern in patterns:
    m = re.search(pattern, text)
    if m:
      return clean_reason(m.group(0))
  return fallback

summary = {}
summary_path = run_dir / "summary.tsv"
if summary_path.exists():
  with summary_path.open(encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
      summary[row["test"]] = row

rows = []
for collect_path in sorted((run_dir / "collect").glob("*.txt")):
  rel = rel_from_safe(collect_path.stem)
  collected = []
  for line in collect_path.read_text(errors="replace").splitlines():
    line = ansi_re.sub("", line).strip()
    if line.startswith(rel + "::"):
      collected.append(line)
  collected = sorted(dict.fromkeys(collected))

  results: dict[str, tuple[str, str]] = {}
  xml_path = run_dir / "xml" / f"{collect_path.stem}.xml"
  if xml_path.exists():
    try:
      root = ET.parse(xml_path).getroot()
      for case in root.iter("testcase"):
        nodeid = xml_nodeid(rel, case.attrib.get("classname", ""), case.attrib.get("name", ""))
        failure = case.find("failure")
        error = case.find("error")
        skipped = case.find("skipped")
        if failure is None and error is None and skipped is None:
          results[nodeid] = ("pass", "")
        else:
          elem = failure if failure is not None else error if error is not None else skipped
          reason = clean_reason(elem.attrib.get("message") or elem.text)
          if skipped is not None:
            results[nodeid] = ("skip", reason)
          else:
            results[nodeid] = ("fail", reason)
    except ET.ParseError as exc:
      fallback = f"JUnit XML parse failed: {exc}"
      for nodeid in collected:
        results.setdefault(nodeid, ("fail", fallback))

  row = summary.get(rel, {})
  log_path = Path(row.get("log", "")) if row.get("log") else run_dir / "logs" / f"{collect_path.stem}.log"
  exit_code = row.get("exit_code", "")
  fallback = "not executed or no JUnit result"
  if exit_code == "124":
    fallback = "timeout"
  elif exit_code == "134":
    fallback = "process aborted"
  elif exit_code and exit_code != "0":
    fallback = f"process failed with exit code {exit_code}"
  fallback = reason_from_log(log_path, fallback)

  all_nodeids = collected or sorted(results)
  for nodeid in all_nodeids:
    status, reason = results.get(nodeid, ("fail", fallback))
    sort_status = {"pass": 0, "skip": 1, "fail": 2}.get(status, 3)
    rows.append((sort_status, rel, subcase_from_nodeid(nodeid, rel), status, reason))

rows.sort(key=lambda item: (item[0], item[1], item[2]))
with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
  writer = csv.writer(f)
  writer.writerow(["测例", "子测例", "是否通过", "报错原因"])
  for _, rel, subcase, status, reason in rows:
    writer.writerow([rel, subcase, status, reason])

pass_count = sum(1 for row in rows if row[3] == "pass")
skip_count = sum(1 for row in rows if row[3] == "skip")
fail_count = sum(1 for row in rows if row[3] == "fail")
print(f"csv={out_csv}")
print(f"total={len(rows)} pass={pass_count} skip={skip_count} fail={fail_count}")
PY

log "CSV complete: ${CSV_FILE}"
touch "${LOG_ROOT}/DONE"
