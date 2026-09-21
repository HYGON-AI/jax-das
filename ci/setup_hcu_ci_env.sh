#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
#
# Provision an HCU build/test environment on top of a bare compile image: the
# BASE_COMPILE_IMAGE_ROCKY_PY311 base ships neither DTK nor a build toolchain.
#
#   1. DTK      -> ${DTK_DIR}          env.sh, rocminfo, hip runtime
#   2. HCU LLVM -> ${DTK_DIR}/aillvm   clang used by --config=hcu
#   3. bazelisk -> /usr/local/bin/bazel
#   4. node     -> /usr/local/node     symlinked into /usr/local/bin
#   5. DTK 26.04: hipBLASLt Tensile kernel dir -> HIPBLASLT_TENSILE_LIBPATH
#
# Required:
#   DTK_VERSION           e.g. 26.04. The download URL is derived from it.
#   RESOURCE_SERVER_URL   internal host serving ai_cc/Nightly/hcu_llvm_installer.sh.
#                         Injected from an organization variable, no default here.
# Optional:
#   DTK_DIR               install prefix, defaults to /opt/dtk
#   AILLVM_MAJOR          26.04 -> 1.0.0, 26.10 and later -> 2.0.0
#   BAZELISK_VERSION      defaults to 1.28.1
#   NODE_VERSION          defaults to 26.7.0

set -euo pipefail

DTK_VERSION="${DTK_VERSION:-26.04}"
RESOURCE_SERVER_URL="${RESOURCE_SERVER_URL:-}"
DTK_DIR="${DTK_DIR:-/opt/dtk}"

BAZELISK_VERSION="${BAZELISK_VERSION:-1.28.1}"
NODE_VERSION="${NODE_VERSION:-26.7.0}"

if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: must run as root, current user id is $(id -u)." >&2
    exit 1
fi

if [[ -z "${RESOURCE_SERVER_URL}" ]]; then
    echo "ERROR: RESOURCE_SERVER_URL is empty, set the RESOURCE_SERVER_URL vars." >&2
    exit 1
fi

derive_aillvm_major() {
    local lowest
    lowest=$(printf '%s\n%s\n' "${DTK_VERSION}" "26.10" | sort -V | head -n1)
    if [[ "${lowest}" == "26.10" ]]; then
        echo "2.0.0"
    else
        echo "1.0.0"
    fi
}

AILLVM_MAJOR="${AILLVM_MAJOR:-$(derive_aillvm_major)}"
echo "DTK ${DTK_VERSION} -> HCU LLVM major ${AILLVM_MAJOR}"

WGET_OPTS=(--timeout=30 --tries=3 -q)

install_dtk() {
    local url="https://download.sourcefind.cn:65024/file/1/DTK-${DTK_VERSION}/Rocky8.6/DTK-${DTK_VERSION}-Rocky8.6-x86_64.tar.gz"

    wget "${WGET_OPTS[@]}" "${url}" -O /tmp/dtk.tar.gz
    mkdir -p "${DTK_DIR}"
    tar -xzf /tmp/dtk.tar.gz -C "${DTK_DIR}" --strip-components=1
    rm -rf /tmp/dtk.tar.gz

    if [[ ! -f "${DTK_DIR}/env.sh" ]]; then
        echo "ERROR: ${DTK_DIR}/env.sh missing after extraction." >&2
        exit 1
    fi
}

install_aillvm() {
    wget "${WGET_OPTS[@]}" --no-check-certificate "${RESOURCE_SERVER_URL%/}/ai_cc/Nightly/hcu_llvm_installer.sh" -O /tmp/hcu_llvm_installer.sh
    chmod +x /tmp/hcu_llvm_installer.sh
    bash /tmp/hcu_llvm_installer.sh --major "${AILLVM_MAJOR}"
    rm -f /tmp/hcu_llvm_installer.sh

    if [[ ! -x "${DTK_DIR}/aillvm/bin/clang" ]]; then
        echo "ERROR: ${DTK_DIR}/aillvm/bin/clang missing after HCU LLVM install." >&2
        exit 1
    fi
}

install_bazelisk() {
    wget "${WGET_OPTS[@]}" \
        "https://github.com/bazelbuild/bazelisk/releases/download/v${BAZELISK_VERSION}/bazelisk-linux-amd64" \
        -O /usr/local/bin/bazel
    chmod +x /usr/local/bin/bazel
}

install_node() {
    wget "${WGET_OPTS[@]}" \
        "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.xz" \
        -O /tmp/node.tar.xz
    mkdir -p /usr/local/node
    tar -xJf /tmp/node.tar.xz -C /usr/local/node --strip-components=1
    rm -f /tmp/node.tar.xz
    ln -sf /usr/local/node/bin/node /usr/local/bin/node
    ln -sf /usr/local/node/bin/npm /usr/local/bin/npm
    ln -sf /usr/local/node/bin/npx /usr/local/bin/npx
}

detect_hcu_arch() {
    local arch=""
    arch=$("${DTK_DIR}/bin/rocminfo" 2>/dev/null \
        | grep -oE 'gfx[0-9a-f]+' \
        | grep -v '^gfx000$' \
        | head -n1) || arch=""
    printf '%s\n' "${arch}"
}

list_tensile_libpaths() {
    local dir names
    echo "Tensile library directories under ${DTK_DIR}/lib/hipblaslt:"
    for dir in "${DTK_DIR}"/lib/hipblaslt/library*/; do
        [[ -d "${dir}" ]] || continue
        names=$(cd "${dir}" && compgen -G 'TensileLibrary_*.dat' | tr '\n' ' ')
        echo "  ${dir%/}: ${names:-<no TensileLibrary_*.dat>}"
    done
}

find_tensile_libpath() {
    local arch="$1"
    local dir
    for dir in "${DTK_DIR}"/lib/hipblaslt/library*/ "${DTK_DIR}/lib/hipblaslt/"; do
        [[ -d "${dir}" ]] || continue
        if [[ -n "${arch}" ]]; then
            [[ -f "${dir}/TensileLibrary_${arch}.dat" ]] || continue
        else
            compgen -G "${dir}/TensileLibrary_*.dat" >/dev/null || continue
        fi
        echo "${dir%/}"
        return 0
    done
    return 1
}

pin_hipblaslt_tensile_libpath() {
    local hipblaslt_dir="${DTK_DIR}/lib/hipblaslt"
    if [[ ! -d "${hipblaslt_dir}" ]]; then
        echo "WARNING: ${hipblaslt_dir} not found, skipping hipBLASLt Tensile library setup." >&2
        return 0
    fi

    local arch
    arch=$(
        set +u
        source "${DTK_DIR}/env.sh"
        detect_hcu_arch
        set -u
    ) || {
        echo "ERROR: Command substitution failed with exit code $?" >&2
        exit 1
    }
    if [[ -z "${arch}" ]]; then
        echo "ERROR: cannot detect an HCU with rocminfo in ${DTK_DIR}/bin." >&2
        exit 1
    fi
    echo "HCU device arch: ${arch}"

    local libpath="${HIPBLASLT_TENSILE_LIBPATH:-}"
    if [[ -n "${libpath}" ]]; then
        if [[ ! -f "${libpath}/TensileLibrary_${arch}.dat" ]]; then
            echo "ERROR: HIPBLASLT_TENSILE_LIBPATH=${libpath} has no TensileLibrary_${arch}.dat." >&2
            list_tensile_libpaths >&2
            exit 1
        fi
        echo "Using the preset HIPBLASLT_TENSILE_LIBPATH."
    elif ! libpath=$(find_tensile_libpath "${arch}"); then
        echo "ERROR: no TensileLibrary_${arch}.dat found for DTK ${DTK_VERSION}." >&2
        list_tensile_libpaths >&2
        echo "       This DTK cannot run hipBLASLt on this device." >&2
        exit 1
    elif [[ "$(basename "${libpath}")" == "library_${arch}" ]]; then
        echo "Per-arch Tensile library (${libpath}), no HIPBLASLT_TENSILE_LIBPATH needed."
        return 0
    fi

    export HIPBLASLT_TENSILE_LIBPATH="${libpath}"
    echo "HIPBLASLT_TENSILE_LIBPATH ${HIPBLASLT_TENSILE_LIBPATH}"

    printf 'export HIPBLASLT_TENSILE_LIBPATH=%s\n' "${HIPBLASLT_TENSILE_LIBPATH}" \
        > "${DTK_DIR}/env.hipblaslt"
    if [[ -n "${GITHUB_ENV:-}" ]]; then
        echo "HIPBLASLT_TENSILE_LIBPATH=${HIPBLASLT_TENSILE_LIBPATH}" >> "${GITHUB_ENV}"
    fi
}

install_dtk
install_aillvm
install_bazelisk
install_node
pin_hipblaslt_tensile_libpath

echo "HCU CI environment ready:"
echo "  DTK        ${DTK_DIR}"
echo "  HCU LLVM   ${DTK_DIR}/aillvm (${AILLVM_MAJOR})"
echo "  bazel      $(command -v bazel) (bazelisk ${BAZELISK_VERSION})"
echo "  node       $(/usr/local/node/bin/node --version)"
echo "  hipblaslt  ${HIPBLASLT_TENSILE_LIBPATH:-<not pinned>}"
