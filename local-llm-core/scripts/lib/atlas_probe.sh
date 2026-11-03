#!/usr/bin/env bash
# =============================================================================
#  scripts/lib/atlas_probe.sh — ATLAS Hardware Fingerprinting  v1.0.0
# =============================================================================
#  ATLAS — Adaptive Target-hardware Lookup & Acceleration Selection
#
#  Probes: CPU vendor/flags/cores, GPU backend, RAM, disk, OS/arch.
#  Outputs: shell variables + optional JSON file.
#  Usage (source into calling script):
#      source "$(dirname "$0")/lib/atlas_probe.sh"
#      atlas_probe          # populates ATLAS_* variables
#      atlas_print_summary  # prints coloured summary
#  Usage (standalone):
#      bash atlas_probe.sh [--json] [--export] [--out /tmp/atlas.env]
# =============================================================================

# NOTE: set -euo pipefail intentionally NOT set at top level.
# This file is designed to be sourced — applying strict mode here would poison
# the caller. Strict mode is applied only in the standalone execution block below.

# ── Output variables (all exported after atlas_probe()) ──────────────────────
ATLAS_OS=""
ATLAS_ARCH=""
ATLAS_CPU_VENDOR=""
ATLAS_CPU_CORES=1
ATLAS_CPU_THREADS=1
ATLAS_HAS_AVX2=false
ATLAS_HAS_AVX512=false
ATLAS_HAS_NEON=false
ATLAS_HAS_SVE=false
ATLAS_IS_APPLE_SILICON=false
ATLAS_GPU_VENDOR="cpu"
ATLAS_GPU_NAME=""
ATLAS_GPU_BACKEND="cpu"
ATLAS_CMAKE_FLAGS=""
ATLAS_LLAMA_CMAKE_ARGS=""
ATLAS_CUDA_AVAILABLE=false
ATLAS_HIP_AVAILABLE=false
ATLAS_METAL_AVAILABLE=false
ATLAS_VULKAN_AVAILABLE=false
ATLAS_RAM_GB=0
ATLAS_SWAP_GB=0
ATLAS_DISK_FREE_GB=0
ATLAS_QUANT_RECOMMENDATION="i2_s"
ATLAS_THREADS=4
ATLAS_PKG_MGR=""
ATLAS_CLANG_VERSION=0
ATLAS_CMAKE_VERSION="0.0.0"
ATLAS_PROBE_VERSION="1.0.0"
ATLAS_PROBE_TS=""

# ── Colours (safe) ───────────────────────────────────────────────────────────
if [[ -t 1 ]] && [[ "${TERM:-dumb}" != "dumb" ]]; then
    _A_GRN='\033[0;32m' _A_YEL='\033[0;33m' _A_CYN='\033[0;36m'
    _A_RED='\033[0;31m' _A_DIM='\033[2m'    _A_RST='\033[0m'
    _A_BLD='\033[1m'
else
    _A_GRN='' _A_YEL='' _A_CYN='' _A_RED='' _A_DIM='' _A_RST='' _A_BLD=''
fi

_a_ok()  { echo -e "  ${_A_GRN}✓${_A_RST}  $*"; }
_a_warn(){ echo -e "  ${_A_YEL}⚠${_A_RST}  $*"; }
_a_info(){ echo -e "  ${_A_CYN}▶${_A_RST}  $*"; }
_a_cmd() { command -v "$1" &>/dev/null; }

# =============================================================================
# atlas_probe_os — OS and architecture detection
# =============================================================================
atlas_probe_os() {
    ATLAS_OS="$(uname -s)"
    local raw_arch; raw_arch="$(uname -m)"
    case "$raw_arch" in
        x86_64|AMD64|x86)       ATLAS_ARCH="x86_64" ;;
        aarch64|arm64|ARM64)    ATLAS_ARCH="arm64"  ;;
        armv7l|armv6l)          ATLAS_ARCH="arm32"  ;;
        *) ATLAS_ARCH="$raw_arch" ;;
    esac

    case "$ATLAS_OS" in
        Linux)
            if   _a_cmd apt-get; then ATLAS_PKG_MGR="apt"
            elif _a_cmd dnf;     then ATLAS_PKG_MGR="dnf"
            elif _a_cmd yum;     then ATLAS_PKG_MGR="yum"
            elif _a_cmd pacman;  then ATLAS_PKG_MGR="pacman"
            elif _a_cmd zypper;  then ATLAS_PKG_MGR="zypper"
            fi ;;
        Darwin) ATLAS_PKG_MGR="brew" ;;
    esac
}

# =============================================================================
# atlas_probe_cpu — CPU vendor, physical cores, SIMD flags
# =============================================================================
atlas_probe_cpu() {
    if [[ "$ATLAS_OS" == "Linux" ]] && [[ -f /proc/cpuinfo ]]; then
        # Vendor
        ATLAS_CPU_VENDOR="$(grep -m1 'vendor_id' /proc/cpuinfo 2>/dev/null \
            | awk '{print $3}' || echo 'unknown')"

        # Flags
        local _flags
        _flags="$(grep -m1 '^flags\|^Features' /proc/cpuinfo 2>/dev/null || echo '')"
        echo "$_flags" | grep -qw 'avx2'    && ATLAS_HAS_AVX2=true
        echo "$_flags" | grep -qw 'avx512f' && ATLAS_HAS_AVX512=true
        echo "$_flags" | grep -qwE 'neon|asimd' && ATLAS_HAS_NEON=true
        echo "$_flags" | grep -qw 'sve'     && ATLAS_HAS_SVE=true

        # Physical cores (unique socket+core_id pairs)
        local _phys
        _phys=$(awk -F: '/^core id/{cores[$2]=1} /^physical id/{sockets[$2]=1}
            END{c=length(cores); s=length(sockets); print (c>0 && s>0 ? c*s : c)}' \
            /proc/cpuinfo 2>/dev/null || echo 0)
        [[ "$_phys" -gt 0 ]] && ATLAS_CPU_CORES="$_phys" || \
            ATLAS_CPU_CORES=$(nproc 2>/dev/null || echo 1)
        ATLAS_CPU_THREADS=$(nproc 2>/dev/null || echo "$ATLAS_CPU_CORES")

    elif [[ "$ATLAS_OS" == "Darwin" ]]; then
        # Apple Silicon check
        if sysctl -n hw.optional.arm64 2>/dev/null | grep -q 1; then
            ATLAS_IS_APPLE_SILICON=true
            ATLAS_HAS_NEON=true
            ATLAS_CPU_VENDOR="apple"
        else
            ATLAS_CPU_VENDOR="intel"
            sysctl -n machdep.cpu.features 2>/dev/null | grep -qi 'AVX2' \
                && ATLAS_HAS_AVX2=true
            sysctl -n machdep.cpu.features 2>/dev/null | grep -qi 'AVX512' \
                && ATLAS_HAS_AVX512=true
        fi
        ATLAS_CPU_CORES=$(sysctl -n hw.physicalcpu 2>/dev/null || echo 2)
        ATLAS_CPU_THREADS=$(sysctl -n hw.logicalcpu 2>/dev/null || echo "$ATLAS_CPU_CORES")
    fi

    # ATLAS thread recommendation: physical cores, capped at 8
    ATLAS_THREADS=$(( ATLAS_CPU_CORES < 8 ? ATLAS_CPU_CORES : 8 ))
    [[ "$ATLAS_THREADS" -lt 1 ]] && ATLAS_THREADS=1
}

# =============================================================================
# atlas_probe_gpu — GPU vendor, backend, cmake flags
# =============================================================================
atlas_probe_gpu() {
    if [[ "$ATLAS_OS" == "Linux" ]]; then
        _atlas_probe_gpu_linux
    elif [[ "$ATLAS_OS" == "Darwin" ]]; then
        _atlas_probe_gpu_macos
    fi
}

_atlas_probe_gpu_linux() {
    # 1. lspci — most reliable on bare metal
    if _a_cmd lspci; then
        local gpu_line
        gpu_line=$(lspci 2>/dev/null | grep -iE "VGA|3D controller|Display" | head -1 || echo "")
        ATLAS_GPU_NAME="$(echo "$gpu_line" | sed 's/.*: //' | head -c 80)"
        if   echo "$gpu_line" | grep -qi 'nvidia'; then ATLAS_GPU_VENDOR="nvidia"
        elif echo "$gpu_line" | grep -qi 'amd\|ati\|radeon'; then ATLAS_GPU_VENDOR="amd"
        elif echo "$gpu_line" | grep -qi 'intel'; then ATLAS_GPU_VENDOR="intel"
        fi
    fi

    # 2. Confirm and augment by driver availability
    # NVIDIA CUDA
    if [[ "$ATLAS_GPU_VENDOR" == "nvidia" ]] || _a_cmd nvidia-smi; then
        if _a_cmd nvidia-smi && nvidia-smi &>/dev/null 2>&1; then
            ATLAS_CUDA_AVAILABLE=true
            ATLAS_GPU_VENDOR="nvidia"
            ATLAS_GPU_BACKEND="cuda"
            ATLAS_CMAKE_FLAGS="-DGGML_CUDA=on"
            ATLAS_LLAMA_CMAKE_ARGS="-DGGML_CUDA=on"
            ATLAS_GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null \
                | head -1 || echo "$ATLAS_GPU_NAME")
        fi
    fi

    # AMD ROCm / HIP
    if [[ "$ATLAS_GPU_VENDOR" == "amd" ]] || [[ -d /opt/rocm ]]; then
        if [[ -d /opt/rocm ]] || _a_cmd rocminfo || _a_cmd hipcc; then
            ATLAS_HIP_AVAILABLE=true
            ATLAS_GPU_VENDOR="amd"
            if [[ "$ATLAS_GPU_BACKEND" == "cpu" ]]; then
                ATLAS_GPU_BACKEND="hip"
                local _rocm_arch=""
                _a_cmd rocminfo && \
                    _rocm_arch=$(rocminfo 2>/dev/null | grep -oP 'gfx\d+' | head -1 || echo "")
                ATLAS_CMAKE_FLAGS="-DGGML_HIP=on"
                [[ -n "$_rocm_arch" ]] && ATLAS_CMAKE_FLAGS+=" -DAMDGPU_TARGETS=${_rocm_arch}"
                ATLAS_LLAMA_CMAKE_ARGS="-DGGML_HIP=on"
            fi
        fi
    fi

    # Vulkan — universal fallback
    local _has_vulkan=false
    for _vlib in /usr/lib/libvulkan.so.1 /usr/lib/x86_64-linux-gnu/libvulkan.so.1 \
                 /usr/lib/aarch64-linux-gnu/libvulkan.so.1; do
        [[ -f "$_vlib" ]] && _has_vulkan=true && break
    done
    _a_cmd vulkaninfo && _has_vulkan=true
    if $_has_vulkan; then
        ATLAS_VULKAN_AVAILABLE=true
        if [[ "$ATLAS_GPU_BACKEND" == "cpu" ]]; then
            ATLAS_GPU_BACKEND="vulkan"
            ATLAS_CMAKE_FLAGS="-DGGML_VULKAN=on"
            ATLAS_LLAMA_CMAKE_ARGS="-DGGML_VULKAN=on"
        fi
    fi
}

_atlas_probe_gpu_macos() {
    if $ATLAS_IS_APPLE_SILICON; then
        ATLAS_GPU_VENDOR="apple_silicon"
        ATLAS_METAL_AVAILABLE=true
        ATLAS_GPU_BACKEND="metal"
        ATLAS_CMAKE_FLAGS="-DGGML_METAL=on"
        ATLAS_LLAMA_CMAKE_ARGS="-DGGML_METAL=on"
        ATLAS_GPU_NAME=$(system_profiler SPDisplaysDataType 2>/dev/null \
            | grep '^ *Chipset Model:' | head -1 | sed 's/.*: //' || echo "Apple Silicon")
    else
        local gpu_info
        gpu_info=$(system_profiler SPDisplaysDataType 2>/dev/null || echo "")
        ATLAS_GPU_NAME=$(echo "$gpu_info" | grep '^ *Chipset Model:' | head -1 \
            | sed 's/.*: //' || echo "Intel")
        if echo "$gpu_info" | grep -qi 'AMD'; then
            ATLAS_GPU_VENDOR="amd"
            ATLAS_VULKAN_AVAILABLE=true
            ATLAS_GPU_BACKEND="vulkan"
            ATLAS_CMAKE_FLAGS="-DGGML_VULKAN=on"
            ATLAS_LLAMA_CMAKE_ARGS="-DGGML_VULKAN=on"
        else
            ATLAS_GPU_VENDOR="intel"
            ATLAS_GPU_BACKEND="cpu"
        fi
    fi
}

# =============================================================================
# atlas_probe_memory — RAM, swap, disk
# =============================================================================
atlas_probe_memory() {
    if [[ "$ATLAS_OS" == "Linux" ]] && [[ -f /proc/meminfo ]]; then
        local _ram_kb _swap_kb
        _ram_kb=$(awk '/^MemTotal/{print $2}' /proc/meminfo 2>/dev/null || echo 0)
        _swap_kb=$(awk '/^SwapTotal/{print $2}' /proc/meminfo 2>/dev/null || echo 0)
        ATLAS_RAM_GB=$(( _ram_kb / 1024 / 1024 ))
        ATLAS_SWAP_GB=$(( _swap_kb / 1024 / 1024 ))
    elif [[ "$ATLAS_OS" == "Darwin" ]]; then
        local _ram_bytes
        _ram_bytes=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
        ATLAS_RAM_GB=$(( _ram_bytes / 1024 / 1024 / 1024 ))
    fi

    # Disk free at $HOME
    if _a_cmd df; then
        local _disk_kb
        _disk_kb=$(df -k "${HOME}" 2>/dev/null | awk 'NR==2{print $4}' || echo 0)
        ATLAS_DISK_FREE_GB=$(( _disk_kb / 1024 / 1024 ))
    fi
}

# =============================================================================
# atlas_probe_toolchain — clang, cmake versions
# =============================================================================
atlas_probe_toolchain() {
    # Best clang >= 18
    for _cb in clang-21 clang-20 clang-19 clang-18 clang; do
        if _a_cmd "$_cb"; then
            local _v
            _v=$("$_cb" --version 2>/dev/null | grep -oP 'version \K[0-9]+' | head -1 || echo 0)
            if [[ "$_v" -gt "$ATLAS_CLANG_VERSION" ]]; then
                ATLAS_CLANG_VERSION="$_v"
            fi
        fi
    done

    # cmake
    if _a_cmd cmake; then
        ATLAS_CMAKE_VERSION=$(cmake --version 2>/dev/null | head -1 | awk '{print $3}' || echo "0.0.0")
    fi
}

# =============================================================================
# atlas_derive_quant — compute quant recommendation
# =============================================================================
atlas_derive_quant() {
    if $ATLAS_IS_APPLE_SILICON || [[ "$ATLAS_ARCH" == "arm64" ]]; then
        ATLAS_QUANT_RECOMMENDATION="tl1"   # ARM NEON-optimised ternary LUT
    elif $ATLAS_HAS_AVX512; then
        ATLAS_QUANT_RECOMMENDATION="tl2"   # x86 AVX-512 ternary LUT
    else
        ATLAS_QUANT_RECOMMENDATION="i2_s"  # INT2 symmetric — safe x86 baseline
    fi
}

# =============================================================================
# atlas_probe — run all probes
# =============================================================================
atlas_probe() {
    ATLAS_PROBE_TS="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    atlas_probe_os
    atlas_probe_cpu
    atlas_probe_gpu
    atlas_probe_memory
    atlas_probe_toolchain
    atlas_derive_quant
}

# =============================================================================
# atlas_export — export all ATLAS_* vars to environment
# =============================================================================
atlas_export() {
    export ATLAS_OS ATLAS_ARCH ATLAS_CPU_VENDOR ATLAS_CPU_CORES ATLAS_CPU_THREADS
    export ATLAS_HAS_AVX2 ATLAS_HAS_AVX512 ATLAS_HAS_NEON ATLAS_HAS_SVE
    export ATLAS_IS_APPLE_SILICON
    export ATLAS_GPU_VENDOR ATLAS_GPU_NAME ATLAS_GPU_BACKEND
    export ATLAS_CMAKE_FLAGS ATLAS_LLAMA_CMAKE_ARGS
    export ATLAS_CUDA_AVAILABLE ATLAS_HIP_AVAILABLE ATLAS_METAL_AVAILABLE ATLAS_VULKAN_AVAILABLE
    export ATLAS_RAM_GB ATLAS_SWAP_GB ATLAS_DISK_FREE_GB
    export ATLAS_QUANT_RECOMMENDATION ATLAS_THREADS
    export ATLAS_PKG_MGR ATLAS_CLANG_VERSION ATLAS_CMAKE_VERSION
    export ATLAS_PROBE_VERSION ATLAS_PROBE_TS
}

# =============================================================================
# atlas_write_env — write sourceable shell file
# =============================================================================
atlas_write_env() {
    local out="${1:-/tmp/atlas.env}"
    cat > "$out" << ENV
# ATLAS probe output — $(date -u)
# Source this file: source ${out}
export ATLAS_OS="${ATLAS_OS}"
export ATLAS_ARCH="${ATLAS_ARCH}"
export ATLAS_CPU_VENDOR="${ATLAS_CPU_VENDOR}"
export ATLAS_CPU_CORES="${ATLAS_CPU_CORES}"
export ATLAS_CPU_THREADS="${ATLAS_CPU_THREADS}"
export ATLAS_HAS_AVX2="${ATLAS_HAS_AVX2}"
export ATLAS_HAS_AVX512="${ATLAS_HAS_AVX512}"
export ATLAS_HAS_NEON="${ATLAS_HAS_NEON}"
export ATLAS_HAS_SVE="${ATLAS_HAS_SVE}"
export ATLAS_IS_APPLE_SILICON="${ATLAS_IS_APPLE_SILICON}"
export ATLAS_GPU_VENDOR="${ATLAS_GPU_VENDOR}"
export ATLAS_GPU_NAME="${ATLAS_GPU_NAME}"
export ATLAS_GPU_BACKEND="${ATLAS_GPU_BACKEND}"
export ATLAS_CMAKE_FLAGS="${ATLAS_CMAKE_FLAGS}"
export ATLAS_LLAMA_CMAKE_ARGS="${ATLAS_LLAMA_CMAKE_ARGS}"
export ATLAS_CUDA_AVAILABLE="${ATLAS_CUDA_AVAILABLE}"
export ATLAS_HIP_AVAILABLE="${ATLAS_HIP_AVAILABLE}"
export ATLAS_METAL_AVAILABLE="${ATLAS_METAL_AVAILABLE}"
export ATLAS_VULKAN_AVAILABLE="${ATLAS_VULKAN_AVAILABLE}"
export ATLAS_RAM_GB="${ATLAS_RAM_GB}"
export ATLAS_SWAP_GB="${ATLAS_SWAP_GB}"
export ATLAS_DISK_FREE_GB="${ATLAS_DISK_FREE_GB}"
export ATLAS_QUANT_RECOMMENDATION="${ATLAS_QUANT_RECOMMENDATION}"
export ATLAS_THREADS="${ATLAS_THREADS}"
export ATLAS_PKG_MGR="${ATLAS_PKG_MGR}"
export ATLAS_CLANG_VERSION="${ATLAS_CLANG_VERSION}"
export ATLAS_CMAKE_VERSION="${ATLAS_CMAKE_VERSION}"
export ATLAS_PROBE_VERSION="${ATLAS_PROBE_VERSION}"
export ATLAS_PROBE_TS="${ATLAS_PROBE_TS}"
ENV
}

# =============================================================================
# atlas_write_json — write JSON report
# =============================================================================
atlas_write_json() {
    local out="${1:-/tmp/atlas.json}"
    cat > "$out" << JSON
{
  "version": "${ATLAS_PROBE_VERSION}",
  "timestamp": "${ATLAS_PROBE_TS}",
  "os": "${ATLAS_OS}",
  "arch": "${ATLAS_ARCH}",
  "cpu": {
    "vendor": "${ATLAS_CPU_VENDOR}",
    "physical_cores": ${ATLAS_CPU_CORES},
    "logical_threads": ${ATLAS_CPU_THREADS},
    "has_avx2": ${ATLAS_HAS_AVX2},
    "has_avx512": ${ATLAS_HAS_AVX512},
    "has_neon": ${ATLAS_HAS_NEON},
    "has_sve": ${ATLAS_HAS_SVE},
    "is_apple_silicon": ${ATLAS_IS_APPLE_SILICON}
  },
  "gpu": {
    "vendor": "${ATLAS_GPU_VENDOR}",
    "name": "${ATLAS_GPU_NAME}",
    "backend": "${ATLAS_GPU_BACKEND}",
    "cmake_flags": "${ATLAS_CMAKE_FLAGS}",
    "cuda": ${ATLAS_CUDA_AVAILABLE},
    "hip": ${ATLAS_HIP_AVAILABLE},
    "metal": ${ATLAS_METAL_AVAILABLE},
    "vulkan": ${ATLAS_VULKAN_AVAILABLE}
  },
  "memory": {
    "ram_gb": ${ATLAS_RAM_GB},
    "swap_gb": ${ATLAS_SWAP_GB},
    "disk_free_gb": ${ATLAS_DISK_FREE_GB}
  },
  "recommendations": {
    "quant_type": "${ATLAS_QUANT_RECOMMENDATION}",
    "threads": ${ATLAS_THREADS}
  },
  "toolchain": {
    "pkg_mgr": "${ATLAS_PKG_MGR}",
    "clang_version": ${ATLAS_CLANG_VERSION},
    "cmake_version": "${ATLAS_CMAKE_VERSION}"
  }
}
JSON
}

# =============================================================================
# atlas_print_summary — coloured summary to stdout
# =============================================================================
atlas_print_summary() {
    echo ""
    echo -e "${_A_BLD}${_A_CYN}  ── ATLAS Hardware Fingerprint v${ATLAS_PROBE_VERSION} ──${_A_RST}"
    echo ""
    _a_ok "OS/Arch      : ${ATLAS_OS} ${ATLAS_ARCH}"
    _a_ok "CPU          : ${ATLAS_CPU_VENDOR} | ${ATLAS_CPU_CORES} physical cores | ${ATLAS_CPU_THREADS} threads"
    _a_ok "CPU flags    : AVX2=${ATLAS_HAS_AVX2}  AVX512=${ATLAS_HAS_AVX512}  NEON=${ATLAS_HAS_NEON}  Apple=${ATLAS_IS_APPLE_SILICON}"
    _a_ok "GPU          : ${ATLAS_GPU_VENDOR} (${ATLAS_GPU_NAME:-unknown}) → backend=${ATLAS_GPU_BACKEND}"
    _a_ok "cmake flags  : ${ATLAS_CMAKE_FLAGS:-none}"
    _a_ok "RAM          : ${ATLAS_RAM_GB} GB  |  Disk free: ${ATLAS_DISK_FREE_GB} GB"
    _a_ok "Clang        : v${ATLAS_CLANG_VERSION}  |  cmake: ${ATLAS_CMAKE_VERSION}"
    echo ""
    echo -e "${_A_BLD}  Recommendations:${_A_RST}"
    echo -e "  ${_A_GRN}★${_A_RST}  Quant type : ${_A_BLD}${ATLAS_QUANT_RECOMMENDATION}${_A_RST}"
    echo -e "  ${_A_GRN}★${_A_RST}  Threads    : ${_A_BLD}${ATLAS_THREADS}${_A_RST}"
    echo ""
}

# =============================================================================
# Standalone execution
# =============================================================================
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -euo pipefail
    _as_json=false
    _out_env=""
    _out_json="/tmp/atlas.json"

    for arg in "$@"; do
        case "$arg" in
            --json)           _as_json=true ;;
            --out=*)          _out_env="${arg#*=}" ;;
            --json-out=*)     _out_json="${arg#*=}" ;;
            --help|-h)
                echo "Usage: atlas_probe.sh [--json] [--out=FILE] [--json-out=FILE]"
                echo "  --json          Write JSON to /tmp/atlas.json (or --json-out=PATH)"
                echo "  --out=FILE      Write sourceable env vars to FILE"
                exit 0 ;;
        esac
    done

    atlas_probe
    atlas_export
    atlas_print_summary

    if [[ -n "$_out_env" ]]; then
        atlas_write_env "$_out_env"
        echo "  Env file written: ${_out_env}"
    fi

    if $_as_json; then
        atlas_write_json "$_out_json"
        echo "  JSON written: ${_out_json}"
    fi
fi
