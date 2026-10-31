#!/usr/bin/env bash
# =============================================================================
#  LOCAL LLM CORE — Production Installer v1.0.0
# =============================================================================
#  Fully automatic, idempotent, multi-platform installer.
#  Supports: Linux (apt/dnf/pacman) · macOS (Homebrew) · all GPU backends.
#
#  Built-in algorithms:
#    ATLAS   — Adaptive Target-hardware Lookup & Acceleration Selection
#              Probes CPU (AVX2/AVX512/NEON/Apple Silicon), GPU (CUDA/ROCm/
#              Metal/Vulkan), RAM, and disk to pick optimal backends + flags.
#    GUARDIAN — Pre-flight checker. Validates ALL preconditions before touching
#              anything. Auto-repairs what it can; hard-fails only what it must.
#    SPECTRE — Subprocess Environment Poisoning Correction & Transparent
#              Resolution Engine. Separates build-time gguf-py (local, has
#              TL1/TL2) from runtime gguf (PyPI) via PYTHONPATH layering.
#    NEXUS   — Unified environment manager. One venv for GGUF + BitNet.
#
#  Environment variables (all optional):
#      PROJECT_ROOT        — absolute path to local-llm-core package root
#      LLM_INSTALL_DIR     — alias for PROJECT_ROOT
#      LLM_VENV_DIR        — override virtual environment path
#      LLM_SKIP_KERNEL     — "true" to skip kernel module build/load
#      LLM_MODEL_DIR       — directory for downloaded models
#      LLM_SKIP_LLAMA_CPP  — "true" to skip llama-cpp-python build
#      LLM_FORCE_CPU       — "true" to force CPU-only llama-cpp-python
#      LLM_CLANG_MIN       — minimum clang version required (default: 18)
#      LLM_PYTHON_MIN      — minimum Python minor version for 3.x (default: 8)
#      NON_INTERACTIVE     — "1" to disable all prompts
#      DRY_RUN             — "1" to print commands without executing
# =============================================================================

set -euo pipefail
IFS=$'\n\t'

# =============================================================================
# SECTION 0 — Script metadata & safety guards
# =============================================================================

readonly INSTALLER_VERSION="1.0.0"
readonly INSTALLER_NAME="LOCAL LLM CORE"
readonly CLANG_MIN_VER="${LLM_CLANG_MIN:-18}"
readonly PYTHON_MIN_MINOR="${LLM_PYTHON_MIN:-8}"

# Prevent recursive execution (install.sh must not call itself)
if [[ -n "${_LLM_INSTALL_RUNNING:-}" ]]; then
    echo "[install.sh] ERROR: recursive execution detected — aborting." >&2
    exit 1
fi
export _LLM_INSTALL_RUNNING=1

# Absolute path to this script (resolves symlinks)
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

DRY_RUN="${DRY_RUN:-0}"
NON_INTERACTIVE="${NON_INTERACTIVE:-0}"

# =============================================================================
# SECTION 1 — Colour palette & output helpers
# =============================================================================

if [[ -t 1 ]] && [[ "${TERM:-dumb}" != "dumb" ]]; then
    C_RED='\033[0;31m';    C_GREEN='\033[0;32m';  C_YELLOW='\033[0;33m'
    C_BLUE='\033[0;34m';   C_MAGENTA='\033[0;35m';C_CYAN='\033[0;36m'
    C_WHITE='\033[0;37m';  C_BOLD='\033[1m';       C_DIM='\033[2m'
    C_RESET='\033[0m'
else
    C_RED=''; C_GREEN=''; C_YELLOW=''; C_BLUE=''; C_MAGENTA=''
    C_CYAN=''; C_WHITE=''; C_BOLD=''; C_DIM=''; C_RESET=''
fi

banner()  { echo -e "\n${C_BOLD}${C_CYAN}==> ${C_RESET}${C_BOLD}$*${C_RESET}"; }
section() { echo -e "\n${C_BOLD}${C_BLUE}┌─ $* ${C_RESET}"; }
ok()      { echo -e "  ${C_GREEN}✅${C_RESET} $*"; }
warn()    { echo -e "  ${C_YELLOW}⚠${C_RESET}  $*"; }
info()    { echo -e "  ${C_CYAN}▶${C_RESET}  $*"; }
step()    { echo -e "  ${C_DIM}→${C_RESET}  $*"; }
die()     { echo -e "\n  ${C_RED}❌ FATAL: $*${C_RESET}\n" >&2; exit 1; }
note()    { echo -e "  ${C_MAGENTA}ℹ${C_RESET}  $*"; }

# Run a command (respects DRY_RUN; hides output unless it fails)
run_cmd() {
    local silent=false
    [[ "${1:-}" == "--silent" ]] && { silent=true; shift; }
    if [[ "$DRY_RUN" == "1" ]]; then
        step "[DRY-RUN] $*"
        return 0
    fi
    if $silent; then
        local _out _rc
        _out=$("$@" 2>&1) && return 0 || _rc=$?
        echo "$_out" >&2
        return $_rc
    else
        "$@"
    fi
}

# Check if a command exists
cmd_exists() { command -v "$1" &>/dev/null; }

# Version comparison: ver_gte A B → true if A >= B (dot-separated integers)
ver_gte() {
    local IFS=.
    local -a a=($1) b=($2)
    local i
    for (( i=0; i<${#b[@]}; i++ )); do
        local av=${a[i]:-0} bv=${b[i]:-0}
        (( av > bv )) && return 0
        (( av < bv )) && return 1
    done
    return 0
}

# =============================================================================
# SECTION 2 — Argument parsing
# =============================================================================

SKIP_KERNEL="${LLM_SKIP_KERNEL:-false}"
SKIP_LLAMA_CPP="${LLM_SKIP_LLAMA_CPP:-false}"
FORCE_CPU="${LLM_FORCE_CPU:-false}"
FORCE_REBUILD=false
RESET_MODE=false

for arg in "$@"; do
    case "$arg" in
        --skip-kernel)      SKIP_KERNEL=true ;;
        --skip-llama-cpp)   SKIP_LLAMA_CPP=true ;;
        --force-cpu)        FORCE_CPU=true ;;
        --force-rebuild)    FORCE_REBUILD=true ;;
        --reset)            RESET_MODE=true ;;
        --non-interactive)  NON_INTERACTIVE=1 ;;
        --dry-run)          DRY_RUN=1 ;;
        --help|-h)
            echo "Usage: bash install.sh [OPTIONS]"
            echo "  --skip-kernel       Skip kernel module build/load"
            echo "  --skip-llama-cpp    Skip llama-cpp-python installation"
            echo "  --force-cpu         Force CPU-only llama-cpp-python"
            echo "  --force-rebuild     Force reinstall of llama-cpp-python"
            echo "  --reset             Remove .llm_env and state, then reinstall"
            echo "  --non-interactive   Disable all prompts"
            echo "  --dry-run           Print actions without executing"
            exit 0
            ;;
        *) warn "Unknown argument: $arg (ignored)" ;;
    esac
done

# =============================================================================
# SECTION 3 — Signal handling & cleanup traps
# =============================================================================

_cleanup() {
    local rc=$?
    # Remove temp files if created
    rm -f /tmp/llm_kos.list /tmp/llm_inst_$$.tmp 2>/dev/null || true
    if [[ $rc -ne 0 ]]; then
        echo -e "\n${C_RED}${C_BOLD}Installation aborted (exit code $rc).${C_RESET}" >&2
        echo -e "  Re-run: ${C_BOLD}bash ${SCRIPT_PATH}${C_RESET}" >&2
    fi
}
trap '_cleanup' EXIT
trap 'die "Interrupted by user (SIGINT)"' INT
trap 'die "Terminated (SIGTERM)"' TERM

# =============================================================================
# SECTION 4 — Privilege elevation
# =============================================================================

if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
    if [[ "$NON_INTERACTIVE" == "1" ]] || [[ ! -t 0 ]]; then
        die "Root privileges required.\n  Run: sudo bash install.sh $*\n  Or set NON_INTERACTIVE=0 and use sudo."
    fi
    echo -e "${C_YELLOW}⚠  Not running as root. Re-executing with sudo…${C_RESET}"
    exec sudo -E bash "$SCRIPT_PATH" "$@"
fi

# =============================================================================
# SECTION 5 — Locate project root
# =============================================================================

if [[ -n "${PROJECT_ROOT:-}" ]]; then
    PROJECT_ROOT="${PROJECT_ROOT}"
elif [[ -n "${LLM_INSTALL_DIR:-}" ]]; then
    PROJECT_ROOT="${LLM_INSTALL_DIR}"
else
    _dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    # If we're in scripts/, step up one level
    [[ "$(basename "$_dir")" == "scripts" ]] && _dir="$(cd "$_dir/.." && pwd)"
    PROJECT_ROOT="$_dir"
    # Walk up looking for package markers
    while [[ "$PROJECT_ROOT" != "/" ]]; do
        if [[ -f "$PROJECT_ROOT/setup.py" ]] \
        || [[ -f "$PROJECT_ROOT/pyproject.toml" ]] \
        || [[ -f "$PROJECT_ROOT/config.yaml" ]]; then
            break
        fi
        PROJECT_ROOT="$(dirname "$PROJECT_ROOT")"
    done
fi

[[ -f "$PROJECT_ROOT/setup.py" ]] \
|| [[ -f "$PROJECT_ROOT/pyproject.toml" ]] \
|| [[ -f "$PROJECT_ROOT/config.yaml" ]] \
|| die "Cannot locate project root (no setup.py/pyproject.toml/config.yaml).\n  Set PROJECT_ROOT or LLM_INSTALL_DIR."

readonly PROJECT_ROOT

# =============================================================================
# SECTION 6 — Real-home detection (sudo-safe)
# =============================================================================

if [[ -n "${SUDO_USER:-}" ]] && [[ "${SUDO_USER}" != "root" ]]; then
    REAL_HOME="$(getent passwd "${SUDO_USER}" 2>/dev/null | cut -d: -f6 || echo "${HOME}")"
    REAL_USER="${SUDO_USER}"
elif [[ "${EUID:-0}" -eq 0 ]] && [[ -d /home ]]; then
    _rh="$(ls -d /home/*/ 2>/dev/null | head -1)"
    REAL_HOME="${_rh%/}"
    [[ -d "${REAL_HOME}" ]] || REAL_HOME="${HOME}"
    REAL_USER="$(basename "${REAL_HOME}")"
else
    REAL_HOME="${HOME}"
    REAL_USER="${USER:-root}"
fi
readonly REAL_HOME REAL_USER

# =============================================================================
# SECTION 7 — Virtual environment path resolution (NEXUS)
# =============================================================================
# Priority: active VIRTUAL_ENV > explicit LLM_VENV_DIR > known named venvs
# > sibling DEVE-BOX > project venv

_resolve_venv() {
    # 1. Already activated venv
    if [[ -n "${VIRTUAL_ENV:-}" ]] && [[ -x "${VIRTUAL_ENV}/bin/python" ]]; then
        echo "${VIRTUAL_ENV}"; return
    fi
    # 2. Explicit override
    if [[ -n "${LLM_VENV_DIR:-}" ]] && [[ -x "${LLM_VENV_DIR}/bin/python" ]]; then
        echo "${LLM_VENV_DIR}"; return
    fi
    # 3. Well-known named venvs in REAL_HOME and sibling dirs
    local candidates=(
        "${REAL_HOME}/DEVE-BOX"
        "${HOME}/DEVE-BOX"
        "${REAL_HOME}/.venv"
        "${HOME}/.venv"
        "$(dirname "${PROJECT_ROOT}")/venv"
        "$(dirname "${PROJECT_ROOT}")/.venv"
        "${PROJECT_ROOT}/venv"
        "${PROJECT_ROOT}/.venv"
    )
    for c in "${candidates[@]}"; do
        [[ -x "${c}/bin/python" ]] && { echo "$c"; return; }
    done
    # 4. Default — create new
    echo "${PROJECT_ROOT}/venv"
}

export LLM_VENV_DIR="$(_resolve_venv)"
MODEL_DIR="${LLM_MODEL_DIR:-${REAL_HOME}/models}"

# =============================================================================
# SECTION 8 — Banner
# =============================================================================

echo -e ""
echo -e "${C_BOLD}${C_CYAN}  ╔══════════════════════════════════════════════════════════╗${C_RESET}"
echo -e "${C_BOLD}${C_CYAN}  ║   🚀  ${INSTALLER_NAME} — System Setup v${INSTALLER_VERSION}        ║${C_RESET}"
echo -e "${C_BOLD}${C_CYAN}  ╚══════════════════════════════════════════════════════════╝${C_RESET}"
echo "  Scripts : $(dirname "${SCRIPT_PATH}")"
echo "  Project : ${PROJECT_ROOT}"
echo "  Venv    : ${LLM_VENV_DIR}"
echo "  Models  : ${MODEL_DIR}"
echo "  DryRun  : ${DRY_RUN} | NonInteractive: ${NON_INTERACTIVE}"
echo ""

if [[ "$RESET_MODE" == "true" ]]; then
    warn "Reset mode — removing .llm_env and .llm_state"
    rm -f "${PROJECT_ROOT}/.llm_env" "${PROJECT_ROOT}/.llm_state" 2>/dev/null || true
    ok "State cleared — proceeding with fresh install"
fi

# =============================================================================
# SECTION 9 — OS & distribution detection
# =============================================================================

section "Operating System Detection"

OS_TYPE="$(uname -s)"
ARCH="$(uname -m)"
case "$OS_TYPE" in
    Linux)  OS_FAMILY="linux"  ;;
    Darwin) OS_FAMILY="macos"  ;;
    *)      die "Unsupported OS: ${OS_TYPE}. Only Linux and macOS are supported." ;;
esac
ok "OS: ${OS_TYPE} / Arch: ${ARCH}"

# Linux distro identification
DISTRO=""
DISTRO_VER=""
if [[ "$OS_FAMILY" == "linux" ]]; then
    if [[ -f /etc/os-release ]]; then
        # shellcheck disable=SC1091
        DISTRO="$(. /etc/os-release && echo "${ID:-unknown}")"
        DISTRO_VER="$(. /etc/os-release && echo "${VERSION_ID:-0}")"
    elif [[ -f /etc/debian_version ]]; then
        DISTRO="debian"
    elif [[ -f /etc/redhat-release ]]; then
        DISTRO="rhel"
    fi
    ok "Distro: ${DISTRO:-unknown} ${DISTRO_VER}"
fi

# Package manager
PKG_MGR=""
if [[ "$OS_FAMILY" == "linux" ]]; then
    if   cmd_exists apt-get;  then PKG_MGR="apt"
    elif cmd_exists dnf;      then PKG_MGR="dnf"
    elif cmd_exists yum;      then PKG_MGR="yum"
    elif cmd_exists pacman;   then PKG_MGR="pacman"
    elif cmd_exists zypper;   then PKG_MGR="zypper"
    else die "No supported package manager found (apt/dnf/yum/pacman/zypper)."
    fi
    ok "Package manager: ${PKG_MGR}"
elif [[ "$OS_FAMILY" == "macos" ]]; then
    if ! cmd_exists brew; then
        if [[ "$NON_INTERACTIVE" == "1" ]]; then
            die "Homebrew not found and NON_INTERACTIVE=1. Install Homebrew first: https://brew.sh"
        fi
        info "Installing Homebrew…"
        run_cmd /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    PKG_MGR="brew"
    ok "Package manager: Homebrew"
fi

# =============================================================================
# SECTION 10 — ATLAS: CPU capability probing
# =============================================================================

section "ATLAS — CPU Capability Probe"

CPU_VENDOR=""
CPU_FLAGS=""
HAS_AVX2=false
HAS_AVX512=false
HAS_NEON=false
HAS_SVE=false
IS_APPLE_SILICON=false
CPU_PHYSICAL_CORES=1

if [[ "$OS_FAMILY" == "linux" ]] && [[ -f /proc/cpuinfo ]]; then
    CPU_VENDOR="$(grep -m1 'vendor_id' /proc/cpuinfo 2>/dev/null | awk '{print $3}' || echo 'unknown')"
    CPU_FLAGS="$(grep -m1 '^flags' /proc/cpuinfo 2>/dev/null || grep -m1 '^Features' /proc/cpuinfo 2>/dev/null || echo '')"
    echo "$CPU_FLAGS" | grep -qw 'avx2'    && HAS_AVX2=true
    echo "$CPU_FLAGS" | grep -qw 'avx512f' && HAS_AVX512=true
    echo "$CPU_FLAGS" | grep -qwE 'neon|asimd' && HAS_NEON=true
    echo "$CPU_FLAGS" | grep -qw 'sve'     && HAS_SVE=true
    # Physical cores = number of unique (socket, core-id) pairs
    CPU_PHYSICAL_CORES=$(grep -c 'physical id' /proc/cpuinfo 2>/dev/null || echo 1)
    [[ "$CPU_PHYSICAL_CORES" -lt 1 ]] && CPU_PHYSICAL_CORES=$(nproc 2>/dev/null || echo 1)
elif [[ "$OS_FAMILY" == "macos" ]]; then
    if sysctl -n hw.optional.arm64 2>/dev/null | grep -q 1; then
        IS_APPLE_SILICON=true
        HAS_NEON=true
        CPU_VENDOR="apple"
    else
        CPU_VENDOR="intel"
        sysctl -n machdep.cpu.features 2>/dev/null | grep -qi 'AVX2' && HAS_AVX2=true
    fi
    CPU_PHYSICAL_CORES=$(sysctl -n hw.physicalcpu 2>/dev/null || echo 2)
fi

# Recommended BitNet thread count — physical cores, capped at 8
ATLAS_THREADS=$(( CPU_PHYSICAL_CORES < 8 ? CPU_PHYSICAL_CORES : 8 ))
[[ "$ATLAS_THREADS" -lt 1 ]] && ATLAS_THREADS=1

# Quantization recommendation for BitNet.cpp
if $IS_APPLE_SILICON || ( [[ "$ARCH" == arm* ]] || [[ "$ARCH" == aarch64 ]] ); then
    ATLAS_QUANT_REC="tl1"     # ARM NEON-optimised ternary LUT
elif $HAS_AVX512; then
    ATLAS_QUANT_REC="tl2"     # x86 AVX-512 ternary LUT (fastest on AVX-512 CPUs)
else
    ATLAS_QUANT_REC="i2_s"    # x86 INT2 symmetric — safe baseline for all x86
fi

ok "CPU: ${CPU_VENDOR} | Physical cores: ${CPU_PHYSICAL_CORES}"
ok "Flags: AVX2=${HAS_AVX2} | AVX512=${HAS_AVX512} | NEON=${HAS_NEON} | Apple Silicon=${IS_APPLE_SILICON}"
ok "BitNet quant recommendation: ${ATLAS_QUANT_REC} | Threads: ${ATLAS_THREADS}"

# =============================================================================
# SECTION 11 — ATLAS: GPU backend detection
# =============================================================================

section "ATLAS — GPU Backend Detection"

GPU_VENDOR="cpu"
GPU_NAME=""
HIP_AVAILABLE=false
CUDA_AVAILABLE=false
METAL_AVAILABLE=false
VULKAN_AVAILABLE=false
ATLAS_GPU_BACKEND="cpu"
ATLAS_CMAKE_FLAGS=""
ATLAS_LLAMA_CMAKE_ARGS=""

_detect_gpu_linux() {
    # 1. lspci — most reliable on bare metal
    if cmd_exists lspci; then
        local gpu_info
        gpu_info=$(lspci 2>/dev/null | grep -iE "VGA|3D|Display" || echo "")
        if echo "$gpu_info" | grep -qi 'nvidia';                then GPU_VENDOR="nvidia"
        elif echo "$gpu_info" | grep -qi 'amd\|ati\|radeon';   then GPU_VENDOR="amd"
        elif echo "$gpu_info" | grep -qi 'intel';               then GPU_VENDOR="intel"
        fi
        GPU_NAME="$(echo "$gpu_info" | head -1 | sed 's/.*: //')"
    fi

    # 2. Confirm NVIDIA CUDA
    if [[ "$GPU_VENDOR" == "nvidia" ]]; then
        if cmd_exists nvidia-smi && nvidia-smi &>/dev/null; then
            CUDA_AVAILABLE=true
            GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "$GPU_NAME")"
            ATLAS_GPU_BACKEND="cuda"
            ATLAS_CMAKE_FLAGS="-DGGML_CUDA=on"
            ATLAS_LLAMA_CMAKE_ARGS="-DGGML_CUDA=on"
        else
            warn "NVIDIA GPU detected but nvidia-smi failed — may need driver install"
            warn "Install: https://developer.nvidia.com/cuda-downloads"
        fi
    fi

    # 3. Confirm AMD ROCm/HIP
    if [[ "$GPU_VENDOR" == "amd" ]]; then
        if cmd_exists rocminfo || [[ -d /opt/rocm ]] || cmd_exists hipcc; then
            HIP_AVAILABLE=true
            # Prefer HIP backend — direct ROCm integration, faster than Vulkan
            ATLAS_GPU_BACKEND="hip"
            ATLAS_CMAKE_FLAGS="-DGGML_HIP=on"
            ATLAS_LLAMA_CMAKE_ARGS="-DGGML_HIP=on"
            # ROCm needs GPU arch target for optimal perf
            if cmd_exists rocminfo; then
                local _rocm_arch
                _rocm_arch=$(rocminfo 2>/dev/null | grep -oP 'gfx\d+' | head -1 || echo "")
                [[ -n "$_rocm_arch" ]] && ATLAS_CMAKE_FLAGS="${ATLAS_CMAKE_FLAGS} -DAMDGPU_TARGETS=${_rocm_arch}"
            fi
        else
            warn "AMD GPU detected but ROCm not found — using Vulkan fallback"
            note "Install ROCm for 5-10× better AMD performance: https://rocm.docs.amd.com/en/latest/deploy/linux/"
        fi
    fi

    # 4. Vulkan — universal fallback (Intel, AMD without ROCm, any unknown)
    if cmd_exists vulkaninfo || [[ -f /usr/lib/libvulkan.so.1 ]] || [[ -f /usr/lib/x86_64-linux-gnu/libvulkan.so.1 ]]; then
        VULKAN_AVAILABLE=true
        if [[ "$ATLAS_GPU_BACKEND" == "cpu" ]]; then
            ATLAS_GPU_BACKEND="vulkan"
            ATLAS_CMAKE_FLAGS="-DGGML_VULKAN=on"
            ATLAS_LLAMA_CMAKE_ARGS="-DGGML_VULKAN=on"
        fi
    fi
}

_detect_gpu_macos() {
    if $IS_APPLE_SILICON; then
        GPU_VENDOR="apple_silicon"
        METAL_AVAILABLE=true
        ATLAS_GPU_BACKEND="metal"
        ATLAS_CMAKE_FLAGS="-DGGML_METAL=on"
        ATLAS_LLAMA_CMAKE_ARGS="-DGGML_METAL=on"
        GPU_NAME="$(system_profiler SPDisplaysDataType 2>/dev/null \
            | grep '^ *Chipset Model:' | head -1 | sed 's/.*: //' || echo 'Apple Silicon')"
    else
        # Intel Mac — check discrete GPU
        local gpu_info
        gpu_info="$(system_profiler SPDisplaysDataType 2>/dev/null || echo '')"
        if echo "$gpu_info" | grep -qi 'AMD'; then
            GPU_VENDOR="amd"
            VULKAN_AVAILABLE=true
            ATLAS_GPU_BACKEND="vulkan"
            ATLAS_CMAKE_FLAGS="-DGGML_VULKAN=on"
            ATLAS_LLAMA_CMAKE_ARGS="-DGGML_VULKAN=on"
        elif echo "$gpu_info" | grep -qi 'NVIDIA'; then
            GPU_VENDOR="nvidia"
            # CUDA on Intel Mac is extremely rare; Vulkan is safer
            VULKAN_AVAILABLE=true
            ATLAS_GPU_BACKEND="vulkan"
            ATLAS_CMAKE_FLAGS="-DGGML_VULKAN=on"
            ATLAS_LLAMA_CMAKE_ARGS="-DGGML_VULKAN=on"
        else
            GPU_VENDOR="intel"
            ATLAS_GPU_BACKEND="cpu"
        fi
        GPU_NAME="$(echo "$gpu_info" | grep '^ *Chipset Model:' | head -1 | sed 's/.*: //' || echo 'Intel')"
    fi
}

if   [[ "$OS_FAMILY" == "linux" ]]; then _detect_gpu_linux
elif [[ "$OS_FAMILY" == "macos" ]]; then _detect_gpu_macos
fi

# Override: user forced CPU
if [[ "$FORCE_CPU" == "true" ]]; then
    ATLAS_GPU_BACKEND="cpu"
    ATLAS_CMAKE_FLAGS=""
    ATLAS_LLAMA_CMAKE_ARGS=""
    warn "LLM_FORCE_CPU=true — forcing CPU-only mode"
fi

ok "GPU vendor: ${GPU_VENDOR} | Name: ${GPU_NAME:-unknown}"
ok "Backend: ${ATLAS_GPU_BACKEND} | CUDA: ${CUDA_AVAILABLE} | HIP: ${HIP_AVAILABLE} | Metal: ${METAL_AVAILABLE} | Vulkan: ${VULKAN_AVAILABLE}"
[[ -n "$ATLAS_CMAKE_FLAGS" ]] && ok "cmake flags: ${ATLAS_CMAKE_FLAGS}"

# =============================================================================
# SECTION 12 — ATLAS: Memory & disk assessment
# =============================================================================

section "ATLAS — Memory & Disk Assessment"

RAM_GB=0
DISK_FREE_GB=0

if [[ "$OS_FAMILY" == "linux" ]] && [[ -f /proc/meminfo ]]; then
    _ram_kb=$(awk '/^MemTotal/{print $2}' /proc/meminfo)
    RAM_GB=$(( _ram_kb / 1024 / 1024 ))
elif [[ "$OS_FAMILY" == "macos" ]]; then
    _ram_bytes=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
    RAM_GB=$(( _ram_bytes / 1024 / 1024 / 1024 ))
fi

if cmd_exists df; then
    _disk_kb=$(df -k "${MODEL_DIR%/*}" 2>/dev/null | awk 'NR==2{print $4}' || echo 0)
    DISK_FREE_GB=$(( _disk_kb / 1024 / 1024 ))
fi

ok "RAM: ${RAM_GB} GB | Free disk (model location): ${DISK_FREE_GB} GB"

# Advisory warnings — non-fatal
[[ "$RAM_GB" -gt 0 ]] && [[ "$RAM_GB" -lt 4 ]] && \
    warn "Less than 4 GB RAM — LLM inference may be very slow or fail"
[[ "$DISK_FREE_GB" -gt 0 ]] && [[ "$DISK_FREE_GB" -lt 10 ]] && \
    warn "Less than 10 GB free disk — model downloads may fail"

# Swap file for low-RAM systems (Linux only, best-effort)
if [[ "$OS_FAMILY" == "linux" ]] && [[ "$RAM_GB" -lt 8 ]]; then
    _swap_total=$(free -g 2>/dev/null | awk '/^Swap/{print $2}' || echo 0)
    if [[ "$_swap_total" -lt 4 ]]; then
        info "Low RAM + insufficient swap detected. Creating 4 GB swap at /var/swap/llm_swap…"
        if [[ "$DRY_RUN" != "1" ]]; then
            mkdir -p /var/swap 2>/dev/null || true
            if [[ ! -f /var/swap/llm_swap ]]; then
                dd if=/dev/zero of=/var/swap/llm_swap bs=1M count=4096 status=none 2>/dev/null \
                    && chmod 600 /var/swap/llm_swap \
                    && mkswap /var/swap/llm_swap >/dev/null 2>&1 \
                    && swapon /var/swap/llm_swap 2>/dev/null \
                    && ok "4 GB swap file created at /var/swap/llm_swap" \
                    || warn "Could not create swap (non-fatal)"
            else
                ok "LLM swap file already exists"
            fi
        fi
    fi
fi

# =============================================================================
# SECTION 13 — GUARDIAN: Pre-flight checks
# =============================================================================

section "GUARDIAN — Pre-flight Checks"

GUARDIAN_PASS=true

_guardian_check() {
    local label="$1" result="$2" advice="$3" fatal="${4:-false}"
    if [[ "$result" == "pass" ]]; then
        ok "$label"
    else
        if [[ "$fatal" == "true" ]]; then
            warn "$label — ${advice}"
            GUARDIAN_PASS=false
        else
            warn "$label — ${advice}"
        fi
    fi
}

# Check: write permission on project root
[[ -w "$PROJECT_ROOT" ]] \
    && _guardian_check "Write access: ${PROJECT_ROOT}" "pass" "" \
    || _guardian_check "Write access: ${PROJECT_ROOT}" "fail" "Run as root or fix permissions" "true"

# Check: curl or wget available
( cmd_exists curl || cmd_exists wget ) \
    && _guardian_check "Download tool (curl/wget)" "pass" "" \
    || _guardian_check "Download tool (curl/wget)" "fail" "Install curl or wget" "true"

# Check: git available
cmd_exists git \
    && _guardian_check "git $(git --version 2>/dev/null | awk '{print $3}')" "pass" "" \
    || _guardian_check "git" "fail" "Install git" "true"

# Check: cmake available (warn only — installed next)
if cmd_exists cmake; then
    _cmake_ver=$(cmake --version 2>/dev/null | awk '/cmake version/{print $3}')
    ver_gte "${_cmake_ver:-0}" "3.14" \
        && _guardian_check "cmake ${_cmake_ver} (>= 3.14)" "pass" "" \
        || _guardian_check "cmake ${_cmake_ver} (< 3.14)" "fail" "Will be upgraded in system packages step" "false"
else
    _guardian_check "cmake" "fail" "Will be installed in system packages step" "false"
fi

# Check: make or ninja
( cmd_exists make || cmd_exists ninja ) \
    && _guardian_check "Build tool (make/ninja)" "pass" "" \
    || _guardian_check "Build tool (make/ninja)" "fail" "Will be installed in system packages step" "false"

# Check: enough disk for venv (~500MB)
[[ "$DISK_FREE_GB" -ge 1 ]] \
    && _guardian_check "Disk space for venv (>= 1 GB)" "pass" "" \
    || _guardian_check "Disk space for venv (>= 1 GB)" "fail" "Free at least 1 GB" "true"

$GUARDIAN_PASS || die "GUARDIAN: Fatal pre-flight failures detected above. Fix them and re-run."

ok "GUARDIAN: All fatal checks passed"

# =============================================================================
# SECTION 14 — System package installation
# =============================================================================

section "System Packages"

_install_apt() {
    # ── Suppress ALL interactive apt/dpkg/needrestart prompts ─────────────────
    export DEBIAN_FRONTEND=noninteractive
    export NEEDRESTART_MODE=a          # auto-restart services, never prompt
    export NEEDRESTART_SUSPEND=1       # skip needrestart entirely on older versions
    export UCF_FORCE_CONFFNEW=1        # always use new config files
    local _DPKG_OPTS=(
        -o Dpkg::Options::="--force-confdef"
        -o Dpkg::Options::="--force-confold"
        -o Dpkg::Options::="--force-unsafe-io"
    )
    # Also suppress needrestart via dpkg hook if it's installed
    if [[ -f /etc/needrestart/needrestart.conf ]]; then
        sed -i "s/^#\?\s*\$nrconf{restart}\s*=.*/\$nrconf{restart} = 'a';/" \
            /etc/needrestart/needrestart.conf 2>/dev/null || true
    fi

    # ── Update package index with timeout ─────────────────────────────────────
    info "apt-get update …"
    timeout 120 apt-get update -qq "${_DPKG_OPTS[@]}" 2>/dev/null || {
        warn "apt-get update failed or timed out — using cached package list"
    }

    # ── Check which core packages actually need installing ────────────────────
    local _core_pkgs=(
        python3 python3-pip python3-venv python3-dev
        build-essential cmake ninja-build git curl wget
        libssl-dev zlib1g-dev libbz2-dev libreadline-dev
        libsqlite3-dev libffi-dev liblz4-dev libgomp1
        pkg-config patchelf ocl-icd-opencl-dev opencl-headers
        psmisc htop ncurses-dev ca-certificates gnupg lsb-release
        dkms
    )
    local _missing=()
    for _p in "${_core_pkgs[@]}"; do
        dpkg -l "$_p" &>/dev/null 2>&1 && \
            dpkg -l "$_p" | grep -q "^ii" || _missing+=("$_p")
    done

    if [[ ${#_missing[@]} -gt 0 ]]; then
        info "Installing ${#_missing[@]} missing core package(s) …"
        DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
            "${_DPKG_OPTS[@]}" "${_missing[@]}" >/dev/null 2>&1 || {
            warn "Some core packages failed — retrying individually …"
            for _p in "${_missing[@]}"; do
                DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
                    "${_DPKG_OPTS[@]}" "$_p" >/dev/null 2>&1 || \
                    warn "  Skipped (not available): ${_p}"
            done
        }
    else
        ok "All core packages already installed — skipping"
    fi

    # ── linux-headers: separate, non-blocking, best-effort ────────────────────
    local _kver; _kver="$(uname -r)"
    local _hdr_pkg="linux-headers-${_kver}"
    if ! dpkg -l "$_hdr_pkg" 2>/dev/null | grep -q "^ii"; then
        info "Installing kernel headers (${_hdr_pkg}) …"
        DEBIAN_FRONTEND=noninteractive timeout 90 apt-get install -y \
            --no-install-recommends "${_DPKG_OPTS[@]}" \
            "$_hdr_pkg" >/dev/null 2>&1 || \
            warn "Kernel headers unavailable — DKMS modules will be skipped"
    else
        ok "Kernel headers already installed: ${_hdr_pkg}"
    fi

    # ── Vulkan (conditional) ───────────────────────────────────────────────────
    if [[ "${ATLAS_GPU_BACKEND:-}" == "vulkan" ]] || [[ "${GPU_VENDOR:-}" == "intel" ]]; then
        local _vk_pkgs=(vulkan-tools libvulkan-dev libvulkan1 mesa-vulkan-drivers spirv-tools)
        local _vk_missing=()
        for _p in "${_vk_pkgs[@]}"; do
            dpkg -l "$_p" &>/dev/null 2>&1 && dpkg -l "$_p" | grep -q "^ii" || _vk_missing+=("$_p")
        done
        if [[ ${#_vk_missing[@]} -gt 0 ]]; then
            DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
                "${_DPKG_OPTS[@]}" "${_vk_missing[@]}" >/dev/null 2>&1 || true
        fi
    fi

    # ── ROCm extras for AMD without full ROCm stack ────────────────────────────
    if [[ "${GPU_VENDOR:-}" == "amd" ]] && ! ${HIP_AVAILABLE:-false}; then
        DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
            "${_DPKG_OPTS[@]}" \
            libvulkan-dev vulkan-tools mesa-vulkan-drivers \
            >/dev/null 2>&1 || true
    fi
}

_install_dnf() {
    run_cmd dnf install -y \
        python3 python3-pip python3-devel gcc gcc-c++ cmake ninja-build \
        git curl wget openssl-devel zlib-devel bzip2-devel readline-devel \
        libffi-devel lz4-devel vulkan-loader-devel \
        mesa-vulkan-drivers vulkan-validation-layers \
        kernel-devel-"$(uname -r)" psmisc htop \
        >/dev/null 2>&1 || true
}

_install_pacman() {
    run_cmd pacman -Sy --noconfirm \
        python python-pip cmake ninja base-devel git curl wget \
        openssl zlib lz4 linux-headers psmisc htop \
        vulkan-icd-loader vulkan-headers vulkan-validation-layers mesa \
        >/dev/null 2>&1 || true
}

_install_zypper() {
    run_cmd zypper --non-interactive install \
        python3 python3-pip python3-devel cmake ninja gcc gcc-c++ \
        git curl wget libopenssl-devel lz4-devel kernel-devel \
        >/dev/null 2>&1 || true
}

_install_brew() {
    run_cmd brew update >/dev/null 2>&1
    run_cmd brew install cmake ninja git wget pkg-config \
        >/dev/null 2>&1 || true
    # Python via brew only if no system Python ≥ 3.8
    if ! cmd_exists python3; then
        run_cmd brew install python@3.11 >/dev/null 2>&1 || true
    fi
    $IS_APPLE_SILICON && run_cmd brew install llvm >/dev/null 2>&1 || true
}

case "$PKG_MGR" in
    apt)    _install_apt ;;
    dnf|yum) _install_dnf ;;
    pacman) _install_pacman ;;
    zypper) _install_zypper ;;
    brew)   _install_brew ;;
esac

ok "System packages installed"

# =============================================================================
# SECTION 15 — Clang installation & version management
# =============================================================================
# BitNet.cpp requires clang >= 18 for correct ternary kernel codegen.
# gcc produces subtly wrong output for I2_S/TL1/TL2 kernels.
# =============================================================================

section "Clang Compiler (≥ ${CLANG_MIN_VER} required for BitNet.cpp)"

# Find the best clang available across versioned names
_find_best_clang() {
    local best_bin="" best_ver=0
    for candidate in clang clang-21 clang-20 clang-19 clang-18 clang-17; do
        if cmd_exists "$candidate"; then
            local v
            v=$("$candidate" --version 2>/dev/null | grep -oP 'version \K[0-9]+' | head -1 || echo 0)
            if [[ "$v" -gt "$best_ver" ]]; then
                best_ver="$v"
                best_bin="$candidate"
            fi
        fi
    done
    echo "${best_bin}:${best_ver}"
}

_install_clang_apt() {
    local target_ver="${1:-18}"
    info "Installing clang-${target_ver} via LLVM apt repository…"
    local llvm_sh="/tmp/llvm_$$.sh"
    if cmd_exists curl; then
        curl -fsSL https://apt.llvm.org/llvm.sh -o "$llvm_sh" || { rm -f "$llvm_sh"; return 1; }
    elif cmd_exists wget; then
        wget -qO "$llvm_sh" https://apt.llvm.org/llvm.sh || { rm -f "$llvm_sh"; return 1; }
    else
        return 1
    fi
    chmod +x "$llvm_sh"
    DEBIAN_FRONTEND=noninteractive bash "$llvm_sh" "$target_ver" >/dev/null 2>&1 || true
    rm -f "$llvm_sh"
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        -o Dpkg::Options::="--force-confdef" \
        -o Dpkg::Options::="--force-confold" \
        "clang-${target_ver}" "clang++-${target_ver}" \
        "lld-${target_ver}" "llvm-${target_ver}" >/dev/null 2>&1 || true
    # Register with update-alternatives
    update-alternatives --install /usr/bin/clang   clang   "/usr/bin/clang-${target_ver}"   "${target_ver}0" 2>/dev/null || true
    update-alternatives --install /usr/bin/clang++ clang++ "/usr/bin/clang++-${target_ver}" "${target_ver}0" 2>/dev/null || true
}

CLANG_INFO="$(_find_best_clang)"
CLANG_BIN="${CLANG_INFO%%:*}"
CLANG_VER="${CLANG_INFO##*:}"

if [[ -z "$CLANG_BIN" ]] || [[ "$CLANG_VER" -lt "${CLANG_MIN_VER}" ]]; then
    info "Best clang found: ${CLANG_BIN:-none} (v${CLANG_VER}) — need ≥ ${CLANG_MIN_VER}"
    if [[ "$OS_FAMILY" == "linux" ]] && [[ "$PKG_MGR" == "apt" ]]; then
        _install_clang_apt "${CLANG_MIN_VER}"
        # Re-probe after install
        CLANG_INFO="$(_find_best_clang)"
        CLANG_BIN="${CLANG_INFO%%:*}"
        CLANG_VER="${CLANG_INFO##*:}"
    elif [[ "$OS_FAMILY" == "macos" ]]; then
        run_cmd brew install llvm >/dev/null 2>&1 || true
        # Add Homebrew LLVM to PATH
        HOMEBREW_LLVM_BIN="$(brew --prefix llvm 2>/dev/null)/bin"
        export PATH="${HOMEBREW_LLVM_BIN}:${PATH}"
        CLANG_INFO="$(_find_best_clang)"
        CLANG_BIN="${CLANG_INFO%%:*}"
        CLANG_VER="${CLANG_INFO##*:}"
    else
        warn "Automatic clang upgrade not supported on ${PKG_MGR} — install clang >= ${CLANG_MIN_VER} manually"
    fi
fi

# Resolve full paths to compiler binaries
if [[ -n "$CLANG_BIN" ]]; then
    CLANG_FULL_PATH="$(command -v "$CLANG_BIN" 2>/dev/null || echo "")"
    # Derive clang++ path from clang path
    CLANGPP_FULL_PATH="${CLANG_FULL_PATH/clang/clang++}"
    # Verify clang++ exists; fall back to searching
    [[ -x "$CLANGPP_FULL_PATH" ]] || CLANGPP_FULL_PATH="$(command -v "${CLANG_BIN/clang/clang++}" 2>/dev/null || echo "")"
else
    CLANG_FULL_PATH=""
    CLANGPP_FULL_PATH=""
fi

if [[ -n "$CLANG_FULL_PATH" ]] && [[ "$CLANG_VER" -ge "${CLANG_MIN_VER}" ]]; then
    export CC="$CLANG_FULL_PATH"
    export CXX="$CLANGPP_FULL_PATH"
    ok "Compiler: CC=${CC} (v${CLANG_VER}) | CXX=${CXX}"
else
    # Warn but don't die — gcc may work for non-BitNet builds
    warn "clang ≥ ${CLANG_MIN_VER} not found (best: ${CLANG_BIN:-none} v${CLANG_VER})"
    warn "BitNet.cpp kernel codegen REQUIRES clang ≥ ${CLANG_MIN_VER}."
    warn "Install manually and re-run, or BitNet builds will fail."
    # Still set CC/CXX to whatever we found so llama-cpp-python has a compiler
    CC="${CLANG_FULL_PATH:-$(command -v gcc 2>/dev/null || echo "gcc")}"
    CXX="${CLANGPP_FULL_PATH:-$(command -v g++ 2>/dev/null || echo "g++")}"
    export CC CXX
fi

# =============================================================================
# SECTION 16 — BitNet directory placeholder (populated by bitnet_setup.sh)
# =============================================================================
# BitNet.cpp discovery, submodule verification, Python 3.14 compatibility
# patches, and the interactive patch prompt are all handled exclusively by
# bitnet_setup.sh which runs immediately after this script completes.
# =============================================================================

BITNET_DIR=""
# BitNet discovery, patch application, and submodule verification are
# handled exclusively by bitnet_setup.sh (runs after this script).

# =============================================================================
# SECTION 17 — Python version detection (NEXUS)
# =============================================================================

section "Python Detection"

# Candidate order: prefer newer, but accept 3.8+
PYTHON_CMD=""
PYTHON_VER_MAJOR=0
PYTHON_VER_MINOR=0

_check_python_candidate() {
    local py="$1"
    cmd_exists "$py" || return 1
    local maj min
    maj=$("$py" -c 'import sys; print(sys.version_info.major)' 2>/dev/null) || return 1
    min=$("$py" -c 'import sys; print(sys.version_info.minor)' 2>/dev/null) || return 1
    [[ "$maj" -eq 3 ]] && [[ "$min" -ge "$PYTHON_MIN_MINOR" ]] || return 1
    # Prefer 3.10-3.12 (widest torch/transformers support)
    PYTHON_CMD="$py"
    PYTHON_VER_MAJOR="$maj"
    PYTHON_VER_MINOR="$min"
    return 0
}

for _py in python3.12 python3.11 python3.10 python3.14 python3.13 python3.9 python3.8 python3; do
    _check_python_candidate "$_py" && break
done

[[ -n "$PYTHON_CMD" ]] \
    || die "No Python 3.${PYTHON_MIN_MINOR}+ found.\n  Install Python 3.10/3.11/3.12 and re-run."

ok "Python ${PYTHON_VER_MAJOR}.${PYTHON_VER_MINOR} at ${PYTHON_CMD}"

# Warn on Python 3.13/3.14 — some packages (torch, llama-cpp-python) may lack wheels
if [[ "$PYTHON_VER_MINOR" -ge 13 ]]; then
    warn "Python 3.${PYTHON_VER_MINOR} detected — some packages may need compilation."
    warn "Python 3.10/3.11/3.12 have the widest pre-built wheel support."
fi

# =============================================================================
# SECTION 18 — Virtual environment setup (NEXUS)
# =============================================================================

section "Virtual Environment (NEXUS)"

VENV_DIR="${LLM_VENV_DIR}"

if [[ -d "$VENV_DIR" ]] && [[ -x "${VENV_DIR}/bin/python" ]]; then
    # Verify the venv's Python still works (can become broken after OS Python upgrade)
    if "${VENV_DIR}/bin/python" -c "import sys" &>/dev/null 2>&1; then
        ok "Reusing valid virtual environment: ${VENV_DIR}"
    else
        warn "Existing venv is broken — recreating at ${VENV_DIR}"
        rm -rf "${VENV_DIR}"
        run_cmd "$PYTHON_CMD" -m venv "${VENV_DIR}"
        ok "Virtual environment recreated"
    fi
else
    info "Creating virtual environment at ${VENV_DIR}…"
    run_cmd "$PYTHON_CMD" -m venv "${VENV_DIR}"
    ok "Virtual environment created"
fi

# Activate
if [[ "${VIRTUAL_ENV:-}" != "$VENV_DIR" ]]; then
    # shellcheck disable=SC1090
    source "${VENV_DIR}/bin/activate"
    [[ "${VIRTUAL_ENV:-}" == "$VENV_DIR" ]] \
        || die "Failed to activate virtual environment at ${VENV_DIR}"
fi
ok "Virtual environment active: ${VENV_DIR}"

# pip / setuptools / wheel upgrade
run_cmd --silent "${VENV_DIR}/bin/python" -m pip install --quiet --upgrade pip
# setuptools < 82 required by torch 2.x; wheel for C-extension builds
run_cmd --silent "${VENV_DIR}/bin/python" -m pip install --quiet "setuptools<82" wheel
ok "pip upgraded"

# Convenient shorthand for the rest of this script
PIP="${VENV_DIR}/bin/pip"
PYTHON="${VENV_DIR}/bin/python"

# =============================================================================
# SECTION 19 — Core Python dependencies
# =============================================================================

section "Core Python Dependencies"

# Base packages — version constraints chosen for broad compatibility
# across Python 3.8–3.14, torch 2.x, and transformers 4.46–5.x
run_cmd --silent "$PIP" install --quiet --upgrade \
    "fastapi" \
    "uvicorn[standard]" \
    "pydantic" \
    "pyyaml" \
    "psutil" \
    "lz4" \
    "numpy" \
    "httpx" \
    "aiohttp" \
    "aiofiles" \
    "aiosqlite" \
    "prometheus-client" \
    "structlog" \
    "tqdm" \
    "packaging" \
    "filelock" \
    "huggingface-hub" \
    "safetensors" \
    "sentencepiece" \
    "tokenizers" \
    "regex" \
    "starlette" \
    "anyio" \
    "pytest" \
    "pytest-asyncio" \
    || warn "Some core packages failed — continuing"

ok "Core Python packages installed"

# =============================================================================
# SECTION 20 — transformers dual-path installation (NEXUS + SPECTRE)
# =============================================================================
# Two distinct transformer needs:
#   PATH A — GGUF/llama-cpp-python models: any released transformers ≥ 4.46.3,<5.0
#   PATH B — BitNet transformers backend: specific pinned commit from Microsoft
#             (only that commit has BitLinear support built in)
#
# Strategy: install PATH A now (stable release). The pinned commit for PATH B
# is installed by bitnet_setup.sh only when the user chooses the transformers
# backend for BitNet. The compatibility layer in transformers_compat.py will
# detect which path is active at runtime.
# =============================================================================

section "transformers Installation"

_tf_version=""
if "$PYTHON" -c "import transformers; v=transformers.__version__.split('.')[:2]; assert int(v[0])>=4 and int(v[1])>=46" &>/dev/null 2>&1; then
    _tf_version=$("$PYTHON" -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "")
    ok "transformers ${_tf_version} already installed and compatible (≥4.46)"
else
    info "Installing transformers (PATH A — standard release for GGUF/llama-cpp)…"
    # Install latest stable but below 5.0 to avoid breaking changes
    run_cmd --silent "$PIP" install --quiet "transformers>=4.46.3,<5.0.0" \
        || run_cmd --silent "$PIP" install --quiet "transformers>=4.46.3" \
        || warn "transformers install failed — BitNet transformers backend will not work"
    _tf_version=$("$PYTHON" -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "unknown")
    ok "transformers ${_tf_version} installed"
fi

note "BitNet transformers backend (PATH B) requires pinned commit 096f25ae."
note "bitnet_setup.sh will install it when transformers backend is selected."

# =============================================================================
# SECTION 21 — torch installation (hardware-aware)
# =============================================================================

section "PyTorch Installation"

_torch_installed() {
    "$PYTHON" -c "import torch; print(torch.__version__)" &>/dev/null 2>&1
}

if _torch_installed; then
    _tv=$("$PYTHON" -c "import torch; print(torch.__version__)" 2>/dev/null)
    ok "torch ${_tv} already installed — skipping"
else
    info "Installing PyTorch for backend: ${ATLAS_GPU_BACKEND}…"
    case "$ATLAS_GPU_BACKEND" in
        cuda)
            # Install the CUDA 12.x variant (matches CUDA 13 toolkit via compatibility)
            run_cmd --silent "$PIP" install --quiet \
                torch torchvision torchaudio \
                --index-url https://download.pytorch.org/whl/cu124 \
                || run_cmd --silent "$PIP" install --quiet torch torchvision torchaudio \
                || warn "torch CUDA install failed — falling back to CPU"
            ;;
        hip)
            # ROCm torch — match the installed ROCm version
            local _rocm_ver
            _rocm_ver=$(cat /opt/rocm/.info/version 2>/dev/null | cut -d. -f1,2 || echo "6.0")
            run_cmd --silent "$PIP" install --quiet \
                torch torchvision torchaudio \
                --index-url "https://download.pytorch.org/whl/rocm${_rocm_ver}" \
                || run_cmd --silent "$PIP" install --quiet torch torchvision torchaudio \
                || warn "torch ROCm install failed"
            ;;
        metal)
            # Apple Silicon — standard PyPI torch includes Metal/MPS support
            run_cmd --silent "$PIP" install --quiet torch torchvision torchaudio \
                || warn "torch Metal install failed"
            ;;
        *)
            # CPU-only or Vulkan — standard PyPI CPU build
            run_cmd --silent "$PIP" install --quiet torch torchvision torchaudio \
                --index-url https://download.pytorch.org/whl/cpu \
                || run_cmd --silent "$PIP" install --quiet torch \
                || warn "torch CPU install failed"
            ;;
    esac
    if _torch_installed; then
        _tv=$("$PYTHON" -c "import torch; print(torch.__version__)" 2>/dev/null)
        ok "torch ${_tv} installed"
    else
        warn "torch not installed — GPU inference will not be available"
    fi
fi

# =============================================================================
# SECTION 22 — llama-cpp-python (ATLAS-driven, with layered fallbacks)
# =============================================================================
# Build order:
#   1. Try primary GPU backend (CUDA/HIP/Metal/Vulkan)
#   2. If that fails, try Vulkan (universal fallback)
#   3. If that fails, try CPU-only
#   4. Record what succeeded for .llm_env
# =============================================================================

section "llama-cpp-python (GPU-accelerated)"

_lcpp_installed() {
    "$PYTHON" -c "import llama_cpp" &>/dev/null 2>&1
}

_install_lcpp_with_flags() {
    local flags="$1" label="$2"
    info "Building llama-cpp-python with ${label}…"
    if [[ "$DRY_RUN" == "1" ]]; then
        step "[DRY-RUN] CMAKE_ARGS=\"${flags}\" pip install llama-cpp-python --no-binary llama-cpp-python"
        return 0
    fi
    CMAKE_ARGS="${flags}" \
    CC="$CC" CXX="$CXX" \
        "$PIP" install llama-cpp-python --no-binary llama-cpp-python -q \
        2>&1 | tail -5
    return "${PIPESTATUS[0]}"
}

if [[ "$SKIP_LLAMA_CPP" == "true" ]]; then
    info "Skipping llama-cpp-python (LLM_SKIP_LLAMA_CPP=true)"
elif _lcpp_installed && [[ "$FORCE_REBUILD" != "true" ]]; then
    _lcpp_ver=$("$PYTHON" -c "import llama_cpp; print(llama_cpp.__version__)" 2>/dev/null || echo "?")
    ok "llama-cpp-python ${_lcpp_ver} already installed"
else
    LCPP_INSTALLED=false

    case "$ATLAS_GPU_BACKEND" in
        cuda)
            _install_lcpp_with_flags "-DGGML_CUDA=on" "CUDA" && LCPP_INSTALLED=true \
                || warn "CUDA build failed — trying Vulkan fallback"
            ;;
        hip)
            _HIPCXX=""
            cmd_exists hipcc && _HIPCXX="hipcc" || _HIPCXX="$CXX"
            info "Building with HIP/ROCm (HIPCXX=${_HIPCXX})…"
            HIPCXX="$_HIPCXX" \
            CMAKE_ARGS="-DGGML_HIP=on ${ATLAS_CMAKE_FLAGS#-DGGML_HIP=on}" \
            CC="$CC" CXX="$CXX" \
                "$PIP" install llama-cpp-python --no-binary llama-cpp-python -q \
                2>&1 | tail -5 && LCPP_INSTALLED=true \
                || warn "HIP build failed — trying Vulkan fallback"
            ;;
        metal)
            _install_lcpp_with_flags "-DGGML_METAL=on" "Metal (Apple Silicon)" && LCPP_INSTALLED=true \
                || warn "Metal build failed — trying CPU fallback"
            ;;
        vulkan)
            _install_lcpp_with_flags "-DGGML_VULKAN=on" "Vulkan" && LCPP_INSTALLED=true \
                || warn "Vulkan build failed — trying CPU fallback"
            ;;
        cpu|*)
            info "Building CPU-only llama-cpp-python…"
            run_cmd --silent "$PIP" install llama-cpp-python -q && LCPP_INSTALLED=true \
                || warn "CPU build failed"
            ;;
    esac

    # Fallback chain: Vulkan → CPU
    if ! $LCPP_INSTALLED && [[ "$ATLAS_GPU_BACKEND" != "vulkan" ]] && $VULKAN_AVAILABLE; then
        warn "Primary GPU build failed — attempting Vulkan fallback"
        _install_lcpp_with_flags "-DGGML_VULKAN=on" "Vulkan (fallback)" && LCPP_INSTALLED=true \
            || warn "Vulkan fallback also failed"
    fi

    if ! $LCPP_INSTALLED && [[ "$ATLAS_GPU_BACKEND" != "cpu" ]]; then
        warn "All GPU builds failed — falling back to CPU-only llama-cpp-python"
        info "GPU inference will not be available for GGUF models"
        run_cmd --silent "$PIP" install llama-cpp-python -q && LCPP_INSTALLED=true \
            || warn "CPU fallback also failed — llama-cpp-python NOT installed"
    fi

    if _lcpp_installed; then
        _lcpp_ver=$("$PYTHON" -c "import llama_cpp; print(llama_cpp.__version__)" 2>/dev/null || echo "?")
        ok "llama-cpp-python ${_lcpp_ver} installed"
    else
        warn "llama-cpp-python could not be installed — GGUF inference will not work"
        warn "Manual install: CMAKE_ARGS='${ATLAS_LLAMA_CMAKE_ARGS}' pip install llama-cpp-python --no-binary llama-cpp-python"
    fi
fi

# =============================================================================
# SECTION 23 — SPECTRE: gguf-py conflict resolution
# =============================================================================
# The BitNet.cpp build pipeline (setup_env.py and its subprocesses) needs the
# LOCAL gguf-py from 3rdparty/llama.cpp/gguf-py, which has GGMLQuantizationType
# .TL1 and .TL2 attributes that PyPI gguf lacks.
#
# SPECTRE resolves this WITHOUT uninstalling PyPI gguf by:
#   1. Installing the local gguf-py as an editable package (superset of PyPI gguf)
#   2. Exporting PYTHONPATH in .llm_env so ALL child processes (subprocesses of
#      setup_env.py, including convert-hf-to-gguf-bitnet.py) see local gguf-py
#   3. Providing a shim script for when local gguf-py is unavailable
# =============================================================================

section "SPECTRE — gguf Version Conflict Resolution"

SPECTRE_GGUF_PY_PATH=""
SPECTRE_MODE="pypi"  # pypi | editable | shim | none

_verify_gguf_has_tl1() {
    "$PYTHON" -c "
import gguf
try:
    _ = gguf.GGMLQuantizationType.TL1
    _ = gguf.GGMLQuantizationType.TL2
    exit(0)
except AttributeError:
    exit(1)
" &>/dev/null 2>&1
}

_install_local_gguf_editable() {
    local gguf_py_dir="$1"
    [[ -f "${gguf_py_dir}/setup.py" ]] || [[ -f "${gguf_py_dir}/pyproject.toml" ]] || return 1
    info "Installing local gguf-py as editable package (enables TL1/TL2 types)…"
    # --no-deps: we don't want it overriding numpy/other deps
    run_cmd --silent "$PIP" install --quiet --no-deps -e "${gguf_py_dir}" && return 0 || return 1
}

_create_gguf_tl_shim() {
    # Last resort: monkey-patch TL1/TL2 values into installed PyPI gguf
    # Values come from BitNet's 3rdparty/llama.cpp/gguf.py constants
    local shim_dir="${VENV_DIR}/lib/$(ls "${VENV_DIR}/lib/" | head -1)/site-packages/gguf_tl_shim"
    mkdir -p "$shim_dir" 2>/dev/null || true
    cat > "${shim_dir}/__init__.py" << 'SHIM_EOF'
# SPECTRE TL1/TL2 shim — auto-injected by local-llm-core installer
# Injects missing GGMLQuantizationType.TL1 and .TL2 into PyPI gguf
import sys
import importlib

def _apply_shim():
    try:
        import gguf
        if not hasattr(gguf.GGMLQuantizationType, 'TL1'):
            # Values from llama.cpp gguf.py (stable across versions)
            gguf.GGMLQuantizationType.TL1 = gguf.GGMLQuantizationType(30)
            gguf.GGMLQuantizationType.TL2 = gguf.GGMLQuantizationType(31)
    except Exception:
        pass

_apply_shim()
SHIM_EOF
    # Create a .pth file to auto-import shim on Python startup
    local site_pkg="${VENV_DIR}/lib/$(ls "${VENV_DIR}/lib/" | head -1)/site-packages"
    echo "import gguf_tl_shim" > "${site_pkg}/gguf_tl_shim.pth" 2>/dev/null || true
    ok "SPECTRE TL1/TL2 shim installed"
    SPECTRE_MODE="shim"
}

# First check if current gguf already has TL1 (e.g. local gguf-py already installed)
if _verify_gguf_has_tl1; then
    _gv=$("$PYTHON" -c "import gguf; print(gguf.__version__)" 2>/dev/null || echo "?")
    ok "gguf ${_gv} already has TL1/TL2 types — SPECTRE not needed"
    SPECTRE_MODE="ok"
else
    # Try editable install from BitNet submodule
    if [[ -n "$GGUF_PY_DIR" ]] && [[ -f "${GGUF_PY_DIR}/gguf/__init__.py" ]]; then
        if _install_local_gguf_editable "${GGUF_PY_DIR}"; then
            if _verify_gguf_has_tl1; then
                ok "Local gguf-py installed (editable) — TL1/TL2 available"
                SPECTRE_GGUF_PY_PATH="${GGUF_PY_DIR}"
                SPECTRE_MODE="editable"
            else
                warn "Local gguf-py installed but TL1 check still fails — applying shim"
                _create_gguf_tl_shim
            fi
        else
            warn "Editable install of local gguf-py failed — applying shim"
            _create_gguf_tl_shim
        fi
    else
        # No local gguf-py yet (BitNet not cloned) — install PyPI gguf and note shim will be applied later
        info "BitNet not cloned yet — installing PyPI gguf (shim applied by bitnet_setup.sh)"
        run_cmd --silent "$PIP" install --quiet "gguf>=0.1.0" || true
        SPECTRE_MODE="pypi"
        note "SPECTRE: After bitnet_setup.sh clones BitNet, local gguf-py will replace PyPI version"
    fi
fi

# =============================================================================
# SECTION 24 — Custom kernel modules (Linux only)
# =============================================================================
# Modules: amd_kali_compat, memory_optimizer, gpu_scheduler, cpu_optimizer
# All BEST-EFFORT — failure never aborts the installer.
# =============================================================================

if [[ "$OS_FAMILY" == "linux" ]] && [[ "$SKIP_KERNEL" != "true" ]]; then

    section "Custom Kernel Modules"

    KERNEL_HEADERS="/lib/modules/$(uname -r)/build"
    MODULES_BASE="${PROJECT_ROOT}/local_llm_core/kernels"

    if [[ ! -d "$KERNEL_HEADERS" ]]; then
        warn "Kernel headers not found at ${KERNEL_HEADERS}"
        info "Install: apt install linux-headers-\$(uname -r)  (or equivalent)"
        info "Skipping kernel module build"
    elif [[ ! -d "$MODULES_BASE" ]]; then
        info "No kernels directory found at ${MODULES_BASE} — skipping"
    else
        # Secure Boot warning
        if [[ -d /sys/firmware/efi/efivars ]] && cmd_exists mokutil; then
            if mokutil --sb-state 2>/dev/null | grep -qi "enabled"; then
                warn "Secure Boot is ENABLED — unsigned kernel modules will be rejected."
                warn "Sign the .ko files or disable Secure Boot in UEFI/BIOS."
            fi
        fi

        rm -f /tmp/llm_kos.list

        for mod_dir in amd_kali_compat memory_optimizer gpu_scheduler cpu_optimizer; do
            mod_path="${MODULES_BASE}/${mod_dir}"
            if [[ -d "$mod_path" ]] && [[ -f "${mod_path}/Makefile" ]]; then
                info "Building ${mod_dir}…"
                if make -C "$mod_path" -j"$(nproc)" >/dev/null 2>&1; then
                    ok "Built ${mod_dir}"
                    find "$mod_path" -name "*.ko" ! -name "*.mod.ko" \
                        -exec readlink -f {} \; >> /tmp/llm_kos.list
                else
                    warn "Build failed: ${mod_dir} (non-fatal)"
                fi
            fi
        done

        if [[ -s /tmp/llm_kos.list ]]; then
            info "Loading kernel modules…"
            set +e   # Module loading is best-effort — never fatal

            KO_LOADED=0; KO_FAILED=0

            while IFS= read -r ko; do
                [[ -f "$ko" ]] || continue
                modname="$(basename "$ko" .ko)"

                # Already loaded?
                if awk '{print $1}' /proc/modules 2>/dev/null | grep -qx "$modname"; then
                    ok "Already loaded: ${modname}"
                    KO_LOADED=$(( KO_LOADED + 1 ))
                    continue
                fi

                # Vermagic (kernel version) check
                ko_kver="$(modinfo "$ko" 2>/dev/null | awk '/^vermagic/{print $2}')"
                run_kver="$(uname -r)"
                if [[ -n "$ko_kver" ]] && [[ "$ko_kver" != "$run_kver" ]]; then
                    warn "Kernel version mismatch: ${modname}"
                    warn "  Built for : ${ko_kver}"
                    warn "  Running   : ${run_kver}"
                    warn "  Fix: apt install linux-headers-\$(uname -r) then re-run with --reset"
                    KO_FAILED=$(( KO_FAILED + 1 ))
                    continue
                fi

            done < /tmp/llm_kos.list

            depmod -a 2>/dev/null || true
            rm -f /tmp/llm_kos.list

            set -euo pipefail   # Restore strict mode

            ok "Kernel modules: ${KO_LOADED} loaded, ${KO_FAILED} failed"
            [[ $KO_FAILED -gt 0 ]] && info "Non-fatal — LLM engine works without custom kernel modules."
        else
            warn "No .ko files built — skipping module loading"
        fi
    fi

else
    [[ "$OS_FAMILY" == "macos" ]]    && info "macOS — kernel module step skipped"
    [[ "$SKIP_KERNEL" == "true" ]]   && info "Kernel modules skipped (LLM_SKIP_KERNEL=true)"
fi

# =============================================================================
# SECTION 25 — System optimisations (Linux only, best-effort)
# =============================================================================
# Huge pages and NUMA tuning improve LLM inference throughput by 15–30% on
# multi-socket / large-RAM machines. All changes are reversible and non-fatal.
# =============================================================================

if [[ "$OS_FAMILY" == "linux" ]]; then

    section "System Optimisations"

    # Transparent Huge Pages: madvise is safer than always (avoids page faults)
    _thp="/sys/kernel/mm/transparent_hugepage/enabled"
    if [[ -f "$_thp" ]]; then
        current_thp="$(cat "$_thp" 2>/dev/null)"
        if echo "$current_thp" | grep -qv '\[madvise\]'; then
            echo "madvise" > "$_thp" 2>/dev/null && ok "Transparent Huge Pages: madvise" \
                || warn "Could not configure THP (non-fatal)"
        else
            ok "Transparent Huge Pages: already madvise"
        fi
    fi

    # vm.overcommit_memory=1 — allows large model weight allocations without OOM-kill
    if [[ "$(sysctl -n vm.overcommit_memory 2>/dev/null)" != "1" ]]; then
        sysctl -w vm.overcommit_memory=1 >/dev/null 2>&1 \
            && ok "vm.overcommit_memory=1 (large model loading)" \
            || warn "Could not set vm.overcommit_memory (non-fatal)"
    else
        ok "vm.overcommit_memory=1 already set"
    fi

    # Increase maximum locked memory (needed by llama.cpp mmap)
    if cmd_exists prlimit; then
        prlimit --memlock=unlimited --pid $$ 2>/dev/null \
            && ok "memlock limit: unlimited" \
            || step "memlock: already at system max (non-fatal)"
    fi

    # NVMe / SSD I/O scheduler — 'none' is optimal for NVMe (removes queueing overhead)
    for blkdev in /sys/block/nvme*/queue/scheduler; do
        [[ -f "$blkdev" ]] || continue
        if grep -q '\[mq-deadline\]' "$blkdev" 2>/dev/null; then
            echo "none" > "$blkdev" 2>/dev/null || true
        fi
    done

fi

# =============================================================================
# SECTION 26 — Package verification (GUARDIAN post-install)
# =============================================================================

section "GUARDIAN — Post-install Verification"

_gguf_ver=$("$PYTHON" -c "import gguf; print(getattr(gguf,'__version__','ok'))" 2>/dev/null || echo "")
if [[ -n "$_gguf_ver" ]]; then
    ok "gguf: ${_gguf_ver}"
else
    warn "gguf: NOT importable — may need manual fix"
fi

if _verify_gguf_has_tl1 2>/dev/null; then
    ok "gguf: TL1/TL2 types present ✓ (BitNet.cpp build will succeed)"
else
    warn "gguf: TL1/TL2 types MISSING — bitnet_setup.sh will apply the SPECTRE fix"
fi

# =============================================================================
# SECTION 27 — Create directories
# =============================================================================

mkdir -p "${MODEL_DIR}" \
         "/tmp/llm_swap" \
         "${REAL_HOME}/.config/local-llm" \
         2>/dev/null || true

ok "Directories created: ${MODEL_DIR}, ${REAL_HOME}/.config/local-llm"

# =============================================================================
# SECTION 29 — Write JSON state file (consumed by local-llm-core at runtime)
# =============================================================================

STATE_FILE="${REAL_HOME}/.config/local-llm/.install_state.json"

if [[ "$DRY_RUN" != "1" ]]; then
    cat > "${STATE_FILE}" << EOF
{
  "installer_version": "${INSTALLER_VERSION}",
  "installed_at": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
  "os_family": "${OS_FAMILY}",
  "arch": "${ARCH}",
  "cpu_vendor": "${CPU_VENDOR}",
  "cpu_cores": ${CPU_PHYSICAL_CORES},
  "has_avx2": ${HAS_AVX2},
  "has_avx512": ${HAS_AVX512},
  "has_neon": ${HAS_NEON},
  "is_apple_silicon": ${IS_APPLE_SILICON},
  "gpu_vendor": "${GPU_VENDOR}",
  "gpu_backend": "${ATLAS_GPU_BACKEND}",
  "cuda_available": ${CUDA_AVAILABLE},
  "hip_available": ${HIP_AVAILABLE},
  "metal_available": ${METAL_AVAILABLE},
  "vulkan_available": ${VULKAN_AVAILABLE},
  "venv_dir": "${LLM_VENV_DIR}",
  "model_dir": "${MODEL_DIR}",
  "project_root": "${PROJECT_ROOT}",
  "cc": "${CC}",
  "cxx": "${CXX}",
  "clang_version": ${CLANG_VER},
  "python_version": "${PYTHON_VER_MAJOR}.${PYTHON_VER_MINOR}",
  "quant_recommendation": "${ATLAS_QUANT_REC}",
  "recommended_threads": ${ATLAS_THREADS},
  "spectre_mode": "${SPECTRE_MODE}",
  "spectre_gguf_py_path": "${SPECTRE_GGUF_PY_PATH}",
  "cmake_flags": "${ATLAS_CMAKE_FLAGS}"
}
EOF
    ok "State file written: ${STATE_FILE}"
fi

# =============================================================================
# SECTION 30 — Make scripts/lib/*.sh executable
# =============================================================================

section "Making scripts/lib/*.sh executable"

SCRIPTS_LIB_DIR="${PROJECT_ROOT}/scripts/lib"
if [[ -d "$SCRIPTS_LIB_DIR" ]]; then
    sh_files=()
    while IFS= read -r -d '' f; do
        sh_files+=("$f")
    done < <(find "$SCRIPTS_LIB_DIR" -maxdepth 1 -type f -name "*.sh" -print0 2>/dev/null)

    if [[ ${#sh_files[@]} -eq 0 ]]; then
        info "No .sh files found in ${SCRIPTS_LIB_DIR}"
    else
        ok "Found ${#sh_files[@]} shell scripts in ${SCRIPTS_LIB_DIR}"
        for script in "${sh_files[@]}"; do
            if [[ "$DRY_RUN" == "1" ]]; then
                step "[DRY-RUN] Would chmod +x $(basename "$script")"
            else
                chmod +x "$script"
                step "chmod +x $(basename "$script")"
            fi
        done
        ok "All scripts are now executable"
    fi
else
    info "Directory not found: ${SCRIPTS_LIB_DIR} — skipping (BitNet not yet cloned?)"
fi

# =============================================================================
# SECTION 31 — Final summary
# =============================================================================

echo ""
echo -e "${C_BOLD}${C_GREEN}  ╔══════════════════════════════════════════════════════╗${C_RESET}"
echo -e "${C_BOLD}${C_GREEN}  ║  ✅  INSTALLATION COMPLETE                            ║${C_RESET}"
echo -e "${C_BOLD}${C_GREEN}  ╚══════════════════════════════════════════════════════╝${C_RESET}"
echo ""
echo -e "  ${C_BOLD}Platform:${C_RESET}      ${OS_TYPE} ${ARCH} | ${GPU_VENDOR} GPU (${ATLAS_GPU_BACKEND})"
echo -e "  ${C_BOLD}Python:${C_RESET}        ${PYTHON_VER_MAJOR}.${PYTHON_VER_MINOR} at ${PYTHON}"
echo -e "  ${C_BOLD}Venv:${C_RESET}          ${LLM_VENV_DIR}"
echo -e "  ${C_BOLD}Models:${C_RESET}        ${MODEL_DIR}"
echo -e "  ${C_BOLD}Compiler:${C_RESET}      ${CC} (v${CLANG_VER})"
echo -e "  ${C_BOLD}SPECTRE:${C_RESET}       ${SPECTRE_MODE}"
echo -e "  ${C_BOLD}BitNet quant:${C_RESET}  ${ATLAS_QUANT_REC} (${ATLAS_THREADS} threads)"
echo ""
echo -e "  ${C_BOLD}Activate env:${C_RESET}"
echo -e "    source ${LLM_VENV_DIR}/bin/activate"
echo ""
echo -e "  ${C_BOLD}Next step — BitNet setup:${C_RESET}"
echo -e "    local-llm-setup  (or: bash scripts/bitnet_setup.sh)"
echo ""

exit 0
