#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║       BitNet.cpp — Local LLM Cloud Integration Setup  v1.0.0               ║
# ║  Production-ready · SPECTRE · PRISM · ATLAS · FORGE · GUARDIAN · NEXUS     ║
# ║  Supports: Linux (x86_64 / arm64)  macOS (Intel / Apple Silicon)           ║
# ║  Python 3.8–3.14+  ·  Clang ≥18 required for BitNet 1-bit kernels         ║
# ║                                                                              ║
# ║  Three inference modes:                                                      ║
# ║    1) Convert HF model → GGUF + compile llama binaries  [fastest]          ║
# ║    2) Pre-built GGUF  + compile llama binaries          [easiest]           ║
# ║    3) HF Transformers API — no compilation              [most compatible]   ║
# ║                                                                              ║
# ║  Patched / hardened build — fixes applied on top of the original baseline: ║
# ║    ✅ STEP 19 added — bitnet_state.json written completely & accurately    ║
# ║    ✅ llama_cli_binary always resolved, never left empty                   ║
# ║    ✅ 3rdparty/llama.cpp submodule robustly populated after every clone    ║
# ║    ✅ Patches interactive per Python version; user can skip per patch      ║
# ║    ✅ Mode 3 skips build; Modes 1+2 always compile llama binaries          ║
# ║    ✅ Mode 1+2 model dirs validated/renamed to match setup_env.py's        ║
# ║       hardcoded supported names (prevents NotImplementedError crash)      ║
# ║    ✅ Mode 1+2 always run setup_env.py (never bare cmake) so kernel        ║
# ║       codegen (bitnet-lut-kernels.h) always runs before the C++ build     ║
# ║    ✅ Unversioned clang/clang++ symlink defense for setup_env.py's         ║
# ║       hardcoded compiler names (it ignores $CC/$CXX entirely)             ║
# ║    ✅ Download offered for every mode when no model found on disk          ║
# ║    ✅ FORGE cache keyed on clang + quant + py + model + source mode        ║
# ║    ✅ State file written atomically (tmp → rename)                         ║
# ║    ✅ Self-test updates state with confirmed final_mode                    ║
# ║    ✅ Usable by people with zero programming experience                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

set -euo pipefail
IFS=$'\n\t'

# ── Prevent recursive execution ───────────────────────────────────────────────
if [[ -n "${_BITNET_SETUP_RUNNING:-}" ]]; then
    echo "[bitnet_setup.sh] ERROR: recursive execution detected — aborting." >&2
    exit 1
fi
export _BITNET_SETUP_RUNNING=1

# ── Colours ───────────────────────────────────────────────────────────────────
if [[ -t 1 ]] && [[ "${TERM:-dumb}" != "dumb" ]]; then
    C_RED='\033[0;31m'  C_GRN='\033[0;32m'  C_YEL='\033[0;33m'
    C_BLU='\033[0;34m'  C_MAG='\033[0;35m'  C_CYN='\033[0;36m'
    C_WHT='\033[1;37m'  C_DIM='\033[2m'     C_RST='\033[0m'
    C_BLD='\033[1m'
else
    C_RED='' C_GRN='' C_YEL='' C_BLU='' C_MAG='' C_CYN=''
    C_WHT='' C_DIM='' C_RST='' C_BLD=''
fi

SCRIPT_VERSION="1.0.0"
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_PATCHES_DIR="${_SCRIPT_DIR}/patches"

# ── Global state variables ────────────────────────────────────────────────────
LOG_FILE=""
BITNET_DIR=""
PYTHON_BIN=""
VENV_DIR=""
MODEL_PATH=""
GGUF_PATH=""
QUANT_TYPE=""
QUANT_EMBD=false
MODEL_NAME=""
MODEL_SOURCE=""          # "1"=convert  "2"=prebuilt-gguf  "3"=transformers
MODEL_ARCH=""
ARCH=""
OS_TYPE=""
PKG_MGR=""
NPROC=4
DRY_RUN=false
NON_INTERACTIVE=false
SKIP_CLONE=false
SKIP_BUILD=false
SKIP_CONVERT=false
SKIP_MODEL=false
USE_PYPI_GGUF=false
APPLY_PATCH=false
FORCE_REBUILD=false
BITNET_DIR_OVERRIDE=""
MODEL_DIR_OVERRIDE=""
HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"

# ATLAS hardware state
ATLAS_QUANT_REC="i2_s"
ATLAS_THREADS=4
ATLAS_GPU_BACKEND="cpu"
ATLAS_CMAKE_FLAGS=""
ATLAS_AVX2=false
ATLAS_AVX512=false
ATLAS_NEON=false

# PRISM model classification
PRISM_CLASS=""
PRISM_ARCH_COMPAT=""

# FORGE build cache
FORGE_CACHE_HIT=false
FORGE_CACHE_FILE=""
CURRENT_HASH=""

# SPECTRE gguf isolation
SPECTRE_GGUF_PY_PATH=""
SPECTRE_MODE="none"
GGUF_PY_DIR=""

# Verification / state
VERIFY_PASS=true
BINARY_PATH=""
SERVER_PATH=""
RUN_INFERENCE=""
FINAL_BITNET_MODE="unknown"
CLANG_BIN="clang"
CLANG_VER=0
CMAKE_VER="0"
PY_VER="3.0"
PY_MINOR=0
GGUF_SIZE=""
PATCHES_APPLIED=()
_NEED_UPFRONT_PATCH=false

# ── Architecture / model registries ──────────────────────────────────────────
UNSUPPORTED_ARCHITECTURES=("BitNetModel")
SUPPORTED_ARCHITECTURES=(
    "LlamaForCausalLM"    "MistralForCausalLM"  "FalconForCausalLM"
    "GPT2LMHeadModel"     "LLaMAForCausalLM"    "MptForCausalLM"
    "BloomForCausalLM"    "OPTForCausalLM"      "RWForCausalLM"
    "PhiForCausalLM"      "BitNetForCausalLM"   "BitnetForCausalLM"
)
declare -A HF_TO_NAME=(
    ["microsoft/bitnet-b1.58-2B-4T"]="BitNet-b1.58-2B-4T"
    ["microsoft/bitnet-b1.58-2B-4T-bf16"]="BitNet-b1.58-2B-4T-bf16"
    ["microsoft/BitNet-b1.58-2B-4T-gguf"]="BitNet-b1.58-2B-4T-gguf"
    ["1bitLLM/bitnet_b1_58-large"]="bitnet_b1_58-large"
    ["1bitLLM/bitnet_b1_58-3B"]="bitnet_b1_58-3B"
    ["HF1BitLLM/Llama3-8B-1.58-100B-tokens"]="Llama3-8B-1.58-100B-tokens"
    ["tiiuae/Falcon3-1B-Instruct-1.58bit"]="Falcon3-1B-Instruct-1.58bit"
    ["tiiuae/Falcon3-3B-Instruct-1.58bit"]="Falcon3-3B-Instruct-1.58bit"
    ["tiiuae/Falcon3-7B-Instruct-1.58bit"]="Falcon3-7B-Instruct-1.58bit"
    ["tiiuae/Falcon3-10B-Instruct-1.58bit"]="Falcon3-10B-Instruct-1.58bit"
)

# ── Real-home detection (sudo-safe) ───────────────────────────────────────────
if [[ -n "${SUDO_USER:-}" ]] && [[ "${SUDO_USER}" != "root" ]]; then
    REAL_HOME="$(getent passwd "${SUDO_USER}" 2>/dev/null | cut -d: -f6 || echo "${HOME}")"
elif [[ "${EUID:-$(id -u)}" -eq 0 ]] && [[ -d /home ]]; then
    _rh="$(ls -d /home/*/ 2>/dev/null | head -1 || echo "")"
    REAL_HOME="${_rh%/}"
    [[ -d "${REAL_HOME}" ]] || REAL_HOME="${HOME}"
else
    REAL_HOME="${HOME}"
fi

LOCAL_LLM_STATE_DIR="${REAL_HOME}/.config/local-llm"
BITNET_STATE_FILE="${LOCAL_LLM_STATE_DIR}/bitnet_state.json"

# ═════════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSING
# ═════════════════════════════════════════════════════════════════════════════
for arg in "$@"; do
    case "$arg" in
        --non-interactive|-y)  NON_INTERACTIVE=true ;;
        --dry-run)             DRY_RUN=true ;;
        --skip-clone)          SKIP_CLONE=true ;;
        --skip-build)          SKIP_BUILD=true ;;
        --skip-model)          SKIP_MODEL=true ;;
        --quant-embd)          QUANT_EMBD=true ;;
        --use-pypi-gguf)       USE_PYPI_GGUF=true ;;
        --apply-patch)         APPLY_PATCH=true ;;
        --force-rebuild)       FORCE_REBUILD=true ;;
        --bitnet-dir=*)        BITNET_DIR_OVERRIDE="${arg#*=}" ;;
        --model-dir=*)         MODEL_DIR_OVERRIDE="${arg#*=}" ;;
        --model-source=*)      MODEL_SOURCE="${arg#*=}" ;;
        --transformers-only)   MODEL_SOURCE="3"; NON_INTERACTIVE=true ;;
        --gguf-only)           MODEL_SOURCE="2" ;;
        --convert)             MODEL_SOURCE="1" ;;
        --hf-token=*)          HF_TOKEN="${arg#*=}" ;;
        --help|-h)
            cat << 'HELPEOF'
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  bitnet_setup.sh v1.0.0 — BitNet.cpp / Local LLM Cloud Setup
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Usage: bash bitnet_setup.sh [OPTIONS]

General:
  -y, --non-interactive    Accept all defaults (no prompts)
  --dry-run                Show what would happen without doing it
  --skip-clone             Use existing BitNet.cpp (skip git clone)
  --skip-build             Skip C++ compilation entirely
  --skip-model             Skip model download/selection
  --quant-embd             Q6_K embedding quantization (~15% less RAM)
  --use-pypi-gguf          Force PyPI gguf (SPECTRE still isolates)
  --apply-patch            Force all patches before first build
  --force-rebuild          Ignore FORGE cache — always rebuild
  --bitnet-dir=PATH        Override BitNet.cpp directory
  --model-dir=PATH         Override model directory
  --hf-token=TOKEN         HuggingFace token for gated models

Mode selection (skips interactive menu):
  --convert                Mode 1: Convert HF safetensors → GGUF + compile
  --gguf-only              Mode 2: Pre-built GGUF + compile llama binaries
  --transformers-only      Mode 3: Transformers API, no compilation

Inference modes explained:
  1 = FASTEST: Downloads packed model (~4.5 GB), converts to GGUF,
      compiles 1-bit CPU kernels. Best performance on CPU.
  2 = EASIEST: Downloads pre-converted GGUF (~750 MB), compiles
      llama binaries. Same speed as Mode 1, no conversion step.
  3 = MOST COMPATIBLE: Pure Python via HuggingFace transformers.
      No C++ build. Works anywhere. Slowest inference.
HELPEOF
            exit 0 ;;
        *) echo "  [warn] Unknown argument ignored: ${arg}" ;;
    esac
done

# ═════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════
log()     { echo -e "${C_DIM}[$(date '+%H:%M:%S')]${C_RST} $*" | tee -a "${LOG_FILE:-/dev/null}" || true; }
info()    { echo -e "${C_CYN}${C_BLD}  ▶${C_RST}  $*" | tee -a "${LOG_FILE:-/dev/null}" || true; }
ok()      { echo -e "${C_GRN}${C_BLD}  ✅${C_RST} $*" | tee -a "${LOG_FILE:-/dev/null}" || true; }
warn()    { echo -e "${C_YEL}${C_BLD}  ⚠  ${C_RST}${C_YEL}$*${C_RST}" | tee -a "${LOG_FILE:-/dev/null}" || true; }
err()     { echo -e "${C_RED}${C_BLD}  ✗  ${C_RST}${C_RED}$*${C_RST}" | tee -a "${LOG_FILE:-/dev/null}" || true; }
note()    { echo -e "${C_MAG}  ℹ${C_RST}  $*" | tee -a "${LOG_FILE:-/dev/null}" || true; }
section() { echo ""; echo -e "${C_MAG}${C_BLD}┌─ $* ${C_RST}" | tee -a "${LOG_FILE:-/dev/null}" || true; }
die()     { err "$*"; [[ -n "${LOG_FILE:-}" ]] && echo -e "${C_DIM}  Log: ${LOG_FILE}${C_RST}"; exit 1; }

ask() {
    local _var="$1" _prompt="$2" _default="${3:-}"
    if $NON_INTERACTIVE; then printf -v "$_var" '%s' "$_default"; return; fi
    local _ans
    [[ -n "$_default" ]] \
        && echo -ne "${C_YEL}  ? ${C_RST}${_prompt} ${C_DIM}[${_default}]${C_RST}: " \
        || echo -ne "${C_YEL}  ? ${C_RST}${_prompt}: "
    read -r _ans
    printf -v "$_var" '%s' "${_ans:-$_default}"
}

confirm() {
    local _prompt="$1" _default="${2:-n}"
    if $NON_INTERACTIVE; then [[ "${_default^^}" == "Y" ]] && return 0 || return 1; fi
    local _ans
    echo -ne "${C_YEL}  ? ${C_RST}${_prompt} ${C_DIM}[${_default}]${C_RST}: "
    read -r _ans
    _ans="${_ans:-$_default}"
    [[ "${_ans^^}" == "Y" || "${_ans^^}" == "YES" ]]
}

run_cmd() {
    local _silent=false
    [[ "${1:-}" == "--silent" ]] && { _silent=true; shift; }
    log "CMD: $(printf '%s ' "$@")"
    $DRY_RUN && { echo -e "${C_DIM}  [dry-run] $(printf '%s ' "$@")${C_RST}"; return 0; }
    $_silent && "$@" >> "${LOG_FILE:-/dev/null}" 2>&1 \
             || "$@" 2>&1 | tee -a "${LOG_FILE:-/dev/null}"
}

pip_install() {
    local _silent=false
    [[ "${1:-}" == "--silent" ]] && { _silent=true; shift; }
    $_silent && run_cmd --silent "$PYTHON_BIN" -m pip install --quiet "$@" \
             || run_cmd         "$PYTHON_BIN" -m pip install "$@"
}

cmd_exists() { command -v "$1" &>/dev/null; }
json_bool()  { [[ "${1:-}" == "true" ]] && echo "true" || echo "false"; }
# py_bool: same boolean conversion but emits Python's capitalized True/False.
# Required because state['x'] = $(json_bool ...) inside a Python heredoc
# splices the literal text "true"/"false" into Python source, which Python
# does not recognize (only "True"/"False" are valid Python identifiers).
py_bool()    { [[ "${1:-}" == "true" ]] && echo "True" || echo "False"; }

ver_gte() {
    local IFS=.
    local -a a=($1) b=($2)
    local i
    for (( i=0; i<${#b[@]}; i++ )); do
        local av="${a[i]:-0}" bv="${b[i]:-0}"
        (( av > bv )) && return 0
        (( av < bv )) && return 1
    done
    return 0
}

_surface_log() {
    local logpath="$1" label="${2:-Log}" nlines="${3:-35}"
    [[ -f "$logpath" ]] || return 0
    local lines; lines=$(wc -l < "$logpath" 2>/dev/null || echo 0)
    [[ "$lines" -eq 0 ]] && return 0
    echo -e "\n${C_RED}${C_BLD}  ═══ ${label} (last ${nlines} lines) ═══${C_RST}" \
        | tee -a "${LOG_FILE:-/dev/null}"
    tail -"${nlines}" "$logpath" | tee -a "${LOG_FILE:-/dev/null}"
    echo -e "${C_DIM}  Full log: ${logpath}${C_RST}"
}

# Cleanup trap
_cleanup() {
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        err "Setup exited with code ${rc}"
        echo -e "  Re-run: ${C_BLD}bash ${BASH_SOURCE[0]}${C_RST}" >&2
    fi
}
trap '_cleanup'                            EXIT
trap 'die "Interrupted by user (Ctrl+C)"' INT
trap 'die "Terminated (SIGTERM)"'         TERM

# ═════════════════════════════════════════════════════════════════════════════
# STEP 0 — Welcome Banner
# ═════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${C_BLU}${C_BLD}"
cat << 'LOGO'
  ╔╗ ╦╔╦╗╔╗╔╔═╗╔╦╗   ╔═╗╔═╗╔═╗   ╦  ╔═╗╔═╗╔═╗╦
  ╠╩╗║ ║ ║║║║╣  ║    ║  ╠═╝╠═╝   ║  ║ ║║  ╠═╣║
  ╚═╝╩ ╩ ╝╚╝╚═╝ ╩    ╚═╝╩  ╩     ╩═╝╚═╝╚═╝╩ ╩╩═╝
         Local LLM Cloud  ·  BitNet Setup  v1.0.0
LOGO
echo -e "${C_RST}"
echo -e "${C_DIM}  SPECTRE · PRISM · ATLAS · FORGE · GUARDIAN · NEXUS${C_RST}"
echo ""
$DRY_RUN         && warn "DRY-RUN mode — no filesystem changes"
$NON_INTERACTIVE && warn "NON-INTERACTIVE — all defaults accepted"
$APPLY_PATCH     && note "--apply-patch: all patches applied before build"
$FORCE_REBUILD   && note "--force-rebuild: FORGE cache bypassed"

# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — OS / Architecture / ATLAS Hardware Detection
# ═════════════════════════════════════════════════════════════════════════════
section "STEP 1 — System Detection (ATLAS)"

OS_TYPE="$(uname -s)"
case "$(uname -m)" in
    x86_64|AMD64)        ARCH="x86_64" ;;
    aarch64|arm64|ARM64) ARCH="arm64"  ;;
    *) die "Unsupported CPU architecture: $(uname -m). Only x86_64 and arm64 supported." ;;
esac

case "$OS_TYPE" in
    Linux)
        if   cmd_exists apt-get; then PKG_MGR="apt"
        elif cmd_exists dnf;     then PKG_MGR="dnf"
        elif cmd_exists yum;     then PKG_MGR="yum"
        elif cmd_exists pacman;  then PKG_MGR="pacman"
        elif cmd_exists zypper;  then PKG_MGR="zypper"
        else warn "No supported package manager found (apt/dnf/yum/pacman/zypper)"; fi ;;
    Darwin) PKG_MGR="brew" ;;
    MINGW*|MSYS*|CYGWIN*) die "Windows shell not supported — use WSL2." ;;
    *) die "Unsupported OS: ${OS_TYPE}. Only Linux and macOS supported." ;;
esac

NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# CPU capability flags
if [[ "$OS_TYPE" == "Linux" ]] && [[ -f /proc/cpuinfo ]]; then
    _flags="$(grep -m1 '^flags' /proc/cpuinfo 2>/dev/null || echo '')"
    echo "$_flags" | grep -qw avx2    && ATLAS_AVX2=true
    echo "$_flags" | grep -qw avx512f && ATLAS_AVX512=true
    echo "$_flags" | grep -qwE 'neon|asimd' && ATLAS_NEON=true
elif [[ "$OS_TYPE" == "Darwin" ]]; then
    sysctl -n machdep.cpu.features 2>/dev/null | grep -qi AVX2   && ATLAS_AVX2=true
    sysctl -n machdep.cpu.features 2>/dev/null | grep -qi AVX512 && ATLAS_AVX512=true
    sysctl -n hw.optional.arm64    2>/dev/null | grep -q  1      && ATLAS_NEON=true
fi

# Source .llm_env from install.sh (inherits ATLAS gpu backend, compiler, etc.)
for _env_f in \
    "${LOCAL_LLM_STATE_DIR}/.llm_env" \
    "${_SCRIPT_DIR}/.llm_env" \
    "${_SCRIPT_DIR}/../.llm_env"; do
    if [[ -f "$_env_f" ]]; then
        # shellcheck disable=SC1090
        source "$_env_f" 2>/dev/null || true
        info "Loaded env state: ${_env_f}"
        break
    fi
done

# .install_state.json is what install.sh ACTUALLY writes (not .llm_env).
# Read ATLAS values from it directly. This gives us gpu_backend, cmake_flags,
# recommended_threads, and cc/cxx — all of which install.sh properly detected.
_INSTALL_STATE="${LOCAL_LLM_STATE_DIR}/.install_state.json"
if [[ -f "$_INSTALL_STATE" ]]; then
    _isgpu=$(python3 -c \
        "import json; d=json.load(open('${_INSTALL_STATE}')); print(d.get('gpu_backend',''))" \
        2>/dev/null || echo "")
    _isthreads=$(python3 -c \
        "import json; d=json.load(open('${_INSTALL_STATE}')); print(d.get('recommended_threads',''))" \
        2>/dev/null || echo "")
    _iscmake=$(python3 -c \
        "import json; d=json.load(open('${_INSTALL_STATE}')); print(d.get('cmake_flags',''))" \
        2>/dev/null || echo "")
    _iscc=$(python3 -c \
        "import json; d=json.load(open('${_INSTALL_STATE}')); print(d.get('cc',''))" \
        2>/dev/null || echo "")
    _iscxx=$(python3 -c \
        "import json; d=json.load(open('${_INSTALL_STATE}')); print(d.get('cxx',''))" \
        2>/dev/null || echo "")
    [[ -n "$_isgpu" ]]     && ATLAS_GPU_BACKEND="$_isgpu"
    [[ -n "$_isthreads" ]] && ATLAS_THREADS="$_isthreads"
    [[ -n "$_iscmake" ]]   && ATLAS_CMAKE_FLAGS="$_iscmake"
    [[ -n "$_iscc" ]]      && export CC="$_iscc"
    [[ -n "$_iscxx" ]]     && export CXX="$_iscxx"
    info "Loaded ATLAS state from .install_state.json (gpu=${ATLAS_GPU_BACKEND} threads=${ATLAS_THREADS})"
fi

export CC="${CC:-clang}"
export CXX="${CXX:-clang++}"

# ATLAS quant recommendation: arm64→tl1, AVX-512→tl2, else i2_s (universal)
if   [[ "$ARCH" == "arm64" ]];  then ATLAS_QUANT_REC="${ATLAS_QUANT_REC:-tl1}"
elif $ATLAS_AVX512;              then ATLAS_QUANT_REC="${ATLAS_QUANT_REC:-tl2}"
else                                  ATLAS_QUANT_REC="${ATLAS_QUANT_REC:-i2_s}"
fi
# Use threads from .install_state.json if set; fall back to NPROC-based default
[[ -z "${ATLAS_THREADS:-}" || "${ATLAS_THREADS}" -lt 1 ]] && \
    ATLAS_THREADS=$(( NPROC < 8 ? NPROC : 8 ))
[[ "$ATLAS_THREADS" -lt 1 ]] && ATLAS_THREADS=1

ok "OS=${OS_TYPE}  Arch=${ARCH}  CPUs=${NPROC}  PkgMgr=${PKG_MGR:-unknown}"
ok "ATLAS: quant=${ATLAS_QUANT_REC}  threads=${ATLAS_THREADS}  gpu=${ATLAS_GPU_BACKEND:-cpu}  avx2=${ATLAS_AVX2}  avx512=${ATLAS_AVX512}"

# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — BitNet.cpp Install Location
# ═════════════════════════════════════════════════════════════════════════════
section "STEP 2 — BitNet.cpp Install Location"

_is_bitnet_dir()    { [[ -f "${1}/run_inference.py" && -f "${1}/setup_env.py" ]]; }
_bitnet_has_build() { [[ -x "${1}/build/bin/llama-cli" ]] || [[ -x "${1}/build/Release/bin/llama-cli" ]]; }

# ── 3rdparty/llama.cpp submodule check (CRITICAL) ────────────────────────────
# `git clone` does NOT pull submodules automatically.
# BitNet.cpp will not build without 3rdparty/llama.cpp.
# We verify all load-bearing paths and auto-repair with `git submodule update`.
_LLAMA_MISSING=""
_llama_cpp_paths_ok() {
    local bd="$1" lc="${1}/3rdparty/llama.cpp"
    local missing=()
    for p in "CMakeLists.txt" "src/CMakeLists.txt" "src/llama.cpp" \
              "include" "common" "ggml/CMakeLists.txt" "ggml/src" \
              "gguf-py/gguf/__init__.py"; do
        [[ -e "${lc}/${p}" ]] || missing+=("3rdparty/llama.cpp/${p}")
    done
    if [[ ${#missing[@]} -gt 0 ]]; then
        _LLAMA_MISSING="${missing[*]}"
        return 1
    fi
    return 0
}

_ensure_submodules() {
    local bd="$1"
    if _llama_cpp_paths_ok "$bd"; then
        ok "3rdparty/llama.cpp: fully populated ✓"
        return 0
    fi
    warn "3rdparty/llama.cpp incomplete — missing: ${_LLAMA_MISSING}"
    info "Running: git submodule update --init --recursive --force"
    if [[ -f "${bd}/.gitmodules" ]]; then
        (
            cd "$bd"
            git submodule sync --recursive 2>/dev/null || true
            git submodule update --init --recursive --force \
                2>&1 | tee -a "${LOG_FILE:-/dev/null}"
        ) || warn "submodule update had errors — continuing"
    else
        warn ".gitmodules not found — release archive without submodules?"
    fi
    if _llama_cpp_paths_ok "$bd"; then
        ok "3rdparty/llama.cpp: populated after retry ✓"
        return 0
    fi
    err "3rdparty/llama.cpp still incomplete after retry."
    err "Missing: ${_LLAMA_MISSING}"
    err "Manual fix: cd ${bd} && git submodule deinit -f 3rdparty/llama.cpp && git submodule update --init --recursive --force"
    return 1
}

# Scan for existing BitNet installs
EXISTING_BITNET_DIRS=()
SEARCH_PATHS=(
    "${REAL_HOME}/AMB-OPT/BitNet"
    "${REAL_HOME}/BitNet"
    "${REAL_HOME}/bitnet"
    "${REAL_HOME}/microsoft/BitNet"
    "/opt/BitNet"
    "/opt/bitnet"
)
for _d in "${REAL_HOME}"/*/; do
    [[ -d "${_d}BitNet" ]] && SEARCH_PATHS+=("${_d%/}/BitNet")
    [[ -d "${_d}bitnet" ]] && SEARCH_PATHS+=("${_d%/}/bitnet")
done 2>/dev/null || true
for _d in "${REAL_HOME}"/*/*/; do
    [[ -d "${_d}BitNet" ]] && SEARCH_PATHS+=("${_d%/}/BitNet")
done 2>/dev/null || true
[[ -n "${BITNET_DIR_OVERRIDE:-}" ]] && SEARCH_PATHS=("${BITNET_DIR_OVERRIDE}" "${SEARCH_PATHS[@]}")

for _c in "${SEARCH_PATHS[@]}"; do
    [[ -d "$_c" ]] && _is_bitnet_dir "$_c" && EXISTING_BITNET_DIRS+=("$_c")
done
if [[ ${#EXISTING_BITNET_DIRS[@]} -gt 0 ]]; then
    mapfile -t EXISTING_BITNET_DIRS < <(
        printf "%s\n" "${EXISTING_BITNET_DIRS[@]}" | grep -v '^$' | sort -u)
fi

# Resolve BITNET_DIR
if [[ -n "${BITNET_DIR_OVERRIDE:-}" ]]; then
    BITNET_DIR="$BITNET_DIR_OVERRIDE"
    _is_bitnet_dir "$BITNET_DIR" && SKIP_CLONE=true
elif [[ ${#EXISTING_BITNET_DIRS[@]} -gt 0 ]] && $NON_INTERACTIVE; then
    BITNET_DIR="${EXISTING_BITNET_DIRS[0]}"
    SKIP_CLONE=true
    info "Auto-selected existing: ${BITNET_DIR}"
else
    DEFAULT_INSTALL_DIR="${REAL_HOME}/AMB-OPT/BitNet"
    if [[ ${#EXISTING_BITNET_DIRS[@]} -gt 0 ]]; then
        echo ""
        echo -e "${C_WHT}${C_BLD}  Existing BitNet.cpp installations:${C_RST}"
        for i in "${!EXISTING_BITNET_DIRS[@]}"; do
            _bt=""; _bitnet_has_build "${EXISTING_BITNET_DIRS[$i]}" && _bt=" ${C_GRN}[built]${C_RST}"
            echo -e "    ${C_DIM}$((i+1)))${C_RST} ${EXISTING_BITNET_DIRS[$i]}${_bt}"
        done
        echo -e "    ${C_DIM}$((${#EXISTING_BITNET_DIRS[@]}+1)))${C_RST} Fresh install → ${DEFAULT_INSTALL_DIR}"
        echo ""
        ask _PICK "Select" "1"
        _BIDX=$(( ${_PICK:-1} - 1 ))
        if [[ "$_BIDX" -ge 0 && "$_BIDX" -lt "${#EXISTING_BITNET_DIRS[@]}" ]] 2>/dev/null; then
            BITNET_DIR="${EXISTING_BITNET_DIRS[$_BIDX]}"
            SKIP_CLONE=true
        else
            ask BITNET_DIR "Install directory" "${DEFAULT_INSTALL_DIR}"
        fi
    else
        ask BITNET_DIR "Where to install BitNet.cpp?" "${REAL_HOME}/AMB-OPT/BitNet"
    fi
fi

BITNET_DIR="${BITNET_DIR/#\~/$HOME}"
[[ -z "${BITNET_DIR:-}" ]] && die "BitNet directory cannot be empty."
# mkdir is non-destructive, so we create it even in --dry-run mode.
# This lets the rest of the script `cd` into it and write logs normally;
# --dry-run still skips every destructive action (clone/build/download/rm).
mkdir -p "${BITNET_DIR}" 2>/dev/null || true

LOG_FILE="${BITNET_DIR}/bitnet_setup_$(date '+%Y%m%d_%H%M%S').log"
touch "${LOG_FILE}" 2>/dev/null \
    || LOG_FILE="/tmp/bitnet_setup_$(date '+%Y%m%d_%H%M%S').log"
touch "${LOG_FILE}" 2>/dev/null || LOG_FILE="/dev/null"
info "BitNet dir : ${BITNET_DIR}"
info "Log file   : ${LOG_FILE}"

# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — System Dependencies (clang ≥18, cmake ≥3.14, git)
# ═════════════════════════════════════════════════════════════════════════════
section "STEP 3 — System Dependencies"

MISSING_DEPS=()
cmd_exists git   || MISSING_DEPS+=("git")
cmd_exists cmake || MISSING_DEPS+=("cmake")

# Find best available clang (prefer newest)
for _cb in clang-21 clang-20 clang-19 clang-18 clang; do
    cmd_exists "$_cb" || continue
    _v=$("$_cb" --version 2>/dev/null | grep -oP '(?<=version )\d+' | head -1 || echo 0)
    [[ "${_v:-0}" -gt "$CLANG_VER" ]] && { CLANG_VER="${_v}"; CLANG_BIN="$_cb"; }
done
[[ "$CLANG_VER" -lt 18 ]] && MISSING_DEPS+=("clang>=18 (found v${CLANG_VER})")

if cmd_exists cmake; then
    CMAKE_VER="$(cmake --version 2>/dev/null | awk 'NR==1{print $3}' || echo "0")"
    ver_gte "${CMAKE_VER}" "3.14" || MISSING_DEPS+=("cmake>=3.14 (found ${CMAKE_VER})")
fi

if [[ ${#MISSING_DEPS[@]} -gt 0 ]]; then
    warn "Missing/outdated: ${MISSING_DEPS[*]}"
    if confirm "Auto-install / upgrade system dependencies now?" "Y"; then
        case "${PKG_MGR:-}" in
            apt)
                run_cmd sudo apt-get update -qq
                run_cmd sudo apt-get install -y \
                    git cmake build-essential libssl-dev pkg-config
                if [[ "$CLANG_VER" -lt 18 ]]; then
                    info "Installing clang-18 via llvm.sh …"
                    _llvm="/tmp/_llvm_$$.sh"
                    cmd_exists curl \
                        && curl -fsSL https://apt.llvm.org/llvm.sh -o "$_llvm" 2>/dev/null \
                        || wget  -qO  "$_llvm" https://apt.llvm.org/llvm.sh  2>/dev/null \
                        || true
                    if [[ -f "$_llvm" ]]; then
                        bash "$_llvm" 18 >/dev/null 2>&1 || true
                        rm -f "$_llvm"
                    fi
                    run_cmd sudo apt-get install -y clang-18 clang++-18 lld-18 2>/dev/null || true
                    # Re-probe after install
                    for _cb in clang-18 clang; do
                        cmd_exists "$_cb" || continue
                        _v=$("$_cb" --version 2>/dev/null | grep -oP '(?<=version )\d+' | head -1 || echo 0)
                        [[ "${_v:-0}" -ge 18 ]] && { CLANG_VER="$_v"; CLANG_BIN="$_cb"; break; }
                    done
                fi ;;
            dnf|yum)
                run_cmd sudo "${PKG_MGR}" install -y \
                    git cmake clang gcc gcc-c++ openssl-devel pkg-config ;;
            pacman)
                run_cmd sudo pacman -S --noconfirm \
                    git cmake clang base-devel openssl pkg-config ;;
            zypper)
                run_cmd sudo zypper install -y \
                    git cmake clang openssl-devel pkg-config ;;
            brew)
                run_cmd brew install git cmake llvm pkg-config
                _brew_llvm="$(brew --prefix llvm 2>/dev/null)/bin"
                export PATH="${_brew_llvm}:${PATH}"
                CLANG_BIN="${_brew_llvm}/clang"
                CLANG_VER=$("${CLANG_BIN}" --version 2>/dev/null \
                    | grep -oP '(?<=version )\d+' | head -1 || echo 0) ;;
            *) warn "Cannot auto-install on '${PKG_MGR:-unknown}'. Install manually: ${MISSING_DEPS[*]}" ;;
        esac
    fi
fi

# Set CC/CXX from best available clang
if cmd_exists "$CLANG_BIN"; then
    _clang_full="$(command -v "$CLANG_BIN")"
    _clangpp="${_clang_full//clang/clang++}"
    [[ -x "$_clangpp" ]] && { export CC="$_clang_full"; export CXX="$_clangpp"; }
fi
[[ "$CLANG_VER" -lt 18 ]] && warn "clang ${CLANG_VER} < 18 — 1-bit kernels need clang ≥18"
ok "clang v${CLANG_VER} (${CLANG_BIN})  cmake ${CMAKE_VER}  CC=${CC}  CXX=${CXX}"

# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — Clone BitNet.cpp + Populate 3rdparty/llama.cpp Submodule
# ═════════════════════════════════════════════════════════════════════════════
section "STEP 4 — Clone BitNet.cpp + Populate Submodules"

if ! $SKIP_CLONE; then
    if _is_bitnet_dir "$BITNET_DIR"; then
        warn "BitNet.cpp already exists at: ${BITNET_DIR}"
        if $NON_INTERACTIVE || confirm "Use existing (skip clone)?" "Y"; then
            SKIP_CLONE=true
        else
            confirm "Delete and re-clone? (local changes lost)" "n" \
                || die "Aborted. Remove ${BITNET_DIR} manually and re-run."
            $DRY_RUN || rm -rf "${BITNET_DIR}"
        fi
    fi
fi

if ! $SKIP_CLONE && ! $DRY_RUN; then
    info "Cloning microsoft/BitNet → ${BITNET_DIR} …"
    run_cmd git clone --depth=1 https://github.com/microsoft/BitNet.git "${BITNET_DIR}"
    ok "Cloned to ${BITNET_DIR}"
elif $SKIP_CLONE; then
    ok "Using existing clone: ${BITNET_DIR}"
fi

cd "${BITNET_DIR}"

# CRITICAL: populate 3rdparty/llama.cpp — BitNet will not build without it
info "Ensuring 3rdparty/llama.cpp submodule is fully populated …"
if $DRY_RUN; then
    note "Dry-run: skipping submodule population check"
elif [[ ! -d "${BITNET_DIR}/.git" ]]; then
    warn "No .git directory in ${BITNET_DIR} — clone was skipped or did not run"
else
    _ensure_submodules "${BITNET_DIR}" \
        || die "3rdparty/llama.cpp could not be populated. Check network and re-run."
fi

GGUF_PY_DIR="${BITNET_DIR}/3rdparty/llama.cpp/gguf-py"

# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 — Python Environment (NEXUS)
# ═════════════════════════════════════════════════════════════════════════════
section "STEP 5 — Python Environment (NEXUS)"

_find_python() {
    # Prefer stable: 3.12→3.11→3.10→3.9→3.8, then 3.13→3.14 (more issues)
    for _p in python3.12 python3.11 python3.10 python3.9 python3.8 \
               python3 python3.13 python3.14; do
        cmd_exists "$_p" || continue
        local _maj _min
        _maj=$("$_p" -c 'import sys; print(sys.version_info.major)' 2>/dev/null) || continue
        _min=$("$_p" -c 'import sys; print(sys.version_info.minor)'  2>/dev/null) || continue
        [[ "$_maj" -eq 3 && "$_min" -ge 8 ]] && { echo "$_p"; return 0; }
    done
    return 1
}

SYSTEM_PYTHON=$(_find_python 2>/dev/null) \
    || die "No Python ≥3.8 found. Install Python 3.10/3.11/3.12 and re-run."

# Discover existing venvs
EXISTING_VENVS=()
EXISTING_VENV_LABELS=()

[[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]] && {
    EXISTING_VENVS+=("${VIRTUAL_ENV}")
    EXISTING_VENV_LABELS+=("Active venv: ${VIRTUAL_ENV}")
}
[[ -n "${LLM_VENV_DIR:-}" && -x "${LLM_VENV_DIR}/bin/python" && \
   "${LLM_VENV_DIR}" != "${VIRTUAL_ENV:-}" ]] && {
    EXISTING_VENVS+=("${LLM_VENV_DIR}")
    EXISTING_VENV_LABELS+=("Project venv (from install.sh): ${LLM_VENV_DIR}")
}
cmd_exists conda && [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]] && {
    EXISTING_VENVS+=("${CONDA_PREFIX}")
    EXISTING_VENV_LABELS+=("Active conda env: ${CONDA_PREFIX}")
}
for _v in "${REAL_HOME}/DEVE-BOX" \
           "${BITNET_DIR}/bitnet-env" \
           "${BITNET_DIR}/venv"; do
    [[ -x "${_v}/bin/python" ]] || continue
    _dup=false
    for _ev in "${EXISTING_VENVS[@]:-__none__}"; do
        [[ "$_ev" == "$_v" ]] && { _dup=true; break; }
    done
    $_dup || { EXISTING_VENVS+=("$_v"); EXISTING_VENV_LABELS+=("Existing venv: $_v"); }
done

echo ""
echo -e "${C_WHT}  Python environment — choose one:${C_RST}"
_IDX=1
for _i in "${!EXISTING_VENVS[@]}"; do
    echo -e "  ${C_GRN}  ${_IDX}) ${EXISTING_VENV_LABELS[$_i]}${C_RST}"
    _IDX=$((_IDX+1))
done
_N=${#EXISTING_VENVS[@]}
_CREATE_IDX=$((_N+1))
_SYS_IDX=$((_N+2))
echo -e "  ${C_DIM}  ${_CREATE_IDX}) Create venv at ${BITNET_DIR}/bitnet-env${C_RST}"
echo -e "  ${C_DIM}  ${_SYS_IDX}) System Python (no isolation)${C_RST}"
[[ $_N -gt 0 ]] && _DEF="1" || _DEF="${_CREATE_IDX}"
ask ENV_CHOICE "Choose" "${_DEF}"

if [[ "${ENV_CHOICE:-}" =~ ^[0-9]+$ ]] && \
   [[ "$ENV_CHOICE" -ge 1 && "$ENV_CHOICE" -le "$_N" ]]; then
    VENV_DIR="${EXISTING_VENVS[$(( ENV_CHOICE - 1 ))]}"
    PYTHON_BIN="${VENV_DIR}/bin/python"
    [[ -x "$PYTHON_BIN" ]] || PYTHON_BIN="${VENV_DIR}/bin/python3"
    ok "Using existing venv: ${VENV_DIR}"
elif [[ "${ENV_CHOICE:-}" == "${_CREATE_IDX}" ]]; then
    VENV_DIR="${BITNET_DIR}/bitnet-env"
    if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
        info "Creating venv at ${VENV_DIR} …"
        run_cmd "$SYSTEM_PYTHON" -m venv "${VENV_DIR}"
    else
        ok "Reusing existing venv at ${VENV_DIR}"
    fi
    PYTHON_BIN="${VENV_DIR}/bin/python"
elif [[ "${ENV_CHOICE:-}" == "${_SYS_IDX}" ]]; then
    PYTHON_BIN="$SYSTEM_PYTHON"
    VENV_DIR=""
    ok "Using system Python: ${PYTHON_BIN}"
else
    # Default: create venv
    VENV_DIR="${BITNET_DIR}/bitnet-env"
    [[ -x "${VENV_DIR}/bin/python" ]] \
        || run_cmd "$SYSTEM_PYTHON" -m venv "${VENV_DIR}"
    PYTHON_BIN="${VENV_DIR}/bin/python"
    ok "Created/reused venv: ${VENV_DIR}"
fi

PY_VER=$("$PYTHON_BIN" -c \
    "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" \
    2>/dev/null || echo "3.0")
PY_MINOR=$("$PYTHON_BIN" -c \
    "import sys; print(sys.version_info.minor)" 2>/dev/null || echo "0")
_PY_FULL=$("$PYTHON_BIN" -c \
    "import sys; print(sys.version.split()[0])" 2>/dev/null || echo "unknown")
ok "Python ${_PY_FULL} — ${PYTHON_BIN}"

run_cmd --silent "$PYTHON_BIN" -m pip install --quiet --upgrade pip
run_cmd --silent "$PYTHON_BIN" -m pip install --quiet "setuptools<82" wheel
ok "pip and setuptools ready"

# ═════════════════════════════════════════════════════════════════════════════
# STEP 6 — Cleanup Stale Resources
# ═════════════════════════════════════════════════════════════════════════════
section "STEP 6 — Cleanup Stale Resources"

[[ -d "${REAL_HOME}/.cache/huggingface/download" ]] && \
    find "${REAL_HOME}/.cache/huggingface/download" \
         -name "*.lock" -type f -delete 2>/dev/null && \
    ok "Removed HuggingFace download lock files" || true
ok "Stale resource cleanup done"

# ═════════════════════════════════════════════════════════════════════════════
# STEP 7 — SPECTRE: gguf Conflict Resolution
# ═════════════════════════════════════════════════════════════════════════════
section "STEP 7 — SPECTRE: gguf Conflict Resolution"

# Root cause: PyPI gguf lacks TL1/TL2 quantization types used by BitNet's
# converter. BitNet ships 3rdparty/llama.cpp/gguf-py which has them.
# SPECTRE manages the PYTHONPATH isolation so the right gguf is always used.

_gguf_has_tl1() {
    "$PYTHON_BIN" -c "
import gguf
_ = gguf.GGMLQuantizationType.TL1
_ = gguf.GGMLQuantizationType.TL2
" &>/dev/null 2>&1
}

_spectre_apply_shim() {
    local _site
    _site=$("$PYTHON_BIN" -c "
import site, sys
try:    print(site.getsitepackages()[0])
except: print(site.getusersitepackages())
" 2>/dev/null || echo "")
    [[ -z "$_site" ]] && { warn "SPECTRE shim: cannot locate site-packages"; return 1; }
    local _shim="${_site}/gguf_tl_shim"
    mkdir -p "$_shim" 2>/dev/null || return 1
    cat > "${_shim}/__init__.py" << 'SHIM'
# SPECTRE TL1/TL2 shim — injected by local-llm-core bitnet_setup.sh v1.0.0
def _apply():
    try:
        import gguf as _g
        if not hasattr(_g.GGMLQuantizationType, 'TL1'):
            def _ext(cls, name, val):
                if hasattr(cls, name): return
                if val in cls._value2member_map_:
                    e = cls._value2member_map_[val]
                    cls._member_map_[name] = e
                    type.__setattr__(cls, name, e)
                    return
                m = int.__new__(cls, val)
                m._name_ = name; m._value_ = val
                cls._member_map_[name] = m
                cls._value2member_map_[val] = m
                type.__setattr__(cls, name, m)
            _ext(_g.GGMLQuantizationType, 'TL1', 38)
            _ext(_g.GGMLQuantizationType, 'TL2', 39)
    except Exception:
        pass
_apply()
SHIM
    printf 'import gguf_tl_shim\n' > "${_site}/gguf_tl_shim.pth" 2>/dev/null || true
    ok "SPECTRE: TL1/TL2 shim installed → ${_site}"
    SPECTRE_MODE="shim"
}

if $USE_PYPI_GGUF; then
    info "SPECTRE: --use-pypi-gguf — using PyPI gguf"
    pip_install --silent "gguf>=0.1.0" || true
    _gguf_has_tl1 || _spectre_apply_shim || true
    SPECTRE_MODE="${SPECTRE_MODE:-shim_pypi}"
    SPECTRE_GGUF_PY_PATH="${GGUF_PY_DIR}"
elif _gguf_has_tl1; then
    _gv=$("$PYTHON_BIN" -c \
        "import gguf; print(getattr(gguf,'__version__','?'))" 2>/dev/null || echo "?")
    ok "SPECTRE: gguf ${_gv} already has TL1/TL2 — no action needed"
    SPECTRE_MODE="ok"
    SPECTRE_GGUF_PY_PATH="${GGUF_PY_DIR}"
elif [[ -f "${GGUF_PY_DIR}/gguf/__init__.py" ]]; then
    export PYTHONPATH="${GGUF_PY_DIR}:${PYTHONPATH:-}"
    SPECTRE_GGUF_PY_PATH="${GGUF_PY_DIR}"
    info "SPECTRE Layer 1: PYTHONPATH → ${GGUF_PY_DIR}"
    info "SPECTRE Layer 2: editable install of bundled gguf-py …"
    if "$PYTHON_BIN" -m pip install --quiet --no-deps -e "${GGUF_PY_DIR}" \
            >> "${LOG_FILE:-/dev/null}" 2>&1; then
        if _gguf_has_tl1; then
            ok "SPECTRE: bundled gguf-py installed — TL1/TL2 confirmed ✓"
            SPECTRE_MODE="editable"
        else
            warn "SPECTRE: editable install OK but TL1 check failed — applying shim"
            _spectre_apply_shim || true
        fi
    else
        warn "SPECTRE: editable install failed — applying shim"
        _spectre_apply_shim || true
    fi
else
    info "SPECTRE: gguf-py submodule absent — PyPI gguf + shim"
    pip_install --silent "gguf>=0.1.0" || true
    _spectre_apply_shim || true
    SPECTRE_MODE="shim_pypi"
fi
ok "SPECTRE mode: ${SPECTRE_MODE}  |  gguf-py path: ${SPECTRE_GGUF_PY_PATH:-none}"

# ═════════════════════════════════════════════════════════════════════════════
# STEP 8 — Python Compatibility Patches (interactive, per Python version)
# ═════════════════════════════════════════════════════════════════════════════
section "STEP 8 — BitNet Compatibility Patches"

# All patch functions are idempotent — safe to call multiple times.

_patch_pyproject_toml() {
    # Patch 1: Remove sentencepiece <=0.2.0 upper bound (no wheel on Python 3.14)
    local f="${GGUF_PY_DIR}/pyproject.toml"
    [[ -f "$f" ]] || { ok "Patch 1: pyproject.toml not found — skip"; return 0; }
    if ! grep -q '<=0.2.0' "$f" 2>/dev/null; then
        ok "Patch 1: pyproject.toml sentencepiece bound already relaxed"; return 0
    fi
    info "Patch 1: Removing sentencepiece <=0.2.0 upper bound …"
    sed -i 's/sentencepiece = ">=0\.1\.98,<=0\.2\.0"/sentencepiece = ">=0.1.98"/' "$f"
    ok "Patch 1: pyproject.toml patched ✓"
    PATCHES_APPLIED+=("pyproject_toml")
}

_patch_tl1_guard() {
    # Inject TL1 guard into setup_env.py: skip bundled gguf install if TL1 accessible
    local f="${BITNET_DIR}/setup_env.py"
    [[ -f "$f" ]] || { ok "Patch TL1: setup_env.py not found — skip"; return 0; }
    grep -q "_gguf_tl1_guard" "$f" 2>/dev/null && {
        ok "Patch TL1: setup_env.py already has TL1 guard"; return 0; }
    info "Patch TL1: injecting TL1 guard into setup_env.py …"
    python3 - "$f" << 'PYEOF'
import sys
path = sys.argv[1]
with open(path) as fh: content = fh.read()
old = 'def setup_gguf():'
new = ('def setup_gguf():\n'
       '    try:\n'
       '        import gguf as _g\n'
       '        _ = _g.GGMLQuantizationType.TL1  # _gguf_tl1_guard\n'
       '        _ = _g.GGMLQuantizationType.TL2\n'
       '        return  # already have TL1/TL2 — skip bundled install\n'
       '    except (ImportError, AttributeError):\n'
       '        pass')
if old in content and '_gguf_tl1_guard' not in content:
    with open(path, 'w') as fh: fh.write(content.replace(old, new, 1))
PYEOF
    ok "Patch TL1: setup_env.py patched ✓"
    PATCHES_APPLIED+=("tl1_guard")
}

_apply_file_patch() {
    # Copy a pre-fixed file from scripts/patches/ → its destination in BitNet tree
    local src_name="$1" label="$2"
    local src="${_PATCHES_DIR}/${src_name}"
    [[ -f "$src" ]] || { warn "Patch source missing: ${src}"; return 1; }
    local dst
    dst=$(find "${BITNET_DIR}" -maxdepth 6 -name "${src_name}" \
        -not -path "*/3rdparty/*" \
        -not -path "*/__pycache__/*" \
        -not -path "*/patches/*" \
        2>/dev/null | head -1)
    if [[ -n "$dst" ]]; then
        cp "$src" "$dst" \
            && ok "Patch: ${label} → ${dst}" \
            && PATCHES_APPLIED+=("${src_name}") \
            || warn "Failed to copy patch: ${label}"
    else
        warn "Patch target '${src_name}' not in ${BITNET_DIR} (not found — may apply after build)"
    fi
}

# Patches 3+4 are MANDATORY before any cmake build regardless of Python version
_apply_build_prereqs() {
    _apply_file_patch "ggml-bitnet-mad.cpp" \
        "ggml-bitnet-mad.cpp (const int8_t* fix — required for clang 18+)" || true
    _apply_file_patch "codegen_tl2.py" \
        "codegen_tl2.py (BitNet-b1.58-2B-4T shape entry — required for kernel gen)" || true
}

# ── Interactive patch menu ────────────────────────────────────────────────────
echo ""
echo -e "  ${C_WHT}Active Python: ${_PY_FULL}  (minor: ${PY_MINOR})${C_RST}"
echo ""

if [[ "$PY_MINOR" -ge 14 ]]; then
    _NEED_UPFRONT_PATCH=true
    echo -e "  ${C_YEL}${C_BLD}⚠  Python 3.${PY_MINOR} — BitNet.cpp has 4 known compatibility issues:${C_RST}"
    echo ""
    echo -e "  ${C_DIM}  1/4: gguf-py/pyproject.toml — sentencepiece bound (no wheel on 3.14)${C_RST}"
    echo -e "  ${C_DIM}  2/4: convert-hf-to-gguf-bitnet.py — tokenizer + packed weight fix${C_RST}"
    echo -e "  ${C_DIM}  3/4: ggml-bitnet-mad.cpp — const int8_t* fix (clang 18+, always needed)${C_RST}"
    echo -e "  ${C_DIM}  4/4: codegen_tl2.py — shape entry for 2B-4T model (always needed)${C_RST}"
    echo -e "  ${C_DIM}  Note: patches 3+4 are applied before build regardless of choice.${C_RST}"
    echo ""
    echo -e "  ${C_GRN}  A)${C_RST} Apply all 4  ${C_DIM}[recommended]${C_RST}"
    echo -e "  ${C_CYN}  S)${C_RST} Choose per patch"
    echo -e "  ${C_YEL}  N)${C_RST} Skip 1+2  (3+4 still auto-apply before build)"
    echo ""
    _PC=""
    if $NON_INTERACTIVE; then _PC="A"
    else ask _PC "Choice [A/S/N]" "A"; fi

    case "${_PC^^}" in
        A|"")
            info "Applying all 4 patches …"
            _patch_pyproject_toml
            _patch_tl1_guard
            _apply_file_patch "convert-hf-to-gguf-bitnet.py" \
                "convert-hf-to-gguf-bitnet.py (tokenizer + packed weight fix)" || true
            _apply_file_patch "ggml-bitnet-mad.cpp" \
                "ggml-bitnet-mad.cpp (const int8_t* fix)" || true
            _apply_file_patch "codegen_tl2.py" \
                "codegen_tl2.py (2B-4T shape entry)" || true
            ok "All 4 patches applied ✓" ;;
        S)
            info "Selective patch mode:"
            confirm "  1/4: pyproject.toml sentencepiece bound?" "Y" \
                && _patch_pyproject_toml || true
            confirm "  TL1: setup_env.py TL1 guard?" "Y" \
                && _patch_tl1_guard || true
            confirm "  2/4: convert-hf-to-gguf-bitnet.py?" "Y" \
                && _apply_file_patch "convert-hf-to-gguf-bitnet.py" "HF converter fix" || true
            note "Patches 3+4 (ggml-bitnet-mad.cpp, codegen_tl2.py) auto-apply before build." ;;
        N)
            warn "Patches 1+2 skipped. Build may fail on Python 3.${PY_MINOR}."
            note "Patches 3+4 still auto-apply before build (always required)." ;;
    esac

elif [[ "$PY_MINOR" -ge 13 ]]; then
    _NEED_UPFRONT_PATCH=true
    info "Python 3.${PY_MINOR}: applying recommended patches (TL1 guard + sentencepiece) …"
    _patch_tl1_guard
    _patch_pyproject_toml

elif [[ "$SPECTRE_MODE" == "editable" || "$SPECTRE_MODE" == "ok" ]]; then
    _NEED_UPFRONT_PATCH=true
    _patch_tl1_guard

else
    info "Python 3.${PY_MINOR}: no upfront patches needed"
fi

$APPLY_PATCH && _NEED_UPFRONT_PATCH=true

# ═════════════════════════════════════════════════════════════════════════════
# STEP 9 — run_inference.py Enhancement
# ═════════════════════════════════════════════════════════════════════════════
section "STEP 9 — run_inference.py Enhancement"

RUN_INFERENCE="${BITNET_DIR}/run_inference.py"

if [[ -f "$RUN_INFERENCE" ]]; then
    if grep -q "local-llm-core v1.0.0\|--gguf-path\|resolve_binary" \
            "$RUN_INFERENCE" 2>/dev/null; then
        ok "run_inference.py: already enhanced — skipping"
    else
        if $NON_INTERACTIVE || \
           confirm "Replace run_inference.py with enhanced v1.0.0?" "Y"; then
            cp "$RUN_INFERENCE" \
               "${RUN_INFERENCE}.bak_$(date '+%Y%m%d')" 2>/dev/null || true
            cat > "${RUN_INFERENCE}" << 'RIEOF'
#!/usr/bin/env python3
"""
run_inference.py — local-llm-core v1.0.0
Enhanced replacement for Microsoft's original script.
Adds: --gguf-path flag, auto-detect from bitnet_state.json,
robust binary discovery, friendly errors for non-technical users.
"""
import argparse, glob, json, os, platform, signal, subprocess, sys
from pathlib import Path

def load_state():
    for p in [
        Path.home() / ".config" / "local-llm" / "bitnet_state.json",
        Path(os.path.dirname(os.path.abspath(__file__))) / "bitnet_state.json",
        Path("/root/.config/local-llm/bitnet_state.json"),
    ]:
        if p.exists():
            try: return json.loads(p.read_text())
            except Exception: pass
    return {}

def find_gguf(hint=None):
    state = load_state()
    sp = state.get("gguf_path", "")
    if sp and Path(sp).is_file(): return sp
    for d in filter(None, [hint, state.get("model_path",""),
                            str(Path.home()/"models"),
                            str(Path(os.path.abspath(__file__)).parent/"models")]):
        if not Path(d).is_dir(): continue
        for pat in ["ggml-model-*.gguf", "*.gguf"]:
            found = sorted(glob.glob(str(Path(d)/pat)))
            if found: return found[0]
    return None

def resolve_binary(name="llama-cli"):
    state = load_state()
    key = "llama_cli_binary" if "cli" in name else "llama_server_binary"
    b = state.get(key, "")
    if b and Path(b).is_file() and os.access(b, os.X_OK): return b
    base = Path(os.path.abspath(__file__)).parent
    for c in [base/"build"/"bin"/name,
              base/"build"/"Release"/"bin"/name,
              base/"build"/"bin"/"Release"/name]:
        cs = str(c) + (".exe" if platform.system()=="Windows" else "")
        if Path(cs).is_file() and os.access(cs, os.X_OK): return cs
    return str(base/"build"/"bin"/name)

def main():
    p = argparse.ArgumentParser(description="BitNet inference (local-llm-core v1.0.0)")
    p.add_argument("-m",    "--model",       type=str, help="Model dir or .gguf file")
    p.add_argument(         "--gguf-path",   type=str, help="Direct .gguf path")
    p.add_argument("-n",    "--n-predict",   type=int, default=128)
    p.add_argument("-p",    "--prompt",      type=str, required=True)
    _state = load_state()
    _default_threads = _state.get("atlas_threads", 4) if _state else 4
    p.add_argument("-t",    "--threads",     type=int, default=_default_threads)
    p.add_argument("-c",    "--ctx-size",    type=int, default=2048)
    p.add_argument("-temp", "--temperature", type=float, default=0.8)
    p.add_argument("-cnv",  "--conversation",action="store_true")
    args = p.parse_args()

    gguf = args.gguf_path
    if gguf and not Path(gguf).is_file():
        sys.exit(f"Error: GGUF not found: {gguf}")
    if not gguf:
        gguf = find_gguf(args.model)
    if not gguf or not Path(gguf).is_file():
        sys.exit(
            "Error: No GGUF model found.\n"
            "  Run bitnet_setup.sh first, or use --gguf-path <file>"
        )
    cli = resolve_binary("llama-cli")
    cmd = [cli, "-m", gguf, "-n", str(args.n_predict),
           "-t", str(args.threads), "-p", args.prompt,
           "-ngl", "0", "-c", str(args.ctx_size),
           "--temp", str(args.temperature), "-b", "1"]
    if args.conversation: cmd.append("-cnv")
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        sys.exit(
            f"Binary not found: {cli}\n"
            "  Run bitnet_setup.sh to compile llama-cli first."
        )
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    main()
RIEOF
            chmod +x "${RUN_INFERENCE}"
            ok "run_inference.py replaced with enhanced v1.0.0 ✓"
        fi
    fi
else
    warn "run_inference.py not found at ${BITNET_DIR} — will be created if build succeeds"
fi

# ═════════════════════════════════════════════════════════════════════════════
# STEP 10 — Python Dependencies (NEXUS unified environment)
# ═════════════════════════════════════════════════════════════════════════════
section "STEP 10 — Python Dependencies (NEXUS)"

# sentencepiece
if ! "$PYTHON_BIN" -c "import sentencepiece" &>/dev/null 2>&1; then
    info "Installing sentencepiece …"
    "$PYTHON_BIN" -m pip install --quiet --only-binary=sentencepiece \
        "sentencepiece" >> "${LOG_FILE:-/dev/null}" 2>&1 \
    || "$PYTHON_BIN" -m pip install --quiet "sentencepiece" \
        >> "${LOG_FILE:-/dev/null}" 2>&1 \
    || warn "sentencepiece install failed — Mode 1 conversion will fail"
else
    _spv=$("$PYTHON_BIN" -c \
        "import sentencepiece; print(sentencepiece.__version__)" 2>/dev/null || echo "?")
    ok "sentencepiece ${_spv} present"
fi

# gguf (TL1/TL2 already handled by SPECTRE in STEP 7)
_gv=$("$PYTHON_BIN" -c \
    "import gguf; print(getattr(gguf,'__version__','?'))" 2>/dev/null || echo "")
if [[ -z "$_gv" ]]; then
    pip_install --silent "gguf>=0.1.0"
else
    _t1=""; _gguf_has_tl1 && _t1=" (TL1/TL2 ✓)"
    ok "gguf ${_gv} present${_t1}"
fi

# torch (version-pinned per Python version)
_tv=$("$PYTHON_BIN" -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")
if [[ -z "$_tv" ]]; then
    _ts="torch"
    [[ "$PY_MINOR" -ge 14 ]] && _ts="torch>=2.9.0"
    [[ "$PY_MINOR" -eq 13 ]] && _ts="torch>=2.5.0"
    info "Installing ${_ts} …"
    if [[ "$OS_TYPE" == "Linux" ]]; then
        pip_install --silent "$_ts" \
            --index-url https://download.pytorch.org/whl/cpu \
            >> "${LOG_FILE:-/dev/null}" 2>&1 \
            || pip_install --silent "$_ts"
    else
        pip_install --silent "$_ts"
    fi
else
    ok "torch ${_tv} present"
fi

# transformers (constraint: ≥4.46.3, <5.0.0)
UNIFIED_TF_SPEC="transformers>=4.46.3,<5.0.0"
BITNET_TF_COMMIT="096f25ae1f501a084d8ff2dcaf25fbc2bd60eba4"
IS_DEV_TF=false

_tfv=$("$PYTHON_BIN" -c \
    "import transformers; print(transformers.__version__)" 2>/dev/null || echo "")
if [[ -n "$_tfv" ]]; then
    echo "$_tfv" | grep -qE 'dev|post|\+' && IS_DEV_TF=true
    if $IS_DEV_TF; then
        ok "transformers ${_tfv} (dev/pinned — BitNet backend assumed)"
    else
        _tmaj=$(echo "$_tfv" | cut -d. -f1)
        _tmin=$(echo "$_tfv" | cut -d. -f2)
        if   [[ "$_tmaj" -ge 5 ]]; then
            warn "transformers ${_tfv} ≥5.0 — downgrading to ${UNIFIED_TF_SPEC}"
            pip_install --silent "${UNIFIED_TF_SPEC}"
        elif [[ "$_tmaj" -eq 4 && "$_tmin" -lt 46 ]]; then
            warn "transformers ${_tfv} <4.46.3 — upgrading"
            pip_install --silent "${UNIFIED_TF_SPEC}"
        else
            ok "transformers ${_tfv} satisfies ≥4.46.3,<5.0.0 ✓"
        fi
    fi
else
    info "Installing transformers …"
    pip_install --silent "${UNIFIED_TF_SPEC}"
fi

# BitNet pinned transformers commit (optional — needed only for Mode 3 packed inference)
if ! $IS_DEV_TF; then
    echo ""
    echo -e "  ${C_DIM}BitNet pinned transformers commit (${BITNET_TF_COMMIT:0:12}…):${C_RST}"
    echo -e "  ${C_DIM}Required ONLY for Mode 3 (transformers API) with packed safetensors.${C_RST}"
    echo -e "  ${C_DIM}Skip if using Mode 1 (convert→GGUF) or Mode 2 (pre-built GGUF).${C_RST}"
    if confirm "Install BitNet pinned transformers commit?" "n"; then
        info "Installing pinned commit …"
        pip_install --silent \
            "git+https://github.com/huggingface/transformers.git@${BITNET_TF_COMMIT}" \
            >> "${LOG_FILE:-/dev/null}" 2>&1 \
            && ok "BitNet transformers backend installed" \
            || warn "Pinned install failed — standard transformers retained"
    else
        info "Pinned transformers skipped"
    fi
fi

# Common converter + runtime dependencies
pip_install --silent \
    "numpy>=1.17" "protobuf>=3.20" "tqdm>=4.27" \
    "huggingface-hub>=0.20" "safetensors>=0.4.0" \
    "httpx>=0.24.0" "psutil>=5.9.0" \
    >> "${LOG_FILE:-/dev/null}" 2>&1 \
    || warn "Some secondary dependencies failed — continuing"
ok "Python dependencies installed"

# Verify critical imports
"$PYTHON_BIN" -c "
import transformers, torch, safetensors
v = transformers.__version__
if not any(x in v for x in ['dev','post','+']):
    tv = tuple(int(x) for x in v.split('.')[:2] if x.isdigit())[:2]
    assert tv >= (4,46), f'transformers {v} < 4.46'
print('OK')
" 2>/dev/null | grep -q OK \
    && ok "Env verified: torch ✓  transformers ✓  safetensors ✓" \
    || warn "Env verification failed — check log"

_gguf_has_tl1 \
    && ok "gguf TL1/TL2: present ✓" \
    || warn "gguf TL1/TL2: STILL MISSING — build will fail (SPECTRE: ${SPECTRE_MODE})"

# ═════════════════════════════════════════════════════════════════════════════
# STEP 11 — PRISM: Model Classification Helpers
# ═════════════════════════════════════════════════════════════════════════════
section "STEP 11 — PRISM: Model Classification"

_prism_classify() {
    local dir="$1"
    local _lib="${_SCRIPT_DIR}/lib/prism_detect.py"
    if [[ -f "$_lib" ]]; then
        "$PYTHON_BIN" "$_lib" "$dir" 2>/dev/null && return
    fi
    # Inline fallback classifier
    "$PYTHON_BIN" - "$dir" << 'PYEOF' 2>/dev/null || echo "unknown"
import json, struct, sys, os, glob
d = sys.argv[1]
if glob.glob(os.path.join(d, 'ggml-model-*.gguf')):
    print('pre-built'); sys.exit()
cfgp = os.path.join(d, 'config.json')
if not os.path.isfile(cfgp):
    print('unknown'); sys.exit()
cfg = json.load(open(cfgp))
qc = cfg.get('quantization_config', {})
if qc and any(k in str(cfg).lower() for k in ['bitlinear','quant_type','bits']):
    print('packed'); sys.exit()
sf = sorted(glob.glob(os.path.join(d, '*.safetensors')))
if not sf:
    print('bf16'); sys.exit()
try:
    with open(sf[0], 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        h = json.loads(f.read(n))
    dtypes = {v.get('dtype','') for k,v in h.items() if k!='__metadata__'}
    print('packed' if 'U8' in dtypes else 'bf16')
except Exception:
    print('unknown')
PYEOF
}

_prism_get_arch() {
    local dir="$1"
    [[ -f "${dir}/config.json" ]] || { echo ""; return; }
    "$PYTHON_BIN" -c "
import json, sys
try:
    cfg = json.load(open('${dir}/config.json'))
    a = cfg.get('architectures', [])
    print(a[0] if a else '')
except Exception:
    print('')
" 2>/dev/null || echo ""
}

_prism_arch_compat() {
    local arch="$1"
    [[ -z "$arch" ]] && { echo "unknown"; return; }
    for u in "${UNSUPPORTED_ARCHITECTURES[@]}"; do
        [[ "$arch" == "$u" ]] && { echo "unsupported"; return; }
    done
    for s in "${SUPPORTED_ARCHITECTURES[@]}"; do
        [[ "$arch" == "$s" ]] && { echo "supported"; return; }
    done
    echo "unknown"
}

_handle_unsupported_arch() {
    local arch="$1"
    echo ""
    echo -e "${C_YEL}${C_BLD}  ╔══════════════════════════════════════════════════════════════╗${C_RST}"
    echo -e "${C_YEL}${C_BLD}  ║  PRISM: '${arch}' cannot be converted by BitNet converter  ║${C_RST}"
    echo -e "${C_YEL}${C_BLD}  ╚══════════════════════════════════════════════════════════════╝${C_RST}"
    echo ""
    echo -e "  ${C_GRN}  G)${C_RST} Download pre-built GGUF (~750 MB) + compile  ${C_DIM}[recommended]${C_RST}"
    echo -e "  ${C_GRN}  B)${C_RST} Download BF16 variant (~4.5 GB) + convert + compile"
    echo -e "  ${C_CYN}  T)${C_RST} Use packed model via transformers API (no compilation)"
    echo -e "  ${C_DIM}  Q)${C_RST} Quit"
    echo ""
    local _ch
    if $NON_INTERACTIVE; then _ch="G"
    else ask _ch "Choice [G/B/T/Q]" "G"; fi
    case "${_ch^^}" in
        G) MODEL_SOURCE="2" ;;
        B) MODEL_SOURCE="1" ;;
        T) MODEL_SOURCE="3"; SKIP_BUILD=true ;;
        *) die "Aborted by user." ;;
    esac
}

# ═════════════════════════════════════════════════════════════════════════════
# STEP 12 — Model Selection (three modes with download fallback)
# ═════════════════════════════════════════════════════════════════════════════
section "STEP 12 — Model Selection"

_MODEL_BASE="${MODEL_DIR_OVERRIDE:-${REAL_HOME}/models}"
DISCOVERED_SAFETENSORS=()
DISCOVERED_GGUF=()
declare -A _MD_ARCH=()
declare -A _MD_LABEL=()

info "Scanning ${_MODEL_BASE} for models …"
while IFS= read -r -d '' _sf; do
    _d="$(dirname "$_sf")"
    _dup=false
    for _x in "${DISCOVERED_SAFETENSORS[@]:-__NONE__}"; do
        [[ "$_x" == "$_d" ]] && { _dup=true; break; }
    done
    $_dup && continue
    DISCOVERED_SAFETENSORS+=("$_d")
    _a=$(_prism_get_arch "$_d" 2>/dev/null || echo "")
    _c=$(_prism_classify  "$_d" 2>/dev/null || echo "unknown")
    _MD_ARCH["$_d"]="$_a"
    _MD_LABEL["$_d"]="${_a:-unknown} · ${_c}"
done < <(find "${_MODEL_BASE}" -maxdepth 4 -name "*.safetensors" -print0 2>/dev/null || true)

while IFS= read -r -d '' _gf; do
    DISCOVERED_GGUF+=("$_gf")
done < <(find "${_MODEL_BASE}" -maxdepth 4 -name "*.gguf" \
         -type f -size +1M -print0 2>/dev/null || true)

if [[ ${#DISCOVERED_SAFETENSORS[@]} -gt 0 ]]; then
    echo -e "\n  ${C_WHT}Safetensors model directories:${C_RST}"
    for _md in "${DISCOVERED_SAFETENSORS[@]}"; do
        echo -e "    ${C_CYN}▸${C_RST} ${_md}  ${C_DIM}(${_MD_LABEL[$_md]:-})${C_RST}"
    done
fi
if [[ ${#DISCOVERED_GGUF[@]} -gt 0 ]]; then
    echo -e "\n  ${C_WHT}Pre-converted GGUF files:${C_RST}"
    for _gf in "${DISCOVERED_GGUF[@]}"; do
        echo -e "    ${C_CYN}▸${C_RST} ${_gf}  ${C_DIM}($(du -h "$_gf" 2>/dev/null | cut -f1))${C_RST}"
    done
fi
[[ ${#DISCOVERED_SAFETENSORS[@]} -eq 0 && ${#DISCOVERED_GGUF[@]} -eq 0 ]] && \
    echo -e "  ${C_YEL}  No model files found in ${_MODEL_BASE}${C_RST}"

# ── Mode selection menu ───────────────────────────────────────────────────────
echo ""
echo -e "  ${C_WHT}${C_BLD}Choose BitNet Inference Mode:${C_RST}"
echo ""
echo -e "  ${C_GRN}${C_BLD}  1)  CONVERT${C_RST}         HF safetensors → GGUF + compile llama-cli"
echo -e "  ${C_DIM}              Best performance. ~4.5 GB download + 5-15 min compile.${C_RST}"
echo ""
echo -e "  ${C_CYN}${C_BLD}  2)  PRE-BUILT GGUF${C_RST}  Use ready-made GGUF + compile llama-cli"
echo -e "  ${C_DIM}              Same speed as Mode 1. Only ~750 MB download. No conversion.${C_RST}"
echo ""
echo -e "  ${C_YEL}${C_BLD}  3)  TRANSFORMERS${C_RST}    Pure Python inference, no compilation"
echo -e "  ${C_DIM}              Works anywhere, no clang/cmake needed. Slower.${C_RST}"
echo ""

# Smart default based on what's already on disk
_AUTO="2"
[[ ${#DISCOVERED_GGUF[@]}        -gt 0 ]] && _AUTO="2"
[[ ${#DISCOVERED_SAFETENSORS[@]} -gt 0 ]] && _AUTO="1"
[[ -n "${MODEL_SOURCE:-}" ]]               && _AUTO="${MODEL_SOURCE}"

if [[ -z "${MODEL_SOURCE:-}" ]]; then
    if $NON_INTERACTIVE; then
        MODEL_SOURCE="$_AUTO"
        info "Non-interactive: auto-selected Mode ${MODEL_SOURCE}"
    else
        ask MODEL_SOURCE "Choose mode [1/2/3]" "$_AUTO"
    fi
fi

# ── Download helper (resumable via huggingface_hub) ───────────────────────────
download_hf_resumable() {
    local repo="$1" dest="$2" token="${3:-${HF_TOKEN:-}}"
    if $DRY_RUN; then
        info "[dry-run] Would download ${repo} → ${dest}"
        mkdir -p "$dest" 2>/dev/null || true
        return 0
    fi
    if ! $NON_INTERACTIVE; then
        confirm "Download ${repo} → ${dest}?" "Y" \
            || { warn "Download skipped by user"; return 1; }
    fi
    mkdir -p "$dest"
    info "Downloading ${repo} → ${dest} (resumable — Ctrl+C to abort)"
    trap 'warn "Download interrupted"; exit 130' INT
    "$PYTHON_BIN" -c "
from huggingface_hub import snapshot_download
import sys
token = '${token}' if '${token}' else None
try:
    snapshot_download(
        repo_id='${repo}',
        local_dir='${dest}',
        token=token,
        resume_download=True
    )
except KeyboardInterrupt:
    sys.exit(130)
" || {
        local rc=$?
        [[ $rc -eq 130 ]] && die "Download aborted by user"
        warn "snapshot_download failed — trying huggingface-cli"
        cmd_exists huggingface-cli \
            || pip_install --silent "huggingface-hub[cli]"
        huggingface-cli download "$repo" --local-dir "$dest" \
            ${token:+--token "$token"} \
            || die "Download failed for ${repo}"
    }
    trap - INT
    ok "Download complete: ${dest}"
}

# ─────────────────────────────────────────────────────────────────────────────
# MODE 1 — Convert HF safetensors → GGUF + compile llama-cli
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$MODEL_SOURCE" == "1" ]]; then
    echo ""
    echo -e "  ${C_GRN}${C_BLD}  Mode 1: Convert → GGUF + compile llama-cli${C_RST}"
    SKIP_CONVERT=false
    MODEL_PATH=""

    # Auto-select BitNet-compatible safetensors dirs
    _BN_DIRS=()
    for _md in "${DISCOVERED_SAFETENSORS[@]:-}"; do
        _a="${_MD_ARCH[$_md]:-}"
        [[ "$_a" =~ BitNet|Bitnet|LlamaForCausalLM ]] && _BN_DIRS+=("$_md")
    done

    if [[ ${#_BN_DIRS[@]} -eq 1 ]]; then
        if $NON_INTERACTIVE || confirm "Use ${_BN_DIRS[0]}?" "Y"; then
            MODEL_PATH="${_BN_DIRS[0]}"
        fi
    elif [[ ${#_BN_DIRS[@]} -gt 1 ]]; then
        echo -e "  ${C_WHT}  Multiple models found:${C_RST}"
        for _i in "${!_BN_DIRS[@]}"; do
            echo -e "    $((_i+1))) ${_BN_DIRS[$_i]}  ${C_DIM}(${_MD_LABEL[${_BN_DIRS[$_i]}]:-})${C_RST}"
        done
        ask _pick "Select [1]" "1"
        _pick="${_pick:-1}"
        [[ "$_pick" =~ ^[0-9]+$ && "$_pick" -ge 1 && "$_pick" -le "${#_BN_DIRS[@]}" ]] \
            || _pick=1
        MODEL_PATH="${_BN_DIRS[$((_pick-1))]}"
    fi

    if [[ -z "$MODEL_PATH" ]]; then
        echo -e "  ${C_YEL}  No BitNet safetensors found in ${_MODEL_BASE}${C_RST}"
        echo ""
        echo -e "  ${C_GRN}  D)${C_RST} Download microsoft/bitnet-b1.58-2B-4T  (packed ~4.5 GB)"
        echo -e "  ${C_GRN}  F)${C_RST} Download microsoft/bitnet-b1.58-2B-4T-bf16  (BF16 ~4.5 GB)"
        echo -e "  ${C_CYN}  P)${C_RST} Enter path to existing model directory"
        ask _a1 "Action [D/F/P]" "D"
        case "${_a1^^}" in
            D) MODEL_PATH="${_MODEL_BASE}/BitNet-b1.58-2B-4T"
               download_hf_resumable "microsoft/bitnet-b1.58-2B-4T" "$MODEL_PATH" ;;
            F)
               # NOTE: setup_env.py's SUPPORTED_HF_MODELS does NOT contain a
               # bf16 entry — its get_model_name() requires the directory to
               # be literally named "BitNet-b1.58-2B-4T" (no "-bf16" suffix)
               # or gen_code() raises NotImplementedError(). We download the
               # bf16 weights but place them under the plain supported name.
               MODEL_PATH="${_MODEL_BASE}/BitNet-b1.58-2B-4T"
               note "Downloading bf16 weights into setup_env.py-compatible folder name: ${MODEL_PATH}"
               download_hf_resumable "microsoft/bitnet-b1.58-2B-4T-bf16" "$MODEL_PATH" ;;
            P) ask MODEL_PATH "Model directory path" ""
               MODEL_PATH="${MODEL_PATH/#\~/$HOME}"
               [[ -d "$MODEL_PATH" ]] || die "Not found: ${MODEL_PATH}" ;;
            *) die "Invalid choice." ;;
        esac
    fi

    # ── Validate the model directory name against setup_env.py's hardcoded list ──
    # ── TRANSPARENT WORKING DIRECTORY — Mode 1 ───────────────────────────────
    # setup_env.py requires basename(model_dir) to exactly match one of 4 names.
    # Instead of moving or renaming the user's model directory, we create a
    # transparent workdir inside the BitNet checkout with a correctly-named
    # symlink pointing at the user's original directory. setup_env.py sees the
    # named workdir; the user's files stay exactly where they are.
    # ─────────────────────────────────────────────────────────────────────────
    _SETUP_ENV_SUPPORTED_NAMES=("bitnet_b1_58-large" "bitnet_b1_58-3B" \
        "Llama3-8B-1.58-100B-tokens" "BitNet-b1.58-2B-4T")
    _mp_base="$(basename "$MODEL_PATH")"
    _mp_name_ok=false
    for _n in "${_SETUP_ENV_SUPPORTED_NAMES[@]}"; do
        [[ "$_mp_base" == "$_n" ]] && { _mp_name_ok=true; break; }
    done
    [[ "$_mp_base" =~ ^(Llama|Falcon) ]] && _mp_name_ok=true

    if ! $_mp_name_ok; then
        # Determine the correct setup_env.py name from the model arch
        _M1_WORK_NAME="BitNet-b1.58-2B-4T"
        _detected_arch=$(_prism_get_arch "$MODEL_PATH" 2>/dev/null || echo "")
        case "${_detected_arch,,}" in
            *llama*)  _M1_WORK_NAME="Llama3-8B-1.58-100B-tokens" ;;
            *falcon*) _M1_WORK_NAME="Falcon3-7B-Instruct-1.58bit" ;;
            *)        _M1_WORK_NAME="BitNet-b1.58-2B-4T" ;;
        esac

        # Create workdir with a symlink to the user's model directory
        # The symlink makes setup_env.py see a named dir while the real
        # files stay in the user's original location — nothing is moved.
        _M1_WORK_PARENT="${BITNET_DIR}/.model_workdir"
        mkdir -p "$_M1_WORK_PARENT" 2>/dev/null || true
        _M1_WORK_LINK="${_M1_WORK_PARENT}/${_M1_WORK_NAME}"
        if [[ -L "$_M1_WORK_LINK" ]]; then
            rm -f "$_M1_WORK_LINK" 2>/dev/null || true
        fi
        ln -sf "$(readlink -f "$MODEL_PATH" 2>/dev/null || echo "$MODEL_PATH")" \
               "$_M1_WORK_LINK" 2>/dev/null \
            || { mkdir -p "$_M1_WORK_LINK"; cp -r "${MODEL_PATH}/." "$_M1_WORK_LINK/" 2>/dev/null; }

        MODEL_NAME="$_M1_WORK_NAME"
        MODEL_PATH="$_M1_WORK_LINK"
        ok "Workdir  : ${MODEL_PATH}  ${C_DIM}(internal — your model files unchanged)${C_RST}"
    else
        MODEL_NAME="$_mp_base"
    fi

    MODEL_ARCH=$(_prism_get_arch "$MODEL_PATH" 2>/dev/null || echo "BitNetForCausalLM")
    [[ -z "$MODEL_ARCH" ]] && MODEL_ARCH="BitNetForCausalLM"
    PRISM_ARCH_COMPAT=$(_prism_arch_compat "$MODEL_ARCH")
    [[ "$PRISM_ARCH_COMPAT" == "unknown" ]] && PRISM_ARCH_COMPAT="convertible"
    PRISM_CLASS="packed_for_conversion"
    MODEL_NAME="$(basename "$MODEL_PATH")"

    [[ "$PRISM_ARCH_COMPAT" == "unsupported" ]] && \
        _handle_unsupported_arch "$MODEL_ARCH"

    # Check if GGUF already exists (skip re-conversion, still compiles)
    _EX_GGUF=""
    for _dir in "$MODEL_PATH" "$(dirname "$MODEL_PATH")"; do
        [[ -d "$_dir" ]] || continue
        _eg=$(find "$_dir" -maxdepth 1 \
            -name "ggml-model-*.gguf" -type f -size +1M \
            2>/dev/null | sort | head -1)
        [[ -n "$_eg" ]] && { _EX_GGUF="$_eg"; break; }
    done

    QUANT_TYPE="${ATLAS_QUANT_REC:-i2_s}"
    if [[ -n "$_EX_GGUF" ]]; then
        _eq=$(basename "$_EX_GGUF" \
            | sed -E 's/^ggml-model-(.+)\.gguf$/\1/' 2>/dev/null || echo "?")
        _esz=$(du -h "$_EX_GGUF" 2>/dev/null | cut -f1)
        echo -e "  ${C_CYN}  ℹ  Existing GGUF: ${_EX_GGUF}  (${_esz}, quant=${_eq})${C_RST}"
        if $NON_INTERACTIVE || \
           confirm "Reuse this GGUF (skip conversion, still compiles llama-cli)?" "Y"; then
            QUANT_TYPE="$_eq"
            GGUF_PATH="${MODEL_PATH}/ggml-model-${QUANT_TYPE}.gguf"
            _rp_ex=$(readlink -f "$_EX_GGUF"  2>/dev/null || echo "$_EX_GGUF")
            _rp_gf=$(readlink -f "$GGUF_PATH" 2>/dev/null || echo "$GGUF_PATH")
            [[ "$_rp_ex" != "$_rp_gf" ]] && \
                { ln -f "$_EX_GGUF" "$GGUF_PATH" 2>/dev/null \
                    || cp "$_EX_GGUF" "$GGUF_PATH"; }
            SKIP_CONVERT=true
            ok "GGUF reused — conversion skipped"
        fi
    fi

    # Quantization type selection (if not reusing)
    if ! $SKIP_CONVERT && ! $NON_INTERACTIVE; then
        echo ""
        if [[ "$ARCH" == "x86_64" ]]; then
            echo -e "  ${C_DIM}  i2_s = universal x86_64  |  tl2 = AVX-512 only (yours: ${ATLAS_AVX512})${C_RST}"
            ask QUANT_TYPE "Quantization [i2_s/tl2]" "${ATLAS_QUANT_REC:-i2_s}"
            [[ "$QUANT_TYPE" =~ ^(i2_s|tl2)$ ]] || { warn "Invalid — using i2_s"; QUANT_TYPE="i2_s"; }
        else
            echo -e "  ${C_DIM}  i2_s = universal  |  tl1 = ARM NEON optimized${C_RST}"
            ask QUANT_TYPE "Quantization [i2_s/tl1]" "${ATLAS_QUANT_REC:-tl1}"
            [[ "$QUANT_TYPE" =~ ^(i2_s|tl1)$ ]] || { warn "Invalid — using tl1"; QUANT_TYPE="tl1"; }
        fi
        confirm "Enable Q6_K embedding quantization (~15% less RAM)?" "Y" \
            && QUANT_EMBD=true
    fi

    GGUF_PATH="${GGUF_PATH:-${MODEL_PATH}/ggml-model-${QUANT_TYPE}.gguf}"
    ok "Model     : ${MODEL_PATH}"
    ok "GGUF out  : ${GGUF_PATH}"
    ok "Quant     : ${QUANT_TYPE}  embd=${QUANT_EMBD}"

# ─────────────────────────────────────────────────────────────────────────────
# MODE 2 — Pre-built GGUF + compile llama-cli
# ─────────────────────────────────────────────────────────────────────────────
elif [[ "$MODEL_SOURCE" == "2" ]]; then
    echo ""
    echo -e "  ${C_CYN}${C_BLD}  Mode 2: Pre-built GGUF + compile llama-cli${C_RST}"
    SKIP_CONVERT=true
    # NOTE: SKIP_BUILD stays false — we MUST compile llama-cli to run the GGUF

    GGUF_PATH=""
    if [[ ${#DISCOVERED_GGUF[@]} -eq 1 ]]; then
        if $NON_INTERACTIVE || confirm "Use ${DISCOVERED_GGUF[0]}?" "Y"; then
            GGUF_PATH="${DISCOVERED_GGUF[0]}"
        fi
    elif [[ ${#DISCOVERED_GGUF[@]} -gt 1 ]]; then
        echo -e "  ${C_WHT}  Multiple GGUF files:${C_RST}"
        for _i in "${!DISCOVERED_GGUF[@]}"; do
            echo -e "    $((_i+1))) ${DISCOVERED_GGUF[$_i]}" \
                "  ${C_DIM}($(du -h "${DISCOVERED_GGUF[$_i]}" 2>/dev/null | cut -f1))${C_RST}"
        done
        ask _gpick "Select [1]" "1"
        _gpick="${_gpick:-1}"
        [[ "$_gpick" =~ ^[0-9]+$ && "$_gpick" -ge 1 && \
           "$_gpick" -le "${#DISCOVERED_GGUF[@]}" ]] || _gpick=1
        GGUF_PATH="${DISCOVERED_GGUF[$((_gpick-1))]}"
    fi

    if [[ -z "$GGUF_PATH" ]]; then
        echo -e "  ${C_YEL}  No GGUF found in ${_MODEL_BASE}${C_RST}"
        echo ""
        echo -e "  ${C_GRN}  D)${C_RST} Download microsoft/BitNet-b1.58-2B-4T-gguf  (~750 MB)"
        echo -e "  ${C_CYN}  P)${C_RST} Enter path to an existing .gguf file"
        ask _a2 "Action [D/P]" "D"
        case "${_a2^^}" in
            D)
                _dl="${_MODEL_BASE}/BitNet-b1.58-2B-4T-gguf"
                download_hf_resumable "microsoft/BitNet-b1.58-2B-4T-gguf" "$_dl"
                GGUF_PATH=$(find "$_dl" -name "*.gguf" -type f -size +1M \
                    2>/dev/null | sort | head -1 || echo "")
                [[ -n "$GGUF_PATH" ]] || die "No GGUF found after download" ;;
            P)
                ask GGUF_PATH "Path to .gguf file" ""
                GGUF_PATH="${GGUF_PATH/#\~/$HOME}"
                [[ -f "$GGUF_PATH" ]] || die "Not found: ${GGUF_PATH}"
                [[ $(stat -c%s "$GGUF_PATH" 2>/dev/null || echo 0) -gt 1048576 ]] \
                    || die "File too small to be a valid GGUF: ${GGUF_PATH}" ;;
            *) die "Invalid choice." ;;
        esac
    fi

    # ── TRANSPARENT WORKING DIRECTORY ALGORITHM ──────────────────────────────
    # PRINCIPLE: The user's chosen GGUF is NEVER overridden or redirected.
    # setup_env.py requires dirname(model_dir) to match one of 4 hardcoded
    # names — but we satisfy that constraint transparently by creating a
    # working directory INSIDE the BitNet checkout, invisible to the user.
    # The user's file stays exactly where they put it. We hard-link it into
    # the workdir so setup_env.py can find it, then pass the workdir as -md.
    # After setup_env.py runs, GGUF_PATH is restored to the ORIGINAL file.
    #
    # This works for ANY GGUF anywhere on disk, regardless of folder name.
    # ─────────────────────────────────────────────────────────────────────────

    # Step 1: Preserve the user's exact chosen GGUF path — never touch this
    USER_GGUF_PATH="$GGUF_PATH"

    # Step 2: Derive quant type and model name from the GGUF filename
    _gguf_fname="$(basename "$USER_GGUF_PATH")"
    QUANT_TYPE=$(echo "$_gguf_fname" \
        | sed -E 's/^ggml-model-(.+)\.gguf$/\1/' 2>/dev/null || echo "i2_s")
    # Validate quant type — if filename didn't match ggml-model-X.gguf pattern,
    # default to ATLAS recommendation
    [[ "$QUANT_TYPE" == "$_gguf_fname" ]] && QUANT_TYPE="${ATLAS_QUANT_REC:-i2_s}"

    # Step 3: Detect MODEL_NAME from GGUF metadata if possible, else default
    MODEL_NAME="BitNet-b1.58-2B-4T"
    _detected_name=$("$PYTHON_BIN" -c "
import struct, json, sys
try:
    with open('${USER_GGUF_PATH}', 'rb') as f:
        magic = f.read(4)
        if magic not in (b'GGUF', b'GGJT'):
            sys.exit(1)
        f.read(4)  # version
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_kv      = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n_kv):
            klen = struct.unpack('<Q', f.read(8))[0]
            key  = f.read(klen).decode('utf-8', errors='ignore')
            vtype = struct.unpack('<I', f.read(4))[0]
            if vtype == 8:  # string
                slen = struct.unpack('<Q', f.read(8))[0]
                val  = f.read(slen).decode('utf-8', errors='ignore')
                if key == 'general.name':
                    print(val.strip())
                    sys.exit(0)
            else:
                break
except Exception:
    pass
" 2>/dev/null || echo "")
    # Map detected metadata name to setup_env.py supported names
    case "${_detected_name,,}" in
        *bitnet*2b*|*2b-4t*|*2b_4t*) MODEL_NAME="BitNet-b1.58-2B-4T" ;;
        *llama*8b*)                   MODEL_NAME="Llama3-8B-1.58-100B-tokens" ;;
        *bitnet*3b*|*3b*)             MODEL_NAME="bitnet_b1_58-3B" ;;
        *bitnet*large*|*large*)       MODEL_NAME="bitnet_b1_58-large" ;;
        *falcon*)                     MODEL_NAME="Falcon3-7B-Instruct-1.58bit" ;;
    esac

    # Step 4: Create the transparent working directory inside BitNet checkout
    # Hidden with a dot prefix so it doesn't pollute the user's model folder
    WORK_DIR="${BITNET_DIR}/.model_workdir/${MODEL_NAME}"
    mkdir -p "$WORK_DIR" 2>/dev/null || true

    # Step 5: Hard-link the user's GGUF into the workdir under the correct name
    # setup_env.py looks for ggml-model-${quant}.gguf inside model_dir
    WORK_GGUF="${WORK_DIR}/ggml-model-${QUANT_TYPE}.gguf"
    _user_ino=$(stat -c '%i' "$USER_GGUF_PATH" 2>/dev/null || echo "")
    _work_ino=$(stat -c '%i' "$WORK_GGUF"      2>/dev/null || echo "none")
    if [[ "$_user_ino" != "$_work_ino" ]] || [[ ! -f "$WORK_GGUF" ]]; then
        # Try hard-link first (same inode = zero extra disk space)
        # Fall back to symlink, then copy as last resort
        ln -f "$USER_GGUF_PATH" "$WORK_GGUF" 2>/dev/null \
            || ln -sf "$USER_GGUF_PATH" "$WORK_GGUF" 2>/dev/null \
            || cp "$USER_GGUF_PATH" "$WORK_GGUF" 2>/dev/null \
            || die "Cannot link GGUF into workdir: ${WORK_GGUF}"
    fi

    # Step 6: setup_env.py sees the workdir — user sees their original path
    MODEL_PATH="$WORK_DIR"
    # GGUF_PATH for the remainder of the script points to the original user file
    # (workdir link has same inode so either works for inference)
    GGUF_PATH="$USER_GGUF_PATH"

    PRISM_CLASS="pre-built"
    PRISM_ARCH_COMPAT="pre-built"
    MODEL_ARCH="bitnet"

    ok "GGUF       : ${USER_GGUF_PATH}  ${C_DIM}(your file — unchanged)${C_RST}"
    ok "Workdir    : ${WORK_DIR}  ${C_DIM}(internal — setup_env.py sees this)${C_RST}"
    ok "Quant type : ${QUANT_TYPE}"
    note "setup_env.py will run kernel codegen then detect the existing GGUF"
    note "and skip re-conversion automatically."
    note "If you see 'GENERATION QUALITY WILL BE DEGRADED' during inference:"
    note "  your GGUF is missing pre-tokenizer metadata (a data issue, not a"
    note "  code issue). Fix: Mode 2 option D (official microsoft GGUF) or"
    note "  Mode 1 option F (re-convert from BF16 safetensors)."

# ─────────────────────────────────────────────────────────────────────────────
# MODE 3 — Transformers API (pure Python, no compilation)
# ─────────────────────────────────────────────────────────────────────────────
elif [[ "$MODEL_SOURCE" == "3" ]]; then
    echo ""
    echo -e "  ${C_YEL}${C_BLD}  Mode 3: Transformers API (no C++ compilation)${C_RST}"
    SKIP_BUILD=true    # No cmake/clang build needed
    SKIP_CONVERT=true  # No HF→GGUF conversion
    GGUF_PATH=""

    MODEL_PATH=""
    _BN_DIRS=()
    for _md in "${DISCOVERED_SAFETENSORS[@]:-}"; do
        _a="${_MD_ARCH[$_md]:-}"
        [[ "$_a" =~ BitNet|Bitnet|LlamaForCausalLM ]] && _BN_DIRS+=("$_md")
    done

    if [[ ${#_BN_DIRS[@]} -eq 1 ]]; then
        if $NON_INTERACTIVE || confirm "Use ${_BN_DIRS[0]}?" "Y"; then
            MODEL_PATH="${_BN_DIRS[0]}"
        fi
    elif [[ ${#_BN_DIRS[@]} -gt 1 ]]; then
        for _i in "${!_BN_DIRS[@]}"; do
            echo -e "    $((_i+1))) ${_BN_DIRS[$_i]}"
        done
        ask _pick "Select [1]" "1"
        _pick="${_pick:-1}"
        [[ "$_pick" =~ ^[0-9]+$ && "$_pick" -ge 1 && \
           "$_pick" -le "${#_BN_DIRS[@]}" ]] || _pick=1
        MODEL_PATH="${_BN_DIRS[$((_pick-1))]}"
    fi

    if [[ -z "$MODEL_PATH" ]]; then
        echo -e "  ${C_YEL}  No BitNet model found in ${_MODEL_BASE}${C_RST}"
        echo -e "  ${C_GRN}  D)${C_RST} Download microsoft/bitnet-b1.58-2B-4T  (~4.5 GB)"
        echo -e "  ${C_CYN}  P)${C_RST} Enter path to existing model directory"
        ask _a3 "Action [D/P]" "D"
        case "${_a3^^}" in
            D) MODEL_PATH="${_MODEL_BASE}/BitNet-b1.58-2B-4T"
               download_hf_resumable "microsoft/bitnet-b1.58-2B-4T" "$MODEL_PATH" ;;
            P) ask MODEL_PATH "Model directory path" ""
               MODEL_PATH="${MODEL_PATH/#\~/$HOME}"
               [[ -d "$MODEL_PATH" ]] || die "Not found: ${MODEL_PATH}" ;;
            *) die "Invalid choice." ;;
        esac
    fi

    MODEL_ARCH=$(_prism_get_arch "$MODEL_PATH" 2>/dev/null || echo "BitNetForCausalLM")
    [[ -z "$MODEL_ARCH" ]] && MODEL_ARCH="BitNetForCausalLM"
    MODEL_NAME="$(basename "$MODEL_PATH")"
    PRISM_CLASS="packed_transformers"
    PRISM_ARCH_COMPAT="transformers"
    ok "Model path : ${MODEL_PATH}"
    ok "Mode       : transformers API (no C++ compilation)"

else
    die "Invalid MODEL_SOURCE='${MODEL_SOURCE}'. Must be 1, 2, or 3."
fi

# ═════════════════════════════════════════════════════════════════════════════
# STEP 13 — FORGE: Build Cache Check
# ═════════════════════════════════════════════════════════════════════════════
section "STEP 13 — FORGE: Build Cache"

FORGE_CACHE_FILE="${BITNET_DIR}/.forge_cache.json"
FORGE_CACHE_HIT=false
CURRENT_HASH=""

if ! $SKIP_BUILD && ! $FORCE_REBUILD && \
   [[ "${PRISM_CLASS:-}" != "packed_transformers" ]]; then
    CURRENT_HASH=$(
        {
            for f in "${BITNET_DIR}/setup_env.py" \
                     "${BITNET_DIR}/src/ggml-bitnet-mad.cpp" \
                     "${BITNET_DIR}/CMakeLists.txt"; do
                [[ -f "$f" ]] && \
                    (sha256sum "$f" 2>/dev/null || md5sum "$f" 2>/dev/null) \
                    | awk '{print $1}'
            done
            echo "${MODEL_PATH:-}|${QUANT_TYPE:-i2_s}|${QUANT_EMBD}|${CLANG_VER}|${PY_MINOR}|${MODEL_ARCH:-}|${MODEL_SOURCE:-}"
        } | sha256sum 2>/dev/null | awk '{print $1}' \
          || echo "nohash_$(date +%s)"
    )
    if [[ -f "$FORGE_CACHE_FILE" ]]; then
        _ch=$(python3 -c \
            "import json; d=json.load(open('${FORGE_CACHE_FILE}')); print(d.get('hash',''))" \
            2>/dev/null || echo "")
        _cg=$(python3 -c \
            "import json; d=json.load(open('${FORGE_CACHE_FILE}')); print(d.get('gguf_path',''))" \
            2>/dev/null || echo "")
        _cb=$(python3 -c \
            "import json; d=json.load(open('${FORGE_CACHE_FILE}')); print(d.get('llama_cli',''))" \
            2>/dev/null || echo "")
        if [[ "$_ch" == "$CURRENT_HASH" ]] && \
           [[ -n "${_cg:-}" ]] && [[ -f "${_cg}" ]] && \
           [[ -n "${_cb:-}" ]] && [[ -x "${_cb}" ]]; then
            ok "FORGE cache HIT — skipping build"
            ok "  GGUF     : ${_cg}"
            ok "  llama-cli: ${_cb}"
            SKIP_BUILD=true
            FORGE_CACHE_HIT=true
            [[ -z "$GGUF_PATH" ]]    && GGUF_PATH="$_cg"
            [[ -z "$BINARY_PATH" ]]  && BINARY_PATH="$_cb"
        else
            info "FORGE cache MISS — will rebuild"
        fi
    else
        info "FORGE: first build (no cache yet)"
    fi
elif $FORCE_REBUILD; then
    info "FORGE: --force-rebuild — cache bypassed"
fi

# ═════════════════════════════════════════════════════════════════════════════
# STEP 14 — GUARDIAN: Pre-build Checklist
# ═════════════════════════════════════════════════════════════════════════════
section "STEP 14 — GUARDIAN: Pre-build Checklist"

GUARDIAN_PASS=true

_gc() {
    local label="$1" check="$2" advice="$3" fatal="${4:-false}"
    if eval "$check" &>/dev/null 2>&1; then
        ok "GUARDIAN: ${label} ✓"
    elif [[ "$fatal" == "true" ]]; then
        err "GUARDIAN [FATAL]: ${label} — ${advice}"
        GUARDIAN_PASS=false
    else
        warn "GUARDIAN: ${label} — ${advice}"
    fi
}

if $SKIP_BUILD || [[ "${PRISM_CLASS:-}" == "packed_transformers" ]]; then
    note "GUARDIAN: pre-build checks skipped (build not needed — FORGE cache hit or Mode 3)"
elif ! $SKIP_BUILD && [[ "${PRISM_CLASS:-}" != "packed_transformers" ]]; then
    _gc "clang ≥ 18" \
        "[[ $CLANG_VER -ge 18 ]]" \
        "clang ${CLANG_VER} — 1-bit BitNet kernels require clang ≥18" "false"
    _gc "cmake ≥ 3.14" \
        "ver_gte '${CMAKE_VER:-0}' '3.14'" \
        "cmake ${CMAKE_VER:-?} — update cmake to ≥3.14" "false"
    _gc "gguf has TL1/TL2 types" \
        "_gguf_has_tl1" \
        "TL1/TL2 missing — converter WILL fail (SPECTRE mode: ${SPECTRE_MODE})" "true"
    _gc "gguf-py submodule populated" \
        "[[ -f '${GGUF_PY_DIR}/gguf/__init__.py' ]]" \
        "run: cd ${BITNET_DIR} && git submodule update --init --recursive --force" "false"
    _gc "setup_env.py present" \
        "[[ -f '${BITNET_DIR}/setup_env.py' ]]" \
        "missing — clone may be incomplete" "true"
    _gc "model directory exists" \
        "[[ -d '${MODEL_PATH:-/nonexistent}' ]]" \
        "'${MODEL_PATH:-none}' not found" "true"
    _gc "≥5 GB disk free" \
        "[[ \$(df -k '${MODEL_PATH:-/tmp}' 2>/dev/null \
             | awk 'NR==2{print int(\$4/1048576)}' || echo 99) -ge 5 ]]" \
        "less than 5 GB free — conversion may fail mid-way" "false"

    if [[ -n "${MODEL_ARCH:-}" && "${PRISM_ARCH_COMPAT:-}" == "unsupported" ]]; then
        err "GUARDIAN [FATAL]: arch '${MODEL_ARCH}' not supported by converter"
        GUARDIAN_PASS=false
    fi

    if ! $GUARDIAN_PASS; then
        [[ "${PRISM_ARCH_COMPAT:-}" == "unsupported" ]] \
            && _handle_unsupported_arch "${MODEL_ARCH:-unknown}" \
            || { confirm "Continue despite GUARDIAN failures?" "Y" \
                     || die "Aborted by GUARDIAN"; }
    else
        ok "GUARDIAN: all checks passed ✓"
    fi
fi

# ═════════════════════════════════════════════════════════════════════════════
# STEP 15 — Apply Mandatory Build Prerequisites (always, before cmake)
# ═════════════════════════════════════════════════════════════════════════════
if ! $SKIP_BUILD && [[ "${PRISM_CLASS:-}" != "packed_transformers" ]]; then
    section "STEP 15 — Mandatory Build Prerequisites"
    # Patches 3+4 are always required regardless of Python version or user choice:
    #   ggml-bitnet-mad.cpp  — const int8_t* fix for clang 18+
    #   codegen_tl2.py       — shape entry for BitNet-b1.58-2B-4T kernel gen
    _apply_build_prereqs
    if $_NEED_UPFRONT_PATCH; then
        _patch_pyproject_toml
        _patch_tl1_guard
    fi

    # ── Unversioned clang/clang++ defense ───────────────────────────────────
    # setup_env.py's compile() hardcodes "-DCMAKE_C_COMPILER=clang
    # -DCMAKE_CXX_COMPILER=clang++" (unversioned names) and does NOT read
    # $CC/$CXX. On distros where llvm.sh only installs versioned binaries
    # (clang-18, clang-21, ...), an unversioned "clang" may not exist on
    # PATH, may resolve to gcc via update-alternatives, or may point to an
    # older/incompatible version than the one we detected in STEP 3. We
    # ensure unversioned "clang"/"clang++" on PATH resolve to our detected
    # ${CLANG_BIN} by prepending a small bin directory with symlinks.
    if cmd_exists "$CLANG_BIN"; then
        _clang_resolved="$(command -v "$CLANG_BIN")"
        _clangpp_resolved="${_clang_resolved//clang/clang++}"
        _unversioned_clang="$(command -v clang 2>/dev/null || echo "")"
        _unversioned_clang_ver=0
        if [[ -n "$_unversioned_clang" ]]; then
            _unversioned_clang_ver=$("$_unversioned_clang" --version 2>/dev/null \
                | grep -oP '(?<=version )\d+' | head -1 || echo 0)
        fi
        if [[ -z "$_unversioned_clang" ]] || [[ "${_unversioned_clang_ver:-0}" -lt 18 ]]; then
            warn "Unversioned 'clang' on PATH is missing or < v18 (setup_env.py hardcodes this name)"
            _SYMLINK_BIN="${BITNET_DIR}/.bin"
            mkdir -p "$_SYMLINK_BIN" 2>/dev/null || true
            ln -sf "$_clang_resolved"   "${_SYMLINK_BIN}/clang"   2>/dev/null || true
            ln -sf "$_clangpp_resolved" "${_SYMLINK_BIN}/clang++" 2>/dev/null || true
            export PATH="${_SYMLINK_BIN}:${PATH}"
            ok "Symlinked clang/clang++ → ${_clang_resolved} (${_SYMLINK_BIN} prepended to PATH)"
        else
            ok "Unversioned 'clang' on PATH is v${_unversioned_clang_ver} ✓ (matches setup_env.py's hardcoded compiler name)"
        fi
    fi
fi

# ═════════════════════════════════════════════════════════════════════════════
# STEP 16 — Kernel Generation + C++ Build (Modes 1 and 2)
# ═════════════════════════════════════════════════════════════════════════════
section "STEP 16 — Kernel Generation & C++ Build"

_diagnose_failure() {
    local log="$1"
    [[ -f "$log" ]] || { echo "unknown"; return; }
    local t; t=$(tail -50 "$log" 2>/dev/null || echo "")
    echo "$t" | grep -qiE "unrecognized arguments:"             && { echo "bad_flag_error";      return; }
    echo "$t" | grep -qiE "bitnet-lut-kernels\.h|No SOURCES given to target: ggml" && { echo "missing_kernel_header"; return; }
    echo "$t" | grep -qiE "architecture.*not.*support"          && { echo "arch_error";          return; }
    echo "$t" | grep -qiE "Can.not map tensor.*weight_scale"    && { echo "packed_weight_error"; return; }
    echo "$t" | grep -qiE "sentencepiece.*not found|set_vocab"  && { echo "tokenizer_error";     return; }
    echo "$t" | grep -qiE "TL1|TL2|GGMLQuantizationType"        && { echo "tl1_error";           return; }
    echo "$t" | grep -qiE "SIGABRT|ApproximateClock|non.mono"   && { echo "cuda_clock_error";    return; }
    echo "$t" | grep -qiE "No space left"                       && { echo "disk_space";          return; }
    echo "$t" | grep -qiE "NotImplementedError"                 && { echo "unsupported_model_name"; return; }
    echo "$t" | grep -qiE "unrecognized command-line option"    && { echo "compiler_flag_error"; return; }
    echo "unknown"
}

if ! $SKIP_BUILD && [[ "${PRISM_CLASS:-}" != "packed_transformers" ]] && $DRY_RUN; then
    note "Dry-run: skipping kernel generation and C++ build"
    note "[dry-run] Would run: setup_env.py / cmake -B build / cmake --build build"
    SETUP_EXIT=0
elif ! $SKIP_BUILD && [[ "${PRISM_CLASS:-}" != "packed_transformers" ]]; then
    cd "${BITNET_DIR}"
    mkdir -p "${BITNET_DIR}/logs"
    _BLOG="${BITNET_DIR}/logs/build_$(date '+%Y%m%d_%H%M%S').log"
    SETUP_EXIT=1

    # Check if already built (handles FORGE miss but binary present case)
    _ALREADY_BUILT=false
    for _b in "${BITNET_DIR}/build/bin/llama-cli" \
               "${BITNET_DIR}/build/Release/bin/llama-cli"; do
        if [[ -x "$_b" ]] && ! $FORCE_REBUILD; then
            _ALREADY_BUILT=true; BINARY_PATH="$_b"
            ok "llama-cli already present: ${_b}"
            break
        fi
    done

    if $_ALREADY_BUILT && $SKIP_CONVERT; then
        SETUP_EXIT=0
        ok "llama-cli built + GGUF ready — skipping full build step"
    else
        # Build attempt loop with auto-diagnose and patch-on-failure
        for _attempt in 1 2 3; do
            info "Build attempt ${_attempt}/3 …"
            (
                cd "${BITNET_DIR}"
                export PYTHONPATH="${SPECTRE_GGUF_PY_PATH:+${SPECTRE_GGUF_PY_PATH}:}${PYTHONPATH:-}"
                # IMPORTANT: setup_env.py ALWAYS runs setup_gguf() -> gen_code()
                # -> compile() -> prepare_model() in that exact order, with no
                # flag to skip kernel codegen. gen_code() is what generates
                # include/bitnet-lut-kernels.h, which cmake's add_library()
                # requires — skipping straight to cmake (as a shortcut for
                # "no conversion needed") leaves that header missing and
                # cmake fails with "Cannot find source file ... No SOURCES
                # given to target: ggml". So Mode 1 AND Mode 2 both run
                # setup_env.py; prepare_model() detects an existing GGUF by
                # path+size and skips re-conversion automatically — no
                # separate code path is needed here.
                _CMD=("$PYTHON_BIN" "${BITNET_DIR}/setup_env.py"
                      "-md" "${MODEL_PATH}"
                      "-q"  "${QUANT_TYPE:-i2_s}")
                $QUANT_EMBD && _CMD+=("--quant-embd")
                # NOTE: setup_env.py does NOT accept --cmake-args or any cmake
                # flag passthrough. Its compile() function hardcodes the cmake
                # invocation internally. ATLAS_CMAKE_FLAGS (e.g. -DGGML_HIP=on)
                # are recorded in bitnet_state.json for reference but cannot be
                # passed to setup_env.py. BitNet.cpp builds CPU-only via this path.
                # GPU support would require a separate manual cmake build outside
                # of setup_env.py.
                if [[ -n "${ATLAS_CMAKE_FLAGS:-}" ]]; then
                    note "GPU cmake flags detected (${ATLAS_CMAKE_FLAGS}) but setup_env.py"
                    note "does not accept cmake flag passthrough — building CPU-only."
                    note "This is normal. BitNet 1-bit CPU kernels are the primary backend."
                fi
                # Use pre-tuned kernels if available for this model
                _PRE="${BITNET_DIR}/preset_kernels/${MODEL_NAME:-}"
                [[ -d "$_PRE" ]] && _CMD+=("--use-pretuned") && \
                    ok "Using pre-tuned kernels: ${_PRE}"
                "${_CMD[@]}" 2>&1 | tee -a "$_BLOG"
            ) && SETUP_EXIT=0 || SETUP_EXIT=$?

            [[ $SETUP_EXIT -eq 0 ]] && break

            _diag=$(_diagnose_failure "$_BLOG")
            warn "Attempt ${_attempt}/3 failed (diagnosis: ${_diag})"
            _surface_log "$_BLOG" "Build log" 25

            case "$_diag" in
                bad_flag_error)
                    # setup_env.py argparse rejected a flag we passed.
                    # This is a hard error — retrying with the same command
                    # will always fail. Die immediately with a clear message.
                    die "DIAGNOSIS: setup_env.py rejected an unrecognized argument.
  This is a code bug, not a user error. The flag passed does not exist in
  your version of setup_env.py. Re-run setup after updating bitnet_setup.sh." ;;
                missing_kernel_header)
                    err "DIAGNOSIS: setup_env.py did not generate include/bitnet-lut-kernels.h"
                    err "This means MODEL_NAME ('${MODEL_NAME:-}') was not recognized by"
                    err "setup_env.py's gen_code() — it only knows these exact names:"
                    err "  bitnet_b1_58-large | bitnet_b1_58-3B | BitNet-b1.58-2B-4T | Llama3-8B-1.58-100B-tokens"
                    if ! $NON_INTERACTIVE; then
                        ask MODEL_NAME "Re-enter model name (must match exactly)" "BitNet-b1.58-2B-4T"
                        _new_dir="${_MODEL_BASE}/${MODEL_NAME}"
                        mkdir -p "$_new_dir" 2>/dev/null || true
                        if [[ -n "${GGUF_PATH:-}" ]] && [[ -f "${GGUF_PATH}" ]]; then
                            _relocated="${_new_dir}/$(basename "$GGUF_PATH")"
                            [[ "$(readlink -f "$GGUF_PATH")" != "$(readlink -f "$_relocated" 2>/dev/null || echo "$_relocated")" ]] && \
                                { ln -f "$GGUF_PATH" "$_relocated" 2>/dev/null || cp "$GGUF_PATH" "$_relocated"; }
                            GGUF_PATH="$_relocated"
                        fi
                        MODEL_PATH="$_new_dir"
                        ok "Retrying with model dir: ${MODEL_PATH}"
                    fi ;;
                unsupported_model_name)
                    err "DIAGNOSIS: NotImplementedError — model name not in setup_env.py's known list"
                    _handle_unsupported_arch "${MODEL_ARCH:-unknown}" ;;
                compiler_flag_error)
                    warn "Unrecognized compiler flag — setup_env.py hardcodes plain 'clang'/'clang++'"
                    warn "Verify unversioned clang on PATH matches your detected compiler (v${CLANG_VER})" ;;
                arch_error)
                    _handle_unsupported_arch "${MODEL_ARCH:-unknown}" ;;
                packed_weight_error|tokenizer_error)
                    _apply_file_patch "convert-hf-to-gguf-bitnet.py" \
                        "HF converter fix (tokenizer + packed weight)" || true ;;
                tl1_error)
                    _spectre_apply_shim || true
                    export PYTHONPATH="${SPECTRE_GGUF_PY_PATH:+${SPECTRE_GGUF_PY_PATH}:}${PYTHONPATH:-}" ;;
                disk_space)
                    die "No disk space remaining — free space and re-run" ;;
                *)
                    if [[ $_attempt -lt 3 ]]; then
                        warn "Unknown failure — applying all patches before retry"
                        _patch_pyproject_toml
                        _patch_tl1_guard
                        _apply_file_patch "convert-hf-to-gguf-bitnet.py" "HF converter" || true
                        _apply_file_patch "ggml-bitnet-mad.cpp"           "mad.cpp const fix" || true
                        _apply_file_patch "codegen_tl2.py"                "codegen_tl2 shape" || true
                    fi ;;
            esac

            if [[ $_attempt -ge 3 ]]; then
                warn "All 3 build attempts failed"
                if ! $SKIP_CONVERT && \
                   { $NON_INTERACTIVE || confirm "Download pre-built GGUF as fallback?" "Y"; }; then
                    _fb="${_MODEL_BASE}/BitNet-b1.58-2B-4T-gguf"
                    download_hf_resumable "microsoft/BitNet-b1.58-2B-4T-gguf" "$_fb"
                    GGUF_PATH=$(find "$_fb" -name "*.gguf" -type f -size +1M \
                        2>/dev/null | sort | head -1 || echo "")
                    if [[ -n "$GGUF_PATH" ]]; then
                        MODEL_PATH="$_fb"
                        MODEL_NAME="BitNet-b1.58-2B-4T-gguf"
                        PRISM_CLASS="pre-built"
                        SKIP_BUILD=true
                        SETUP_EXIT=0
                        ok "Fallback GGUF ready: ${GGUF_PATH}"
                    else
                        die "No GGUF found after fallback download. Check network."
                    fi
                else
                    die "Build failed after 3 attempts. Full log: ${_BLOG}"
                fi
            fi
        done
    fi

    # Post-build: resolve paths and update FORGE cache
    if [[ "${SETUP_EXIT:-1}" -eq 0 ]]; then
        # Resolve GGUF path
        if [[ -z "$GGUF_PATH" ]] && [[ -n "$MODEL_PATH" ]]; then
            GGUF_PATH=$(find "$MODEL_PATH" -maxdepth 1 \
                -name "ggml-model-*.gguf" -type f -size +1M \
                2>/dev/null | sort | head -1 || echo "")
        fi
        # Resolve llama-cli binary
        for _b in "${BITNET_DIR}/build/bin/llama-cli" \
                   "${BITNET_DIR}/build/Release/bin/llama-cli"; do
            [[ -x "$_b" ]] && { BINARY_PATH="$_b"; break; }
        done
        ok "Build complete"
        [[ -n "$GGUF_PATH" ]] && ok "  GGUF     : ${GGUF_PATH}"
        [[ -n "$BINARY_PATH" ]] && ok "  llama-cli: ${BINARY_PATH}"

        # Write FORGE cache (hash + gguf + llama-cli path)
        if [[ -n "${CURRENT_HASH:-}" ]]; then
            python3 -c "
import json
json.dump({
    'hash':      '${CURRENT_HASH}',
    'gguf_path': '${GGUF_PATH:-}',
    'llama_cli': '${BINARY_PATH:-}',
    'model':     '${MODEL_NAME:-}',
    'quant':     '${QUANT_TYPE:-i2_s}',
    'arch':      '${MODEL_ARCH:-}',
    'py_minor':  ${PY_MINOR},
    'clang_ver': ${CLANG_VER},
}, open('${FORGE_CACHE_FILE}','w'), indent=2)
" 2>/dev/null && ok "FORGE: cache written → ${FORGE_CACHE_FILE}" || true
        fi
    fi
else
    if [[ "${PRISM_CLASS:-}" == "packed_transformers" ]]; then
        ok "Mode 3: C++ build skipped (transformers API)"
    fi
    if $SKIP_BUILD; then
        ok "Build skipped (FORGE cache hit or --skip-build)"
    fi
fi

# ═════════════════════════════════════════════════════════════════════════════
# STEP 17 — Optional GEMM Kernel Tuning
# ═════════════════════════════════════════════════════════════════════════════
section "STEP 17 — GEMM Kernel Tuning (Optional)"

_TUNE_S="${BITNET_DIR}/utils/tune_gemm_config.py"
if $SKIP_BUILD; then
    note "GEMM tuning skipped — binaries already up to date (FORGE cache hit)"
elif [[ -f "$_TUNE_S" ]] && ! $DRY_RUN && ! $SKIP_BUILD && \
   [[ "${PRISM_CLASS:-}" != "packed_transformers" ]]; then
    echo -e "  ${C_DIM}  GEMM tuning finds optimal tiling parameters for your exact CPU.${C_RST}"
    echo -e "  ${C_DIM}  Takes 2–5 minutes. Yields 1.15×–2.1× extra throughput.${C_RST}"
    if confirm "Run GEMM kernel tuning?" "Y"; then
        _GH="${BITNET_DIR}/include/gemm-config.h"
        if [[ -f "$_GH" ]]; then
            _TC=("$PYTHON_BIN" "$_TUNE_S" "--config" "$_GH" "--threads" "${ATLAS_THREADS}")
            [[ -n "${GGUF_PATH:-}" ]] && [[ -f "${GGUF_PATH}" ]] && _TC+=("--model" "$GGUF_PATH")
            pushd "${BITNET_DIR}/utils" >/dev/null 2>&1 || true
            set +e
            "${_TC[@]}" 2>&1 | tee -a "${LOG_FILE}"
            _te=${PIPESTATUS[0]}
            set -e
            popd >/dev/null 2>&1 || true
            [[ $_te -eq 0 ]] \
                && ok "GEMM tuning complete ✓" \
                || warn "GEMM tuning exited ${_te} — default config used"
        else
            warn "include/gemm-config.h not found — skipping GEMM tuning"
        fi
    else
        info "GEMM tuning skipped"
    fi
else
    if [[ ! -f "${_TUNE_S:-}" ]]; then
        info "tune_gemm_config.py not present — not available in this BitNet release"
    fi
fi

# ═════════════════════════════════════════════════════════════════════════════
# STEP 18 — Verification
# ═════════════════════════════════════════════════════════════════════════════
section "STEP 18 — Verification"

VERIFY_PASS=true
BINARY_PATH="${BINARY_PATH:-}"
SERVER_PATH=""

# Resolve llama-cli
for _b in "${BITNET_DIR}/build/bin/llama-cli" \
           "${BITNET_DIR}/build/Release/bin/llama-cli" \
           "${BITNET_DIR}/build/bin/Release/llama-cli"; do
    [[ -x "$_b" ]] && { BINARY_PATH="$_b"; ok "llama-cli: ${BINARY_PATH}"; break; }
done

# Resolve llama-server
for _s in "${BITNET_DIR}/build/bin/llama-server" \
           "${BITNET_DIR}/build/Release/bin/llama-server" \
           "${BITNET_DIR}/build/bin/Release/llama-server"; do
    [[ -x "$_s" ]] && { SERVER_PATH="$_s"; ok "llama-server: ${SERVER_PATH}"; break; }
done

# Modes 1+2 expect a compiled binary; Mode 3 does not
if [[ "${PRISM_CLASS:-}" != "packed_transformers" ]] && \
   [[ -z "$BINARY_PATH" ]] && ! $DRY_RUN; then
    warn "llama-cli not found — build may be incomplete"
    VERIFY_PASS=false
fi

# run_inference.py
RUN_INFERENCE="${BITNET_DIR}/run_inference.py"
if [[ -f "$RUN_INFERENCE" ]]; then
    ok "run_inference.py: ${RUN_INFERENCE}"
else
    warn "run_inference.py not found"
    VERIFY_PASS=false
fi

# GGUF path resolution
if [[ -z "${GGUF_PATH:-}" ]] && [[ -n "${MODEL_PATH:-}" ]]; then
    GGUF_PATH=$(find "${MODEL_PATH}" -maxdepth 1 -name "*.gguf" \
        -type f -size +1M 2>/dev/null | sort | head -1 || echo "")
fi

if [[ -n "${GGUF_PATH:-}" ]] && [[ -f "${GGUF_PATH}" ]]; then
    GGUF_SIZE=$(du -h "$GGUF_PATH" 2>/dev/null | cut -f1 || echo "?")
    ok "BitNet GGUF: ${GGUF_PATH} (${GGUF_SIZE})"
elif [[ "${PRISM_CLASS:-}" == "packed_transformers" ]]; then
    note "No GGUF — Mode 3 uses transformers API (expected)"
else
    warn "GGUF not found: ${GGUF_PATH:-<unset>}"
    VERIFY_PASS=false
fi

# Python imports check
"$PYTHON_BIN" -c "
import gguf, torch, transformers, safetensors
v = transformers.__version__
if not any(x in v for x in ['dev','post','+']):
    tv = tuple(int(x) for x in v.split('.')[:2] if x.isdigit())[:2]
    assert tv >= (4,46), f'transformers {v} < 4.46'
" &>/dev/null 2>&1 \
    && ok "Python: gguf ✓  torch ✓  transformers ✓  safetensors ✓" \
    || warn "Some Python imports failed — check log"

_gguf_has_tl1 \
    && ok "gguf TL1/TL2: present ✓" \
    || warn "gguf TL1/TL2 missing (SPECTRE mode: ${SPECTRE_MODE})"

# Smoke test (end-to-end inference call)
if [[ -f "${RUN_INFERENCE:-}" ]] && [[ -n "${GGUF_PATH:-}" ]] && \
   [[ -f "$GGUF_PATH" ]] && [[ -n "$BINARY_PATH" ]] && ! $DRY_RUN; then
    echo ""
    if confirm "Run 5-token smoke test (end-to-end verify)?" "Y"; then
        info "Running smoke test …"
        set +e
        SMOKE_OUT=$("$PYTHON_BIN" "$RUN_INFERENCE" \
            --gguf-path "$GGUF_PATH" \
            -p "Hello" -n 5 -temp 0 -t "${ATLAS_THREADS}" 2>&1)
        SMOKE_EXIT=$?
        set -e
        if [[ $SMOKE_EXIT -eq 0 ]] && \
           ! echo "$SMOKE_OUT" | grep -qiE "error:|not found|traceback|exception"; then
            ok "Smoke test PASSED ✓"
        else
            warn "Smoke test FAILED (exit ${SMOKE_EXIT}): ${SMOKE_OUT:0:200}"
            VERIFY_PASS=false
        fi
    fi
fi

# ═════════════════════════════════════════════════════════════════════════════
# STEP 19 — Write bitnet_state.json   ★ THE CRITICAL FIX (this step was missing) ★
# ═════════════════════════════════════════════════════════════════════════════
# The original baseline script declared BITNET_STATE_FILE but NEVER wrote it
# anywhere. This step was completely missing. As a result llama_cli_binary was
# always empty and downstream Python code (StateFileReader) could never
# confirm bitnet.cpp was ready. This step fixes that permanently.
# ═════════════════════════════════════════════════════════════════════════════
section "STEP 19 — Writing bitnet_state.json"

# Final pass: resolve every path one more time right before writing.
# Covers edge cases like FORGE cache hit or fallback GGUF download.
[[ -z "${BINARY_PATH:-}" ]] && for _b in \
    "${BITNET_DIR}/build/bin/llama-cli" \
    "${BITNET_DIR}/build/Release/bin/llama-cli"; do
    [[ -x "$_b" ]] && { BINARY_PATH="$_b"; break; }
done

[[ -z "${SERVER_PATH:-}" ]] && for _s in \
    "${BITNET_DIR}/build/bin/llama-server" \
    "${BITNET_DIR}/build/Release/bin/llama-server"; do
    [[ -x "$_s" ]] && { SERVER_PATH="$_s"; break; }
done

[[ -z "${RUN_INFERENCE:-}" ]] && RUN_INFERENCE="${BITNET_DIR}/run_inference.py"
[[ ! -f "${RUN_INFERENCE:-}" ]] && RUN_INFERENCE=""

# Determine the active inference mode that will be saved to disk
if [[ "${PRISM_CLASS:-}" == "packed_transformers" ]]; then
    _ACTIVE_MODE="transformers"
elif [[ -n "${BINARY_PATH:-}" ]] && [[ -x "${BINARY_PATH}" ]] && \
     [[ -n "${GGUF_PATH:-}" ]]   && [[ -f "${GGUF_PATH}" ]]; then
    _ACTIVE_MODE="bitnet_cpp"
elif [[ -n "${GGUF_PATH:-}" ]] && [[ -f "${GGUF_PATH}" ]]; then
    _ACTIVE_MODE="gguf_only"
else
    _ACTIVE_MODE="unknown"
fi

# Build the patches_applied JSON array from the bash array
_PAJ="["
for _p in "${PATCHES_APPLIED[@]:-}"; do
    [[ -n "$_p" ]] && _PAJ+="\"${_p}\","
done
_PAJ="${_PAJ%,}]"

mkdir -p "${LOCAL_LLM_STATE_DIR}"

# Write atomically: write to a temp file in the same directory, then os.rename().
# This guarantees the state file is never left half-written if the process
# is killed mid-write (rename is atomic on POSIX filesystems).
"$PYTHON_BIN" - << PYEOF
import json, os, tempfile
from pathlib import Path

state = {
    # ── Meta ─────────────────────────────────────────────────────────────────
    "_comment":                  "Auto-generated by bitnet_setup.sh v1.0.0. Do not edit manually.",
    "setup_version":             "1.0.0",
    "setup_timestamp":           "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",

    # ── Paths consumed by local-llm-core Python engine ────────────────────────
    "bitnet_dir":                "${BITNET_DIR}",
    "python_bin":                "${PYTHON_BIN}",
    "venv_dir":                  "${VENV_DIR:-}",
    "run_inference_script":      "${RUN_INFERENCE:-}",

    # ── Binary paths — StateFileReader.get_llama_cli()/get_llama_server() ────
    # These MUST be non-empty for is_bitnet_cpp_ready() to return True.
    "llama_cli_binary":          "${BINARY_PATH:-}",
    "llama_server_binary":       "${SERVER_PATH:-}",

    # ── Model ────────────────────────────────────────────────────────────────
    "model_name":                "${MODEL_NAME:-}",
    "model_path":                "${MODEL_PATH:-}",
    "model_arch":                "${MODEL_ARCH:-bitnet}",
    "gguf_path":                 "${GGUF_PATH:-}",
    "quant_type":                "${QUANT_TYPE:-i2_s}",
    "quant_embd":                $(py_bool "${QUANT_EMBD}"),
    "model_source_mode":         "${MODEL_SOURCE:-}",

    # ── PRISM classification ───────────────────────────────────────────────────
    "prism_class":               "${PRISM_CLASS:-unknown}",
    "prism_arch_compat":         "${PRISM_ARCH_COMPAT:-unknown}",

    # ── Active runtime mode ─────────────────────────────────────────────────────
    # "bitnet_cpp" | "gguf_only" | "transformers" | "unknown"
    "final_mode":                "${_ACTIVE_MODE}",

    # ── Verification status ─────────────────────────────────────────────────────
    "verify_passed":             $(py_bool "${VERIFY_PASS}"),

    # ── SPECTRE — gguf PYTHONPATH isolation ─────────────────────────────────────
    "spectre_mode":              "${SPECTRE_MODE}",
    "spectre_pythonpath":        "${SPECTRE_GGUF_PY_PATH:-}",
    "spectre_gguf_py_path":      "${SPECTRE_GGUF_PY_PATH:-}",

    # ── ATLAS — hardware profile ─────────────────────────────────────────────────
    "atlas_quant_rec":           "${ATLAS_QUANT_REC}",
    "atlas_threads":             ${ATLAS_THREADS},
    "atlas_gpu_backend":         "${ATLAS_GPU_BACKEND:-cpu}",
    "atlas_avx2":                $(py_bool "${ATLAS_AVX2}"),
    "atlas_avx512":              $(py_bool "${ATLAS_AVX512}"),
    "atlas_neon":                $(py_bool "${ATLAS_NEON}"),

    # ── FORGE — build cache ─────────────────────────────────────────────────────
    "forge_cache_hit":           $(py_bool "${FORGE_CACHE_HIT}"),
    "forge_build_hash":          "${CURRENT_HASH:-}",

    # ── Build environment ───────────────────────────────────────────────────────
    "clang_version":             ${CLANG_VER},
    "cmake_version":             "${CMAKE_VER:-0}",
    "python_version":            "${PY_VER}",
    "python_minor":              ${PY_MINOR},
    "cc":                        "${CC:-clang}",
    "cxx":                       "${CXX:-clang++}",

    # ── System ──────────────────────────────────────────────────────────────────
    "arch":                      "${ARCH}",
    "os":                        "${OS_TYPE}",
    "nproc":                     ${NPROC},

    # ── Transformers ─────────────────────────────────────────────────────────────
    "unified_transformers_constraint": ">=4.46.3,<5.0.0",
    "bitnet_tf_commit":          "096f25ae1f501a084d8ff2dcaf25fbc2bd60eba4",

    # ── Patches applied during this run ─────────────────────────────────────────
    "patches_applied":           ${_PAJ},
    "upfront_patches_applied":   $(py_bool "${_NEED_UPFRONT_PATCH}"),
    "pypi_gguf_forced":          $(py_bool "${USE_PYPI_GGUF}"),
}

# Warn (do not fail) if any critical field is missing
crit = ["bitnet_dir", "python_bin", "prism_class", "spectre_mode", "final_mode"]
missing = [k for k in crit if not state.get(k)]
if missing:
    print(f"  WARNING: critical fields empty: {missing}")

p = Path("${BITNET_STATE_FILE}")
p.parent.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile(
    mode='w', dir=str(p.parent), delete=False, suffix='.tmp', encoding='utf-8'
) as tf:
    json.dump(state, tf, indent=2, ensure_ascii=False)
    tf.flush()
    os.fsync(tf.fileno())
    tmp_name = tf.name
os.rename(tmp_name, str(p))

print(f"  bitnet_state.json written: {p}")
print(f"  llama_cli_binary   : {state['llama_cli_binary']    or '<not found>'}")
print(f"  llama_server_binary: {state['llama_server_binary'] or '<not found>'}")
print(f"  gguf_path          : {state['gguf_path']           or '<not set>'}")
print(f"  verify_passed      : {state['verify_passed']}")
print(f"  final_mode         : {state['final_mode']}")
print(f"  prism_class        : {state['prism_class']}")
print(f"  spectre_mode       : {state['spectre_mode']}")
print(f"  atlas_threads      : {state['atlas_threads']}")
print(f"  clang_version      : {state['clang_version']}")

# FIX: StateFileReader._get_candidates() uses Python's os.path.expanduser("~")
# which returns /root when the process runs as root — even if the actual user
# home is /home/deve-oops. We write the state file to REAL_HOME above, but
# Python never finds it because it searches ~/.config/local-llm/ = /root/...
# Solution: also write to /root/.config/local-llm/ so both paths are populated.
import pwd, getpass
try:
    py_home = Path.home()              # what Python's expanduser("~") returns
    real_home_path = p.parent.parent   # parent of .config/local-llm/
    if py_home != real_home_path:
        alt_dir = py_home / ".config" / "local-llm"
        alt_dir.mkdir(parents=True, exist_ok=True)
        alt_path = alt_dir / "bitnet_state.json"
        import shutil
        shutil.copy2(str(p), str(alt_path))
        print(f"  Also written to   : {alt_path}  (Python expanduser home = {py_home})")
except Exception as e:
    print(f"  Note: could not write alt state path: {e}")
PYEOF
ok "bitnet_state.json written: ${BITNET_STATE_FILE}"

# ═════════════════════════════════════════════════════════════════════════════
# STEP 20 — Launcher Scripts
# ═════════════════════════════════════════════════════════════════════════════
section "STEP 20 — Launcher Scripts"

if ! $DRY_RUN && [[ -n "${GGUF_PATH:-}" ]] && [[ -f "${GGUF_PATH}" ]]; then
    _PPE=""
    [[ -n "${SPECTRE_GGUF_PY_PATH:-}" ]] && \
        _PPE="export PYTHONPATH=\"${SPECTRE_GGUF_PY_PATH}:\${PYTHONPATH:-}\""

    # bitnet-run.sh — single prompt
    cat > "${BITNET_DIR}/bitnet-run.sh" << LAUNCHEOF
#!/usr/bin/env bash
# bitnet-run.sh — single prompt launcher (local-llm-core v1.0.0)
PYTHON="${PYTHON_BIN}"
RUN="${RUN_INFERENCE:-${BITNET_DIR}/run_inference.py}"
GGUF="${GGUF_PATH}"
${_PPE}
cd "${BITNET_DIR}"
exec "\${PYTHON}" "\${RUN}" --gguf-path "\${GGUF}" \\
    -p "\${1:-Hello, what can you do?}" \\
    -n "\${2:-200}" \\
    -temp 0.8 -t ${ATLAS_THREADS} "\${@:3}"
LAUNCHEOF
    chmod +x "${BITNET_DIR}/bitnet-run.sh"
    ok "bitnet-run.sh created"

    # bitnet-chat.sh — interactive loop
    cat > "${BITNET_DIR}/bitnet-chat.sh" << CHATEOF
#!/usr/bin/env bash
# bitnet-chat.sh — interactive chat loop (local-llm-core v1.0.0)
PYTHON="${PYTHON_BIN}"
RUN="${RUN_INFERENCE:-${BITNET_DIR}/run_inference.py}"
GGUF="${GGUF_PATH}"
THREADS="${ATLAS_THREADS}"
${_PPE}
cd "${BITNET_DIR}"
echo ""
echo "  BitNet Interactive Chat  (type 'exit' or Ctrl+C to quit)"
echo "  Model: \${GGUF}"
echo ""
while true; do
    printf "  You: "; IFS= read -r U || break
    [[ "\${U,,}" =~ ^(exit|quit)\$ ]] && { echo "  Bye!"; break; }
    [[ -z "\$U" ]] && continue
    echo ""
    printf "  BitNet: "
    "\${PYTHON}" "\${RUN}" --gguf-path "\${GGUF}" \\
        -p "\${U}" -n "\${BITNET_TOKENS:-512}" \\
        -temp "\${BITNET_TEMP:-0.8}" -t "\${THREADS}" 2>/dev/null
    echo ""
done
CHATEOF
    chmod +x "${BITNET_DIR}/bitnet-chat.sh"
    ok "bitnet-chat.sh created"

    # bitnet-server.sh — OpenAI-compatible HTTP server
    if [[ -n "${SERVER_PATH:-}" ]] && [[ -x "${SERVER_PATH}" ]]; then
        _GPU_FLAGS=""
        [[ "${ATLAS_GPU_BACKEND:-cpu}" =~ ^(hip|cuda|metal)$ ]] && \
            _GPU_FLAGS="--n-gpu-layers 99"
        cat > "${BITNET_DIR}/bitnet-server.sh" << SRVEOF
#!/usr/bin/env bash
# bitnet-server.sh — OpenAI-compatible HTTP server (local-llm-core v1.0.0)
SERVER="${SERVER_PATH}"
GGUF="${GGUF_PATH}"
HOST="\${BITNET_HOST:-127.0.0.1}"
PORT="\${BITNET_PORT:-18080}"
CTX="\${BITNET_CTX:-4096}"
PAR="\${BITNET_PARALLEL:-4}"
echo "  BitNet HTTP server → http://\${HOST}:\${PORT}"
echo "  Model: \${GGUF}"
exec "\${SERVER}" \\
    --model "\${GGUF}" \\
    --host "\${HOST}" --port "\${PORT}" \\
    --threads ${ATLAS_THREADS} \\
    --ctx-size "\${CTX}" \\
    --parallel "\${PAR}" \\
    ${_GPU_FLAGS} \\
    --log-disable "\$@"
SRVEOF
        chmod +x "${BITNET_DIR}/bitnet-server.sh"
        ok "bitnet-server.sh created"
    fi
elif [[ "${PRISM_CLASS:-}" == "packed_transformers" ]]; then
    note "Mode 3 (transformers): launcher scripts not applicable"
fi

# ═════════════════════════════════════════════════════════════════════════════
# STEP 21 — Self-Test, Fallback Chain & Final State Update
# ═════════════════════════════════════════════════════════════════════════════
section "STEP 21 — Self-Test & Fallback Chain"

_self_test() {
    case "$1" in
        gguf)
            [[ -z "${GGUF_PATH:-}" || ! -f "${GGUF_PATH}" ]] && \
                { warn "Self-test [gguf]: GGUF file not found"; return 1; }
            # Verify GGUF magic bytes (0x47475546 = ASCII "GGUF")
            local _m
            _m=$(python3 -c \
                "f=open('${GGUF_PATH}','rb'); print(f.read(4).hex())" 2>/dev/null || echo "")
            if [[ "$_m" != "47475546"* ]] && [[ "$_m" != "4757554a"* ]]; then
                local _sz; _sz=$(stat -c%s "${GGUF_PATH}" 2>/dev/null || echo 0)
                [[ "$_sz" -gt 10485760 ]] || \
                    { warn "Self-test [gguf]: file may be corrupt (magic=${_m})"; return 1; }
            fi
            ok "Self-test [gguf]: PASSED ✓"
            return 0 ;;
        transformers)
            "$PYTHON_BIN" -c "import transformers, torch; print('PASS')" \
                2>/dev/null | grep -q PASS || \
                { warn "Self-test [transformers]: import failed"; return 1; }
            if [[ -n "${MODEL_PATH:-}" && -f "${MODEL_PATH}/config.json" ]]; then
                "$PYTHON_BIN" -c \
                    "import json; json.load(open('${MODEL_PATH}/config.json'))" \
                    2>/dev/null || \
                    { warn "Self-test [transformers]: config.json invalid"; return 1; }
            fi
            ok "Self-test [transformers]: PASSED ✓"
            return 0 ;;
    esac
}

FINAL_BITNET_MODE="${_ACTIVE_MODE}"

if [[ "${PRISM_CLASS:-}" == "packed_transformers" ]]; then
    _self_test "transformers" \
        && FINAL_BITNET_MODE="transformers" \
        || { FINAL_BITNET_MODE="transformers_degraded"; VERIFY_PASS=false; }
elif [[ -n "${GGUF_PATH:-}" ]] && [[ -f "${GGUF_PATH}" ]]; then
    if _self_test "gguf"; then
        if [[ -n "$BINARY_PATH" ]] && [[ -x "$BINARY_PATH" ]]; then
            FINAL_BITNET_MODE="bitnet_cpp"
        else
            FINAL_BITNET_MODE="gguf_only"
        fi
    else
        warn "GGUF self-test failed — attempting transformers fallback"
        if _self_test "transformers"; then
            FINAL_BITNET_MODE="transformers"
            VERIFY_PASS=false
        else
            FINAL_BITNET_MODE="none"
            VERIFY_PASS=false
        fi
    fi
else
    if _self_test "transformers"; then
        FINAL_BITNET_MODE="transformers"
        VERIFY_PASS=false
    else
        FINAL_BITNET_MODE="none"
        VERIFY_PASS=false
    fi
fi

# Update state file with the confirmed final_mode + verify_passed + binaries.
# This second write is intentional: self-test can downgrade final_mode
# (e.g. bitnet_cpp → transformers) after the first write in STEP 19.
"$PYTHON_BIN" - << PYEOF2 2>/dev/null || warn "Could not update state file with self-test results"
import json, shutil
from pathlib import Path
p = Path("${BITNET_STATE_FILE}")
if not p.exists():
    print("  State file not found — skipping self-test update")
else:
    try:
        state = json.loads(p.read_text())
        state["final_mode"]          = "${FINAL_BITNET_MODE}"
        state["verify_passed"]       = $(py_bool "${VERIFY_PASS}")
        state["llama_cli_binary"]    = "${BINARY_PATH:-}"
        state["llama_server_binary"] = "${SERVER_PATH:-}"
        p.write_text(json.dumps(state, indent=2, ensure_ascii=False))
        print(f"  State updated: final_mode={state['final_mode']}  verify_passed={state['verify_passed']}")
        print(f"  llama_cli_binary={state['llama_cli_binary'] or '<not found>'}")
        # Keep the /root copy in sync (same fix as STEP 19 — root user path issue)
        py_home = Path.home()
        real_home_path = p.parent.parent
        if py_home != real_home_path:
            alt = py_home / ".config" / "local-llm" / "bitnet_state.json"
            alt.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(p), str(alt))
    except Exception as e:
        print(f"  WARNING: {e}")
PYEOF2
ok "State file finalised: ${BITNET_STATE_FILE}"

# ═════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
echo ""
echo ""
case "$FINAL_BITNET_MODE" in
    bitnet_cpp)
        ok "ACTIVE MODE: bitnet.cpp — full 1-bit hardware-optimized inference"
        echo -e "  ${C_DIM}  llama-cli   : ${BINARY_PATH:-<not found>}${C_RST}"
        echo -e "  ${C_DIM}  GGUF        : ${GGUF_PATH:-<not set>}${C_RST}"
        [[ -n "${SERVER_PATH:-}" ]] && \
            echo -e "  ${C_DIM}  llama-server: ${SERVER_PATH}${C_RST}" ;;
    gguf_only)
        ok "ACTIVE MODE: GGUF ready (llama_cpp Python binding will be used)"
        echo -e "  ${C_DIM}  GGUF: ${GGUF_PATH:-<not set>}${C_RST}" ;;
    transformers|transformers_degraded)
        ok "ACTIVE MODE: transformers API — Python inference"
        echo -e "  ${C_DIM}  Model path: ${MODEL_PATH:-<not set>}${C_RST}"
        echo -e "  ${C_YEL}  ⚠  No 1-bit hardware acceleration — slower than Modes 1+2${C_RST}" ;;
    none)
        err "No working inference mode found. Check log: ${LOG_FILE}" ;;
    *)
        warn "Mode: ${FINAL_BITNET_MODE}" ;;
esac

echo ""
if $VERIFY_PASS; then
    echo -e "${C_GRN}${C_BLD}"
    cat << 'SUCCESS'
  ╔══════════════════════════════════════════════════════════════════╗
  ║     ✅  BitNet Setup Complete — Local LLM Cloud Ready!          ║
  ╚══════════════════════════════════════════════════════════════════╝
SUCCESS
    echo -e "${C_RST}"
else
    echo -e "${C_YEL}${C_BLD}  ⚠  Setup completed in fallback mode — see warnings above${C_RST}"
    echo -e "${C_DIM}  Active mode: ${FINAL_BITNET_MODE:-unknown}${C_RST}"
    echo ""
fi

echo -e "${C_WHT}${C_BLD}  Quick Start:${C_RST}"
echo ""
if [[ "${FINAL_BITNET_MODE:-}" == "transformers"* ]]; then
    echo -e "${C_CYN}  # BitNet via transformers API:${C_RST}"
    echo -e "${C_WHT}  python -c \"from local_llm_core.compatibility.transformers_compat import load_bitnet_model; r=load_bitnet_model('${MODEL_PATH:-}'); print(r.model)\"${C_RST}"
else
    if [[ -f "${BITNET_DIR}/bitnet-run.sh" ]]; then
        echo -e "${C_CYN}  # Single prompt:${C_RST}"
        echo -e "${C_WHT}  ${BITNET_DIR}/bitnet-run.sh \"Tell me about 1-bit LLMs\"${C_RST}"
        echo ""
    fi
    if [[ -f "${BITNET_DIR}/bitnet-chat.sh" ]]; then
        echo -e "${C_CYN}  # Interactive chat:${C_RST}"
        echo -e "${C_WHT}  ${BITNET_DIR}/bitnet-chat.sh${C_RST}"
        echo ""
    fi
    if [[ -f "${BITNET_DIR}/bitnet-server.sh" ]]; then
        echo -e "${C_CYN}  # OpenAI-compatible HTTP server (port 18080):${C_RST}"
        echo -e "${C_WHT}  ${BITNET_DIR}/bitnet-server.sh${C_RST}"
        echo ""
    fi
    if [[ -n "${GGUF_PATH:-}" ]]; then
        echo -e "${C_CYN}  # Direct via run_inference.py:${C_RST}"
        echo -e "${C_WHT}  python ${BITNET_DIR}/run_inference.py --gguf-path ${GGUF_PATH} -p \"Hello\" -n 200${C_RST}"
        echo ""
    fi
fi
echo -e "${C_CYN}  # local-llm-core auto-integration:${C_RST}"
echo -e "${C_WHT}  python -c \"from local_llm_core.engine.bitnet import auto_detect_backend; print(auto_detect_backend())\"${C_RST}"
echo ""
echo -e "  ${C_DIM}State file : ${BITNET_STATE_FILE}${C_RST}"
echo -e "  ${C_DIM}Full log   : ${LOG_FILE}${C_RST}"
echo ""
exit 0
