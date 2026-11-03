#!/usr/bin/env bash
# =============================================================================
#  scripts/lib/guardian_check.sh — GUARDIAN Pre-flight Checker  v1.0.0
# =============================================================================
#  GUARDIAN — Gated Universal Assertion & Repair for Dependencies In a
#             Neutral environment
#
#  Pure bash, zero Python required. Checks ALL preconditions before the
#  BitNet build. Produces a colour-coded report + JSON to /tmp/guardian_report.json
#
#  Usage (standalone):
#      bash scripts/lib/guardian_check.sh [--bitnet-dir=PATH] [--model-dir=PATH]
#  Usage (sourced by bitnet_setup.sh):
#      source lib/guardian_check.sh
#      guardian_run_all --bitnet-dir="$BITNET_DIR" --model-dir="$MODEL_PATH"
#      $GUARDIAN_PASS || die "Pre-flight failed"
#
#  Variables set after guardian_run_all():
#      GUARDIAN_PASS        true|false
#      GUARDIAN_TOTAL       total checks run
#      GUARDIAN_PASSED      checks that passed
#      GUARDIAN_FAILED      checks that failed (fatal or warning)
#      GUARDIAN_FATAL       checks that produced fatal failures
#      GUARDIAN_REPORT_JSON path to JSON report
# =============================================================================

# NOTE: set -euo pipefail intentionally NOT set at top level.
# This file is designed to be sourced — applying strict mode here would poison
# the caller. Strict mode is applied only in the standalone execution block below.

GUARDIAN_VERSION="1.0.0"
GUARDIAN_PASS=true
GUARDIAN_TOTAL=0
GUARDIAN_PASSED=0
GUARDIAN_FAILED=0
GUARDIAN_FATAL=0
GUARDIAN_REPORT_JSON="/tmp/guardian_report.json"

# ── Colours ───────────────────────────────────────────────────────────────────
if [[ -t 1 ]] && [[ "${TERM:-dumb}" != "dumb" ]]; then
    G_GRN='\033[0;32m' G_YEL='\033[0;33m' G_RED='\033[0;31m'
    G_CYN='\033[0;36m' G_DIM='\033[2m'    G_RST='\033[0m'
    G_BLD='\033[1m'
else
    G_GRN='' G_YEL='' G_RED='' G_CYN='' G_DIM='' G_RST='' G_BLD=''
fi

# ── Internal check registry ───────────────────────────────────────────────────
declare -a _G_RESULTS=()   # "PASS|WARN|FAIL|SKIP:label:detail"

_g_record() {
    local status="$1" label="$2" detail="$3"
    _G_RESULTS+=("${status}:${label}:${detail}")
    GUARDIAN_TOTAL=$(( GUARDIAN_TOTAL + 1 ))
    case "$status" in
        PASS) GUARDIAN_PASSED=$(( GUARDIAN_PASSED + 1 ))
              echo -e "  ${G_GRN}✓${G_RST}  ${G_BLD}${label}${G_RST} ${G_DIM}${detail}${G_RST}" ;;
        WARN) GUARDIAN_FAILED=$(( GUARDIAN_FAILED + 1 ))
              echo -e "  ${G_YEL}⚠${G_RST}  ${G_BLD}${label}${G_RST} ${G_YEL}${detail}${G_RST}" ;;
        FAIL) GUARDIAN_FAILED=$(( GUARDIAN_FAILED + 1 ))
              GUARDIAN_FATAL=$(( GUARDIAN_FATAL + 1 ))
              GUARDIAN_PASS=false
              echo -e "  ${G_RED}✗${G_RST}  ${G_BLD}${label}${G_RST} ${G_RED}${detail}${G_RST}" ;;
        SKIP) echo -e "  ${G_DIM}–  ${label} ${detail}${G_RST}" ;;
    esac
}

_g_pass() { _g_record "PASS" "$1" "${2:-}"; }
_g_warn() { _g_record "WARN" "$1" "${2:-}"; }
_g_fail() { _g_record "FAIL" "$1" "${2:-}"; }
_g_skip() { _g_record "SKIP" "$1" "${2:-}"; }

_g_cmd()  { command -v "$1" &>/dev/null; }

_g_ver_gte() {
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

# =============================================================================
# Individual check functions
# =============================================================================

guardian_check_os() {
    local os; os="$(uname -s)"
    local arch; arch="$(uname -m)"
    case "$os" in
        Linux|Darwin) _g_pass "OS" "${os} ${arch}" ;;
        *) _g_fail "OS" "Unsupported: ${os}. Only Linux and macOS supported." ;;
    esac
}

guardian_check_clang() {
    local best_ver=0
    for cb in clang-21 clang-20 clang-19 clang-18 clang; do
        _g_cmd "$cb" || continue
        local v; v=$("$cb" --version 2>/dev/null | grep -oP 'version \K[0-9]+' | head -1 || echo 0)
        [[ "$v" -gt "$best_ver" ]] && best_ver="$v"
    done
    if [[ "$best_ver" -ge 18 ]]; then
        _g_pass "clang >= 18" "Found clang v${best_ver}"
    elif [[ "$best_ver" -gt 0 ]]; then
        _g_warn "clang < 18" "v${best_ver} found — BitNet kernels may miscompile; upgrade to >= 18"
    else
        _g_fail "clang" "Not found. Install: apt install clang-18 or brew install llvm"
    fi
}

guardian_check_cmake() {
    if ! _g_cmd cmake; then
        _g_fail "cmake" "Not found. Install: apt install cmake or brew install cmake"
        return
    fi
    local ver; ver=$(cmake --version 2>/dev/null | head -1 | awk '{print $3}' || echo "0")
    if _g_ver_gte "$ver" "3.14"; then
        _g_pass "cmake >= 3.14" "Found ${ver}"
    else
        _g_warn "cmake < 3.14" "${ver} found — update recommended"
    fi
}

guardian_check_git() {
    _g_cmd git && _g_pass "git" "$(git --version 2>/dev/null)" || \
        _g_fail "git" "Not found. Install: apt install git or brew install git"
}

guardian_check_make_ninja() {
    if _g_cmd ninja; then
        _g_pass "build tool" "ninja $(ninja --version 2>/dev/null)"
    elif _g_cmd make; then
        _g_pass "build tool" "make $(make --version 2>/dev/null | head -1)"
    else
        _g_warn "build tool" "Neither ninja nor make found — cmake may fail"
    fi
}

guardian_check_python() {
    local found_py=""
    for py in python3.14 python3.13 python3.12 python3.11 python3 python3.10 python3.9 python3.8; do
        _g_cmd "$py" || continue
        local maj min
        maj=$("$py" -c 'import sys; print(sys.version_info.major)' 2>/dev/null) || continue
        min=$("$py" -c 'import sys; print(sys.version_info.minor)' 2>/dev/null) || continue
        if [[ "$maj" -eq 3 && "$min" -ge 8 ]]; then
            found_py="$py"
            local ver; ver=$("$py" --version 2>/dev/null)
            _g_pass "Python >= 3.8" "${ver} at $(command -v $py)"
            break
        fi
    done
    [[ -z "$found_py" ]] && _g_fail "Python" "No Python 3.8+ found in PATH"
}

guardian_check_gguf_tl1() {
    local py=""
    for p in python3 python3.14 python3.13 python3.12 python3.11 python3.10; do _g_cmd "$p" && py="$p" && break; done
    [[ -z "$py" ]] && { _g_skip "gguf TL1/TL2" "(no Python)"; return; }

    # If build is being skipped (pre-built GGUF or transformers path), TL1/TL2 not needed
    if [[ "${SKIP_BUILD:-false}" == "true" ]] || \
       [[ "${MINIMAL_BUILD:-false}" == "true" ]] || \
       [[ "${PRISM_CLASS:-}" == "pre-built" ]] || \
       [[ "${PRISM_CLASS:-}" == "packed_transformers" ]]; then
        _g_skip "gguf TL1/TL2" "(not needed — build skipped, pre-built GGUF or transformers path)"
        return
    fi

    if "$py" -c "
import gguf
gguf.GGMLQuantizationType.TL1
gguf.GGMLQuantizationType.TL2
" &>/dev/null 2>&1; then
        local gver; gver=$("$py" -c "import gguf; print(getattr(gguf,'__version__','?'))" 2>/dev/null)
        _g_pass "gguf TL1/TL2" "gguf ${gver} has TL1/TL2 — BitNet build WILL succeed"
    else
        _g_fail "gguf TL1/TL2" "MISSING — convert-hf-to-gguf-bitnet.py WILL fail. Run SPECTRE fix."
    fi
}

guardian_check_gguf_py_submodule() {
    local bitnet_dir="${_G_BITNET_DIR:-}"
    [[ -z "$bitnet_dir" ]] && { _g_skip "gguf-py submodule" "(BITNET_DIR not set)"; return; }
    local gguf_py="${bitnet_dir}/3rdparty/llama.cpp/gguf-py/gguf/__init__.py"
    if [[ -f "$gguf_py" ]]; then
        _g_pass "gguf-py submodule" "Populated at 3rdparty/llama.cpp/gguf-py"
    else
        _g_warn "gguf-py submodule" "Empty — run: git submodule update --init --recursive"
    fi
}

guardian_check_setup_env() {
    local bitnet_dir="${_G_BITNET_DIR:-}"
    [[ -z "$bitnet_dir" ]] && { _g_skip "setup_env.py" "(BITNET_DIR not set)"; return; }
    [[ -f "${bitnet_dir}/setup_env.py" ]] && \
        _g_pass "setup_env.py" "Present at ${bitnet_dir}/setup_env.py" || \
        _g_fail "setup_env.py" "MISSING — clone may be incomplete"
}

guardian_check_model_path() {
    local model_dir="${_G_MODEL_DIR:-}"
    [[ -z "$model_dir" ]] && { _g_skip "model path" "(MODEL_DIR not set)"; return; }
    if [[ -d "$model_dir" ]]; then
        _g_pass "model directory" "$model_dir"
    elif [[ -f "$model_dir" && "$model_dir" == *.gguf ]]; then
        _g_pass "model (GGUF file)" "$model_dir"
    else
        _g_fail "model path" "Not found: ${model_dir}"
    fi
}

guardian_check_safetensors() {
    local model_dir="${_G_MODEL_DIR:-}"
    [[ -z "$model_dir" ]] || [[ ! -d "$model_dir" ]] && return
    local count; count=$(find "$model_dir" -maxdepth 1 -name "*.safetensors" 2>/dev/null | wc -l)
    if [[ "$count" -gt 0 ]]; then
        _g_pass "safetensors" "${count} file(s) in ${model_dir}"
    else
        # Check if pre-built GGUF exists
        local gguf_count; gguf_count=$(find "$model_dir" -name "*.gguf" 2>/dev/null | wc -l)
        if [[ "$gguf_count" -gt 0 ]]; then
            _g_pass "pre-built GGUF" "${gguf_count} .gguf file(s) found"
        else
            _g_warn "safetensors" "No .safetensors in ${model_dir} — converter will fail"
        fi
    fi
}

guardian_check_disk_space() {
    local model_dir="${_G_MODEL_DIR:-${HOME}}"
    [[ ! -d "$model_dir" ]] && model_dir="$HOME"
    local free_kb; free_kb=$(df -k "$model_dir" 2>/dev/null | awk 'NR==2{print $4}' || echo 0)
    local free_gb=$(( free_kb / 1024 / 1024 ))
    if [[ "$free_gb" -ge 10 ]]; then
        _g_pass "disk space" "${free_gb} GB free (>= 10 GB)"
    elif [[ "$free_gb" -ge 5 ]]; then
        _g_warn "disk space" "${free_gb} GB free — marginal (10 GB recommended)"
    else
        _g_warn "disk space" "Only ${free_gb} GB free — model conversion may fail"
    fi
}

guardian_check_ram() {
    local ram_gb=0
    if [[ "$(uname -s)" == "Linux" ]] && [[ -f /proc/meminfo ]]; then
        local ram_kb; ram_kb=$(awk '/^MemTotal/{print $2}' /proc/meminfo || echo 0)
        ram_gb=$(( ram_kb / 1024 / 1024 ))
    elif [[ "$(uname -s)" == "Darwin" ]]; then
        local ram_bytes; ram_bytes=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
        ram_gb=$(( ram_bytes / 1024 / 1024 / 1024 ))
    fi
    if [[ "$ram_gb" -ge 8 ]]; then
        _g_pass "RAM" "${ram_gb} GB (>= 8 GB)"
    elif [[ "$ram_gb" -ge 4 ]]; then
        _g_warn "RAM" "${ram_gb} GB — may be slow; 8 GB recommended"
    else
        _g_warn "RAM" "${ram_gb} GB — very low; conversion and inference may fail"
    fi
}

guardian_check_write_permission() {
    local bitnet_dir="${_G_BITNET_DIR:-}"
    [[ -z "$bitnet_dir" ]] && return
    if [[ -w "$bitnet_dir" ]]; then
        _g_pass "write permission" "${bitnet_dir}"
    else
        _g_fail "write permission" "Cannot write to ${bitnet_dir}"
    fi
}

guardian_check_huggingface_cli() {
    if _g_cmd huggingface-cli; then
        _g_pass "huggingface-cli" "$(huggingface-cli --version 2>/dev/null || echo 'present')"
    else
        _g_warn "huggingface-cli" "Not found — model download will need manual install"
    fi
}

guardian_check_network() {
    local url="https://huggingface.co"
    if _g_cmd curl; then
        if curl -fsSL --max-time 5 --head "$url" &>/dev/null; then
            _g_pass "network" "HuggingFace reachable"
        else
            _g_warn "network" "Cannot reach ${url} — model downloads may fail"
        fi
    elif _g_cmd wget; then
        if wget -q --spider --timeout=5 "$url" &>/dev/null; then
            _g_pass "network" "HuggingFace reachable (wget)"
        else
            _g_warn "network" "Cannot reach ${url}"
        fi
    else
        _g_warn "network" "Neither curl nor wget found — cannot verify connectivity"
    fi
}

guardian_check_secure_boot() {
    [[ "$(uname -s)" != "Linux" ]] && return
    if [[ -d /sys/firmware/efi/efivars ]] && _g_cmd mokutil; then
        if mokutil --sb-state 2>/dev/null | grep -qi "enabled"; then
            _g_warn "Secure Boot" "ENABLED — unsigned kernel modules will be REJECTED"
        else
            _g_pass "Secure Boot" "Not enabled"
        fi
    else
        _g_skip "Secure Boot" "(EFI/mokutil not available)"
    fi
}

guardian_check_transformers_version() {
    local py=""
    for p in python3 python3.14 python3.13 python3.12 python3.11 python3.10; do _g_cmd "$p" && py="$p" && break; done
    [[ -z "$py" ]] && return
    local ver
    ver=$("$py" -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "")
    if [[ -z "$ver" ]]; then
        _g_warn "transformers" "Not installed — GGUF inference path will be limited"
        return
    fi
    local major minor
    major=$(echo "$ver" | cut -d. -f1)
    minor=$(echo "$ver" | cut -d. -f2)
    # Accept dev/post/+ builds unconditionally — these are pinned BitNet forks
    if echo "$ver" | grep -qE 'dev|post|\+'; then
        _g_pass "transformers" "v${ver} (dev/pinned build — BitNet PATH B active)"
    elif [[ "$major" -eq 4 && "$minor" -ge 46 && "$major" -lt 5 ]]; then
        _g_pass "transformers" "v${ver} satisfies >=4.46,<5.0.0"
    elif [[ "$major" -ge 5 ]]; then
        _g_warn "transformers" "v${ver} >= 5.0 — may conflict with llama-cpp-python; downgrade recommended"
    else
        _g_warn "transformers" "v${ver} < 4.46 — upgrade recommended"
    fi
}

# =============================================================================
# guardian_run_all — run all checks, build JSON report
# =============================================================================
_G_BITNET_DIR=""
_G_MODEL_DIR=""

guardian_run_all() {
    # Parse args
    for arg in "$@"; do
        case "$arg" in
            --bitnet-dir=*) _G_BITNET_DIR="${arg#*=}" ;;
            --model-dir=*)  _G_MODEL_DIR="${arg#*=}"  ;;
        esac
    done

    GUARDIAN_PASS=true
    GUARDIAN_TOTAL=0
    GUARDIAN_PASSED=0
    GUARDIAN_FAILED=0
    GUARDIAN_FATAL=0
    _G_RESULTS=()

    echo ""
    echo -e "${G_BLD}${G_CYN}┌─ GUARDIAN Pre-flight Checker v${GUARDIAN_VERSION} ────────────────────────────${G_RST}"
    echo ""

    guardian_check_os
    guardian_check_clang
    guardian_check_cmake
    guardian_check_git
    guardian_check_make_ninja
    guardian_check_python
    guardian_check_gguf_tl1
    guardian_check_gguf_py_submodule
    guardian_check_setup_env
    guardian_check_model_path
    guardian_check_safetensors
    guardian_check_disk_space
    guardian_check_ram
    guardian_check_write_permission
    guardian_check_huggingface_cli
    guardian_check_network
    guardian_check_secure_boot
    guardian_check_transformers_version

    # Summary
    echo ""
    if $GUARDIAN_PASS; then
        echo -e "${G_BLD}${G_GRN}  GUARDIAN: ${GUARDIAN_PASSED}/${GUARDIAN_TOTAL} checks passed — build should succeed ✓${G_RST}"
    else
        echo -e "${G_BLD}${G_RED}  GUARDIAN: ${GUARDIAN_FATAL} fatal failure(s), ${GUARDIAN_FAILED} total issue(s)/${GUARDIAN_TOTAL} checks${G_RST}"
        echo -e "${G_YEL}  Fix fatal failures before building.${G_RST}"
    fi
    echo ""

    # Emit JSON report
    _guardian_write_json
}

_guardian_write_json() {
    local ts; ts=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
    local pass_str; $GUARDIAN_PASS && pass_str="true" || pass_str="false"

    {
        printf '{\n'
        printf '  "version": "%s",\n' "$GUARDIAN_VERSION"
        printf '  "timestamp": "%s",\n' "$ts"
        printf '  "pass": %s,\n' "$pass_str"
        printf '  "total": %d,\n' "$GUARDIAN_TOTAL"
        printf '  "passed": %d,\n' "$GUARDIAN_PASSED"
        printf '  "failed": %d,\n' "$GUARDIAN_FAILED"
        printf '  "fatal": %d,\n' "$GUARDIAN_FATAL"
        printf '  "bitnet_dir": "%s",\n' "${_G_BITNET_DIR}"
        printf '  "model_dir": "%s",\n' "${_G_MODEL_DIR}"
        printf '  "checks": [\n'
        local first=true
        for entry in "${_G_RESULTS[@]:-}"; do
            IFS=':' read -r status label detail <<< "$entry"
            $first || printf ',\n'
            printf '    {"status": "%s", "label": "%s", "detail": "%s"}' \
                "$status" "$label" "${detail//\"/\'}"
            first=false
        done
        printf '\n  ]\n'
        printf '}\n'
    } > "$GUARDIAN_REPORT_JSON"
}

# =============================================================================
# Standalone execution
# =============================================================================
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -euo pipefail
    ARGS=()
    for arg in "$@"; do
        case "$arg" in
            --help|-h)
                echo "Usage: guardian_check.sh [--bitnet-dir=PATH] [--model-dir=PATH]"
                echo "  Runs all pre-flight checks and emits ${GUARDIAN_REPORT_JSON}"
                exit 0 ;;
            *) ARGS+=("$arg") ;;
        esac
    done
    guardian_run_all "${ARGS[@]:-}"
    echo "  JSON report: ${GUARDIAN_REPORT_JSON}"
    $GUARDIAN_PASS && exit 0 || exit 1
fi
