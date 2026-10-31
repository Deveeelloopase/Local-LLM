#!/usr/bin/env bash
# =============================================================================
#  scripts/lib/forge_cache.sh — FORGE Build Cache  v1.0.0
# =============================================================================
#  FORGE — Fingerprint-driven Optimised Rebuild Guard Engine
#
#  Prevents unnecessary 5-30 minute BitNet.cpp rebuilds by caching a
#  content-hash of all build inputs. A cache HIT means source+config is
#  unchanged AND the output GGUF exists — skip the build entirely.
#
#  Usage (sourced by bitnet_setup.sh):
#      source lib/forge_cache.sh
#      forge_init  "$BITNET_DIR"
#      forge_check "$MODEL_PATH" "$QUANT_TYPE" "$QUANT_EMBD" && skip_build=true
#      # ... build ...
#      forge_record "$GGUF_PATH"
#
#  Standalone test:
#      bash forge_cache.sh --test --bitnet-dir=/path/to/BitNet
#
#  Cache file: ${BITNET_DIR}/.forge_cache.json
#  Hash covers: setup_env.py, CMakeLists.txt, ggml-bitnet-mad.cpp,
#               model path, quant type, quant_embd flag, clang version.
# =============================================================================

FORGE_VERSION="1.0.0"
FORGE_CACHE_FILE=""
FORGE_BITNET_DIR=""
FORGE_CURRENT_HASH=""
FORGE_CACHE_HIT=false

# ── Colours (safe, no-op if no terminal) ─────────────────────────────────────
if [[ -t 1 ]] && [[ "${TERM:-dumb}" != "dumb" ]]; then
    _FC_GRN='\033[0;32m' _FC_YEL='\033[0;33m' _FC_CYN='\033[0;36m'
    _FC_DIM='\033[2m'    _FC_RST='\033[0m'    _FC_BLD='\033[1m'
else
    _FC_GRN='' _FC_YEL='' _FC_CYN='' _FC_DIM='' _FC_RST='' _FC_BLD=''
fi

_fc_ok()   { echo -e "  ${_FC_GRN}✓${_FC_RST}  FORGE: $*"; }
_fc_info() { echo -e "  ${_FC_CYN}▶${_FC_RST}  FORGE: $*"; }
_fc_warn() { echo -e "  ${_FC_YEL}⚠${_FC_RST}  FORGE: $*"; }
_fc_cmd()  { command -v "$1" &>/dev/null; }

# =============================================================================
# forge_hash_file — hash a single file, returns hex string
# =============================================================================
forge_hash_file() {
    local path="$1"
    [[ -f "$path" ]] || { echo "missing"; return; }
    if _fc_cmd sha256sum; then
        sha256sum "$path" 2>/dev/null | cut -d' ' -f1
    elif _fc_cmd sha256;  then
        sha256 -q "$path" 2>/dev/null
    elif _fc_cmd md5sum;  then
        md5sum "$path" 2>/dev/null | cut -d' ' -f1
    elif _fc_cmd md5;     then
        md5 -q "$path" 2>/dev/null
    elif _fc_cmd openssl; then
        openssl dgst -sha256 "$path" 2>/dev/null | awk '{print $2}'
    else
        # Last resort: stat mtime+size (less reliable but works everywhere)
        stat -c '%Y-%s' "$path" 2>/dev/null || stat -f '%m-%z' "$path" 2>/dev/null || echo "nostat"
    fi
}

# =============================================================================
# forge_hash_string — hash an arbitrary string
# =============================================================================
forge_hash_string() {
    local str="$1"
    if _fc_cmd sha256sum; then
        printf '%s' "$str" | sha256sum 2>/dev/null | cut -d' ' -f1
    elif _fc_cmd md5sum; then
        printf '%s' "$str" | md5sum 2>/dev/null | cut -d' ' -f1
    elif _fc_cmd openssl; then
        printf '%s' "$str" | openssl dgst -sha256 2>/dev/null | awk '{print $2}'
    else
        # Pure bash fallback: simple djb2-style hash (collision-prone but usable)
        local h=5381 c
        while IFS= read -r -n1 c; do
            [[ -z "$c" ]] && continue
            h=$(( (h * 33 + $(printf '%d' "'$c")) & 0xFFFFFFFF ))
        done <<< "$str"
        printf '%08x' "$h"
    fi
}

# =============================================================================
# forge_compute_hash — compute the combined build-input hash
#
# Inputs hashed:
#   1. setup_env.py content
#   2. CMakeLists.txt content
#   3. ggml-bitnet-mad.cpp content (key BitNet kernel file)
#   4. run_inference.py content
#   5. Model path string (so different models get different cache entries)
#   6. Quant type (i2_s | tl1 | tl2)
#   7. Quant-embd flag (true|false)
#   8. Clang version (different compilers = different output)
#   9. OS + arch (cross-platform safety)
# =============================================================================
forge_compute_hash() {
    local model_path="${1:-}"
    local quant_type="${2:-i2_s}"
    local quant_embd="${3:-false}"
    local bitnet_dir="${FORGE_BITNET_DIR}"

    local combined=""

    # Source file hashes
    for rel_path in \
        "setup_env.py" \
        "CMakeLists.txt" \
        "src/ggml-bitnet-mad.cpp" \
        "run_inference.py" \
        "utils/convert-hf-to-gguf-bitnet.py"; do
        local abs="${bitnet_dir}/${rel_path}"
        local h; h="$(forge_hash_file "$abs")"
        combined+="${rel_path}=${h}|"
    done

    # Config inputs
    combined+="model=${model_path}|"
    combined+="quant=${quant_type}|"
    combined+="embd=${quant_embd}|"

    # Toolchain
    local clang_ver=0
    for cb in clang-21 clang-20 clang-19 clang-18 clang; do
        _fc_cmd "$cb" && {
            clang_ver=$("$cb" --version 2>/dev/null | grep -oP 'version \K[0-9]+' | head -1 || echo 0)
            break
        }
    done
    combined+="clang=${clang_ver}|"

    # Platform
    combined+="os=$(uname -s)|arch=$(uname -m)|"

    FORGE_CURRENT_HASH="$(forge_hash_string "$combined")"
    echo "$FORGE_CURRENT_HASH"
}

# =============================================================================
# forge_init — set up FORGE for a given bitnet directory
# =============================================================================
forge_init() {
    local bitnet_dir="${1:-}"
    if [[ -z "$bitnet_dir" ]]; then
        _fc_warn "forge_init: BITNET_DIR not provided"
        return 1
    fi
    FORGE_BITNET_DIR="$bitnet_dir"
    FORGE_CACHE_FILE="${bitnet_dir}/.forge_cache.json"
    FORGE_CACHE_HIT=false
    FORGE_CURRENT_HASH=""
}

# =============================================================================
# forge_check — check if build inputs are unchanged AND output GGUF exists
# Returns: 0 = cache HIT (skip build), 1 = cache MISS (must build)
#
# Usage:
#   if forge_check "$MODEL_PATH" "$QUANT_TYPE" "$QUANT_EMBD"; then
#       echo "Build not needed"
#   else
#       echo "Must rebuild"
#   fi
# =============================================================================
forge_check() {
    local model_path="${1:-}"
    local quant_type="${2:-i2_s}"
    local quant_embd="${3:-false}"

    FORGE_CACHE_HIT=false

    if [[ -z "$FORGE_BITNET_DIR" ]]; then
        _fc_warn "forge_init not called — cache check skipped"
        return 1
    fi

    # Compute current hash of all inputs
    forge_compute_hash "$model_path" "$quant_type" "$quant_embd" > /dev/null

    if [[ ! -f "$FORGE_CACHE_FILE" ]]; then
        _fc_info "No cache file found → MISS (first build)"
        return 1
    fi

    # Read cached values
    local cached_hash cached_gguf cached_model cached_quant cached_ts
    cached_hash=$(  python3 -c "import json; d=json.load(open('${FORGE_CACHE_FILE}')); print(d.get('hash',''))"      2>/dev/null || echo "")
    cached_gguf=$(  python3 -c "import json; d=json.load(open('${FORGE_CACHE_FILE}')); print(d.get('gguf_path',''))"  2>/dev/null || echo "")
    cached_model=$( python3 -c "import json; d=json.load(open('${FORGE_CACHE_FILE}')); print(d.get('model',''))"      2>/dev/null || echo "")
    cached_quant=$( python3 -c "import json; d=json.load(open('${FORGE_CACHE_FILE}')); print(d.get('quant',''))"      2>/dev/null || echo "")
    cached_ts=$(    python3 -c "import json; d=json.load(open('${FORGE_CACHE_FILE}')); print(d.get('timestamp',''))"  2>/dev/null || echo "")

    # Fallback: grep-based JSON reading (no Python)
    if [[ -z "$cached_hash" ]]; then
        cached_hash=$(grep -oP '"hash"\s*:\s*"\K[^"]+' "$FORGE_CACHE_FILE" 2>/dev/null || echo "")
        cached_gguf=$(grep -oP '"gguf_path"\s*:\s*"\K[^"]+' "$FORGE_CACHE_FILE" 2>/dev/null || echo "")
    fi

    # Hash match check
    if [[ "$cached_hash" != "$FORGE_CURRENT_HASH" ]]; then
        _fc_info "Hash mismatch → MISS (source or config changed)"
        _fc_info "  Cached: ${cached_hash:0:16}…"
        _fc_info "  Current: ${FORGE_CURRENT_HASH:0:16}…"
        return 1
    fi

    # GGUF existence check
    if [[ -z "$cached_gguf" ]] || [[ ! -f "$cached_gguf" ]]; then
        _fc_info "GGUF not found at cached path → MISS"
        _fc_info "  Expected: ${cached_gguf:-<empty>}"
        return 1
    fi

    # All checks passed → HIT
    FORGE_CACHE_HIT=true
    _fc_ok "Cache HIT — hash=${FORGE_CURRENT_HASH:0:16}…  built ${cached_ts:-unknown}"
    _fc_ok "GGUF: ${cached_gguf}"
    _fc_ok "Use --force-rebuild to ignore cache"
    return 0
}

# =============================================================================
# forge_record — write a successful build to cache
#
# Usage:
#   forge_record "$GGUF_PATH" "$MODEL_PATH" "$QUANT_TYPE"
# =============================================================================
forge_record() {
    local gguf_path="${1:-}"
    local model_path="${2:-}"
    local quant_type="${3:-i2_s}"
    local quant_embd="${4:-false}"

    if [[ -z "$FORGE_BITNET_DIR" ]]; then
        _fc_warn "forge_init not called — cannot record cache"
        return 1
    fi

    # Recompute hash if not already done
    if [[ -z "$FORGE_CURRENT_HASH" ]]; then
        forge_compute_hash "$model_path" "$quant_type" "$quant_embd" > /dev/null
    fi

    local ts; ts="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    local gguf_size=""
    [[ -f "$gguf_path" ]] && gguf_size="$(du -h "$gguf_path" 2>/dev/null | cut -f1)"

    cat > "$FORGE_CACHE_FILE" << JSON
{
  "_comment": "FORGE build cache v${FORGE_VERSION}. Do not edit manually.",
  "version": "${FORGE_VERSION}",
  "hash": "${FORGE_CURRENT_HASH}",
  "timestamp": "${ts}",
  "gguf_path": "${gguf_path}",
  "gguf_size": "${gguf_size}",
  "model": "${model_path}",
  "quant": "${quant_type}",
  "quant_embd": ${quant_embd},
  "bitnet_dir": "${FORGE_BITNET_DIR}",
  "os": "$(uname -s)",
  "arch": "$(uname -m)"
}
JSON

    _fc_ok "Build hash recorded → ${FORGE_CACHE_FILE}"
    _fc_ok "  Hash: ${FORGE_CURRENT_HASH:0:32}…"
}

# =============================================================================
# forge_invalidate — force-delete the cache (used with --force-rebuild)
# =============================================================================
forge_invalidate() {
    if [[ -n "${FORGE_CACHE_FILE:-}" ]] && [[ -f "$FORGE_CACHE_FILE" ]]; then
        rm -f "$FORGE_CACHE_FILE"
        _fc_info "Cache invalidated (deleted ${FORGE_CACHE_FILE})"
    fi
    FORGE_CACHE_HIT=false
    FORGE_CURRENT_HASH=""
}

# =============================================================================
# forge_status — print current cache status (for diagnostics)
# =============================================================================
forge_status() {
    echo ""
    echo -e "${_FC_BLD}${_FC_CYN}  FORGE Cache Status  v${FORGE_VERSION}${_FC_RST}"
    echo -e "  ${_FC_DIM}Bitnet dir : ${FORGE_BITNET_DIR:-<not set>}${_FC_RST}"
    echo -e "  ${_FC_DIM}Cache file : ${FORGE_CACHE_FILE:-<not set>}${_FC_RST}"
    if [[ -n "${FORGE_CACHE_FILE:-}" ]] && [[ -f "$FORGE_CACHE_FILE" ]]; then
        echo -e "  ${_FC_GRN}Cache exists${_FC_RST}"
        cat "$FORGE_CACHE_FILE" | grep -E '"hash"|"timestamp"|"gguf_path"|"quant"' \
            | sed 's/^/    /' || true
    else
        echo -e "  ${_FC_YEL}No cache file${_FC_RST}"
    fi
    echo -e "  ${_FC_DIM}Current hash : ${FORGE_CURRENT_HASH:0:32}${FORGE_CURRENT_HASH:+…}${_FC_RST}"
    $FORGE_CACHE_HIT && echo -e "  ${_FC_GRN}Status: HIT${_FC_RST}" || \
        echo -e "  ${_FC_YEL}Status: MISS (or not checked)${_FC_RST}"
    echo ""
}

# =============================================================================
# Standalone test / usage
# =============================================================================
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    _RUN_TEST=false
    _BITNET_DIR=""
    _MODEL_PATH=""
    _QUANT="i2_s"
    _QUANT_EMBD="false"

    for arg in "$@"; do
        case "$arg" in
            --test)          _RUN_TEST=true ;;
            --bitnet-dir=*)  _BITNET_DIR="${arg#*=}" ;;
            --model-dir=*)   _MODEL_PATH="${arg#*=}" ;;
            --quant=*)       _QUANT="${arg#*=}" ;;
            --quant-embd)    _QUANT_EMBD="true" ;;
            --invalidate)
                FORGE_CACHE_FILE="${_BITNET_DIR}/.forge_cache.json"
                forge_invalidate
                exit 0 ;;
            --status)
                forge_init "${_BITNET_DIR:-$(pwd)}"
                forge_status
                exit 0 ;;
            --help|-h)
                cat << 'HELP'
Usage: forge_cache.sh [OPTIONS]
  --test             Run self-test
  --bitnet-dir=PATH  BitNet.cpp directory
  --model-dir=PATH   Model directory
  --quant=TYPE       Quant type (i2_s|tl1|tl2)
  --quant-embd       Enable embedding quant
  --invalidate       Delete cache file
  --status           Show cache status
HELP
                exit 0 ;;
        esac
    done

    if $_RUN_TEST; then
        echo -e "${_FC_BLD}FORGE self-test${_FC_RST}"
        [[ -z "$_BITNET_DIR" ]] && { echo "Error: --bitnet-dir required for test"; exit 1; }
        forge_init "$_BITNET_DIR"
        echo "Computing hash..."
        forge_compute_hash "${_MODEL_PATH:-/tmp/test_model}" "$_QUANT" "$_QUANT_EMBD" > /dev/null
        echo "Hash: ${FORGE_CURRENT_HASH}"
        forge_status
        echo "Checking cache..."
        if forge_check "${_MODEL_PATH:-/tmp/test_model}" "$_QUANT" "$_QUANT_EMBD"; then
            echo "Result: HIT"
        else
            echo "Result: MISS"
            echo "Recording fake entry..."
            forge_record "/tmp/fake.gguf" "${_MODEL_PATH:-/tmp/test_model}" "$_QUANT" "$_QUANT_EMBD"
            forge_status
        fi
        exit 0
    fi

    echo "Usage: source forge_cache.sh  (then call forge_init, forge_check, forge_record)"
    echo "       bash forge_cache.sh --help"
fi
