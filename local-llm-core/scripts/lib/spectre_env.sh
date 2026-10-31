#!/usr/bin/env bash
# =============================================================================
#  scripts/lib/spectre_env.sh — SPECTRE Environment Manager  v1.0.0
# =============================================================================
#  SPECTRE — Subprocess Environment Poisoning Correction &
#            Transparent Resolution Engine
#
#  Manages the gguf-py version isolation that prevents the TL1/TL2
#  AttributeError in convert-hf-to-gguf-bitnet.py.
#
#  Root cause (confirmed from your logs):
#    gguf.GGMLQuantizationType.TL1 — AttributeError
#    PyPI gguf (any version) lacks TL1/TL2.  Only 3rdparty/llama.cpp/gguf-py
#    has them.  sys.path.insert in setup_env.py does NOT propagate to child
#    processes.  PYTHONPATH in os.environ DOES propagate.
#
#  Functions:
#    spectre_activate_build   Set PYTHONPATH + install local gguf-py editable.
#                             Call before ANY setup_env.py invocation.
#    spectre_activate_runtime Restore PYTHONPATH for runtime use.
#                             Call after build is complete.
#    spectre_status           Print current SPECTRE state.
#    spectre_verify_tl1       Test if TL1/TL2 are accessible.
#    spectre_write_env FILE   Append SPECTRE vars to an env file.
#
#  Variables exported after spectre_activate_build():
#    SPECTRE_GGUF_PY_PATH     path to local gguf-py (or empty)
#    SPECTRE_MODE             ok | editable | shim | shim_pypi | none
#    PYTHONPATH               updated with local gguf-py at front
#    SPECTRE_ORIGINAL_PP      original PYTHONPATH (for restore)
# =============================================================================

SPECTRE_VERSION="1.0.0"
SPECTRE_GGUF_PY_PATH=""
SPECTRE_MODE="none"
SPECTRE_ORIGINAL_PP="${PYTHONPATH:-}"
SPECTRE_BUILD_ACTIVE=false
SPECTRE_TL1_VERIFIED=false

# ── Colours ───────────────────────────────────────────────────────────────────
if [[ -t 1 ]] && [[ "${TERM:-dumb}" != "dumb" ]]; then
    _SP_GRN='\033[0;32m' _SP_YEL='\033[0;33m' _SP_CYN='\033[0;36m'
    _SP_RED='\033[0;31m' _SP_DIM='\033[2m'    _SP_RST='\033[0m'
    _SP_BLD='\033[1m'
else
    _SP_GRN='' _SP_YEL='' _SP_RED='' _SP_CYN='' _SP_DIM='' _SP_RST='' _SP_BLD=''
fi

_sp_ok()   { echo -e "  ${_SP_GRN}✓${_SP_RST}  SPECTRE: $*"; }
_sp_info() { echo -e "  ${_SP_CYN}▶${_SP_RST}  SPECTRE: $*"; }
_sp_warn() { echo -e "  ${_SP_YEL}⚠${_SP_RST}  SPECTRE: $*"; }
_sp_err()  { echo -e "  ${_SP_RED}✗${_SP_RST}  SPECTRE: $*"; }
_sp_cmd()  { command -v "$1" &>/dev/null; }

# =============================================================================
# _spectre_find_python — locate best Python 3.8+
# =============================================================================
_spectre_find_python() {
    local py=""
    # Check in order: PYTHON_BIN env first, then newest stable versions, then system python3
    for p in "${PYTHON_BIN:-}" python3.14 python3.13 python3.12 python3.11 python3 python3.10 python3.9 python3.8; do
        [[ -z "$p" ]] && continue
        _sp_cmd "$p" || continue
        local maj min
        maj=$("$p" -c 'import sys; print(sys.version_info.major)' 2>/dev/null) || continue
        min=$("$p" -c 'import sys; print(sys.version_info.minor)' 2>/dev/null) || continue
        [[ "$maj" -eq 3 && "$min" -ge 8 ]] && { py="$p"; break; }
    done
    echo "$py"
}

# =============================================================================
# _spectre_locate_gguf_py — find local gguf-py directory
#   Priority: BITNET_DIR/3rdparty/llama.cpp/gguf-py
#             > common search paths
#             > bitnet_state.json
# =============================================================================
_spectre_locate_gguf_py() {
    local bitnet_dir="${BITNET_DIR:-${1:-}}"

    # Priority 1: explicit BITNET_DIR
    if [[ -n "$bitnet_dir" ]]; then
        local candidate="${bitnet_dir}/3rdparty/llama.cpp/gguf-py"
        if [[ -f "${candidate}/gguf/__init__.py" ]]; then
            echo "$candidate"; return
        fi
    fi

    # Priority 2: SPECTRE_GGUF_PY_PATH if already set
    if [[ -n "${SPECTRE_GGUF_PY_PATH:-}" ]] && \
       [[ -f "${SPECTRE_GGUF_PY_PATH}/gguf/__init__.py" ]]; then
        echo "$SPECTRE_GGUF_PY_PATH"; return
    fi

    # Priority 3: scan common BitNet install locations
    local search_roots=(
        "${HOME}/AMB-OPT/BitNet"
        "${HOME}/BitNet"
        "${HOME}/bitnet"
        "/opt/BitNet"
    )
    for _d in /home/*/; do
        [[ -d "${_d}BitNet" ]] && search_roots+=("${_d}BitNet")
    done 2>/dev/null || true
    for root in "${search_roots[@]}"; do
        local candidate="${root}/3rdparty/llama.cpp/gguf-py"
        [[ -f "${candidate}/gguf/__init__.py" ]] && { echo "$candidate"; return; }
    done

    # Priority 4: bitnet_state.json
    local state_file="${HOME}/.config/local-llm/bitnet_state.json"
    if [[ -f "$state_file" ]]; then
        local sp
        sp=$(grep -oP '"spectre_gguf_py_path"\s*:\s*"\K[^"]+' "$state_file" 2>/dev/null || echo "")
        if [[ -n "$sp" ]] && [[ -f "${sp}/gguf/__init__.py" ]]; then
            echo "$sp"; return
        fi
        local bdir
        bdir=$(grep -oP '"bitnet_dir"\s*:\s*"\K[^"]+' "$state_file" 2>/dev/null || echo "")
        if [[ -n "$bdir" ]]; then
            local candidate="${bdir}/3rdparty/llama.cpp/gguf-py"
            [[ -f "${candidate}/gguf/__init__.py" ]] && { echo "$candidate"; return; }
        fi
    fi

    # Priority 5: scan existing PYTHONPATH entries that look like gguf-py
    for entry in ${PYTHONPATH//:/ }; do
        [[ -f "${entry}/gguf/__init__.py" ]] && { echo "$entry"; return; }
    done

    echo ""
}

# =============================================================================
# spectre_verify_tl1 — test if gguf has TL1/TL2
# Returns 0 if available, 1 if missing
# =============================================================================
spectre_verify_tl1() {
    local py; py="$(_spectre_find_python)"
    [[ -z "$py" ]] && return 1
    "$py" -c "
import gguf
try:
    _ = gguf.GGMLQuantizationType.TL1
    _ = gguf.GGMLQuantizationType.TL2
    exit(0)
except AttributeError:
    exit(1)
" &>/dev/null 2>&1 && SPECTRE_TL1_VERIFIED=true || SPECTRE_TL1_VERIFIED=false
    $SPECTRE_TL1_VERIFIED
}

# =============================================================================
# spectre_activate_build — LAYER 1+2: PYTHONPATH + editable install
#
# LAYER 1: Export PYTHONPATH with local gguf-py at front.
#   This propagates to ALL child processes (setup_env.py, the converter,
#   llama-quantize wrappers) via os.environ inheritance.
#
# LAYER 2: pip install --editable local gguf-py.
#   Makes `import gguf` in any new Python process resolve to local gguf-py
#   regardless of PYTHONPATH (installed into site-packages as editable).
#
# Sets: SPECTRE_GGUF_PY_PATH, SPECTRE_MODE, PYTHONPATH
# =============================================================================
spectre_activate_build() {
    local bitnet_dir="${1:-${BITNET_DIR:-}}"
    SPECTRE_ORIGINAL_PP="${PYTHONPATH:-}"
    SPECTRE_BUILD_ACTIVE=true

    _sp_info "Activating build-phase environment …"

    # Check if TL1 already available (e.g., local gguf-py already installed)
    if spectre_verify_tl1; then
        local py; py="$(_spectre_find_python)"
        local gver=""
        [[ -n "$py" ]] && gver=$("$py" -c "import gguf; print(getattr(gguf,'__version__','?'))" 2>/dev/null || echo "?")
        _sp_ok "TL1/TL2 already available (gguf ${gver}) — no action needed"
        SPECTRE_MODE="ok"
        return 0
    fi

    # Locate local gguf-py
    local gguf_py_dir
    gguf_py_dir="$(_spectre_locate_gguf_py "$bitnet_dir")"

    if [[ -z "$gguf_py_dir" ]]; then
        _sp_warn "Local gguf-py not found — will apply shim (TL1/TL2 via monkey-patch)"
        SPECTRE_MODE="shim_pypi"
        _spectre_apply_shim
        return 0
    fi

    SPECTRE_GGUF_PY_PATH="$gguf_py_dir"
    export SPECTRE_GGUF_PY_PATH

    # ── LAYER 1: PYTHONPATH export ────────────────────────────────────────────
    if [[ -n "${PYTHONPATH:-}" ]]; then
        # Validate existing entries (remove stale/missing paths)
        local clean_pp=""
        while IFS= read -r entry; do
            entry="${entry// /}"
            [[ -z "$entry" ]] && continue
            [[ -d "$entry" ]] && clean_pp+="${entry}:"
        done <<< "${PYTHONPATH//:/$'\n'}"
        export PYTHONPATH="${gguf_py_dir}:${clean_pp%:}"
    else
        export PYTHONPATH="${gguf_py_dir}"
    fi
    _sp_info "Layer 1: PYTHONPATH = ${PYTHONPATH:0:80}…"

    # ── LAYER 2: editable install ─────────────────────────────────────────────
    local py; py="$(_spectre_find_python)"
    if [[ -n "$py" ]]; then
        local has_setup has_pyproject
        has_setup=false;    [[ -f "${gguf_py_dir}/setup.py" ]]      && has_setup=true
        has_pyproject=false; [[ -f "${gguf_py_dir}/pyproject.toml" ]] && has_pyproject=true

        if $has_setup || $has_pyproject; then
            _sp_info "Layer 2: installing local gguf-py editable …"
            if "$py" -m pip install --quiet --no-deps -e "${gguf_py_dir}" \
                    >> "${LOG_FILE:-/dev/null}" 2>&1; then
                if spectre_verify_tl1; then
                    _sp_ok "Editable install succeeded — TL1/TL2 confirmed ✓"
                    SPECTRE_MODE="editable"
                    return 0
                else
                    _sp_warn "Editable install done but TL1 check still fails → applying shim"
                    _spectre_apply_shim
                fi
            else
                _sp_warn "Editable install failed (non-fatal) → applying shim"
                _spectre_apply_shim
            fi
        else
            _sp_warn "No setup.py or pyproject.toml in ${gguf_py_dir} → PYTHONPATH only"
            SPECTRE_MODE="pythonpath"
        fi
    else
        _sp_warn "No Python found for editable install → PYTHONPATH only"
        SPECTRE_MODE="pythonpath"
    fi

    # Final TL1 check
    if spectre_verify_tl1; then
        _sp_ok "TL1/TL2 available after SPECTRE activation (mode=${SPECTRE_MODE})"
    else
        _sp_warn "TL1/TL2 still not detected — build may fail at converter step"
        _sp_warn "PYTHONPATH=${PYTHONPATH:-<empty>}"
        _sp_warn "Run: python3 scripts/lib/gguf_tl_shim.py"
    fi
}

# =============================================================================
# _spectre_apply_shim — LAYER 3: invoke gguf_tl_shim.py
# =============================================================================
_spectre_apply_shim() {
    local shim_script
    # Search for gguf_tl_shim.py relative to this script
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    shim_script="${script_dir}/gguf_tl_shim.py"

    local py; py="$(_spectre_find_python)"
    [[ -z "$py" ]] && { _sp_warn "No Python for shim — skipping Layer 3"; return 1; }

    if [[ -f "$shim_script" ]]; then
        _sp_info "Layer 3: applying gguf TL1/TL2 shim via ${shim_script} …"
        if "$py" "$shim_script" --install >> "${LOG_FILE:-/dev/null}" 2>&1; then
            _sp_ok "Shim installed"
            SPECTRE_MODE="shim"
        else
            _sp_warn "Shim install failed — build will likely fail at TL1 step"
            SPECTRE_MODE="failed"
        fi
    else
        _sp_warn "gguf_tl_shim.py not found at ${shim_script}"
        _sp_warn "Attempting inline shim …"
        # Inline minimal shim as last resort
        "$py" -c "
import sys
def _ext(cls, name, val):
    if hasattr(cls, name): return
    if val in cls._value2member_map_:
        e=cls._value2member_map_[val]; cls._member_map_[name]=e; type.__setattr__(cls,name,e); return
    m=int.__new__(cls,val); m._name_=name; m._value_=val
    cls._member_map_[name]=m; cls._value2member_map_[val]=m; type.__setattr__(cls,name,m)
try:
    import gguf as _g
    if not hasattr(_g.GGMLQuantizationType, 'TL1'):
        _ext(_g.GGMLQuantizationType, 'TL1', 38)
        _ext(_g.GGMLQuantizationType, 'TL2', 39)
        print('Inline shim: TL1/TL2 patched')
    else:
        print('TL1/TL2 already present')
except Exception as e:
    print(f'Inline shim failed: {e}')
    sys.exit(1)
" 2>&1 | tee -a "${LOG_FILE:-/dev/null}" && SPECTRE_MODE="shim_inline" || SPECTRE_MODE="failed"
    fi
}

# =============================================================================
# spectre_activate_runtime — restore env after build is complete
# =============================================================================
spectre_activate_runtime() {
    if [[ -n "${SPECTRE_ORIGINAL_PP}" ]]; then
        export PYTHONPATH="${SPECTRE_ORIGINAL_PP}"
        _sp_info "Runtime mode: PYTHONPATH restored to original"
    else
        # Keep SPECTRE gguf-py in PYTHONPATH for runtime too (harmless)
        _sp_info "Runtime mode: PYTHONPATH kept (was empty before SPECTRE)"
    fi
    SPECTRE_BUILD_ACTIVE=false

    # Install PyPI gguf for runtime use (llama-cpp-python / local-llm-core)
    local py; py="$(_spectre_find_python)"
    if [[ -n "$py" ]]; then
        local gver
        gver=$("$py" -c "import gguf; print(getattr(gguf,'__version__','?'))" 2>/dev/null || echo "")
        if [[ -z "$gver" ]]; then
            _sp_info "Installing PyPI gguf for runtime …"
            "$py" -m pip install --quiet "gguf>=0.1.0" >> "${LOG_FILE:-/dev/null}" 2>&1 || true
        else
            _sp_ok "Runtime gguf: ${gver} (TL1=${SPECTRE_TL1_VERIFIED})"
        fi
    fi
}

# =============================================================================
# spectre_write_env — append SPECTRE state to an env file
# =============================================================================
spectre_write_env() {
    local env_file="${1:-}"
    [[ -z "$env_file" ]] && return
    [[ -f "$env_file" ]] || return

    # Don't duplicate if already written
    grep -q "SPECTRE_GGUF_PY_PATH" "$env_file" 2>/dev/null && return

    {
        echo ""
        echo "# SPECTRE written by spectre_env.sh v${SPECTRE_VERSION} on $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
        echo "export SPECTRE_GGUF_PY_PATH=\"${SPECTRE_GGUF_PY_PATH}\""
        echo "export SPECTRE_MODE=\"${SPECTRE_MODE}\""
        echo "export SPECTRE_VERSION=\"${SPECTRE_VERSION}\""
        if [[ -n "${SPECTRE_GGUF_PY_PATH:-}" ]]; then
            echo "export PYTHONPATH=\"${SPECTRE_GGUF_PY_PATH}:\${PYTHONPATH:-}\""
        fi
    } >> "$env_file"

    _sp_ok "SPECTRE vars appended to ${env_file}"
}

# =============================================================================
# spectre_status — print current state
# =============================================================================
spectre_status() {
    echo ""
    echo -e "${_SP_BLD}${_SP_CYN}  ── SPECTRE Status v${SPECTRE_VERSION} ──${_SP_RST}"
    echo -e "  Mode        : ${_SP_BLD}${SPECTRE_MODE}${_SP_RST}"
    echo -e "  gguf-py dir : ${SPECTRE_GGUF_PY_PATH:-<not found>}"
    echo -e "  TL1/TL2     : ${SPECTRE_TL1_VERIFIED}"
    echo -e "  Build active: ${SPECTRE_BUILD_ACTIVE}"
    echo -e "  PYTHONPATH  : ${PYTHONPATH:-<empty>}"
    local py; py="$(_spectre_find_python)"
    if [[ -n "$py" ]]; then
        local gver
        gver=$("$py" -c "import gguf; print(getattr(gguf,'__version__','?'))" 2>/dev/null || echo "?")
        echo -e "  gguf version: ${gver}"
    fi
    echo ""
}

# =============================================================================
# Standalone execution
# =============================================================================
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    _cmd="status"
    _BITNET=""
    for arg in "$@"; do
        case "$arg" in
            activate-build)   _cmd="activate_build" ;;
            activate-runtime) _cmd="activate_runtime" ;;
            status)           _cmd="status" ;;
            verify)           _cmd="verify" ;;
            --bitnet-dir=*)   _BITNET="${arg#*=}" ;;
            --help|-h)
                echo "Usage: spectre_env.sh [COMMAND] [--bitnet-dir=PATH]"
                echo "  activate-build    Set PYTHONPATH + install local gguf-py"
                echo "  activate-runtime  Restore env post-build"
                echo "  status            Print current SPECTRE state"
                echo "  verify            Test if TL1/TL2 are accessible"
                exit 0 ;;
        esac
    done

    case "$_cmd" in
        activate_build)
            [[ -n "$_BITNET" ]] && export BITNET_DIR="$_BITNET"
            spectre_activate_build "$_BITNET"
            spectre_status ;;
        activate_runtime)
            spectre_activate_runtime
            spectre_status ;;
        verify)
            if spectre_verify_tl1; then
                echo -e "  ${_SP_GRN}✓ TL1/TL2 available${_SP_RST}"
                exit 0
            else
                echo -e "  ${_SP_RED}✗ TL1/TL2 MISSING${_SP_RST}"
                exit 1
            fi ;;
        *)
            spectre_status ;;
    esac
fi
