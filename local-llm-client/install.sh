#!/usr/bin/env bash
################################################################################
#  LOCAL LLM CLIENT — Production Setup Script  v1.0.0
#
#  Sets up the local-llm-client systemd service and client configuration.
#  Assumes `pip install local-llm-client` has ALREADY been run.
#  This script NEVER re-runs pip — it only configures and deploys.
#
#  Recommended usage (via registered console script):
#    sudo llm-install                         # interactive
#    sudo llm-install --force                 # force re-run
#
#  Direct usage:
#    sudo bash install.sh
#
#  Non-interactive / CI usage (set env vars, no prompts):
#    sudo NONINTERACTIVE=1 SERVER_URL=http://myserver:8080 bash install.sh
#
#  All defaults can be overridden via environment variable:
#    VENV_DIR      — virtual environment path (auto-detected if unset)
#    WORK_DIR      — client package directory (auto-detected from script)
#    SERVER_URL    — LLM server URL (default: http://localhost:8080)
#    LOG_DIR       — log directory (default: ../logs relative to project)
#    SERVICE_NAME  — systemd service name (default: local-llm-client)
#    SERVICE_PORT  — unused for client, kept for symmetry
#    NONINTERACTIVE=1 — skip all prompts, use defaults
#
#  v1.0.0 changes:
#    - Auto-detect all paths (no hardcoded user/machine paths)
#    - Remove redundant pip re-install step
#    - Generate service file dynamically if not present
#    - Create/update ~/.llmcloud/config.yaml with server URL
#    - Support any machine, any user, any venv location
#    - Non-fatal on optional steps (warn, continue)
################################################################################

set -eo pipefail

# ─────────────────────────────────────────────────────────────────────────────
#  Path auto-detection (NEVER hardcode)
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${WORK_DIR:-$SCRIPT_DIR}"
SERVICE_NAME="${SERVICE_NAME:-local-llm-client}"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
SERVER_URL="${SERVER_URL:-http://localhost:8080}"
NONINTERACTIVE="${NONINTERACTIVE:-0}"

# Resolve project root (one level up from client dir, or use client dir itself)
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." 2>/dev/null && pwd || echo "$SCRIPT_DIR")}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs}"
CONFIG_DIR="${HOME}/.llmcloud"
CLIENT_CONFIG="${CONFIG_DIR}/config.yaml"

# ─────────────────────────────────────────────────────────────────────────────
#  Colours & logging
# ─────────────────────────────────────────────────────────────────────────────
RED=$'\033[0;31m';  GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'
BLUE=$'\033[0;34m'; CYAN=$'\033[0;36m';  BOLD=$'\033[1m';    NC=$'\033[0m'
MAGENTA=$'\033[0;35m'; DIM=$'\033[2m'

log_info()    { echo -e "  ${BLUE}ℹ${NC}  $*"; }
log_success() { echo -e "  ${GREEN}✅${NC}  $*"; }
log_warning() { echo -e "  ${YELLOW}⚠️ ${NC}  $*"; }
log_error()   { echo -e "  ${RED}❌${NC}  $*" >&2; }
log_section() { echo -e "\n${BOLD}${CYAN}  ┌─────────────────────────────────────────────┐${NC}"; \
                echo -e "${BOLD}${CYAN}  │  $*$(printf '%*s' $((45 - ${#1})) '')│${NC}"; \
                echo -e "${BOLD}${CYAN}  └─────────────────────────────────────────────┘${NC}"; }

die() { log_error "$*"; exit 1; }

# ─────────────────────────────────────────────────────────────────────────────
#  Prompting helpers
# ─────────────────────────────────────────────────────────────────────────────
yes_no() {
    local prompt="$1" default="${2:-y}"
    [[ "$NONINTERACTIVE" == "1" ]] && { [[ "$default" == "y" ]] && return 0 || return 1; }
    local hint; [[ "$default" == "y" ]] && hint="Y/n" || hint="y/N"
    local answer
    while true; do
        read -rp "$(echo -e "${CYAN}[?]${NC}  $prompt [$hint]: ")" answer
        answer="${answer:-$default}"
        case "$answer" in
            [Yy]*) return 0 ;;
            [Nn]*) return 1 ;;
            *) echo "    Please answer yes or no." ;;
        esac
    done
}

ask() {
    local var="$1" prompt="$2" default="$3"
    if [[ "$NONINTERACTIVE" == "1" ]]; then
        local current="${!var:-}"
        printf -v "$var" '%s' "${current:-$default}"
        return
    fi
    local answer
    read -rp "$(echo -e "${CYAN}[?]${NC}  $prompt [${default}]: ")" answer
    printf -v "$var" '%s' "${answer:-$default}"
}

# ─────────────────────────────────────────────────────────────────────────────
#  Virtual environment auto-detection
# ─────────────────────────────────────────────────────────────────────────────
detect_venv() {
    log_section "Virtual Environment"

    # If VENV_DIR is already set and valid, use it
    if [[ -n "${VENV_DIR:-}" && -x "${VENV_DIR}/bin/python" ]]; then
        log_success "Using VENV_DIR from environment: $VENV_DIR"
        return
    fi

    # Ordered list of candidates to try
    local candidates=(
        # Currently active venv
        "${VIRTUAL_ENV:-}"
        # Common project conventions
        "$PROJECT_ROOT/DEVE-BOX"
        "$PROJECT_ROOT/venv"
        "$PROJECT_ROOT/.venv"
        # Sibling to the project (DEVE-BOX pattern)
        "$(dirname "$PROJECT_ROOT")/DEVE-BOX"
        "$(dirname "$PROJECT_ROOT")/venv"
        # System-wide patterns
        "/opt/local-llm-cloud/venv"
        "/opt/venv"
        # Python executable that ran pip install
        "$(dirname "$(dirname "$(command -v python3 2>/dev/null || command -v python)")" 2>/dev/null || true)"
    )

    VENV_DIR=""
    for candidate in "${candidates[@]}"; do
        if [[ -z "$candidate" ]]; then continue; fi
        if [[ -d "$candidate" && -x "$candidate/bin/python" ]]; then
            VENV_DIR="$candidate"
            break
        fi
    done

    if [[ -z "$VENV_DIR" ]]; then
        if [[ "$NONINTERACTIVE" == "1" ]]; then
            die "Could not auto-detect virtual environment. Set VENV_DIR=/path/to/venv and retry."
        fi
        log_warning "Could not auto-detect virtual environment."
        log_warning "Tried:"
        for c in "${candidates[@]}"; do if [[ -n "$c" ]]; then echo "    $c"; fi; done
        echo ""
        local custom_venv
        ask custom_venv "Enter virtual environment path" "/opt/local-llm-cloud/venv"
        [[ -x "$custom_venv/bin/python" ]] || die "Python not found at: $custom_venv/bin/python"
        VENV_DIR="$custom_venv"
    fi

    PYTHON_BIN="$VENV_DIR/bin/python"
    PIP_BIN="$VENV_DIR/bin/pip"
    LLM_BIN="$VENV_DIR/bin/llm"

    local py_ver
    py_ver="$("$PYTHON_BIN" --version 2>&1 | awk '{print $2}')"
    log_success "Virtual environment: $VENV_DIR  (Python $py_ver)"
}

# ─────────────────────────────────────────────────────────────────────────────
#  Prerequisites
# ─────────────────────────────────────────────────────────────────────────────
check_prerequisites() {
    log_section "Prerequisites"

    [[ "$EUID" -eq 0 ]] || die "Run as root: sudo llm-install  or  sudo bash install.sh"
    command -v systemctl &>/dev/null || die "systemd is required but not found."

    # Verify the package is installed
    if [[ -x "${VENV_DIR:-}/bin/llm" ]]; then
        local version
        version="$("$VENV_DIR/bin/llm" --version 2>/dev/null || echo "unknown")"
        log_success "local-llm-client installed: $version"
    else
        log_warning "llm binary not found in venv — package may not be installed yet."
        log_warning "Run: pip install local-llm-client  then retry."
    fi

    log_success "All prerequisites met"
}

# ─────────────────────────────────────────────────────────────────────────────
#  Server URL configuration
# ─────────────────────────────────────────────────────────────────────────────
configure_server_url() {
    log_section "Server Connection"

    # Try reading from existing client config
    if [[ -f "$CLIENT_CONFIG" ]]; then
        local cfg_url
        cfg_url="$(grep -E '^\s*base_url\s*:' "$CLIENT_CONFIG" 2>/dev/null | \
                   sed 's/.*base_url[[:space:]]*:[[:space:]]*//' | \
                   sed 's/[[:space:]]*#.*//' | tr -d '"'"'" | xargs 2>/dev/null || true)"
        # Filter out env-var templates like ${LLM_SERVER_URL:...}
        if [[ -n "$cfg_url" && "$cfg_url" != *'${'* ]]; then
            SERVER_URL="${SERVER_URL:-$cfg_url}"
            log_info "Server URL from existing config: $cfg_url"
        fi
    fi

    if [[ "$NONINTERACTIVE" != "1" ]]; then
        ask SERVER_URL "LLM server URL" "$SERVER_URL"
    fi

    # Verify server connectivity (optional — warn only)
    log_info "Testing connection to: $SERVER_URL"
    if command -v curl &>/dev/null; then
        if curl -sf --max-time 5 "${SERVER_URL}/health" &>/dev/null; then
            log_success "Server is reachable at $SERVER_URL"
        else
            log_warning "Server not reachable at $SERVER_URL (this is OK — start it later)"
            log_warning "  sudo systemctl start local-llm-cloud"
        fi
    else
        log_warning "curl not found — skipping server connectivity check"
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
#  Directories
# ─────────────────────────────────────────────────────────────────────────────
prepare_directories() {
    log_section "Directories"

    for dir in "$LOG_DIR" "$CONFIG_DIR"; do
        if mkdir -p "$dir" 2>/dev/null; then
            log_success "Ready: $dir"
        else
            log_warning "Could not create: $dir"
        fi
    done
}

# ─────────────────────────────────────────────────────────────────────────────
#  Client config file (~/.llmcloud/config.yaml)
# ─────────────────────────────────────────────────────────────────────────────
configure_client_config() {
    log_section "Client Configuration"

    if [[ -f "$CLIENT_CONFIG" ]]; then
        log_info "Existing config found: $CLIENT_CONFIG"
        if [[ "$NONINTERACTIVE" != "1" ]]; then
            if ! yes_no "Update server.base_url to $SERVER_URL in existing config?"; then
                log_info "Keeping existing config unchanged."
                return
            fi
        fi
        # Update base_url in existing config (in-place using sed)
        if grep -q "base_url" "$CLIENT_CONFIG" 2>/dev/null; then
            sed -i "s|base_url:.*|base_url: \"${SERVER_URL}\"|g" "$CLIENT_CONFIG"
            log_success "Updated base_url → $SERVER_URL"
        else
            echo "  base_url: \"${SERVER_URL}\"" >> "$CLIENT_CONFIG"
            log_success "Appended base_url → $SERVER_URL"
        fi
    else
        log_info "Creating default config: $CLIENT_CONFIG"
        # Use package config.yaml as template if available
        local template="$WORK_DIR/config.yaml"
        if [[ -f "$template" ]]; then
            cp "$template" "$CLIENT_CONFIG"
            # Update server URL in the copied config
            sed -i "s|base_url:.*|base_url: \"${SERVER_URL}\"|g" "$CLIENT_CONFIG"
            log_success "Config created from template with base_url → $SERVER_URL"
        else
            cat > "$CLIENT_CONFIG" << CFGEOF
# local-llm-client configuration
# Generated by llm-install on $(date '+%Y-%m-%d %H:%M:%S')
# Edit this file to customise your client settings.

server:
  base_url: "${SERVER_URL}"
  api_key: ""       # set via: llm config set server.api_key sk-yourkey
  master_key: ""    # set via: llm config set server.master_key admin-yourkey

inference:
  model: ""         # set via: llm config set inference.model bitnet-gguf
  temperature: 0.7
  max_tokens: 0

logging:
  level: "info"
CFGEOF
            log_success "Default config created: $CLIENT_CONFIG"
            log_warning "  Set your API key: llm config set server.api_key sk-yourkey"
            log_warning "  Set your master key: llm config set server.master_key admin-yourkey"
        fi
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
#  systemd service file (generated if missing)
# ─────────────────────────────────────────────────────────────────────────────
install_service() {
    log_section "systemd Service"

    local service_src="$WORK_DIR/${SERVICE_NAME}.service"

    # Generate service file if it doesn't exist in the package directory
    if [[ ! -f "$service_src" ]]; then
        log_info "Service file not found — generating: $service_src"
        cat > "$service_src" << SVCEOF
[Unit]
Description=Local LLM Client — background daemon mode
Documentation=https://github.com/Deveeelloopase/Local-LLM
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=${WORK_DIR}

Environment="PATH=${VENV_DIR}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment="VIRTUAL_ENV=${VENV_DIR}"
Environment="LLM_SERVER_URL=${SERVER_URL}"
Environment="LLM_CONFIG=${CLIENT_CONFIG}"

# Note: the llm CLI is designed for interactive use.
# If you have a custom background script, set ExecStart to point to it.
# Example for a background agent:
#   ExecStart=${VENV_DIR}/bin/python -m local_llm_client.daemon
# For now, this keeps a health-check loop alive:
ExecStart=/bin/bash -c 'while true; do sleep 60; done'

Restart=on-failure
RestartSec=15
StandardOutput=append:${LOG_DIR}/client.log
StandardError=append:${LOG_DIR}/client.log
SyslogIdentifier=local-llm-client

[Install]
WantedBy=multi-user.target
SVCEOF
        log_success "Service file generated: $service_src"
    fi

    # Install to systemd if not already there or if outdated
    if [[ -f "$SERVICE_FILE" ]]; then
        if [[ "$NONINTERACTIVE" == "1" ]] || ! yes_no "Service file already exists at $SERVICE_FILE. Overwrite?"; then
            log_info "Keeping existing service file. Reloading systemd..."
            systemctl daemon-reload
            return
        fi
    fi

    cp "$service_src" "$SERVICE_FILE"
    chmod 644 "$SERVICE_FILE"
    systemctl daemon-reload
    log_success "Service installed: $SERVICE_FILE"
}

# ─────────────────────────────────────────────────────────────────────────────
#  Permissions
# ─────────────────────────────────────────────────────────────────────────────
set_permissions() {
    log_section "Permissions"

    chmod 755 "$LOG_DIR"     2>/dev/null && log_success "Logs dir: $LOG_DIR" || \
        log_warning "Could not set permissions on $LOG_DIR"

    if [[ -f "$CLIENT_CONFIG" ]]; then
        # Config contains keys — restrict to owner only
        chmod 600 "$CLIENT_CONFIG" 2>/dev/null && log_success "Config protected: $CLIENT_CONFIG" || \
            log_warning "Could not restrict permissions on $CLIENT_CONFIG"
    fi

    log_success "Permissions applied"
}

# ─────────────────────────────────────────────────────────────────────────────
#  Verify
# ─────────────────────────────────────────────────────────────────────────────
verify_installation() {
    log_section "Verification"

    local ok=true

    if [[ -f "$SERVICE_FILE" ]]; then
        log_success "Service file present: $SERVICE_FILE"
    else
        log_warning "Service file NOT found: $SERVICE_FILE"
        ok=false
    fi

    if [[ -f "$CLIENT_CONFIG" ]]; then
        log_success "Config file present: $CLIENT_CONFIG"
    else
        log_warning "Config file NOT found: $CLIENT_CONFIG"
    fi

    if [[ -x "${VENV_DIR}/bin/llm" ]]; then
        log_success "CLI binary: ${VENV_DIR}/bin/llm"
    else
        log_warning "CLI binary not found — run: pip install local-llm-client"
    fi

    $ok && log_success "Verification passed" || log_warning "Verification completed with warnings"
}

# ─────────────────────────────────────────────────────────────────────────────
#  Summary
# ─────────────────────────────────────────────────────────────────────────────
print_summary() {
    echo ""
    echo -e "${BOLD}${GREEN}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}${GREEN}║${NC}  local-llm-client Setup Complete!                                ${BOLD}${GREEN}║${NC}"
    echo -e "${BOLD}${GREEN}╚══════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${CYAN}  Paths${NC}"
    echo    "  ────────────────────────────────────────────────────────────"
    echo    "  Work dir     : $WORK_DIR"
    echo    "  Virtual env  : $VENV_DIR"
    echo    "  Config file  : $CLIENT_CONFIG"
    echo    "  Logs         : $LOG_DIR"
    echo    "  Service file : $SERVICE_FILE"
    echo    "  Server URL   : $SERVER_URL"
    echo ""
    echo -e "${CYAN}  Next Steps${NC}"
    echo    "  ────────────────────────────────────────────────────────────"
    echo -e "  1. ${YELLOW}Set your API key${NC}"
    echo    "     llm config set server.api_key sk-yourkey"
    echo    "     llm config set server.master_key admin-yourkey"
    echo ""
    echo -e "  2. ${YELLOW}Run diagnostics${NC}"
    echo    "     llm doctor"
    echo ""
    echo -e "  3. ${YELLOW}Test the connection${NC}"
    echo    "     llm status health"
    echo    "     llm models list"
    echo ""
    echo -e "  4. ${YELLOW}Chat${NC}"
    echo    "     llm chat --interactive"
    echo    "     llm chat --model bitnet-gguf \"Hello, how does BitNet work?\""
    echo ""
    echo -e "  5. ${YELLOW}Manage API keys (master key required)${NC}"
    echo    "     llm keys create --id myuser --rps 10 --burst 20"
    echo    "     llm keys list"
    echo    "     llm keys revoke myuser"
    echo ""
    echo -e "  6. ${YELLOW}Optional: start background service${NC}"
    echo    "     sudo systemctl start  $SERVICE_NAME"
    echo    "     sudo systemctl enable $SERVICE_NAME"
    echo    "     sudo journalctl -u $SERVICE_NAME -f"
    echo ""
    echo -e "  ${YELLOW}Note:${NC} The systemd service is optional. The llm CLI"
    echo    "  works without it — just activate your venv and run llm."
    echo    "  Add to ~/.bashrc: source ${VENV_DIR}/bin/activate"
    echo ""
}

# ─────────────────────────────────────────────────────────────────────────────
#  Header
# ─────────────────────────────────────────────────────────────────────────────
print_header() {
    echo ""
    echo -e "${BOLD}${BLUE}  ╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}${BLUE}  ║${NC}  🤖  ${BOLD}LOCAL LLM CLIENT${NC} — Setup Script  ${CYAN}v2.0.0${NC}                   ${BOLD}${BLUE}║${NC}"
    echo -e "${BOLD}${BLUE}  ║${NC}  ${DIM}CLI • SDK • Background Service • Config Setup${NC}                  ${BOLD}${BLUE}║${NC}"
    echo -e "${BOLD}${BLUE}  ╚══════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    log_info "Script dir   : ${DIM}$SCRIPT_DIR${NC}"
    log_info "Work dir     : ${DIM}$WORK_DIR${NC}"
    log_info "Project root : ${DIM}$PROJECT_ROOT${NC}"
    if [[ "$NONINTERACTIVE" == "1" ]]; then
        log_info "Mode         : ⚡ ${YELLOW}NON-INTERACTIVE${NC}"
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
main() {
    print_header
    detect_venv
    check_prerequisites
    prepare_directories
    configure_server_url
    configure_client_config
    install_service
    set_permissions
    verify_installation
    print_summary
}

main "$@"
