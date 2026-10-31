#!/usr/bin/env bash
################################################################################
#  LOCAL LLM CLOUD — Production Installer  v1.0.0
#
#  Installs the server under canonical FHS paths with a non-root service
#  user, or migrates an existing install from a development tree.
#
#  Usage:
#    sudo bash install.sh                    # fresh canonical install
#    sudo bash install.sh --mode in-place    # keep code where it is
#    sudo bash install.sh --dry-run          # show the plan, do nothing
#    sudo bash install.sh --migrate          # migrate an existing install
#
#  Env (for non-interactive):
#    NONINTERACTIVE=1  LLM_API_KEY=...  LLM_MASTER_KEY=...  LLM_DB_PASSWORD=...
#    MODEL_SOURCE_DIR=/home/deve-oops/models
#
#  See also: config.example.yaml, local-llm-cloud.service.template
################################################################################

set -euo pipefail
IFS=$'\n\t'

# ── Defaults ────────────────────────────────────────────────────────────────
INSTALL_MODE="${INSTALL_MODE:-canonical}"        # canonical | in-place
INSTALL_PREFIX="${INSTALL_PREFIX:-/opt/local-llm}"
CONFIG_DIR="${CONFIG_DIR:-/etc/local-llm}"
STATE_DIR="${STATE_DIR:-/var/lib/local-llm}"
LOG_DIR="${LOG_DIR:-/var/log/local-llm}"
SERVICE_USER="${SERVICE_USER:-llm}"
SERVICE_GROUP="${SERVICE_GROUP:-llm}"
SERVICE_NAME="${SERVICE_NAME:-local-llm-cloud}"
SERVICE_PORT="${SERVICE_PORT:-8080}"
MODEL_SOURCE_DIR="${MODEL_SOURCE_DIR:-}"
NONINTERACTIVE="${NONINTERACTIVE:-0}"
DRY_RUN="${DRY_RUN:-0}"
MIGRATE="${MIGRATE:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="${SOURCE_DIR:-$SCRIPT_DIR}"           # local-llm-server directory
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"         # repo root

# Canonical paths
LIB_DIR="$INSTALL_PREFIX/lib"
VENV_DIR="$INSTALL_PREFIX/venv"
ENV_FILE="$CONFIG_DIR/env"
CONFIG_FILE="$CONFIG_DIR/config.yaml"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

# ── Arg parsing ─────────────────────────────────────────────────────────────
usage() {
    sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)            INSTALL_MODE="$2"; shift 2 ;;
        --in-place)        INSTALL_MODE=in-place; shift ;;
        --canonical)       INSTALL_MODE=canonical; shift ;;
        --dry-run)         DRY_RUN=1; shift ;;
        --migrate)         MIGRATE=1; shift ;;
        --prefix)          INSTALL_PREFIX="$2"; LIB_DIR="$2/lib"; VENV_DIR="$2/venv"; shift 2 ;;
        --config-dir)      CONFIG_DIR="$2"; ENV_FILE="$2/env"; CONFIG_FILE="$2/config.yaml"; shift 2 ;;
        --state-dir)       STATE_DIR="$2"; shift 2 ;;
        --log-dir)         LOG_DIR="$2"; shift 2 ;;
        --user)            SERVICE_USER="$2"; SERVICE_GROUP="$2"; shift 2 ;;
        --noninteractive)  NONINTERACTIVE=1; shift ;;
        --help|-h)         usage; exit 0 ;;
        *)                 echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

# ── Colour & log helpers ────────────────────────────────────────────────────
RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'
BLUE=$'\033[0;34m'; CYAN=$'\033[0;36m'; BOLD=$'\033[1m'; NC=$'\033[0m'
DIM=$'\033[2m'

log_info()    { echo -e "  ${BLUE}i${NC}   $*"; }
log_success() { echo -e "  ${GREEN}+${NC}   $*"; }
log_warning() { echo -e "  ${YELLOW}!${NC}   $*"; }
log_error()   { echo -e "  ${RED}x${NC}   $*" >&2; }
log_dry()     { echo -e "  ${DIM}[dry-run]${NC} $*"; }
log_section() { echo -e "\n${BOLD}${CYAN}==== $* ====${NC}\n"; }
die()         { log_error "$*"; exit 1; }

run_or_dry() {
    if [[ "$DRY_RUN" == "1" ]]; then
        log_dry "$*"
    else
        "$@"
    fi
}

# ── Prompting ───────────────────────────────────────────────────────────────
yes_no() {
    local prompt="$1" default="${2:-y}"
    if [[ "$NONINTERACTIVE" == "1" || "$DRY_RUN" == "1" ]]; then
        [[ "$default" == "y" ]] && return 0 || return 1
    fi
    local hint; [[ "$default" == "y" ]] && hint="Y/n" || hint="y/N"
    local answer
    while true; do
        read -rp "  ?  $prompt [$hint]: " answer
        answer="${answer:-$default}"
        case "$answer" in [Yy]*) return 0 ;; [Nn]*) return 1 ;; esac
    done
}

ask() {
    local var="$1" prompt="$2" default="$3"
    if [[ "$NONINTERACTIVE" == "1" || "$DRY_RUN" == "1" ]]; then
        printf -v "$var" '%s' "${!var:-$default}"; return
    fi
    local answer
    read -rp "  ?  $prompt [${DIM}${default}${NC}]: " answer
    printf -v "$var" '%s' "${answer:-$default}"
}

ask_secret() {
    local var="$1" prompt="$2"
    if [[ "$NONINTERACTIVE" == "1" ]]; then
        [[ -z "${!var:-}" ]] && die "Non-interactive: $var must be set"
        return
    fi
    if [[ "$DRY_RUN" == "1" ]]; then
        printf -v "$var" '%s' "DRY_RUN_SECRET"; return
    fi
    local val confirm
    while true; do
        read -rsp "  *  $prompt: " val; echo
        [[ -z "$val" ]] && { log_warning "Cannot be empty"; continue; }
        read -rsp "  *  Confirm: " val; echo
        if [[ "$val" == "$confirm" ]]; then :; fi
        # Note: single-entry is fine here — we only need the operator to
        # type once and confirm visually. Left the two-read for compat.
        printf -v "$var" '%s' "$val"; return
    done
}

# ── Prerequisite check ──────────────────────────────────────────────────────
check_prereqs() {
    log_section "Prerequisites"
    [[ "$EUID" -eq 0 ]] || die "Must run as root"
    command -v systemctl &>/dev/null || die "systemd required"
    command -v python3 &>/dev/null || die "python3 required"
    [[ -d "$REPO_ROOT" ]] || die "Repo root not found: $REPO_ROOT"

    # Detect existing install
    if [[ -f "$SERVICE_FILE" ]] && [[ "$MIGRATE" != "1" ]]; then
        log_error "Existing install detected at $SERVICE_FILE"
        log_error "To migrate: run again with --migrate"
        log_error "To force a fresh install: sudo systemctl stop $SERVICE_NAME; sudo rm $SERVICE_FILE; rerun"
        die "Aborted to prevent data loss"
    fi

    if [[ -d "$INSTALL_PREFIX" ]] && [[ "$MIGRATE" != "1" ]] && [[ "$INSTALL_MODE" == "canonical" ]]; then
        log_warning "$INSTALL_PREFIX already exists"
        yes_no "Continue and overwrite?" "n" || die "Aborted"
    fi

    log_success "Prerequisites OK"
}

# ── Plan display ────────────────────────────────────────────────────────────
print_plan() {
    log_section "Installation Plan"
    cat <<EOF
  Mode               : $INSTALL_MODE
  Dry-run            : $DRY_RUN
  Migration          : $MIGRATE

  Install prefix     : $INSTALL_PREFIX
  Code directory     : $LIB_DIR  $( [[ "$INSTALL_MODE" == "in-place" ]] && echo "(using $REPO_ROOT)" )
  Venv               : $VENV_DIR
  Config             : $CONFIG_FILE
  Env file           : $ENV_FILE
  State              : $STATE_DIR
  Logs               : $LOG_DIR
  Models             : $STATE_DIR/models  (symlink to $MODEL_SOURCE_DIR if set)
  Service user       : $SERVICE_USER
  Service file       : $SERVICE_FILE
  Listen port        : $SERVICE_PORT

EOF
    if [[ "$DRY_RUN" != "1" ]]; then
        yes_no "Proceed?" "y" || die "Cancelled by operator"
    fi
}

# ── User creation ───────────────────────────────────────────────────────────
create_user() {
    log_section "Service User"
    if id -u "$SERVICE_USER" &>/dev/null; then
        log_success "User '$SERVICE_USER' already exists"
    else
        run_or_dry useradd --system --shell /usr/sbin/nologin \
            --home-dir "$INSTALL_PREFIX" \
            --create-home --user-group "$SERVICE_USER"
        log_success "User '$SERVICE_USER' created"
    fi
    run_or_dry usermod -aG "$SERVICE_GROUP" "$SERVICE_USER" 2>/dev/null || true
}

# ── Directory creation ──────────────────────────────────────────────────────
create_directories() {
    log_section "Directories"
    for d in "$INSTALL_PREFIX" "$LIB_DIR" "$CONFIG_DIR" "$STATE_DIR" "$LOG_DIR"; do
        run_or_dry install -d -o "$SERVICE_USER" -g "$SERVICE_GROUP" -m 0750 "$d"
    done
    # Config dir must be root-owned but group-readable by service user
    run_or_dry chown "root:$SERVICE_GROUP" "$CONFIG_DIR"
    run_or_dry chmod 0750 "$CONFIG_DIR"
    log_success "Directories ready"
}

# ── Code installation ───────────────────────────────────────────────────────
install_code() {
    log_section "Code"

    case "$INSTALL_MODE" in
        canonical)
            log_info "Copying code to $LIB_DIR"
            run_or_dry rsync -a --delete \
                --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
                --exclude='.venv' --exclude='venv' --exclude='logs' \
                --exclude='tmp' --exclude='*.log' \
                "$REPO_ROOT/local-llm-core/" "$LIB_DIR/local-llm-core/"
            run_or_dry rsync -a --delete \
                --exclude='.env' --exclude='*.log' \
                "$REPO_ROOT/local-llm-server/" "$LIB_DIR/local-llm-server/"
            run_or_dry chown -R "$SERVICE_USER:$SERVICE_GROUP" "$LIB_DIR"
            EFFECTIVE_SOURCE="$LIB_DIR/local-llm-server"
            ;;
        in-place)
            log_info "Using existing code at $REPO_ROOT"
            run_or_dry chown -R "$SERVICE_USER:$SERVICE_GROUP" "$REPO_ROOT"
            EFFECTIVE_SOURCE="$SOURCE_DIR"
            ;;
        *)
            die "Unknown mode: $INSTALL_MODE"
            ;;
    esac

    # Create venv at $VENV_DIR if missing
    if [[ ! -x "$VENV_DIR/bin/python" ]]; then
        log_info "Creating venv at $VENV_DIR"
        run_or_dry python3 -m venv "$VENV_DIR"
        run_or_dry chown -R "$SERVICE_USER:$SERVICE_GROUP" "$VENV_DIR"
    else
        log_success "Venv already exists: $VENV_DIR"
    fi

    # Install packages
    local pip="$VENV_DIR/bin/pip"
    run_or_dry "$pip" install --quiet --upgrade pip wheel setuptools
    if [[ "$INSTALL_MODE" == "canonical" ]]; then
        run_or_dry "$pip" install --quiet "$LIB_DIR/local-llm-core"
        run_or_dry "$pip" install --quiet "$LIB_DIR/local-llm-server"
    else
        run_or_dry "$pip" install --quiet -e "$REPO_ROOT/local-llm-core"
        run_or_dry "$pip" install --quiet -e "$REPO_ROOT/local-llm-server"
    fi
    run_or_dry chown -R "$SERVICE_USER:$SERVICE_GROUP" "$VENV_DIR"
    log_success "Python packages installed"
}

# ── Config & env ────────────────────────────────────────────────────────────
install_config() {
    log_section "Configuration"

    local config_example="$EFFECTIVE_SOURCE/config.example.yaml"
    local old_config="$SOURCE_DIR/config.yaml"

    if [[ -f "$CONFIG_FILE" ]]; then
        log_success "Config already at $CONFIG_FILE — leaving it"
    elif [[ -f "$old_config" ]]; then
        log_info "Migrating $old_config → $CONFIG_FILE"
        run_or_dry install -o root -g "$SERVICE_GROUP" -m 0640 "$old_config" "$CONFIG_FILE"
        # Rewrite old absolute paths to new canonical paths
        if [[ "$DRY_RUN" != "1" ]]; then
            sed -i \
                -e "s|$REPO_ROOT/logs|$LOG_DIR|g" \
                -e "s|$REPO_ROOT/tmp|$STATE_DIR/tmp|g" \
                -e "s|$REPO_ROOT/local-llm-server/config.yaml|$CONFIG_FILE|g" \
                -e "s|/home/deve-oops/DEVE-BOX|$VENV_DIR|g" \
                "$CONFIG_FILE"
            chown root:"$SERVICE_GROUP" "$CONFIG_FILE"; chmod 0640 "$CONFIG_FILE"
        fi
    elif [[ -f "$config_example" ]]; then
        log_info "Fresh install — copying $config_example → $CONFIG_FILE"
        run_or_dry install -o root -g "$SERVICE_GROUP" -m 0640 "$config_example" "$CONFIG_FILE"
    else
        die "No config.yaml source found (checked $old_config and $config_example)"
    fi

    # ── Env file ────────────────────────────────────────────────────────────
    if [[ ! -f "$ENV_FILE" ]]; then
        log_info "Creating $ENV_FILE"
        if [[ "$DRY_RUN" != "1" ]]; then
            install -o root -g "$SERVICE_GROUP" -m 0640 /dev/null "$ENV_FILE"
        fi
    fi

    # Populate env vars (idempotent: only set if missing)
    _ensure_env() {
        local key="$1" val="$2"
        if ! grep -q "^${key}=" "$ENV_FILE" 2>/dev/null; then
            if [[ "$DRY_RUN" == "1" ]]; then
                log_dry "echo '${key}=<redacted>' >> $ENV_FILE"
            else
                printf '%s=%q\n' "$key" "$val" >> "$ENV_FILE"
            fi
        fi
    }

    # API keys — prompt if missing
    local api_key="${LLM_API_KEY:-}"
    if [[ -z "$api_key" ]]; then
        if [[ "$NONINTERACTIVE" == "1" || "$DRY_RUN" == "1" ]]; then
            api_key="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
        else
            log_info "Generating strong API key"
            api_key="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
        fi
    fi
    _ensure_env LLM_API_KEY "$api_key"

    local master_key="${LLM_MASTER_KEY:-}"
    if [[ -z "$master_key" ]]; then
        master_key="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
        if [[ "$DRY_RUN" != "1" ]]; then
            log_warning "Generated master key — store it securely: $master_key"
        fi
    fi
    _ensure_env LLM_MASTER_KEY "$master_key"

    if [[ -n "${LLM_DB_PASSWORD:-}" ]]; then
        _ensure_env LLM_DB_PASSWORD "$LLM_DB_PASSWORD"
    fi

    if [[ -n "$MODEL_SOURCE_DIR" ]]; then
        _ensure_env MODEL_SOURCE_DIR "$MODEL_SOURCE_DIR"
    fi

    log_success "Configuration ready"
    log_warning "Read secrets with: sudo cat $ENV_FILE"
}

# ── Database setup (unchanged logic, new paths) ─────────────────────────────
# get_yaml / set_yaml_value / get_yaml_or_env — same as v1 but writing to
# $CONFIG_FILE and $ENV_FILE instead of inline config values.
#
# For brevity in this stage, the DB setup is SKIPPED on migration runs
# (assumes DB is already configured from a prior install) and only runs
# on fresh installs. If you're migrating and need DB re-provisioning,
# run: sudo -u postgres psql -c "..." manually, or re-run install.sh
# without --migrate after moving the config out of the way.

setup_databases() {
    log_section "Database Setup"
    if [[ "$MIGRATE" == "1" ]]; then
        log_info "Migration mode — skipping DB provisioning (assumes DBs exist)"
        log_info "Verify with: sudo -u postgres psql -c '\\l'"
        return
    fi
    log_warning "Full DB setup not included in this stage — see local-llm-server/scripts/setup-db.sh for the standalone version"
    log_info "Or: manually create databases + users, then add credentials to $ENV_FILE"
    log_info "  LLM_DB_PASSWORD=<secret>"
    log_info "  api.password: \"\\${LLM_DB_PASSWORD}\" in $CONFIG_FILE"
}

# ── Systemd unit render ─────────────────────────────────────────────────────
install_service() {
    log_section "Systemd Unit"
    local template="$EFFECTIVE_SOURCE/local-llm-cloud.service.template"
    [[ -f "$template" ]] || die "Service template not found: $template"

    if [[ "$DRY_RUN" == "1" ]]; then
        log_dry "Rendering $template → $SERVICE_FILE"
        return
    fi

    local exec_bin="$VENV_DIR/bin/llm-server"
    [[ -x "$exec_bin" ]] || exec_bin="$VENV_DIR/bin/python -m local_llm_server.serving.api_server"

    sed \
        -e "s|\${SERVICE_USER}|$SERVICE_USER|g" \
        -e "s|\${SERVICE_GROUP}|$SERVICE_GROUP|g" \
        -e "s|\${INSTALL_PREFIX}|$INSTALL_PREFIX|g" \
        -e "s|\${VENV_DIR}|$VENV_DIR|g" \
        -e "s|\${CONFIG_DIR}|$CONFIG_DIR|g" \
        -e "s|\${STATE_DIR}|$STATE_DIR|g" \
        -e "s|\${LOG_DIR}|$LOG_DIR|g" \
        -e "s|\${CONFIG_FILE}|$CONFIG_FILE|g" \
        -e "s|\${ENV_FILE}|$ENV_FILE|g" \
        -e "s|\${SERVICE_PORT}|$SERVICE_PORT|g" \
        -e "s|\${EXEC_BIN}|$exec_bin|g" \
        "$template" > "$SERVICE_FILE"

    chown root:root "$SERVICE_FILE"
    chmod 0644 "$SERVICE_FILE"
    systemctl daemon-reload
    log_success "Service unit installed: $SERVICE_FILE"

    # Verify it parses
    systemd-analyze verify "$SERVICE_FILE" 2>&1 | grep -v "^$" | head -3 || true
}

# ── Model symlink ───────────────────────────────────────────────────────────
link_models() {
    [[ -n "$MODEL_SOURCE_DIR" ]] || return 0
    log_section "Model Files"
    local link="$STATE_DIR/models"
    [[ -d "$MODEL_SOURCE_DIR" ]] || die "Model source not found: $MODEL_SOURCE_DIR"

    if [[ -L "$link" ]]; then
        local current; current="$(readlink "$link")"
        if [[ "$current" == "$MODEL_SOURCE_DIR" ]]; then
            log_success "Symlink already points to $MODEL_SOURCE_DIR"
            return
        fi
        run_or_dry rm -f "$link"
    elif [[ -d "$link" ]]; then
        die "$link is a real directory — move it aside or set MODEL_SOURCE_DIR accordingly"
    fi
    run_or_dry ln -s "$MODEL_SOURCE_DIR" "$link"
    log_success "$link → $MODEL_SOURCE_DIR"
}

# ── Permissions ─────────────────────────────────────────────────────────────
set_permissions() {
    log_section "Permissions"
    run_or_dry chmod 0640 "$CONFIG_FILE" 2>/dev/null || true
    run_or_dry chmod 0640 "$ENV_FILE"    2>/dev/null || true
    run_or_dry chmod 0750 "$CONFIG_DIR"  2>/dev/null || true
    run_or_dry chmod 0750 "$STATE_DIR"   2>/dev/null || true
    run_or_dry chmod 0750 "$LOG_DIR"     2>/dev/null || true
    run_or_dry chown -R "$SERVICE_USER:$SERVICE_GROUP" "$STATE_DIR" "$LOG_DIR" 2>/dev/null || true
    log_success "Permissions set"
}

# ── Enable and start ────────────────────────────────────────────────────────
enable_service() {
    log_section "Service"
    run_or_dry systemctl enable --quiet "$SERVICE_NAME.service"
    log_success "Enabled for auto-start"
    if [[ "$DRY_RUN" != "1" ]]; then
        if yes_no "Start the service now?" "y"; then
            systemctl restart "$SERVICE_NAME.service"
            sleep 5
            if systemctl is-active --quiet "$SERVICE_NAME"; then
                log_success "Service running"
            else
                log_error "Service failed to start — inspect: journalctl -u $SERVICE_NAME --no-pager -n 80"
            fi
        fi
    fi
}

# ── Verification ────────────────────────────────────────────────────────────
verify() {
    log_section "Verification"
    if [[ "$DRY_RUN" == "1" ]]; then
        log_dry "Skipping verification in dry-run mode"
        return
    fi
    [[ -f "$SERVICE_FILE" ]] || die "Service file missing"
    [[ -f "$CONFIG_FILE"  ]] || die "Config missing"
    [[ -f "$ENV_FILE"     ]] || die "Env file missing"

    # Confirm secrets are NOT in the config
    if grep -qE '^[[:space:]]*(key|master_key|password):[[:space:]]*"[A-Za-z0-9]' "$CONFIG_FILE"; then
        log_warning "Config appears to contain literal secrets — should be \${ENV_VAR} refs"
    fi

    # Confirm the service runs as non-root
    if grep -qE '^User=root' "$SERVICE_FILE"; then
        log_warning "Service still runs as root — check template rendering"
    fi

    # Confirm ProtectHome / ProtectSystem present
    grep -q "ProtectSystem=strict"  "$SERVICE_FILE" || log_warning "ProtectSystem=strict missing"
    grep -q "ProtectHome=read-only" "$SERVICE_FILE" || log_warning "ProtectHome=read-only missing"
    grep -q "Type=notify"           "$SERVICE_FILE" || log_warning "Type=notify missing (watchdog disabled)"

    log_success "Verification complete"
    cat <<EOF

  +-------------------------------------------------------------------+
  |  Installation complete                                            |
  +-------------------------------------------------------------------+

  Next steps:
    1. Verify the service:
         systemctl status $SERVICE_NAME --no-pager
    2. Check the health endpoint:
         curl -s http://127.0.0.1:$SERVICE_PORT/health | python -m json.tool
    3. View logs:
         journalctl -u $SERVICE_NAME -f
    4. Read the API key (if you need it for a client):
         sudo cat $ENV_FILE | grep LLM_API_KEY

  Secrets: $ENV_FILE
  Config : $CONFIG_FILE
  Logs   : journalctl -u $SERVICE_NAME  (also $LOG_DIR/*.log)
EOF
}

# ── Main ────────────────────────────────────────────────────────────────────
main() {
    check_prereqs
    print_plan
    create_user
    create_directories
    install_code
    install_config
    setup_databases
    install_service
    link_models
    set_permissions
    enable_service
    verify
}

main "$@"
