#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# entrypoint.sh
# AI Financial Behavior Project — Docker entrypoint
#
# Behaviour is controlled by the APP_MODE environment variable:
#   APP_MODE=dashboard  (default) → launch Streamlit dashboard
#   APP_MODE=train                → run main.py (full model training)
#   APP_MODE=validate             → run validate_new_dataset.py
#   APP_MODE=test                 → run pytest test suite
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Logging setup ─────────────────────────────────────────────────────────────
LOG_DIR="${LOG_DIR:-/app/logs}"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/app_${TIMESTAMP}.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "═══════════════════════════════════════════════════════"
log "AI Financial Behavior Project — starting"
log "APP_MODE = ${APP_MODE:-dashboard}"
log "Python   = $(python --version)"
log "Working  = $(pwd)"
log "═══════════════════════════════════════════════════════"

# ── Mode dispatch ─────────────────────────────────────────────────────────────
case "${APP_MODE:-dashboard}" in

    dashboard)
        log "Launching Streamlit dashboard on port 8501..."
        exec streamlit run app/dashboard.py \
            --server.port=8501 \
            --server.address=0.0.0.0 \
            --server.headless=true \
            --browser.gatherUsageStats=false \
            2>&1 | tee -a "$LOG_FILE"
        ;;

    train)
        log "Starting full model training pipeline..."
        exec python main.py 2>&1 | tee -a "$LOG_FILE"
        ;;

    validate)
        log "Running new dataset validation (2025 data)..."
        if [ ! "$(ls -A models/ 2>/dev/null)" ]; then
            log "ERROR: models/ directory is empty."
            log "       Run APP_MODE=train first to train and save models."
            exit 1
        fi
        exec python validate_new_dataset.py 2>&1 | tee -a "$LOG_FILE"
        ;;

    test)
        log "Running pytest test suite..."
        exec python -m pytest tests/ \
            -v \
            --tb=short \
            --log-file="${LOG_DIR}/pytest_${TIMESTAMP}.log" \
            --log-file-level=INFO \
            2>&1 | tee -a "$LOG_FILE"
        ;;

    generate)
        log "Running generate_outputs.py (load saved models, regenerate outputs)..."
        exec python generate_outputs.py 2>&1 | tee -a "$LOG_FILE"
        ;;

    *)
        log "ERROR: Unknown APP_MODE='${APP_MODE}'"
        log "Valid values: dashboard | train | validate | test | generate"
        exit 1
        ;;

esac