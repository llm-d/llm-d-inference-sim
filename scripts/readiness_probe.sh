#!/bin/bash
#
# vLLM Readiness Probe Script
#
# This script checks if the vLLM inference server is fully ready by verifying:
# 1. The basic health endpoint is responding
# 2. The /v1/models endpoint is available (indicates model is loaded)
# 3. The model metadata is properly returned
#
# This addresses the three stages of readiness:
# - Container is running (handled by Kubernetes)
# - vLLM API server is up (checked via /health)
# - Model-specific API routes are ready (checked via /v1/models)
#
# Exit codes:
#   0 - Service is ready
#   1 - Service is not ready
#
# Usage:
#   readiness_probe.sh PORT [HOST]
#
# Arguments:
#   PORT - Port where vLLM server is listening (required)
#   HOST - Host to connect to (optional, default: localhost)

set -euo pipefail

# Validate required arguments
if [ -z "${1:-}" ]; then
    echo "[readiness_probe] ERROR: PORT argument is required" >&2
    echo "Usage: readiness_probe.sh PORT [HOST]" >&2
    exit 1
fi

# Configuration
PORT="${1}"
HOST="${2:-localhost}"
TIMEOUT="${READINESS_TIMEOUT:-5}"
HEALTH_ENDPOINT="http://${HOST}:${PORT}/health"
MODELS_ENDPOINT="http://${HOST}:${PORT}/v1/models"

# Helper function for logging
log() {
    echo "[readiness_probe] $*" >&2
}

# Check if curl is available
if ! command -v curl &> /dev/null; then
    log "ERROR: curl command not found"
    exit 1
fi

# Step 1: Check basic health endpoint
log "Checking health endpoint: ${HEALTH_ENDPOINT}"
if ! curl -sf --max-time "${TIMEOUT}" "${HEALTH_ENDPOINT}" > /dev/null 2>&1; then
    log "Health endpoint not responding"
    exit 1
fi
log "Health endpoint is responding"

# Step 2: Check /v1/models endpoint (indicates model is loaded)
log "Checking models endpoint: ${MODELS_ENDPOINT}"
MODELS_RESPONSE=$(curl -sf --max-time "${TIMEOUT}" "${MODELS_ENDPOINT}" 2>&1)
CURL_EXIT_CODE=$?

if [ ${CURL_EXIT_CODE} -ne 0 ]; then
    log "Models endpoint not responding (curl exit code: ${CURL_EXIT_CODE})"
    exit 1
fi

# Step 3: Verify response contains model data
if ! echo "${MODELS_RESPONSE}" | grep -q '"data"'; then
    log "Models endpoint response does not contain expected 'data' field"
    exit 1
fi

# Check if data array is not empty
if echo "${MODELS_RESPONSE}" | grep -q '"data":\s*\[\s*\]'; then
    log "Models endpoint returned empty data array - model not loaded yet"
    exit 1
fi

log "vLLM server is ready - model is loaded and API is responding"
exit 0

