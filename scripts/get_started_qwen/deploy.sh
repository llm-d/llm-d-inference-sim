#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-llm-d-sim}"
TIMEOUT="${TIMEOUT:-180s}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

kubectl create namespace "${NAMESPACE}" \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl apply -n "${NAMESPACE}" -f "${SCRIPT_DIR}/deployment.yaml"

kubectl rollout status deployment/inference-sim \
  -n "${NAMESPACE}" \
  --timeout="${TIMEOUT}"

kubectl get pods -n "${NAMESPACE}" -o wide
