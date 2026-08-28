#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-llm-d-sim}"

kubectl delete namespace "${NAMESPACE}" --ignore-not-found
