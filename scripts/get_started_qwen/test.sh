#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-llm-d-sim}"
LOCAL_PORT="${LOCAL_PORT:-8000}"
MODEL="${MODEL:-test-sim-model}"
PROMPT="${PROMPT:-Hello from llm-d simulation}"

PORT_FORWARD_LOG="$(mktemp)"
RESPONSE_FILE="$(mktemp)"
PF_PID=""

cleanup() {
  if [[ -n "${PF_PID}" ]]; then
    kill "${PF_PID}" 2>/dev/null || true
    wait "${PF_PID}" 2>/dev/null || true
  fi
  rm -f "${PORT_FORWARD_LOG}" "${RESPONSE_FILE}"
}
trap cleanup EXIT

kubectl port-forward \
  -n "${NAMESPACE}" \
  svc/inference-sim \
  "${LOCAL_PORT}:8000" \
  >"${PORT_FORWARD_LOG}" 2>&1 &
PF_PID=$!

for _ in $(seq 1 30); do
  if ! kill -0 "${PF_PID}" 2>/dev/null; then
    cat "${PORT_FORWARD_LOG}" >&2
    exit 1
  fi

  if curl -sS \
    "http://127.0.0.1:${LOCAL_PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"${PROMPT}\"}],\"max_tokens\":32}" \
    >"${RESPONSE_FILE}"; then
    cat "${RESPONSE_FILE}"
    printf '\n'
    exit 0
  fi

  sleep 1
done

cat "${PORT_FORWARD_LOG}" >&2
exit 1
