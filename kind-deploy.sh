#!/bin/bash

# This shell script deploys a kind cluster with the vLLM renderer and the simulator
set -eo pipefail

# ------------------------------------------------------------------------------
# Check variables
# ------------------------------------------------------------------------------
require_non_empty() {
  local name=$1
  if [ -z "${!name-}" ]; then
    echo "Error: $name must not be empty" >&2
    exit 1
  fi
}

require_non_empty CLUSTER_NAME
require_non_empty HOST_PORT
require_non_empty MODEL_NAME
require_non_empty VLLM_SIMULATOR_IMAGE

export CLUSTER_NAME
export HOST_PORT
export MODEL_NAME
export VLLM_SIMULATOR_IMAGE
export HF_TOKEN

# ------------------------------------------------------------------------------
# Setup & Requirement Checks
# ------------------------------------------------------------------------------

# Check for a supported container runtime if an explicit one was not set
if [ -z "${CONTAINER_RUNTIME}" ]; then
  if command -v docker &> /dev/null; then
    CONTAINER_RUNTIME="docker"
  elif command -v podman &> /dev/null; then
    CONTAINER_RUNTIME="podman"
  else
    echo "Neither docker nor podman could be found in PATH" >&2
    exit 1
  fi
fi

set -u

# Check for required programs
for cmd in kind kubectl ${CONTAINER_RUNTIME}; do
    if ! command -v "$cmd" &> /dev/null; then
        echo "Error: $cmd is not installed or not in the PATH."
        exit 1
    fi
done

# ------------------------------------------------------------------------------
# Cluster Deployment
# ------------------------------------------------------------------------------

# Check if the cluster already exists
if kind get clusters 2>/dev/null | grep -q "^${CLUSTER_NAME}$"; then
    echo "Cluster '${CLUSTER_NAME}' already exists, re-using"
else
    kind create cluster --name "${CLUSTER_NAME}" --config - << EOF
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- role: control-plane
  extraPortMappings:
  - containerPort: ${HOST_PORT}
    hostPort: ${HOST_PORT}
    protocol: TCP
EOF
fi

# Set the kubectl context to the kind cluster
KUBE_CONTEXT="kind-${CLUSTER_NAME}"
kubectl config set-context ${KUBE_CONTEXT} --namespace=default

set -x

# Hotfix for https://github.com/kubernetes-sigs/kind/issues/3880
CONTAINER_NAME="${CLUSTER_NAME}-control-plane"
${CONTAINER_RUNTIME} exec -it ${CONTAINER_NAME} /bin/bash -c "sysctl net.ipv4.conf.all.arp_ignore=0"

# Wait for all pods to be ready
kubectl --context ${KUBE_CONTEXT} -n kube-system wait --for=condition=Ready --all pods --timeout=600s

echo "Waiting for local-path-storage pods to be created..."
until kubectl --context ${KUBE_CONTEXT} -n local-path-storage get pods -o name | grep -q pod/; do
  sleep 2
done
kubectl --context ${KUBE_CONTEXT} -n local-path-storage wait --for=condition=Ready --all pods --timeout=300s

# ------------------------------------------------------------------------------
# Load Container Images
# ------------------------------------------------------------------------------

# Load the vllm simulator image into the cluster (only if it's a locally built image)

if [ -n "$(${CONTAINER_RUNTIME} images -q "${VLLM_SIMULATOR_IMAGE}")" ]; then
    echo "INFO: Loading vllm-sim image into KIND cluster..."
    if [ "${CONTAINER_RUNTIME}" == "podman" ]; then
        podman save ${VLLM_SIMULATOR_IMAGE} -o /dev/stdout | kind --name ${CLUSTER_NAME} load image-archive /dev/stdin
    else
        kind --name ${CLUSTER_NAME} load docker-image ${VLLM_SIMULATOR_IMAGE}
    fi
fi

# ------------------------------------------------------------------------------
# Development Environment
# ------------------------------------------------------------------------------

kubectl kustomize deploy \
	| envsubst '${MODEL_NAME} ${VLLM_SIMULATOR_IMAGE} ${HOST_PORT} ${HF_TOKEN}' \
  | kubectl --context ${KUBE_CONTEXT} apply -f -

# ------------------------------------------------------------------------------
# Check & Verify
# ------------------------------------------------------------------------------

# Wait for all deployments to be ready
kubectl --context ${KUBE_CONTEXT} -n default wait --for=condition=available --timeout=300s deployment --all

cat <<EOF
-----------------------------------------
Deployment completed!

* Kind Cluster Name: ${CLUSTER_NAME}
* Kubectl Context: ${KUBE_CONTEXT}

Status:

* The vllm simulator is running
* The vllm renderer is running

You can watch the Simulator logs with:

  $ kubectl --context ${KUBE_CONTEXT} logs -f deployments/vllm-sim

With that running in the background, you can make requests:

  $ curl -s -w '\n' http://localhost:${HOST_PORT}/v1/completions -H 'Content-Type: application/json' -d '{"model":"${MODEL_NAME}","prompt":"hi","max_tokens":10,"temperature":0}' | jq

-----------------------------------------
EOF
