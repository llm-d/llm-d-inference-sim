# Configuration variables
TARGETOS ?= $(shell go env GOOS)
TARGETARCH ?= $(shell go env GOARCH)
CONTAINER_TOOL := $(shell { command -v docker >/dev/null 2>&1 && echo docker; } || { command -v podman >/dev/null 2>&1 && echo podman; } || echo "")
IMAGE_REGISTRY ?= quay.io/mayab
# ghcr.io/llm-d
IMAGE_NAME ?= zmq-listener
IMAGE_TAG ?= latest
NAMESPACE ?= default
IMG = $(IMAGE_REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)

# Docker targets
.PHONY: image-build
image-build:
	$(CONTAINER_TOOL) build \
	--platform linux/amd64,linux/arm64 \
	--build-arg TARGETOS=linux \
	--build-arg TARGETARCH=$(TARGETARCH) \
	-t $(IMG) .

.PHONY: image-push
image-push: image-build
	docker push $(IMG)

# Kubernetes targets
.PHONY: deploy-listener
deploy-listener:
	kubectl apply -f deploy_listener.yaml -n $(NAMESPACE)

.PHONY: deploy-sim
deploy-sim:
	kubectl apply -f deploy_simulator.yaml -n $(NAMESPACE)

.PHONY: deploy-vllm
deploy-vllm:
	kubectl apply -f deploy_vllm.yaml -n $(NAMESPACE)

.PHONY: deploy-all
deploy-all: deploy-listener deploy-sim deploy-vllm

.PHONY: delete-listener
delete-listener:
	kubectl delete -f deploy_listener.yaml -n $(NAMESPACE) || true

.PHONY: delete-sim
delete-sim:
	kubectl delete -f deploy_simulator.yaml -n $(NAMESPACE) || true

.PHONY: delete-vllm
delete-vllm:
	kubectl delete -f deploy_vllm.yaml -n $(NAMESPACE) || true

.PHONY: delete-all
delete-all: delete-listener delete-sim delete-vllm

.PHONY: clean
clean: delete-all
	docker rmi $(IMG) || true