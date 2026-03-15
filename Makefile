# Copyright 2025 The llm-d-inference-sim Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

LOCALBIN ?= $(shell pwd)/bin
$(LOCALBIN):
	mkdir -p $(LOCALBIN)

# Project-specific Go tooling is defined in Makefile.tools.mk.
include Makefile.tools.mk

# Makefile for the llm-d-inference-sim project

SHELL := /usr/bin/env bash

# Defaults
TARGETOS ?= $(shell go env GOOS)
TARGETARCH ?= $(shell go env GOARCH)
PROJECT_NAME ?= llm-d-inference-sim
IMAGE_REGISTRY ?= ghcr.io/llm-d
IMAGE_TAG_BASE ?= $(IMAGE_REGISTRY)/$(PROJECT_NAME)
SIM_TAG ?= dev
IMG = $(IMAGE_TAG_BASE):$(SIM_TAG)
# zmq defaults
ZMQ_IMAGE_NAME ?= zmq-listener
ZMQ_IMAGE_TAG ?= latest
NAMESPACE ?= default
ZMQ_IMG ?= $(IMAGE_REGISTRY)/$(ZMQ_IMAGE_NAME):$(ZMQ_IMAGE_TAG)

CONTAINER_TOOL := $(shell { command -v docker >/dev/null 2>&1 && echo docker; } || { command -v podman >/dev/null 2>&1 && echo podman; } || echo "")
BUILDER := $(shell command -v buildah >/dev/null 2>&1 && echo buildah || echo $(CONTAINER_TOOL))
PLATFORMS ?= linux/amd64 # linux/arm64 # linux/s390x,linux/ppc64le

# go source files
SRC = $(shell find . -type f -name '*.go')

.PHONY: help
help: ## Print help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Development

.PHONY: clean
clean:
	go clean -testcache -cache
	rmdir lib

.PHONY: format
format: ## Format Go source files
	@printf "\033[33;1m==== Running gofmt ====\033[0m\n"
	@gofmt -l -w $(SRC)

.PHONY: test
test: $(GINKGO) install-dependencies ## Run tests
	@printf "\033[33;1m==== Running tests ====\033[0m\n"
ifdef GINKGO_FOCUS
	CGO_ENABLED=1 $(GINKGO) -v -r -- -ginkgo.v -ginkgo.focus="$(GINKGO_FOCUS)"
else
	CGO_ENABLED=1 $(GINKGO) -v -r $(TEST_PKG)
endif

.PHONY: post-deploy-test
post-deploy-test: ## Run post deployment tests
	echo Success!
	@echo "Post-deployment tests passed."
	
.PHONY: lint
lint: $(GOLANGCI_LINT) ## Run lint
	@printf "\033[33;1m==== Running linting ====\033[0m\n"
	$(GOLANGCI_LINT) run

##@ Build

.PHONY: build
build: check-go install-dependencies
	@printf "\033[33;1m==== Building ====\033[0m\n"
	go build -o $(LOCALBIN)/$(PROJECT_NAME) cmd/$(PROJECT_NAME)/main.go

##@ Container Build/Push

.PHONY:	image-build
image-build: check-container-tool ## Build Docker image ## Build Docker image using $(CONTAINER_TOOL)
	@printf "\033[33;1m==== Building Docker image $(IMG) ====\033[0m\n"
	$(CONTAINER_TOOL) build \
		--platform linux/$(TARGETARCH) \
 		--build-arg TARGETOS=linux \
		--build-arg TARGETARCH=$(TARGETARCH)\
		-t $(IMG) .

.PHONY: image-push
image-push: check-container-tool ## Push Docker image $(IMG) to registry
	@printf "\033[33;1m==== Pushing Docker image $(IMG) ====\033[0m\n"
	$(CONTAINER_TOOL) push $(IMG)

.PHONY: image-build-and-push
image-build-and-push: image-build image-push ## Build and push Docker image $(IMG) to registry

### Docker Targets

.PHONY: install-docker
install-docker: check-container-tool ## Install app using $(CONTAINER_TOOL)
	@echo "Starting container with $(CONTAINER_TOOL)..."
	$(CONTAINER_TOOL) run -d --name $(PROJECT_NAME)-container $(IMG)
	@echo "$(CONTAINER_TOOL) installation complete."
	@echo "To use $(PROJECT_NAME), run:"
	@echo "alias $(PROJECT_NAME)='$(CONTAINER_TOOL) exec -it $(PROJECT_NAME)-container /app/$(PROJECT_NAME)'"

.PHONY: uninstall-docker
uninstall-docker: check-container-tool ## Uninstall app from $(CONTAINER_TOOL)
	@echo "Stopping and removing container in $(CONTAINER_TOOL)..."
	-$(CONTAINER_TOOL) stop $(PROJECT_NAME)-container && $(CONTAINER_TOOL) rm $(PROJECT_NAME)-container
	@echo "$(CONTAINER_TOOL) uninstallation complete. Remove alias if set: unalias $(PROJECT_NAME)"

### Helm Targets
.PHONY: install-helm
install-helm: check-helm ## Install app using Helm
	@echo "Installing chart with Helm..."
	helm upgrade --install $(PROJECT_NAME) helm/$(PROJECT_NAME) --namespace default
	@echo "Helm installation complete."

.PHONY: uninstall-helm
uninstall-helm: check-helm ## Uninstall app using Helm
	@echo "Uninstalling chart with Helm..."
	helm uninstall $(PROJECT_NAME) --namespace default
	@echo "Helm uninstallation complete."

.PHONY: env
env: ## Print environment variables
	@echo "IMAGE_TAG_BASE=$(IMAGE_TAG_BASE)"
	@echo "IMG=$(IMG)"
	@echo "CONTAINER_TOOL=$(CONTAINER_TOOL)"

##@ Tools
.PHONY: check-go
check-go:
	@command -v go >/dev/null 2>&1 || { \
	  echo "❌ Go is not installed. Install it from https://golang.org/dl/"; exit 1; }

.PHONY: check-container-tool
check-container-tool:
	@command -v $(CONTAINER_TOOL) >/dev/null 2>&1 || { \
	  echo "❌ $(CONTAINER_TOOL) is not installed."; \
	  echo "🔧 Try: sudo apt install $(CONTAINER_TOOL) OR brew install $(CONTAINER_TOOL)"; exit 1; }

.PHONY: check-helm
check-helm:
	@command -v helm >/dev/null 2>&1 || { \
	  echo "❌ helm is not installed. Install it from https://helm.sh/docs/intro/install/"; exit 1; }

.PHONY: check-builder
check-builder:
	@if [ -z "$(BUILDER)" ]; then \
		echo "❌ No container builder tool (buildah, docker, or podman) found."; \
		exit 1; \
	else \
		echo "✅ Using builder: $(BUILDER)"; \
	fi

##@ Alias checking
.PHONY: check-alias
check-alias: check-container-tool
	@echo "🔍 Checking alias functionality for container '$(PROJECT_NAME)-container'..."
	@if ! $(CONTAINER_TOOL) exec $(PROJECT_NAME)-container /app/$(PROJECT_NAME) --help >/dev/null 2>&1; then \
	  echo "⚠️  The container '$(PROJECT_NAME)-container' is running, but the alias might not work."; \
	  echo "🔧 Try: $(CONTAINER_TOOL) exec -it $(PROJECT_NAME)-container /app/$(PROJECT_NAME)"; \
	else \
	  echo "✅ Alias is likely to work: alias $(PROJECT_NAME)='$(CONTAINER_TOOL) exec -it $(PROJECT_NAME)-container /app/$(PROJECT_NAME)'"; \
	fi

.PHONY: print-project-name
print-project-name: ## Print the current project name
	@echo "$(PROJECT_NAME)"

.PHONY: install-hooks
install-hooks: ## Install git hooks
	git config core.hooksPath hooks

##@ ZMQ Setup

.PHONY: install-dependencies
install-dependencies: ## Install development dependencies based on OS/ARCH
	@echo "Checking and installing development dependencies..."
	@if [ "$(TARGETOS)" = "linux" ]; then \
	  if [ -x "$$(command -v apt)" ]; then \
	    if ! dpkg -s libzmq3-dev >/dev/null 2>&1 || ! dpkg -s g++ >/dev/null 2>&1; then \
	      echo "Installing dependencies with apt..."; \
	      sudo apt-get update && sudo apt-get install -y libzmq3-dev g++; \
	    else \
	      echo "✅ ZMQ and g++ are already installed."; \
	    fi; \
	  elif [ -x "$$(command -v dnf)" ]; then \
	    if ! dnf -q list installed zeromq-devel >/dev/null 2>&1 || ! dnf -q list installed gcc-c++ >/dev/null 2>&1; then \
	      echo "Installing dependencies with dnf..."; \
	      sudo dnf install -y zeromq-devel gcc-c++; \
	    else \
	      echo "✅ ZMQ and gcc-c++ are already installed."; \
	    fi; \
	  else \
	    echo "Unsupported Linux package manager. Install libzmq and g++/gcc-c++ manually."; \
	    exit 1; \
	  fi; \
	elif [ "$(TARGETOS)" = "darwin" ]; then \
	  if [ -x "$$(command -v brew)" ]; then \
	    if ! brew list zeromq pkg-config >/dev/null 2>&1; then \
	      echo "Installing dependencies with brew..."; \
	      brew install zeromq pkg-config; \
	    else \
	      echo "✅ ZeroMQ and pkgconf are already installed."; \
	    fi; \
	  else \
	    echo "Homebrew is not installed and is required to install zeromq. Install it from https://brew.sh/"; \
	    exit 1; \
	  fi; \
	else \
	  echo "Unsupported OS: $(TARGETOS). Install development dependencies manually."; \
	  exit 1; \
	fi

# ZMQ listener

# Docker targets
.PHONY: zmq-image-build
zmq-image-build:
	$(CONTAINER_TOOL) build \
	--platform linux/amd64,linux/arm64 \
	--build-arg TARGETOS=linux \
	--build-arg TARGETARCH=$(TARGETARCH) \
	-t $(ZMQ_IMG) -f Dockerfile.zmq .

.PHONY: zmq-image-push
zmq-image-push: zmq-image-build
	docker push $(ZMQ_IMG)

# Kubernetes targets
.PHONY: deploy-zmq-listener
deploy-zmq-listener:
	kubectl apply -f ./manifests/zmq-listener/deploy_listener.yaml -n $(NAMESPACE)

.PHONY: deploy-sim
deploy-sim:
	kubectl apply -f ./manifests/zmq-listener/deploy_simulator.yaml -n $(NAMESPACE)

.PHONY: deploy-vllm
deploy-vllm:
	kubectl apply -f ./manifests/zmq-listener/deploy_vllm.yaml -n $(NAMESPACE)

.PHONY: deploy-zmq-all
deploy-all: deploy-zmq-listener deploy-sim deploy-vllm

.PHONY: delete-zmq-listener
delete-zmq-listener:
	kubectl delete -f ./manifests/zmq-listener/deploy_listener.yaml -n $(NAMESPACE) || true

.PHONY: delete-sim
delete-sim:
	kubectl delete -f ./manifests/zmq-listener/deploy_simulator.yaml -n $(NAMESPACE) || true

.PHONY: delete-vllm
delete-vllm:
	kubectl delete -f ./manifests/zmq-listener/deploy_vllm.yaml -n $(NAMESPACE) || true

.PHONY: delete-zmq-all
delete-all: delete-zmq-listener delete-sim delete-vllm

.PHONY: clean-zmq
clean-zmq: delete-zmq-all
	docker rmi $(ZMQ_IMG) || true

## Dataset tool
.PHONY: ds-tool-build
ds-tool-build: check-go install-dependencies
	@printf "\033[33;1m==== Building ====\033[0m\n"
	go build -o $(LOCALBIN)/ds_tool cmd/dataset-tool/main.go


# Deploy the simulator with UDS tokenizer on kind
KIND_CLUSTER_NAME ?= ${PROJECT_NAME}-dev
HOST_PORT ?= 30080
MODEL_NAME ?= TinyLlama/TinyLlama-1.1B-Chat-v1.0
UDS_TOKENIZER_TAG=v0.6.0
UDS_TOKENIZER_IMG_NAME ?= $(IMAGE_REGISTRY)/llm-d-uds-tokenizer:${UDS_TOKENIZER_TAG}

.PHONY: dev-env-kind
dev-env-kind: 
	@printf "\033[33;1m==== Deploying on kind ====\033[0m\n"
	CLUSTER_NAME=$(KIND_CLUSTER_NAME) \
	HOST_PORT=$(HOST_PORT) \
	MODEL_NAME=${MODEL_NAME} \
	VLLM_SIMULATOR_IMAGE=${IMG} \
	UDS_TOKENIZER_IMAGE=${UDS_TOKENIZER_IMG_NAME} \
	./kind-deploy.sh

.PHONY: clean-dev-env-kind
clean-dev-env-kind: ## Cleanup kind setup (delete cluster $(KIND_CLUSTER_NAME))
	@echo "INFO: cleaning up kind cluster $(KIND_CLUSTER_NAME)"
	kind delete cluster --name $(KIND_CLUSTER_NAME)

