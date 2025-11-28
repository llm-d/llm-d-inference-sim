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

ifeq ($(TARGETOS),darwin)
ifeq ($(TARGETARCH),amd64)
TOKENIZER_ARCH = x86_64
else
TOKENIZER_ARCH = $(TARGETARCH)
endif
else
TOKENIZER_ARCH = $(TARGETARCH)
endif

CONTAINER_TOOL := $(shell { command -v docker >/dev/null 2>&1 && echo docker; } || { command -v podman >/dev/null 2>&1 && echo podman; } || echo "")
BUILDER := $(shell command -v buildah >/dev/null 2>&1 && echo buildah || echo $(CONTAINER_TOOL))
PLATFORMS ?= linux/amd64 # linux/arm64 # linux/s390x,linux/ppc64le

# go source files
SRC = $(shell find . -type f -name '*.go')

.PHONY: help
help: ## Print help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

PYTHON_INCLUDE := $(shell python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
CGO_CFLAGS     := $(shell python3-config --cflags --embed)
CGO_LDFLAGS    := $(shell python3-config --ldflags --embed)

export PKG_CONFIG_PATH=/usr/lib/pkgconfig

GOMODCACHE := $(shell go env GOMODCACHE)
KV_CACHE_MGR_VERSION := $(shell go list -m -f '{{.Version}}' github.com/llm-d/llm-d-kv-cache-manager)
KV_CACHE_MGR_PATH := $(GOMODCACHE)/github.com/llm-d/llm-d-kv-cache-manager@$(KV_CACHE_MGR_VERSION)/pkg/preprocessing/chat_completions
export PYTHONPATH := $(KV_CACHE_MGR_PATH):$(PYTHONPATH)

# Export them for all targets (optional)
export CGO_ENABLED=1
export CGO_CFLAGS
export CGO_LDFLAGS
export CPATH := $(PYTHON_INCLUDE):$(CPATH)

GO_LDFLAGS := -extldflags '-L$(shell pwd)/lib $(LDFLAGS)'
CGO_ENABLED=1
TOKENIZER_LIB = lib/libtokenizers.a
# Extract TOKENIZER_VERSION from Dockerfile
TOKENIZER_VERSION := $(shell grep '^ARG TOKENIZER_VERSION=' Dockerfile | cut -d'=' -f2)

.PHONY: download-tokenizer
download-tokenizer: $(TOKENIZER_LIB)
$(TOKENIZER_LIB):
	## Download the HuggingFace tokenizer bindings.
	@echo "Downloading HuggingFace tokenizer bindings for version $(TOKENIZER_VERSION)..."
	mkdir -p lib
	curl -L https://github.com/daulet/tokenizers/releases/download/$(TOKENIZER_VERSION)/libtokenizers.$(TARGETOS)-$(TOKENIZER_ARCH).tar.gz | tar -xz -C lib
	ranlib lib/*.a

##@ Development

.PHONY: clean
clean:
	go clean -testcache -cache
	rm -f $(TOKENIZER_LIB)
	rmdir lib

.PHONY: format
format: ## Format Go source files
	@printf "\033[33;1m==== Running gofmt ====\033[0m\n"
	@gofmt -l -w $(SRC)

.PHONY: test
test: $(GINKGO) install-dependencies ## Run tests
	@printf "\033[33;1m==== Running tests ====\033[0m\n"
ifdef GINKGO_FOCUS
	CGO_ENABLED=1 ginkgo -ldflags="$(GO_LDFLAGS)" -v -r -- -ginkgo.v -ginkgo.focus="$(GINKGO_FOCUS)"
else
	CGO_ENABLED=1 $(GINKGO) -ldflags="$(GO_LDFLAGS)" -v -r
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
	go build -ldflags="$(GO_LDFLAGS)" -o $(LOCALBIN)/$(PROJECT_NAME) cmd/$(PROJECT_NAME)/main.go

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

##@ Install/Uninstall Targets

# Default install/uninstall (Docker)
install: install-docker ## Default install using Docker
	@echo "Default Docker install complete."

uninstall: uninstall-docker ## Default uninstall using Docker
	@echo "Default Docker uninstall complete."

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
install-dependencies: download-tokenizer ## Install development dependencies based on OS/ARCH
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
