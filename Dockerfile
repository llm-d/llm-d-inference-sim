# Build Stage: using Go 1.24.1 image
FROM quay.io/projectquay/golang:1.26 AS builder
ARG TARGETOS
ARG TARGETARCH

# Install build tools
# The builder is based on UBI8, so we need epel-release-8.
RUN dnf install -y 'https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm' && \
    dnf install -y gcc-c++ libstdc++ libstdc++-devel clang zeromq-devel pkgconfig python3.12-devel python3.12-pip git && \
    dnf clean all
# python3.12-devel needed for CGO compilation (Python headers and python3.12-config for linker flags)

WORKDIR /workspace
# Copy the Go Modules manifests
COPY go.mod go.mod
COPY go.sum go.sum
# cache deps before building and copying source so that we don't need to re-download as much
# and so that source changes don't invalidate our downloaded layer
RUN go mod download

# Copy the go source
COPY cmd/llm-d-inference-sim/main.go cmd/cmd.go
COPY . .

# HuggingFace tokenizer bindings
RUN mkdir -p lib
# Ensure that the TOKENIZER_VERSION matches the one used in the imported llm-d-kv-cache-manager version
ARG TOKENIZER_VERSION=v1.22.1
RUN curl -L https://github.com/daulet/tokenizers/releases/download/${TOKENIZER_VERSION}/libtokenizers.${TARGETOS}-${TARGETARCH}.tar.gz | tar -xz -C lib
RUN ranlib lib/*.a

# Copy Python wrapper and requirements from kv-cache-manager dependency
# Extract version dynamically and copy to a known location
RUN KV_CACHE_MGR_VERSION=$(go list -m -f '{{.Version}}' github.com/llm-d/llm-d-kv-cache-manager) && \
    mkdir -p /workspace/kv-cache-manager-wrapper && \
    cp /go/pkg/mod/github.com/llm-d/llm-d-kv-cache-manager@${KV_CACHE_MGR_VERSION}/pkg/preprocessing/chat_completions/render_jinja_template_wrapper.py \
       /workspace/kv-cache-manager-wrapper/ && \
    cp /go/pkg/mod/github.com/llm-d/llm-d-kv-cache-manager@${KV_CACHE_MGR_VERSION}/pkg/preprocessing/chat_completions/requirements.txt \
       /workspace/kv-cache-manager-wrapper/

# Build
# the GOARCH has not a default value to allow the binary be built according to the host where the command
# was called. For example, if we call make image-build in a local env which has the Apple Silicon M1 SO
# the docker BUILDPLATFORM arg will be linux/arm64 when for Apple x86 it will be linux/amd64. Therefore,
# by leaving it empty we can ensure that the container and binary shipped on it will have the same platform.
ENV CGO_ENABLED=1
ENV GOOS=${TARGETOS:-linux}
ENV GOARCH=${TARGETARCH}
ENV PYTHON=python3.12
ENV PYTHONPATH=/usr/lib64/python3.12/site-packages:/usr/lib/python3.12/site-packages

RUN export CGO_CFLAGS="$(python3.12-config --cflags) -I/workspace/lib" && \
    export CGO_LDFLAGS="$(python3.12-config --ldflags --embed) -L/workspace/lib -ltokenizers -ldl -lm" && \
    go build -a -o bin/llm-d-inference-sim -ldflags="-extldflags '-L$(pwd)/lib'" cmd/cmd.go

# Runtime stage
# Use ubi9 as a minimal base image to package the manager binary
# Refer to https://catalog.redhat.com/software/containers/ubi9/ubi-minimal/615bd9b4075b022acc111bf5 for more details
FROM registry.access.redhat.com/ubi9/ubi-minimal:latest

WORKDIR /

# Install zeromq runtime library and Python runtime needed by the manager.
# The final image is UBI9, so we need epel-release-9.
# Using microdnf for minimal image size
USER root
RUN curl -L -o /tmp/epel-release.rpm https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm && \
    rpm -i /tmp/epel-release.rpm && \
    rm /tmp/epel-release.rpm && \
    microdnf install -y --setopt=install_weak_deps=0 zeromq python3.12 python3.12-libs python3.12-pip && \
    microdnf clean all && \
    rm -rf /var/cache/yum /var/lib/yum && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.12 /usr/bin/python

# Install wrapper as a module in site-packages
RUN mkdir -p /usr/local/lib/python3.12/site-packages/
COPY --from=builder /workspace/kv-cache-manager-wrapper/render_jinja_template_wrapper.py /usr/local/lib/python3.12/site-packages/

# Python deps (no cache, single target) – filter out torch
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY --from=builder /workspace/kv-cache-manager-wrapper/requirements.txt /tmp/requirements.txt
RUN sed '/^torch\b/d' /tmp/requirements.txt > /tmp/requirements.notorch.txt && \
    python3.12 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python3.12 -m pip install --no-cache-dir --target /usr/local/lib/python3.12/site-packages -r /tmp/requirements.notorch.txt && \
    rm /tmp/requirements.txt /tmp/requirements.notorch.txt && \
    rm -rf /root/.cache/pip

# Python env
ENV PYTHONPATH="/usr/local/lib/python3.12/site-packages:/usr/lib/python3.12/site-packages"
ENV PYTHON=python3.12

COPY --from=builder /workspace/bin/llm-d-inference-sim /app/llm-d-inference-sim

USER 65532:65532

ENTRYPOINT ["/app/llm-d-inference-sim"]
