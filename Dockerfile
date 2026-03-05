# Build Stage: using Go 1.24.1 image
FROM quay.io/projectquay/golang:1.24 AS builder
ARG TARGETOS
ARG TARGETARCH

# Install build tools
# The builder is based on UBI8, so we need epel-release-8.
RUN dnf install -y 'https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm' && \
    dnf install -y zeromq-devel pkgconfig && \
    dnf clean all

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

# Build simulator with CGO for ZMQ only (no Python, no embedded tokenizer)
# The default kv-cache build uses UDS tokenizer (//go:build !embedded_tokenizers)
RUN CGO_ENABLED=1 GOOS=${TARGETOS} GOARCH=${TARGETARCH} go build -o bin/llm-d-inference-sim cmd/cmd.go

# Build simulator with CGO for ZMQ only (no Python, no embedded tokenizer)
# The default kv-cache build uses UDS tokenizer (//go:build !embedded_tokenizers)
FROM registry.access.redhat.com/ubi9/ubi-minimal:9.7

WORKDIR /

# Install ZMQ runtime library only (no Python needed)
RUN curl -L -o /tmp/epel-release.rpm https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm && \
    rpm -i /tmp/epel-release.rpm && \
    rm /tmp/epel-release.rpm && \
    microdnf install -y --setopt=install_weak_deps=0 zeromq && \
    microdnf clean all && \
    rm -rf /var/cache/yum /var/lib/yum

COPY --from=builder /workspace/bin/llm-d-inference-sim /app/llm-d-inference-sim

USER 65532:65532

ENTRYPOINT ["/app/llm-d-inference-sim"]
