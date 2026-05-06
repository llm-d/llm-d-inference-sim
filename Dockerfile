# Build Stage: using Go 1.25.0 image
## Pure Go build, no CGO required
FROM quay.io/projectquay/golang:1.25 AS builder
ARG TARGETOS
ARG TARGETARCH

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

# Build simulator, no Python, no embedded tokenizer, no CGO for ZMQ, use pure Go
RUN CGO_ENABLED=0 GOOS=${TARGETOS} GOARCH=${TARGETARCH} go build -o bin/llm-d-inference-sim cmd/cmd.go

FROM registry.access.redhat.com/ubi9/ubi-micro:9.7

WORKDIR /

COPY --from=builder /workspace/bin/llm-d-inference-sim /app/llm-d-inference-sim

USER 65532:65532

ENTRYPOINT ["/app/llm-d-inference-sim"]
