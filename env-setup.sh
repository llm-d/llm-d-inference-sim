#!/bin/bash

export PYTHONPATH=${PYTHONPATH}:$(go env GOPATH)/pkg/mod/github.com/llm-d/llm-d-kv-cache-manager@$(go list -m -f '{{.Version}}' github.com/llm-d/llm-d-kv-cache-manager)/pkg/preprocessing/chat_completions

