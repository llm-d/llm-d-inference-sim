#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2026 The llm-d-inference-sim Authors.
"""Export the pinned vLLM request field schemas without loading the engine.

Only data declarations are evaluated. Methods (including Python validators) do
not contribute JSON Schema constraints; their checks live in the Go validator.
See pkg/engine/vllm/schema/README.md for the source and dependency pins.
"""

import argparse
import ast
import builtins
import json
import subprocess
import typing
import uuid
from dataclasses import field
from pathlib import Path

import pydantic
import pydantic.dataclasses
import typing_extensions
from openai_harmony import Message as OpenAIHarmonyMessage
from PIL import Image

VLLM_COMMIT = "ad7125a431e176d4161099480a66f0169609a690"
SOURCE_FILES = (
    "vllm/entrypoints/openai/engine/protocol.py",
    "vllm/entrypoints/chat_utils.py",
    "vllm/sampling_params.py",
    "vllm/entrypoints/openai/chat_completion/protocol.py",
    "vllm/entrypoints/openai/completion/protocol.py",
)


def export(source: Path) -> dict:
    namespace = {
        "__name__": "vllm_request_schema",
        "random_uuid": lambda: uuid.uuid4().hex,
        "field": field,
        "Image": Image,
        "OpenAIHarmonyMessage": OpenAIHarmonyMessage,
    }
    for module in (typing, typing_extensions, pydantic):
        namespace.update({key: getattr(module, key) for key in dir(module) if not key.startswith("_")})
    namespace["dataclass"] = pydantic.dataclasses.dataclass
    nodes = {}
    for filename in SOURCE_FILES:
        for node in ast.parse((source / filename).read_text()).body:
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("openai."):
                exec(compile(ast.Module([node], []), filename, "exec"), namespace)
            if isinstance(node, ast.ClassDef):
                node.body = [item for item in node.body if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))]
                nodes[node.name] = node
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                nodes[node.target.id] = node
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        nodes[target.id] = node

    def resolve(name):
        if name in namespace or hasattr(builtins, name):
            return
        node = nodes[name]
        while True:
            try:
                module = ast.fix_missing_locations(ast.Module([node], []))
                exec(compile(module, "vllm_request_schema", "exec"), namespace)
                return
            except NameError as error:
                resolve(error.name)

    document = {
        "openapi": "3.1.0",
        "info": {"title": "vLLM request schemas", "version": "0.21.0"},
        "paths": {},
        "components": {"schemas": {}},
        "x-vllm-commit": VLLM_COMMIT,
    }
    for path, name in (
        ("/v1/chat/completions", "ChatCompletionRequest"),
        ("/v1/completions", "CompletionRequest"),
    ):
        resolve(name)
        schema = namespace[name].model_json_schema(ref_template="#/components/schemas/{model}")
        document["components"]["schemas"].update(schema.pop("$defs", {}))
        document["components"]["schemas"][name] = schema
        document["paths"][path] = {"post": {"requestBody": {
            "required": True,
            "content": {"application/json": {"schema": {"$ref": "#/components/schemas/" + name}}},
        }}}
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vllm_source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "-C", str(args.vllm_source), "rev-parse", "HEAD"], text=True).strip()
    if revision != VLLM_COMMIT:
        parser.error(f"expected vLLM commit {VLLM_COMMIT}, got {revision}")
    subprocess.run(["git", "-C", str(args.vllm_source), "diff", "--exit-code", "HEAD", "--", *SOURCE_FILES], check=True)
    args.output.write_text(json.dumps(export(args.vllm_source), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
