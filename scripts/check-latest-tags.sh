#!/usr/bin/env bash
# Scan YAML files for image references using the ':latest' tag.
# Default: warn only. --strict makes findings a hard error.
set -e

STRICT=0
if [ "${1:-}" = "--strict" ]; then STRICT=1; fi

# Match `image: something:latest` (quoted or not), ignoring commented lines.
matches=$(grep -RInE --include='*.yaml' --include='*.yml' \
  '^[[:space:]]*[^#]*image:[[:space:]]*"?[^[:space:]"]+:latest"?' . \
  --exclude-dir=.git --exclude-dir=venv --exclude-dir=node_modules || true)

if [ -z "$matches" ]; then
  echo "No ':latest' image tags found."
  exit 0
fi

echo "Found ':latest' image tags:"
echo "$matches"

if [ "$STRICT" -eq 1 ]; then
  echo
  echo "ERROR: ':latest' tags are not allowed. Pin an explicit version."
  exit 1
fi
echo "WARNING: ':latest' tags found. Pin explicit versions."
