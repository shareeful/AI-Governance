#!/usr/bin/env bash
set -euo pipefail

LOCAL_CONFIG="${1:-configs/local.yaml}"

if [ ! -f "$LOCAL_CONFIG" ]; then
    echo "local configuration not found at $LOCAL_CONFIG" >&2
    echo "copy configs/example.local.yaml and set every path to a real resource" >&2
    exit 1
fi

CFG=(--config configs/default.yaml)
for system_config in configs/systems/*.yaml; do
    CFG+=(--config "$system_config")
done
CFG+=(--config "$LOCAL_CONFIG")

agp "${CFG[@]}" keygen
agp "${CFG[@]}" compile
agp "${CFG[@]}" train
agp "${CFG[@]}" shadow-pool
agp "${CFG[@]}" certify
agp "${CFG[@]}" experiment
agp "${CFG[@]}" report
