#!/usr/bin/env bash
set -euo pipefail

echo "Removing generated directories..."

dirs=("data" "models" "plots" "logs")

for dir in "${dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo "Removing $dir/"
        rm -rf "$dir"
    fi
done

echo "Cleanup complete."
