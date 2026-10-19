#!/usr/bin/env bash

set -euo pipefail

scripts=(
    simulate.py
    generate_data.py
    train.py
    plot.py
)

for script in "${scripts[@]}"; do
    echo "Running $script"
    python "$script"
done

echo "All scripts finished successfully."
