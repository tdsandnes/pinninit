#!/usr/bin/env bash

set -euo pipefail

scripts=(
    simulate.py
    generate_data.py
    train.py
    plot_losses.py
    plot_snapshots.py
    plot_error_grid.py
)

for script in "${scripts[@]}"; do
    echo "Running $script"
    python "$script"
done

echo "All scripts finished successfully."
