#!/bin/bash

# Create data directory if it doesn't exist
mkdir -p ./data


# Base URL for Hugging Face dataset files
BASE_URL="https://huggingface.co/datasets/tdsandnes/pinninit-examples/resolve/main/cylinder_2d"

# Files to download
FILES=(
    "raissi2019_cylinder_nektar_t0_vorticity.mat"
    "raissi2019_cylinder_nektar_wake.mat"
)

# Download each file
for file in "${FILES[@]}"; do
    echo "Downloading $file..."
    curl -L "$BASE_URL/$file" -o "./data/$file"
done

echo "All files downloaded to ./data"
