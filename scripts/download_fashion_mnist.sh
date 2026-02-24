#!/bin/bash

# Download Fashion-MNIST dataset
# Usage: ./scripts/download_fashion_mnist.sh

set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Usage: ./scripts/download_fashion_mnist.sh"
    echo "Downloads Fashion-MNIST IDX files into ./data."
    exit 0
fi

DATA_DIR="data"
BASE_URL="https://fashion-mnist.s3-website.eu-central-1.amazonaws.com"

FILES=(
    "train-images-idx3-ubyte.gz"
    "train-labels-idx1-ubyte.gz"
    "t10k-images-idx3-ubyte.gz"
    "t10k-labels-idx1-ubyte.gz"
)

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "Downloading Fashion-MNIST dataset..."

for file in "${FILES[@]}"; do
    if [ ! -f "${file%.gz}" ]; then
        echo "Downloading $file..."
        curl -fsSLO "$BASE_URL/$file"
        echo "Extracting $file..."
        gunzip -f "$file"
    else
        echo "${file%.gz} already exists, skipping."
    fi
done

echo ""
echo "Fashion-MNIST dataset downloaded to $DATA_DIR/"
ls -lh *.ubyte 2>/dev/null || echo "No .ubyte files found"
