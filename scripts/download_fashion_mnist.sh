#!/bin/bash

# Download Fashion-MNIST dataset
# Usage: ./scripts/download_fashion_mnist.sh

set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Usage: ./scripts/download_fashion_mnist.sh"
    echo "Downloads Fashion-MNIST IDX files into ./data."
    exit 0
fi

DATA_DIR="$(dirname "$0")/../data"
RETRIES=8
MIRRORS=(
    "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion"
    "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion"
    "https://fashion-mnist.s3-website.eu-central-1.amazonaws.com"
    "https://ossci-datasets.s3.amazonaws.com/fashion-mnist"
)

FILES=(
    "train-images-idx3-ubyte.gz"
    "train-labels-idx1-ubyte.gz"
    "t10k-images-idx3-ubyte.gz"
    "t10k-labels-idx1-ubyte.gz"
)

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "Downloading Fashion-MNIST dataset..."

download_with_fallback() {
    local file="$1"

    for base_url in "${MIRRORS[@]}"; do
        echo "Trying mirror: $base_url"
        if curl -fL --retry "$RETRIES" --retry-all-errors --retry-delay 2 \
            --connect-timeout 20 --continue-at - -o "$file" "$base_url/$file"; then
            return 0
        fi
        echo "Mirror failed for $file: $base_url"
    done

    return 1
}

for file in "${FILES[@]}"; do
    if [ ! -f "${file%.gz}" ]; then
        echo "Downloading $file..."
        if ! download_with_fallback "$file"; then
            echo "Failed to download $file from all mirrors." >&2
            exit 1
        fi
        echo "Extracting $file..."
        gunzip -f "$file"
    else
        echo "${file%.gz} already exists, skipping."
    fi
done

echo ""
echo "Fashion-MNIST dataset downloaded to $DATA_DIR/"
ls -lh *ubyte 2>/dev/null || echo "No ubyte files found"
