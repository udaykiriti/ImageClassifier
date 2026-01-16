#!/bin/bash

# Download MNIST dataset
# Usage: ./scripts/download_mnist.sh

DATA_DIR="data"
BASE_URL="http://yann.lecun.com/exdb/mnist"

FILES=(
    "train-images-idx3-ubyte.gz"
    "train-labels-idx1-ubyte.gz"
    "t10k-images-idx3-ubyte.gz"
    "t10k-labels-idx1-ubyte.gz"
)

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "Downloading MNIST dataset..."

for file in "${FILES[@]}"; do
    if [ ! -f "${file%.gz}" ]; then
        echo "Downloading $file..."
        curl -O "$BASE_URL/$file"
        echo "Extracting $file..."
        gunzip -f "$file"
    else
        echo "${file%.gz} already exists, skipping."
    fi
done

echo ""
echo "MNIST dataset downloaded to $DATA_DIR/"
ls -lh *.ubyte 2>/dev/null || echo "No .ubyte files found"
