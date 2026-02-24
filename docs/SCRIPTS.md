# Scripts Documentation

This document describes all utility scripts in the `scripts/` folder.

---

## Overview

| Script | Language | Purpose |
|--------|----------|---------|
| download_fashion_mnist.sh | Bash | Download Fashion-MNIST dataset |
| preprocess.py | Python | Convert images to text format |

---

## download_fashion_mnist.sh

Downloads the Fashion-MNIST dataset from the official source.

### Location

```
scripts/download_fashion_mnist.sh
```

### Usage

```bash
./scripts/download_fashion_mnist.sh
```

### What It Does

1. Creates `data/` directory if not exists
2. Downloads 4 gzipped files from the Fashion-MNIST mirror
3. Extracts files to `data/` directory
4. Skips files that already exist

### Files Downloaded

| File | Size | Description |
|------|------|-------------|
| train-images-idx3-ubyte.gz | 9.9 MB | 60,000 training images |
| train-labels-idx1-ubyte.gz | 29 KB | Training labels |
| t10k-images-idx3-ubyte.gz | 1.6 MB | 10,000 test images |
| t10k-labels-idx1-ubyte.gz | 5 KB | Test labels |

### Source Code

```bash
#!/bin/bash

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
```

### Requirements

- curl (for downloading)
- gunzip (for extraction)

### Example Output

```
Downloading Fashion-MNIST dataset...
Downloading train-images-idx3-ubyte.gz...
Extracting train-images-idx3-ubyte.gz...
Downloading train-labels-idx1-ubyte.gz...
Extracting train-labels-idx1-ubyte.gz...
Downloading t10k-images-idx3-ubyte.gz...
Extracting t10k-images-idx3-ubyte.gz...
Downloading t10k-labels-idx1-ubyte.gz...
Extracting t10k-labels-idx1-ubyte.gz...

Fashion-MNIST dataset downloaded to data/
-rw-r--r-- 1 user user 45M Jan 16 10:00 train-images-idx3-ubyte
-rw-r--r-- 1 user user 59K Jan 16 10:00 train-labels-idx1-ubyte
-rw-r--r-- 1 user user 7.5M Jan 16 10:00 t10k-images-idx3-ubyte
-rw-r--r-- 1 user user 9.8K Jan 16 10:00 t10k-labels-idx1-ubyte
```

### Troubleshooting

#### Permission denied

```bash
chmod +x scripts/download_fashion_mnist.sh
```

#### curl not found

Ubuntu/Debian:
```bash
sudo apt install curl
```

macOS:
```bash
brew install curl
```

#### Connection timeout

The Fashion-MNIST mirror may be slow. Try running the script again or download manually from:
https://github.com/zalandoresearch/fashion-mnist

---

## preprocess.py

Converts image files to text format for prediction.

### Location

```
scripts/preprocess.py
```

### Usage

```bash
# Default: data/image.png -> data/image.txt
python scripts/preprocess.py

# Custom input
python scripts/preprocess.py path/to/fashion_item.png

# Custom input and output
python scripts/preprocess.py input.png output.txt
```

### What It Does

1. Loads input image (any format supported by PIL)
2. Converts to grayscale
3. Resizes to 28x28 pixels
4. Flattens to 784 values
5. Saves as space-separated integers (0-255)

### Source Code

```python
#!/usr/bin/env python3
"""
Convert an image file to a text file of pixel values for class prediction.
Usage: python scripts/preprocess.py [input_image] [output_file]
"""

import sys
from PIL import Image
import numpy as np

def convert_image(input_path="data/image.png", output_path="data/image.txt"):
    img = Image.open(input_path).convert("L")
    img = img.resize((28, 28))
    pixels = np.array(img).flatten()
    np.savetxt(output_path, pixels, fmt="%d")
    print(f"Converted: {input_path} -> {output_path}")
    print(f"Shape: 28x28 = 784 pixels")

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        convert_image(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 2:
        convert_image(sys.argv[1])
    else:
        convert_image()
```

### Requirements

- Python 3.x
- PIL/Pillow
- NumPy

### Installation

```bash
pip install pillow numpy
```

### Input Format

Any image format supported by PIL:
- PNG
- JPEG
- BMP
- GIF
- TIFF

### Output Format

Text file with 784 space-separated integers:

```
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 128 255 255 128 0 0 0 0 0 0 0 0 0 0 0
...
```

Each value represents pixel intensity:
- 0 = Black (background)
- 255 = White (foreground)

### Example Workflow

1. Create or obtain a item image (e.g., sample fashion item)

2. Convert to text:
   ```bash
   python scripts/preprocess.py my_item.png data/image.txt
   ```

3. Run prediction:
   ```bash
   ./build/predict --show
   ```

### Tips for Best Results

1. Use high contrast images (black item on white background)
2. Center the fashion item in the image
3. Use square images for best aspect ratio
4. For dark backgrounds, invert the image first

### Troubleshooting

#### ModuleNotFoundError: No module named 'PIL'

```bash
pip install pillow
```

#### Image too small/large

The script automatically resizes to 28x28. Original size does not matter.

#### Wrong colors (inverted)

Fashion-MNIST uses white items on black background. If your image is inverted:

```python
# Add this line after loading
img = Image.open(input_path).convert("L")
img = ImageOps.invert(img)  # Add this line
```

---

## Creating Custom Scripts

### Template for New Scripts

Bash script template:
```bash
#!/bin/bash
# Description: [what this script does]
# Usage: ./scripts/script_name.sh [arguments]

set -e  # Exit on error

# Your code here
```

Python script template:
```python
#!/usr/bin/env python3
"""
Description: [what this script does]
Usage: python scripts/script_name.py [arguments]
"""

import sys

def main():
    # Your code here
    pass

if __name__ == "__main__":
    main()
```

### Adding to Repository

1. Create script in `scripts/` directory
2. Make executable: `chmod +x scripts/your_script.sh`
3. Add documentation to this file
4. Update README if user-facing
