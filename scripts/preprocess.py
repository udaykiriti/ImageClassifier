#!/usr/bin/env python3
"""
Convert an image file to a text file of pixel values for Fashion-MNIST prediction.
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
