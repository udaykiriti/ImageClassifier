# ImageClassifier Documentation

This folder contains detailed documentation for the ImageClassifier project.

## Table of Contents

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Project structure and design |
| [API.md](API.md) | Header files and class documentation |
| [BUILD.md](BUILD.md) | Build instructions and Makefile targets |
| [SCRIPTS.md](SCRIPTS.md) | Utility scripts documentation |
| [USAGE.md](USAGE.md) | Command-line usage and examples |

## Quick Start

1. Clone and setup:
   ```bash
   git clone --recurse-submodules https://github.com/udaykiriti/ImageClassifier.git
   cd ImageClassifier
   ./scripts/download_mnist.sh
   ```

2. Build:
   ```bash
   make
   ```

3. Train:
   ```bash
   ./build/train --model nn --train 5000
   ```

4. Predict:
   ```bash
   ./build/predict --show
   ```
