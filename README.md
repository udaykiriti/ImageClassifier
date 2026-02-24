# ImageClassifier

A C++17 Fashion-MNIST classifier project with:
- an in-house neural network implementation,  
- a KNN baseline with OpenMP acceleration, and  
- both CLI and interactive TUI workflows.

## Highlights

- In-house neural network (no external ML runtime)
- KNN with distance-weighted voting (optional uniform voting)
- Robust dataset/model/input validation
- Interactive terminal UI (`./build/tui`)
- Top-k probability output for predictions

## Project Layout

```text
ImageClassifier/
├── src/
│   ├── apps/               # train / predict / tui entrypoints
│   ├── classifiers/        # neural_net + knn implementations
│   └── core/               # dataset loader + classifier base logic
├── include/                # public headers
├── scripts/
│   ├── download_fashion_mnist.sh
│   └── preprocess.py
├── data/                   # local datasets and sample inputs
├── models/                 # generated model files
├── Makefile
├── CMakeLists.txt
└── README.md
```

## Requirements

- C++17 compiler (`g++` or `clang++`)
- OpenMP (optional but recommended for KNN)
- Python 3 + Pillow + NumPy (required only when using non-`.txt` image inputs like `.png`/`.jpg`)

## Quick Start

```bash
git clone https://github.com/udaykiriti/ImageClassifier.git
cd ImageClassifier
./scripts/download_fashion_mnist.sh
make -j4
./build/tui
```

## Build

```bash
make train
make predict
make tui
make            # builds all
make clean      # remove build outputs and generated models
```

## Usage

### 1. Train / Evaluate

```bash
# Neural network (default)
./build/train

# Neural network with explicit hyperparameters
./build/train --model nn --train 5000 --test 1000 --epochs 15 --batch 64 --hidden 256 --lr 0.01 --l2 0.0001 --seed 42

# KNN
./build/train --model knn --train 2000 --test 500 --k 5

# KNN with uniform voting
./build/train --model knn --train 2000 --test 500 --k 5 --uniform-knn
```

Train flags:

| Flag | Description | Default |
|---|---|---|
| `--model` | `nn` or `knn` | `nn` |
| `--train` | Training samples | `2000` |
| `--test` | Test samples | `500` |
| `--epochs` | NN epochs | `10` |
| `--batch` | NN batch size | `32` |
| `--hidden` | NN hidden size | `128` |
| `--lr` | NN learning rate | `0.01` |
| `--l2` | NN L2 regularization | `1e-4` |
| `--seed` | NN random seed | `42` |
| `--k` | KNN neighbors | `3` |
| `--uniform-knn` | Use uniform voting (disable distance weighting) | off |

### 2. Predict

```bash
# Basic prediction
./build/predict

# Show ASCII image
./build/predict --image data/image.txt --show

# Predict directly from PNG/JPG (auto-preprocessed)
./build/predict --image data/image.png --topk 3

# Compare against known class
./build/predict --image data/image.txt --label 8 --show

# Show top-k probabilities
./build/predict --image data/image.txt --topk 3
```

Predict flags:

| Flag | Description | Default |
|---|---|---|
| `--image` | Input path (`.txt` or image file like `.png/.jpg`) | `./data/image.txt` |
| `--model` | Model file path | `./models/neural_net.model` |
| `--label` | True class id (0-9) | omitted |
| `--topk` | Show top-k probabilities | `0` |
| `--show` | Print ASCII image | off |

Image text format:
- exactly 784 values (28x28)
- values accepted in either `0..255` or normalized `0..1`
- non-`.txt` files are auto-preprocessed via `scripts/preprocess.py`
  - automatically inverts colors if light background is detected (Fashion-MNIST is white-on-black)
- if preprocessing dependencies are missing: `pip install pillow numpy`

### 3. Interactive TUI

```bash
./build/tui
```

Menu-driven options:
- Train / Evaluate model
- Predict from image
- Show class labels

## Fashion-MNIST Class Map

| ID | Class |
|---|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

## Notes

- Datasets and trained models are not tracked as source artifacts.
- If `predict` reports missing/unsupported model, run `./build/train --model nn ...` once to regenerate `models/neural_net.model`.
- If `predict` results are poor for your own images, ensure the background is dark (or rely on auto-inversion in `preprocess.py`).
- If dataset files are missing, run `./scripts/download_fashion_mnist.sh`.

## License

MIT. See [LICENSE](LICENSE).

## Reference

- Fashion-MNIST: https://github.com/zalandoresearch/fashion-mnist
