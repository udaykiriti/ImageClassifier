# ImageClassifier

A C++ machine learning project for classifying fashion items from the Fashion-MNIST dataset.

## Features

- In-house Neural Network classifier (784 -> 128 -> 10)
- K-Nearest Neighbors (KNN) classifier with OpenMP parallelization
- Command-line interface with configurable parameters
- Strong input validation for dataset files and CLI arguments
- ASCII visualization of item images
- Model save/load functionality

## Project Structure

```
ImageClassifier/
├── src/
│   ├── apps/               # Application entry points
│   │   ├── train.cpp       # Training and evaluation
│   │   └── predict.cpp     # Single image prediction
│   ├── classifiers/        # Classifier implementations
│   │   ├── neural_net.cpp
│   │   └── knn.cpp
│   └── core/               # Core functionality
│       ├── classifier.cpp  # Base classifier interface
│       └── dataset.cpp     # Fashion-MNIST data loading
├── include/                # Header files
│   ├── types.hpp           # Common types and constants
│   ├── classifier.hpp      # Classifier interface
│   ├── dataset.hpp         # Dataset class
│   ├── neural_net.hpp      # Neural network classifier
│   └── knn.hpp             # KNN classifier
├── scripts/                # Utility scripts
│   ├── download_fashion_mnist.sh   # Download Fashion-MNIST dataset
│   └── preprocess.py       # Image preprocessing
├── data/                   # Fashion-MNIST dataset files
├── models/                 # Saved model files
├── Makefile
├── CMakeLists.txt
├── LICENSE
└── README.md
```

## Requirements

- C++17 compiler (g++, clang++)
- OpenMP (optional, for parallel processing)
- Python 3 with PIL and NumPy (for image preprocessing)

## Installation

```bash
git clone https://github.com/udaykiriti/ImageClassifier.git
cd ImageClassifier
```

Download Fashion-MNIST dataset:

```bash
./scripts/download_fashion_mnist.sh
```

## Build

```bash
make train      # Build training app
make predict    # Build prediction app
make tui        # Build interactive terminal UI
make            # Build all
make clean      # Clean build files
```

## Usage

### Training

```bash
# Neural Network (default)
./build/train

# Neural Network with custom parameters
./build/train --model nn --train 5000 --test 1000 --epochs 15 --batch 64 --hidden 256 --lr 0.01 --l2 0.0001 --seed 42

# KNN classifier
./build/train --model knn --train 1000 --test 200 --k 5
```

Options:
| Flag | Description | Default |
|------|-------------|---------|
| --model | Classifier type (nn/knn) | nn |
| --train | Training samples | 2000 |
| --test | Test samples | 500 |
| --epochs | NN training epochs | 10 |
| --batch | NN batch size | 32 |
| --hidden | NN hidden layer size | 128 |
| --lr | NN learning rate | 0.01 |
| --l2 | NN L2 regularization | 1e-4 |
| --seed | NN random seed | 42 |
| --k | KNN neighbors | 3 |
| --uniform-knn | Use uniform voting for KNN (otherwise distance-weighted) | false |

### Prediction

```bash
# Basic prediction
./build/predict

# With visualization
./build/predict --show

# With true class comparison
./build/predict --image data/image.txt --label 5 --show

# Show top-3 probabilities
./build/predict --image data/image.txt --topk 3
```

Options:
| Flag | Description | Default |
|------|-------------|---------|
| --image | Image file path | ./data/image.txt |
| --model | Model file path | ./models/neural_net.model |
| --label | True class for comparison | - |
| --topk | Show top-k class probabilities | 0 |
| --show | Show ASCII visualization | false |

`--image` accepts text pixels in either `0..255` or already-normalized `0..1` format.

### Interactive TUI

```bash
./build/tui
```

Menu options:
- Train/Evaluate model
- Predict from image
- Show Fashion-MNIST class labels

### Image Preprocessing

Convert a PNG image to text format:

```bash
python scripts/preprocess.py input.png data/image.txt
```

## Example Output

```
[NeuralNet] Training with 2000 samples...
Training completed (10 epochs)

Accuracy: 94.6%

Sample Predictions:
----------------------------------------
Image 0 | True: 7 (Sneaker) | Predicted: 7 (Sneaker) [OK]
............................
..............@@@#..........
.............@@@@@..........
............@@@@@@..........
...........@@@@@@@..........
```

## Architecture

### Neural Network
- Input: 784 neurons (28x28 pixels)
- Hidden: 128 neurons (ReLU)
- Output: 10 neurons (Softmax)
- Optimizer: Mini-batch SGD with learning-rate decay
- Loss: Cross-entropy
- Regularization: L2 weight decay

### KNN
- Distance: Euclidean
- Voting: Distance-weighted by default (or uniform with `--uniform-knn`)
- Parallelized with OpenMP

## License

MIT License - see [LICENSE](LICENSE)

## References

- [Fashion-MNIST Database](https://github.com/zalandoresearch/fashion-mnist)
