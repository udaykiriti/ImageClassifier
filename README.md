# ImageClassifier

A C++ machine learning project for classifying handwritten digits from the MNIST dataset.

## Features

- Neural Network classifier (784 -> 128 -> 10) using Tiny-dnn
- K-Nearest Neighbors (KNN) classifier with OpenMP parallelization
- Command-line interface with configurable parameters
- ASCII visualization of digit images
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
│       └── dataset.cpp     # MNIST data loading
├── include/                # Header files
│   ├── types.hpp           # Common types and constants
│   ├── classifier.hpp      # Classifier interface
│   ├── dataset.hpp         # Dataset class
│   ├── neural_net.hpp      # Neural network classifier
│   └── knn.hpp             # KNN classifier
├── scripts/                # Utility scripts
│   ├── download_mnist.sh   # Download MNIST dataset
│   └── preprocess.py       # Image preprocessing
├── data/                   # MNIST dataset files
├── models/                 # Saved model files
├── tiny-dnn/               # Tiny-dnn library (submodule)
├── Makefile
├── CMakeLists.txt
├── LICENSE
└── README.md
```

## Requirements

- C++17 compiler (g++, clang++)
- Tiny-dnn (included as git submodule)
- OpenMP (optional, for parallel processing)
- Python 3 with PIL and NumPy (for image preprocessing)

## Installation

```bash
git clone --recurse-submodules https://github.com/udaykiriti/ImageClassifier.git
cd ImageClassifier
```

Download MNIST dataset:

```bash
./scripts/download_mnist.sh
```

## Build

```bash
make train      # Build training app
make predict    # Build prediction app
make            # Build all
make clean      # Clean build files
```

## Usage

### Training

```bash
# Neural Network (default)
./build/train

# Neural Network with custom parameters
./build/train --model nn --train 5000 --test 1000 --epochs 15

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
| --k | KNN neighbors | 3 |

### Prediction

```bash
# Basic prediction
./build/predict

# With visualization
./build/predict --show

# With true label comparison
./build/predict --image data/image.txt --label 5 --show
```

Options:
| Flag | Description | Default |
|------|-------------|---------|
| --image | Image file path | ./data/image.txt |
| --model | Model file path | ./models/neural_net.model |
| --label | True label for comparison | - |
| --show | Show ASCII visualization | false |

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
Image 0 | True: 7 | Predicted: 7 [OK]
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
- Optimizer: Adagrad
- Loss: MSE

### KNN
- Distance: Euclidean
- Voting: Majority
- Parallelized with OpenMP

## License

MIT License - see [LICENSE](LICENSE)

## References

- [MNIST Database](http://yann.lecun.com/exdb/mnist/)
- [Tiny-dnn](https://github.com/tiny-dnn/tiny-dnn)
