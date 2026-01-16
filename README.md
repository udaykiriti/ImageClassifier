# ImageClassifier

A C++ machine learning project for classifying handwritten digits from the MNIST dataset.

## Overview

This project implements two classification algorithms:

- **Neural Network** - A fully connected network (784 -> 128 -> 10) using Tiny-dnn
- **K-Nearest Neighbors (KNN)** - A distance-based classifier with parallel processing

## Features

- MNIST dataset loader with binary file parsing
- ASCII visualization of digit images
- Model persistence (save/load trained models)
- Single image prediction from text files
- OpenMP parallel processing for faster training and inference

## Project Structure

```
ImageClassifier/
├── src/
│   ├── main.cpp              # Train and evaluate models
│   ├── predict.cpp           # Predict single digit
│   ├── predict_visualize.cpp # Predict with ASCII display
│   ├── mnist_loader.cpp      # MNIST binary file loader
│   ├── neural_network.cpp    # Neural network implementation
│   └── knn.cpp               # KNN implementation
├── include/
│   ├── mnist_loader.hpp
│   ├── neural_network.hpp
│   └── knn.hpp
├── data/                     # MNIST dataset files
├── models/                   # Saved model files
├── scripts/
│   └── preprocess.py         # Image to text converter
├── tiny-dnn/                 # Tiny-dnn library (submodule)
├── Makefile
├── CMakeLists.txt
└── README.md
```

## Requirements

- C++17 compiler (g++, clang++)
- Tiny-dnn (included as git submodule)
- OpenMP (optional)
- Python 3 with PIL and NumPy (for preprocessing)

## Installation

```bash
git clone --recurse-submodules https://github.com/udaykiriti/ImageClassifier.git
cd ImageClassifier
```

If already cloned:

```bash
git submodule update --init
```

## Dataset

Download MNIST dataset files and place in `data/`:

| File | Description |
|------|-------------|
| train-images-idx3-ubyte | 60,000 training images |
| train-labels-idx1-ubyte | Training labels |
| t10k-images-idx3-ubyte | 10,000 test images |
| t10k-labels-idx1-ubyte | Test labels |

Download: http://yann.lecun.com/exdb/mnist/

## Build

```bash
make nn        # Neural network classifier
make knn       # KNN classifier
make predict   # Single image predictor
make visualize # Predictor with ASCII output
make clean     # Remove build files
```

Or with CMake:

```bash
mkdir build && cd build
cmake ..
make
```

## Usage

### Train and Evaluate

```bash
./build/classifier
```

Output:
```
MNIST loader initialized: MNIST Training
Loaded 60000 images (28x28)
Loaded 60000 labels
Training completed.
Accuracy: 94.6%

Sample Predictions:
--------------------
Image 0 | True: 7 | Predicted: 7 [OK]
```

### Predict Single Image

1. Convert image to text:
```bash
python scripts/preprocess.py path/to/digit.png data/image.txt
```

2. Run prediction:
```bash
./build/predict
```

### Predict with Visualization

```bash
./build/visualize 5  # Optional: pass true label
```

Output:
```
ASCII Image:
............................
...........@@@...#@@#.......
...........#@@...#@@#.......

Predicted digit: 5
True label: 5
CORRECT
```

## Architecture

### Neural Network
- Input: 784 neurons (28x28 pixels)
- Hidden: 128 neurons with ReLU activation
- Output: 10 neurons with Softmax
- Optimizer: Adagrad
- Loss: MSE

### KNN
- Distance metric: Euclidean
- Default K: 3
- Parallelized with OpenMP

## License

MIT License

## References

- [MNIST Database](http://yann.lecun.com/exdb/mnist/)
- [Tiny-dnn](https://github.com/tiny-dnn/tiny-dnn)
