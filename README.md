# ImageClassifier

A C++ machine learning project for classifying handwritten digits from the MNIST dataset using two approaches:

- **K-Nearest Neighbors (KNN)** - A simple distance-based classifier
- **Neural Network** - A fully connected network implemented with Tiny-dnn

## Features

- Load and preprocess MNIST dataset (60,000 training + 10,000 test images)
- Train and evaluate KNN and Neural Network classifiers
- Display images in ASCII format
- Save and load trained neural network models
- Predict single images from text files
- Display prediction results with confidence

## Project Structure

```
ImageClassifier/
├── src/
│   ├── main.cpp             # Train and evaluate models
│   ├── Predict.cpp          # Predict single image from text file
│   ├── PredictProb.cpp      # Predict with ASCII visualization
│   ├── dataset.cpp          # MNIST dataset loader
│   ├── knn_classifier.cpp   # KNN implementation
│   ├── SimpleNN.cpp         # Neural network implementation
│   └── preprocess.py        # Convert PNG to text file
├── include/
│   ├── dataset.hpp
│   ├── knn_classifier.hpp
│   └── SimpleNN.hpp
├── data/                    # MNIST dataset and input images
├── build/                   # Compiled binaries and saved models
├── tiny-dnn/                # Tiny-dnn library (git submodule)
├── Makefile
├── CMakeLists.txt
└── README.md
```

## Requirements

- C++17 compatible compiler (g++, clang++)
- [Tiny-dnn](https://github.com/tiny-dnn/tiny-dnn) (included as submodule)
- OpenMP (optional, for parallel processing)
- Python 3 with PIL/Pillow and NumPy (for image preprocessing)

## Installation

### 1. Clone the repository

```bash
git clone --recurse-submodules https://github.com/udaykiriti/ImageClassifier.git
cd ImageClassifier
```

If you already cloned without submodules:

```bash
git submodule update --init
```

### 2. Download MNIST dataset

Place the following files in the `data/` folder:

| File                        | Description      |
|-----------------------------|------------------|
| `train-images-idx3-ubyte`   | Training images  |
| `train-labels-idx1-ubyte`   | Training labels  |
| `t10k-images-idx3-ubyte`    | Test images      |
| `t10k-labels-idx1-ubyte`    | Test labels      |

Download from: http://yann.lecun.com/exdb/mnist/

### 3. Build the project

**Using Makefile (recommended):**

```bash
make nn            # Build Neural Network version
make knn           # Build KNN version
make predict       # Build single image predictor
make predictprob   # Build predictor with ASCII output
make clean         # Remove build artifacts
```

**Using CMake:**

```bash
mkdir -p build && cd build
cmake ..
make
```

## Usage

### Train and evaluate

```bash
./build/ImageClassifier
```

This will:
1. Load the MNIST dataset
2. Train the model (or load existing model)
3. Evaluate accuracy on test set
4. Display sample predictions

### Predict a single image

1. Convert your image to text format:

```bash
python src/preprocess.py
```

2. Run prediction:

```bash
./build/ImagePredict
```

### Predict with ASCII visualization

```bash
./build/ImagePredictProb
```

## Example Output

### Training output

```
Dataset module initialized: MNIST Training Data
Loaded 60000 images of size 28x28
Loaded 60000 labels
Training completed.
Accuracy: 94.6%

Sample Predictions:
Test Image 0 - True Label: 7, Predicted: 7
Test Image 1 - True Label: 2, Predicted: 2
Test Image 2 - True Label: 1, Predicted: 1
```

### ASCII visualization

```
Predicted label: 5
True label: 5
Prediction is correct!

............................
...........@@@...#@@#.......
...........#@@...#@@#.......
...........#@@*..#@@#.......
............##...*##........
.............****##.........
............................
```

## How It Works

### KNN Classifier
- Stores all training images in memory
- For each test image, finds K nearest neighbors using Euclidean distance
- Predicts the most common label among neighbors
- Uses OpenMP for parallel distance computation

### Neural Network
- Architecture: 784 -> 128 (ReLU) -> 10 (Softmax)
- Optimizer: Adagrad
- Loss: Mean Squared Error
- Trained for 10 epochs with batch size 32

## License

This project is licensed under the MIT License.

## References

- [MNIST Database](http://yann.lecun.com/exdb/mnist/)
- [Tiny-dnn Library](https://github.com/tiny-dnn/tiny-dnn)
