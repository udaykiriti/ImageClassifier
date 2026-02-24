# Usage Guide

This document covers command-line usage for all executables.

---

## Executables

| Binary | Purpose | Location |
|--------|---------|----------|
| train | Train and evaluate classifiers | build/train |
| predict | Predict class from image file | build/predict |

---

## train

Training and evaluation application for neural network and KNN classifiers.

### Basic Usage

```bash
# Train with defaults (Neural Network)
./build/train

# Train KNN
./build/train --model knn
```

### Command-Line Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| --model | string | nn | Classifier type: "nn" or "knn" |
| --train | integer | 2000 | Number of training samples |
| --test | integer | 500 | Number of test samples |
| --epochs | integer | 10 | Training epochs (NN only) |
| --k | integer | 3 | Number of neighbors (KNN only) |
| --help | flag | - | Show help message |

### Examples

#### Train Neural Network

```bash
# Default settings
./build/train

# More training data and epochs
./build/train --model nn --train 10000 --test 2000 --epochs 20

# Quick test
./build/train --train 500 --test 100 --epochs 5
```

#### Train KNN

```bash
# Default k=3
./build/train --model knn

# Custom k value
./build/train --model knn --k 5 --train 2000 --test 500

# Larger training set for better accuracy
./build/train --model knn --k 7 --train 5000 --test 1000
```

### Output Format

```
Loaded 60000 images (28x28)
Loaded 10000 images (28x28)

[NeuralNet] Training with 2000 samples...

Training completed (10 epochs)
Saved model: ./models/neural_net.model

Accuracy: 94.6%

Sample Predictions:
----------------------------------------
Image 0 | True: 7 (Sneaker) | Predicted: 7 (Sneaker) [OK]
............................
..............@@@#..........
.............@@@@@..........
............@@@@@@..........
...
----------------------------------------
Image 1 | True: 2 (Pullover) | Predicted: 2 (Pullover) [OK]
...
```

### Model Persistence

- Neural Network models are saved to `./models/neural_net.model`
- If model exists, it is loaded instead of training
- Delete model file to retrain: `rm models/neural_net.model`
- KNN does not save models (stores training data in memory)

---

## predict

Prediction application for single images.

### Basic Usage

```bash
# Predict with defaults
./build/predict

# Show ASCII visualization
./build/predict --show

# Compare with known label
./build/predict --label 5 --show
```

### Command-Line Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| --image | string | ./data/image.txt | Path to image text file |
| --model | string | ./models/neural_net.model | Path to trained model |
| --label | integer | -1 | True class for comparison |
| --show | flag | false | Display ASCII visualization |
| --help | flag | - | Show help message |

### Examples

#### Basic Prediction

```bash
./build/predict
```

Output:
```
Predicted class: 7 (Sneaker)
```

#### With Visualization

```bash
./build/predict --show
```

Output:
```
ASCII Image:
............................
..............@@@#..........
.............@@@@@..........
............@@@@@@..........
...

Predicted class: 7 (Sneaker)
```

#### With Label Comparison

```bash
./build/predict --label 7 --show
```

Output:
```
ASCII Image:
...

Predicted class: 7 (Sneaker)
True class: 7 (Sneaker)
CORRECT
```

#### Custom Image and Model

```bash
./build/predict --image data/my_item.txt --model models/my_model.model --show
```

### Image File Format

The image text file must contain:
- Exactly 784 values (28x28 pixels)
- Values in range 0-255 or already-normalized 0-1
- Space or newline separated
- Row-major order (left to right, top to bottom)

Example (simplified, actual file has 784 values):
```
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 128 255 255 128 0 0 0 0 0 0 0 0 0 0 0
...
```

### Creating Image Files

#### Method 1: Using preprocess.py

```bash
python scripts/preprocess.py input.png data/image.txt
./build/predict --show
```

#### Method 2: Extract from Fashion-MNIST

Write a simple program to extract and save test images:

```cpp
Dataset test("./data");
test.loadTest();

// Save first image
std::ofstream out("data/image.txt");
for (double val : test.images()[0]) {
    out << static_cast<int>(val * 255) << " ";
}
```

---

## Workflow Examples

### Complete Training Workflow

```bash
# 1. Setup
git clone https://github.com/udaykiriti/ImageClassifier.git
cd ImageClassifier

# 2. Download dataset
./scripts/download_fashion_mnist.sh

# 3. Build
make

# 4. Train neural network
./build/train --model nn --train 10000 --epochs 15

# 5. Check accuracy
# (Accuracy displayed after training)
```

### Complete Prediction Workflow

```bash
# 1. Prepare image
python scripts/preprocess.py my_fashion_item.png data/image.txt

# 2. Run prediction
./build/predict --show --label 7
```

### Comparing Classifiers

```bash
# Train and evaluate Neural Network
./build/train --model nn --train 5000 --test 1000

# Train and evaluate KNN
./build/train --model knn --train 5000 --test 1000 --k 5

# Compare accuracy results
```

---

## Performance Tips

### Neural Network

| Setting | Effect |
|---------|--------|
| More epochs | Higher accuracy, longer training |
| More training data | Higher accuracy, longer training |
| Batch size 32-64 | Good balance of speed and accuracy |

Recommended for best accuracy:
```bash
./build/train --train 60000 --test 10000 --epochs 20
```

### KNN

| Setting | Effect |
|---------|--------|
| Higher k | Smoother decision boundary |
| More training data | Higher accuracy, slower prediction |
| k = 3-7 | Good default range |

Recommended for best accuracy:
```bash
./build/train --model knn --train 10000 --k 5
```

Note: KNN prediction is slow with large training sets.

---

## Error Messages

### Dataset not found

```
Cannot open: ./data/train-images-idx3-ubyte
Failed to load dataset
```

Solution: Download Fashion-MNIST dataset
```bash
./scripts/download_fashion_mnist.sh
```

### Model not found

```
Cannot open: ./models/neural_net.model
```

Solution: Train the model first
```bash
./build/train
```

### Invalid image size

```
Error: Expected 784 pixels, got 100
```

Solution: Ensure image file has exactly 784 values

### Cannot open image file

```
Error: Cannot open ./data/image.txt
```

Solution: Create image file using preprocess.py
```bash
python scripts/preprocess.py input.png data/image.txt
```
