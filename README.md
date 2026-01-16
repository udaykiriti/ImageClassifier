# ![ImageClassifier](https://img.shields.io/badge/ImageClassifier-MNIST-blue) ImageClassifier

**ImageClassifier** is a C++ project for classifying **MNIST dataset** using:

- **K-Nearest Neighbors (KNN)**
- **Simple Neural Network** implemented with **Tiny-dnn**

---

## Flow...

- Load and preprocess MNIST dataset

- Train and evaluate:

  - KNN classifier
  - Neural Network classifier

- Display images in ASCII format

- Evaluate accuracy on test dataset

- Save and load trained neural network models

- Predict single images from text files

- Display true and predicted labels

---

## Project Structure

```
ImageClassifier/
├─ src/                  # Source code
│  ├─ main.cpp           # Entry point (train models)
│  ├─ Predict.cpp        # Predict single image from text file
│  ├─ PredictProb.cpp    # Predict and display true label with ASCII image
│  ├─ dataset.cpp        # MNIST dataset loader
│  ├─ knn_classifier.cpp # KNN implementation
│  ├─ SimpleNN.cpp       # Neural network (Tiny-dnn)
│  ├─ preprocess.py      # Convert image to text file
├─ include/              # Header files
│  ├─ dataset.hpp
│  ├─ knn_classifier.hpp
│  ├─ SimpleNN.hpp
├─ data/                 # MNIST dataset files / image.txt
├─ build/                # Compiled executable output / saved model
├─ tiny-dnn/             # Tiny-dnn library (git submodule)
└─ README.md             # Project documentation
```

---

## Libraries

- C++17 compatible compiler (e.g., `g++`)
- [Tiny-dnn](https://github.com/tiny-dnn/tiny-dnn) library (for neural network)
- OpenMP (optional, for multi-threading)

---

## Installation & Compilation

### 1. Clone the repository

```bash
git clone "https://github.com/udaykiriti/ImageClassifier.git"
cd ImageClassifier
```

### 2. Initialize submodules

```bash
git submodule update --init
```

### 3. Compile

**Using Makefile (recommended):**

```bash
make nn          # Neural Network version
make knn         # KNN version
make predict     # Predict single image
make predictprob # Predict with ASCII output
```

**Or using CMake:**

```bash
mkdir -p build && cd build
cmake ..
make
```

**Or manually:**

**KNN version:**

```bash
g++ src/main.cpp src/dataset.cpp src/knn_classifier.cpp -Iinclude -O2 -std=c++17 -fopenmp -o build/ImageClassifier
./build/ImageClassifier
```

**Neural Network version:**

```bash
g++ src/main.cpp src/dataset.cpp src/SimpleNN.cpp -Itiny-dnn -Iinclude -O2 -std=c++17 -fopenmp -o build/ImageClassifier
./build/ImageClassifier
```

**Predict single image:**

```bash
g++ src/Predict.cpp src/SimpleNN.cpp src/dataset.cpp -Itiny-dnn -Iinclude -O2 -std=c++17 -fopenmp -o build/ImagePredict
./build/ImagePredict
```

**Predict with true label and ASCII image:**

```bash
g++ src/PredictProb.cpp src/SimpleNN.cpp src/dataset.cpp -Itiny-dnn -Iinclude -O2 -std=c++17 -fopenmp -o build/ImagePredictProb
./build/ImagePredictProb
```

---

## Dataset

The project uses the **MNIST dataset** of handwritten digits. Place the dataset files in the `data/` folder:

| File                      | Description     |
| ------------------------- | --------------- |
| `train-images-idx3-ubyte` | Training images |
| `train-labels-idx1-ubyte` | Training labels |
| `t10k-images-idx3-ubyte`  | Test images     |
| `t10k-labels-idx1-ubyte`  | Test labels     |

For `Predict.cpp` and `PredictProb.cpp`, images must be converted to **text files** of 784 normalized pixel values (0-255).

---

## Usage

1. Compile as described above.
2. Train the model using `main.cpp`.
3. Predict a single image:

```bash
./build/ImagePredict
```

4. Predict with true label and ASCII image:

```bash
./build/ImagePredictProb
```

**Example Output (`PredictProb.exe`):**

```
Predicted label: 5
True label: 5
Prediction is correct!

ASCII Image:
............................
...........@@@...#@@#.......
...........#@@...#@@#.......
...........#@@*..#@@#.......
............##...*##........
.............****##.........
............................
```

---

## Example Output

```
Loading MNIST dataset...
Training KNN classifier...
Accuracy on test set: 95.2%
Sample predictions:
Image 0: 7, Predicted: 7
Image 1: 2, Predicted: 2
...
```

---

## License

This project is licensed under the **MIT License**.

---

## References

- [Tiny-dnn library](https://github.com/tiny-dnn/tiny-dnn)

