# Project Architecture

## Directory Structure

```
ImageClassifier/
├── src/                    # Source code
│   ├── apps/               # Application entry points
│   │   ├── train.cpp       # Training and evaluation app
│   │   └── predict.cpp     # Prediction app
│   ├── classifiers/        # Classifier implementations
│   │   ├── neural_net.cpp  # Neural network implementation
│   │   └── knn.cpp         # K-Nearest Neighbors implementation
│   └── core/               # Core functionality
│       ├── classifier.cpp  # Base classifier implementation
│       └── dataset.cpp     # Fashion-MNIST dataset loader
├── include/                # Header files
│   ├── types.hpp           # Common types and constants
│   ├── classifier.hpp      # Abstract classifier interface
│   ├── dataset.hpp         # Dataset class declaration
│   ├── neural_net.hpp      # Neural network class declaration
│   └── knn.hpp             # KNN class declaration
├── scripts/                # Utility scripts
│   ├── download_fashion_mnist.sh   # Fashion-MNIST dataset downloader
│   └── preprocess.py       # Image preprocessing utility
├── data/                   # Dataset files
│   ├── train-images-idx3-ubyte
│   ├── train-labels-idx1-ubyte
│   ├── t10k-images-idx3-ubyte
│   ├── t10k-labels-idx1-ubyte
│   └── image.txt           # Custom image for prediction
├── models/                 # Saved model files
│   └── neural_net.model    # Trained neural network
├── build/                  # Compiled binaries
│   ├── train               # Training executable
│   └── predict             # Prediction executable
├── docs/                   # Documentation
├── Makefile                # Build configuration
├── CMakeLists.txt          # CMake configuration
├── LICENSE                 # MIT License
└── README.md               # Project overview
```

## Design Patterns

### 1. Abstract Interface Pattern

The `Classifier` class provides an abstract interface that all classifiers implement:

```
          +----------------+
          |   Classifier   |  (Abstract Base Class)
          +----------------+
          | + train()      |
          | + predict()    |
          | + evaluate()   |
          | + save()       |
          | + load()       |
          +----------------+
                 ^
                 |
        +--------+--------+
        |                 |
+---------------+  +---------------+
|   NeuralNet   |  |      KNN      |
+---------------+  +---------------+
| - network_    |  | - k_          |
| - epochs_     |  | - train_data_ |
+---------------+  +---------------+
```

### 2. Separation of Concerns

| Layer | Purpose | Files |
|-------|---------|-------|
| Apps | User interface, CLI parsing | src/apps/*.cpp |
| Classifiers | ML algorithm implementations | src/classifiers/*.cpp |
| Core | Data loading, base classes | src/core/*.cpp |
| Headers | Type definitions, interfaces | include/*.hpp |

### 3. Namespace Organization

All project code is contained in the `mnist` namespace:

```cpp
namespace mnist {
    // Types
    using Image = std::vector<double>;
    using ImageSet = std::vector<Image>;
    using Labels = std::vector<int>;

    // Classes
    class Classifier;
    class Dataset;
    class NeuralNet;
    class KNN;
}
```

## Data Flow

### Training Flow

```
1. Load Dataset
   Dataset::loadTraining() -> images_, labels_

2. Create Classifier
   NeuralNet(epochs, batch_size) or KNN(k)

3. Train Model
   classifier->train(images, labels)

4. Evaluate
   classifier->evaluate(test_images, test_labels)

5. Save Model (Neural Network only)
   classifier->save("./models/neural_net.model")
```

### Prediction Flow

```
1. Load Image
   loadImageFromFile("./data/image.txt") -> Image

2. Load Model
   model.load("./models/neural_net.model")

3. Predict
   model.predict(image) -> int (0-9)

4. Display Result
   Dataset::printImage(image) (optional)
```

## Dependencies

### External Libraries

| Library | Purpose | Location |
|---------|---------|----------|
| OpenMP | Parallel processing for KNN | System library |

### Standard Library Headers

| Header | Used For |
|--------|----------|
| vector | Image and label storage |
| string | File paths, names |
| fstream | File I/O |
| algorithm | min, max, distance |
| memory | unique_ptr for polymorphism |
| cstdint | uint32_t for Fashion-MNIST format |

## File Formats

### Fashion-MNIST Binary Format

Images file header:
```
[4 bytes] Magic number (2051)
[4 bytes] Number of images
[4 bytes] Number of rows (28)
[4 bytes] Number of columns (28)
[n * 784 bytes] Pixel data (0-255)
```

Labels file header:
```
[4 bytes] Magic number (2049)
[4 bytes] Number of labels
[n bytes] Label data (0-9)
```

### Image Text Format

For custom image prediction:
```
784 space-separated integers (0-255)
Each value represents one pixel
Row-major order (left to right, top to bottom)
```

### Model Format

Neural network models are saved using the project's custom binary format.
Contains network architecture and trained weights.
