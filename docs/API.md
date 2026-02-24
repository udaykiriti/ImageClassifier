# API Documentation

This document describes all header files and their classes, methods, and types.

---

## types.hpp

Common types and constants used throughout the project.

### Namespace: mnist

#### Type Aliases

| Type | Definition | Description |
|------|------------|-------------|
| `Image` | `std::vector<double>` | Single 28x28 image as 784 normalized values (0.0-1.0) |
| `ImageSet` | `std::vector<Image>` | Collection of images |
| `Labels` | `std::vector<int>` | Collection of labels (0-9) |

#### Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `IMAGE_SIZE` | 28 | Width and height of Fashion-MNIST images |
| `IMAGE_PIXELS` | 784 | Total pixels per image (28 * 28) |
| `NUM_CLASSES` | 10 | Number of fashion classes (0-9) |
| `FASHION_CLASS_NAMES` | array[10] | Class-name mapping for labels 0-9 |

#### Utility Function

`className(int label)` returns the human-readable Fashion-MNIST class name.

#### Example Usage

```cpp
#include "types.hpp"

mnist::Image img(mnist::IMAGE_PIXELS, 0.0);
mnist::Labels labels = {0, 1, 2, 3};
std::cout << mnist::className(7);  // "Sneaker"
```

---

## classifier.hpp

Abstract base class for all classifiers.

### Class: Classifier

#### Public Methods

| Method | Return | Description |
|--------|--------|-------------|
| `train(images, labels)` | void | Train the classifier on dataset |
| `predict(image)` | int | Predict label for single image |
| `evaluate(images, labels)` | double | Calculate accuracy on test set |
| `save(path)` | void | Save model to file (optional) |
| `load(path)` | void | Load model from file (optional) |
| `name()` | string | Get classifier name |

#### Method Signatures

```cpp
virtual void train(const ImageSet& images, const Labels& labels) = 0;
virtual int predict(const Image& image) = 0;
virtual double evaluate(const ImageSet& images, const Labels& labels);
virtual void save(const std::string& path);
virtual void load(const std::string& path);
virtual std::string name() const = 0;
```

#### Notes

- `train()`, `predict()`, and `name()` are pure virtual (must be implemented)
- `evaluate()` has default implementation using `predict()`
- `save()` and `load()` have empty default implementations

---

## dataset.hpp

Fashion-MNIST dataset loading and image utilities.

### Class: Dataset

#### Constructor

```cpp
explicit Dataset(const std::string& path = "./data");
```

Creates dataset loader with specified data directory.

#### Public Methods

| Method | Return | Description |
|--------|--------|-------------|
| `load(images_file, labels_file)` | bool | Load specific image and label files |
| `loadTraining()` | bool | Load training set (60,000 images) |
| `loadTest()` | bool | Load test set (10,000 images) |
| `printImage(index, out)` | void | Print image at index as ASCII |
| `printImage(image, out)` | void | Print image vector as ASCII (static) |
| `size()` | size_t | Number of loaded images |
| `rows()` | int | Image height (28) |
| `cols()` | int | Image width (28) |
| `images()` | const ImageSet& | Get all images |
| `labels()` | const Labels& | Get all labels |
| `label(index)` | int | Get label at index |
| `getImages(start, count)` | ImageSet | Get subset of images |
| `getLabels(start, count)` | Labels | Get subset of labels |

#### Example Usage

```cpp
#include "dataset.hpp"

mnist::Dataset train("./data");
train.loadTraining();

auto images = train.getImages(0, 1000);  // First 1000 images
auto labels = train.getLabels(0, 1000);  // First 1000 labels

train.printImage(0);  // Print first image as ASCII
```

#### ASCII Characters

| Value Range | Character |
|-------------|-----------|
| > 0.75 | @ |
| > 0.50 | # |
| > 0.25 | * |
| <= 0.25 | . |

---

## neural_net.hpp

Neural network classifier implemented in-project (no external ML dependency).

### Class: NeuralNet

Inherits from: `Classifier`

#### Constructor

```cpp
NeuralNet(int epochs = 10, int batch_size = 32);
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| epochs | 10 | Number of training iterations |
| batch_size | 32 | Samples per batch |

#### Network Architecture

```
Layer 1: Fully Connected (784 -> 128)
Layer 2: ReLU Activation
Layer 3: Fully Connected (128 -> 10)
Layer 4: Softmax Activation
```

#### Training Configuration

| Setting | Value |
|---------|-------|
| Optimizer | Adagrad |
| Loss Function | Mean Squared Error (MSE) |
| Input Size | 784 |
| Hidden Size | 128 |
| Output Size | 10 |

#### Public Methods

| Method | Description |
|--------|-------------|
| `train(images, labels)` | Train network on dataset |
| `predict(image)` | Get predicted class (0-9) |
| `save(path)` | Save trained model to file |
| `load(path)` | Load model from file |
| `name()` | Returns "NeuralNet" |

#### Example Usage

```cpp
#include "neural_net.hpp"

mnist::NeuralNet model(15, 64);  // 15 epochs, batch size 64
model.train(train_images, train_labels);

int predicted_class = model.predict(test_image);
model.save("./models/neural_net.model");
```

---

## knn.hpp

K-Nearest Neighbors classifier with OpenMP parallelization.

### Class: KNN

Inherits from: `Classifier`

#### Constructor

```cpp
explicit KNN(int k = 3);
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| k | 3 | Number of neighbors to consider |

#### Algorithm Details

| Setting | Value |
|---------|-------|
| Distance Metric | Euclidean |
| Voting Method | Majority |
| Parallelization | OpenMP |

#### Public Methods

| Method | Description |
|--------|-------------|
| `train(images, labels)` | Store training data |
| `predict(image)` | Get predicted class using k-NN |
| `name()` | Returns "KNN" |

#### Notes

- KNN does not support save/load (stores all training data in memory)
- Prediction time scales with training set size
- Uses OpenMP for parallel distance computation

#### Example Usage

```cpp
#include "knn.hpp"

mnist::KNN model(5);  // k=5 neighbors
model.train(train_images, train_labels);

int predicted_class = model.predict(test_image);
double accuracy = model.evaluate(test_images, test_labels);
```

---

## Include Order

Recommended include order for application files:

```cpp
// Project headers
#include "types.hpp"
#include "dataset.hpp"
#include "classifier.hpp"
#include "neural_net.hpp"
#include "knn.hpp"

// Standard library
#include <iostream>
#include <vector>
#include <string>
#include <memory>
```

---

## Polymorphism Example

Using the Classifier interface for runtime selection:

```cpp
#include "classifier.hpp"
#include "neural_net.hpp"
#include "knn.hpp"
#include <memory>

std::unique_ptr<mnist::Classifier> createClassifier(const std::string& type) {
    if (type == "knn") {
        return std::make_unique<mnist::KNN>(3);
    } else {
        return std::make_unique<mnist::NeuralNet>(10, 32);
    }
}

int main() {
    auto classifier = createClassifier("nn");
    classifier->train(images, labels);
    int prediction = classifier->predict(test_image);
    double accuracy = classifier->evaluate(test_images, test_labels);
    return 0;
}
```
