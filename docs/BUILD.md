# Build Instructions

This document covers all build methods and Makefile targets.

---

## Requirements

### Compiler

- C++17 compatible compiler
- Tested with: GCC 9+, Clang 10+

### Libraries

| Library | Required | Purpose |
|---------|----------|---------|
| tiny-dnn | Yes | Neural network implementation |
| OpenMP | Optional | Parallel processing for KNN |
| Python 3 | Optional | Image preprocessing |
| PIL/Pillow | Optional | Image loading in Python |
| NumPy | Optional | Array operations in Python |

---

## Build Methods

### Method 1: Makefile (Recommended)

```bash
# Build all targets
make

# Build specific target
make train
make predict

# Clean build files
make clean

# Show help
make help
```

### Method 2: CMake

```bash
# Create build directory
mkdir build && cd build

# Generate build files
cmake ..

# Build
make

# Executables will be in build/bin/
```

### Method 3: Manual Compilation

```bash
# Build train executable
g++ -std=c++17 -O2 -Wall -Wextra \
    src/apps/train.cpp \
    src/core/classifier.cpp \
    src/core/dataset.cpp \
    src/classifiers/neural_net.cpp \
    src/classifiers/knn.cpp \
    -Iinclude -Itiny-dnn \
    -fopenmp \
    -o build/train

# Build predict executable
g++ -std=c++17 -O2 -Wall -Wextra \
    src/apps/predict.cpp \
    src/core/classifier.cpp \
    src/core/dataset.cpp \
    src/classifiers/neural_net.cpp \
    -Iinclude -Itiny-dnn \
    -fopenmp \
    -o build/predict
```

---

## Makefile Reference

### File: Makefile

```makefile
# Compiler settings
CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra
LDFLAGS = -fopenmp

# Directories
BUILD_DIR = build
MODEL_DIR = models
INC = -Iinclude -Itiny-dnn

# Source files
CORE_SRC = src/core/classifier.cpp src/core/dataset.cpp
NN_SRC = src/classifiers/neural_net.cpp
KNN_SRC = src/classifiers/knn.cpp
```

### Targets

| Target | Description | Output |
|--------|-------------|--------|
| `all` | Build train and predict | build/train, build/predict |
| `train` | Build training app | build/train |
| `predict` | Build prediction app | build/predict |
| `dirs` | Create build and models directories | - |
| `clean` | Remove build artifacts | - |
| `help` | Show available targets | - |

### Target Details

#### make train

Builds the training and evaluation application.

```bash
make train
```

Compiles:
- src/apps/train.cpp
- src/core/classifier.cpp
- src/core/dataset.cpp
- src/classifiers/neural_net.cpp
- src/classifiers/knn.cpp

Output: `build/train`

#### make predict

Builds the prediction application.

```bash
make predict
```

Compiles:
- src/apps/predict.cpp
- src/core/classifier.cpp
- src/core/dataset.cpp
- src/classifiers/neural_net.cpp

Output: `build/predict`

#### make clean

Removes all build artifacts.

```bash
make clean
```

Deletes:
- build/*
- models/*.model

#### make help

Displays usage information.

```bash
make help
```

Output:
```
Usage:
  make train    - Build training app
  make predict  - Build prediction app
  make clean    - Remove build files

Run:
  ./build/train --model nn --train 5000 --epochs 10
  ./build/train --model knn --train 1000 --k 5
  ./build/predict --image data/image.txt --show
```

---

## CMake Reference

### File: CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.14)
project(ImageClassifier LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
```

### CMake Targets

| Target | Description |
|--------|-------------|
| train | Training executable |
| predict | Prediction executable |

### CMake Options

| Option | Description |
|--------|-------------|
| CMAKE_BUILD_TYPE | Release, Debug, RelWithDebInfo |
| CMAKE_CXX_COMPILER | Specify compiler |

### Example CMake Build

```bash
# Release build
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make

# Debug build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
```

---

## Compiler Flags

### Optimization Flags

| Flag | Description |
|------|-------------|
| -O2 | Optimization level 2 |
| -O3 | Maximum optimization |
| -Os | Optimize for size |

### Warning Flags

| Flag | Description |
|------|-------------|
| -Wall | Enable common warnings |
| -Wextra | Enable extra warnings |
| -Werror | Treat warnings as errors |

### Debug Flags

| Flag | Description |
|------|-------------|
| -g | Include debug symbols |
| -fsanitize=address | Enable AddressSanitizer |

---

## Troubleshooting

### Issue: tiny-dnn not found

```
fatal error: tiny_dnn/tiny_dnn.h: No such file or directory
```

Solution:
```bash
git submodule update --init
```

### Issue: OpenMP not found

```
fatal error: omp.h: No such file or directory
```

Solution (Ubuntu/Debian):
```bash
sudo apt install libomp-dev
```

Solution (macOS):
```bash
brew install libomp
```

### Issue: C++17 not supported

```
error: unrecognized command line option '-std=c++17'
```

Solution: Update compiler or use -std=c++1z

### Issue: Makefile tab errors

```
Makefile:XX: *** missing separator. Stop.
```

Solution: Ensure Makefile uses tabs, not spaces, for indentation.

---

## Build Outputs

### Executable Sizes (Approximate)

| Binary | Size | Notes |
|--------|------|-------|
| train | ~3 MB | Includes NN and KNN |
| predict | ~3 MB | Includes NN only |

### Build Time (Approximate)

| Target | Time | Notes |
|--------|------|-------|
| train | 60-120s | Includes tiny-dnn templates |
| predict | 60-120s | Includes tiny-dnn templates |

Most compile time is spent instantiating tiny-dnn templates.
