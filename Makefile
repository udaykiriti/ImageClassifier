# ImageClassifier Makefile

CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra
LDFLAGS = -fopenmp

BUILD_DIR = build
MODEL_DIR = models
INC = -Iinclude -Itiny-dnn

CORE_SRC = src/core/classifier.cpp src/core/dataset.cpp
NN_SRC = src/classifiers/neural_net.cpp
KNN_SRC = src/classifiers/knn.cpp

.PHONY: all clean dirs train predict help

all: dirs train predict

dirs:
	@mkdir -p $(BUILD_DIR) $(MODEL_DIR)

train: dirs
	$(CXX) $(CXXFLAGS) src/apps/train.cpp $(CORE_SRC) $(NN_SRC) $(KNN_SRC) $(INC) $(LDFLAGS) -o $(BUILD_DIR)/train

predict: dirs
	$(CXX) $(CXXFLAGS) src/apps/predict.cpp $(CORE_SRC) $(NN_SRC) $(INC) $(LDFLAGS) -o $(BUILD_DIR)/predict

clean:
	rm -rf $(BUILD_DIR)/* $(MODEL_DIR)/*.model

help:
	@echo "Usage:"
	@echo "  make train    - Build training app"
	@echo "  make predict  - Build prediction app"
	@echo "  make clean    - Remove build files"
	@echo ""
	@echo "Run:"
	@echo "  ./build/train --model nn --train 5000 --epochs 10"
	@echo "  ./build/train --model knn --train 1000 --k 5"
	@echo "  ./build/predict --image data/image.txt --show"
