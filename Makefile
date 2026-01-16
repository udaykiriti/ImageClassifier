# ImageClassifier Makefile

CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra
LDFLAGS = -fopenmp

SRC_DIR = src
INC_DIR = include
BUILD_DIR = build
MODEL_DIR = models

INCLUDES = -I$(INC_DIR) -Itiny-dnn

MNIST_SRC = $(SRC_DIR)/mnist_loader.cpp
NN_SRC = $(SRC_DIR)/neural_network.cpp
KNN_SRC = $(SRC_DIR)/knn.cpp

.PHONY: all clean dirs nn knn predict visualize

all: dirs nn

dirs:
@mkdir -p $(BUILD_DIR) $(MODEL_DIR)

nn: dirs
$(CXX) $(CXXFLAGS) $(SRC_DIR)/main.cpp $(MNIST_SRC) $(NN_SRC) \
$(INCLUDES) $(LDFLAGS) -o $(BUILD_DIR)/classifier

knn: dirs
$(CXX) $(CXXFLAGS) $(SRC_DIR)/main.cpp $(MNIST_SRC) $(KNN_SRC) \
$(INCLUDES) $(LDFLAGS) -o $(BUILD_DIR)/classifier_knn

predict: dirs
$(CXX) $(CXXFLAGS) $(SRC_DIR)/predict.cpp $(MNIST_SRC) $(NN_SRC) \
$(INCLUDES) $(LDFLAGS) -o $(BUILD_DIR)/predict

visualize: dirs
$(CXX) $(CXXFLAGS) $(SRC_DIR)/predict_visualize.cpp $(MNIST_SRC) $(NN_SRC) \
$(INCLUDES) $(LDFLAGS) -o $(BUILD_DIR)/visualize

clean:
rm -rf $(BUILD_DIR)/* $(MODEL_DIR)/*

help:
@echo "Targets:"
@echo "  nn        - Build neural network classifier (default)"
@echo "  knn       - Build KNN classifier"
@echo "  predict   - Build single image predictor"
@echo "  visualize - Build predictor with ASCII output"
@echo "  clean     - Remove build artifacts"
