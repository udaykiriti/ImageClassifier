# ImageClassifier Makefile

CXX = g++
CXXFLAGS = -std=c++17 -O2 -fopenmp
INCLUDES = -Iinclude -Itiny-dnn
BUILD_DIR = build

# Source files
SRC_DIR = src
DATASET_SRC = $(SRC_DIR)/dataset.cpp
KNN_SRC = $(SRC_DIR)/knn_classifier.cpp
NN_SRC = $(SRC_DIR)/SimpleNN.cpp

# Targets
.PHONY: all clean knn nn predict predictprob

all: nn

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# KNN version
knn: $(BUILD_DIR)
	$(CXX) $(SRC_DIR)/main.cpp $(DATASET_SRC) $(KNN_SRC) -Iinclude $(CXXFLAGS) -o $(BUILD_DIR)/ImageClassifier

# Neural Network version
nn: $(BUILD_DIR)
	$(CXX) $(SRC_DIR)/main.cpp $(DATASET_SRC) $(NN_SRC) $(INCLUDES) $(CXXFLAGS) -o $(BUILD_DIR)/ImageClassifier

# Predict single image
predict: $(BUILD_DIR)
	$(CXX) $(SRC_DIR)/Predict.cpp $(NN_SRC) $(DATASET_SRC) $(INCLUDES) $(CXXFLAGS) -o $(BUILD_DIR)/ImagePredict

# Predict with true label and ASCII image
predictprob: $(BUILD_DIR)
	$(CXX) $(SRC_DIR)/PredictProb.cpp $(NN_SRC) $(DATASET_SRC) $(INCLUDES) $(CXXFLAGS) -o $(BUILD_DIR)/ImagePredictProb

clean:
	rm -rf $(BUILD_DIR)/*
