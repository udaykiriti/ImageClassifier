cmake_minimum_required(VERSION 3.10)
project(ImageClassifier)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenMP
find_package(OpenMP)

# Include directories
include_directories(include)
include_directories(tiny-dnn)

# Neural Network version
add_executable(ImageClassifier
    src/main.cpp
    src/dataset.cpp
    src/SimpleNN.cpp
)

# Predict executable
add_executable(ImagePredict
    src/Predict.cpp
    src/dataset.cpp
    src/SimpleNN.cpp
)

# PredictProb executable
add_executable(ImagePredictProb
    src/PredictProb.cpp
    src/dataset.cpp
    src/SimpleNN.cpp
)

# KNN version
add_executable(ImageClassifierKNN
    src/main.cpp
    src/dataset.cpp
    src/knn_classifier.cpp
)

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(ImageClassifier OpenMP::OpenMP_CXX)
    target_link_libraries(ImagePredict OpenMP::OpenMP_CXX)
    target_link_libraries(ImagePredictProb OpenMP::OpenMP_CXX)
    target_link_libraries(ImageClassifierKNN OpenMP::OpenMP_CXX)
endif()

# Compiler optimization
target_compile_options(ImageClassifier PRIVATE -O2)
target_compile_options(ImagePredict PRIVATE -O2)
target_compile_options(ImagePredictProb PRIVATE -O2)
target_compile_options(ImageClassifierKNN PRIVATE -O2)
