cmake_minimum_required(VERSION 3.10)
project(ImageClassifier LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(OpenMP)

include_directories(include)
include_directories(tiny-dnn)

# Common sources
set(MNIST_SRC src/mnist_loader.cpp)
set(NN_SRC src/neural_network.cpp)
set(KNN_SRC src/knn.cpp)

# Neural Network Classifier
add_executable(classifier
    src/main.cpp
    ${MNIST_SRC}
    ${NN_SRC}
)

# KNN Classifier
add_executable(classifier_knn
    src/main.cpp
    ${MNIST_SRC}
    ${KNN_SRC}
)

# Predict
add_executable(predict
    src/predict.cpp
    ${MNIST_SRC}
    ${NN_SRC}
)

# Visualize
add_executable(visualize
    src/predict_visualize.cpp
    ${MNIST_SRC}
    ${NN_SRC}
)

# Link OpenMP
if(OpenMP_CXX_FOUND)
    target_link_libraries(classifier OpenMP::OpenMP_CXX)
    target_link_libraries(classifier_knn OpenMP::OpenMP_CXX)
    target_link_libraries(predict OpenMP::OpenMP_CXX)
    target_link_libraries(visualize OpenMP::OpenMP_CXX)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall -Wextra")

# Create models directory
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/models)
