cmake_minimum_required(VERSION 3.14)
project(ImageClassifier LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(OpenMP)

include_directories(include)

set(CORE_SRC
    src/core/classifier.cpp
    src/core/dataset.cpp
)

set(CLASSIFIERS_SRC
    src/classifiers/neural_net.cpp
    src/classifiers/knn.cpp
)

add_executable(train
    src/apps/train.cpp
    ${CORE_SRC}
    ${CLASSIFIERS_SRC}
)

add_executable(predict
    src/apps/predict.cpp
    ${CORE_SRC}
    src/classifiers/neural_net.cpp
)

add_executable(tui
    src/apps/tui.cpp
    ${CORE_SRC}
    ${CLASSIFIERS_SRC}
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(train OpenMP::OpenMP_CXX)
    target_link_libraries(predict OpenMP::OpenMP_CXX)
    target_link_libraries(tui OpenMP::OpenMP_CXX)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall -Wextra")

file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/models)
