#include "mnist_loader.hpp"
#include "knn.hpp"
#include <iostream>
#include <algorithm>

int main()
{
    const std::string data_path = "./data";

    MnistLoader train_data("MNIST Training", data_path);
    MnistLoader test_data("MNIST Test", data_path);

    train_data.loadImages("train-images-idx3-ubyte");
    train_data.loadLabels("train-labels-idx1-ubyte");
    test_data.loadImages("t10k-images-idx3-ubyte");
    test_data.loadLabels("t10k-labels-idx1-ubyte");

    int train_size = std::min(1000, train_data.numImages());
    int test_size = std::min(200, test_data.numImages());

    std::vector<std::vector<double>> train_images(
        train_data.images().begin(),
        train_data.images().begin() + train_size);
    std::vector<int> train_labels(
        train_data.labels().begin(),
        train_data.labels().begin() + train_size);

    std::vector<std::vector<double>> test_images(
        test_data.images().begin(),
        test_data.images().begin() + test_size);
    std::vector<int> test_labels(
        test_data.labels().begin(),
        test_data.labels().begin() + test_size);

    std::cout << "\nTraining KNN classifier (k=3)..." << std::endl;
    KNN knn(3);
    knn.fit(train_images, train_labels);

    std::cout << "Evaluating on " << test_size << " test images..." << std::endl;
    double accuracy = knn.evaluate(test_images, test_labels);
    std::cout << "Accuracy: " << accuracy * 100 << "%" << std::endl;

    std::cout << "\nSample Predictions:\n";
    std::cout << "--------------------\n";
    for (int i = 0; i < 10; ++i)
    {
        int predicted = knn.predict(test_images[i]);
        std::cout << "Image " << i
                  << " | True: " << test_labels[i]
                  << " | Predicted: " << predicted
                  << (predicted == test_labels[i] ? " [OK]" : " [X]")
                  << std::endl;
    }

    return 0;
}
