#include "dataset.hpp"
#include "neural_net.hpp"
#include "knn.hpp"
#include <iostream>
#include <fstream>
#include <memory>
#include <cstring>

using namespace mnist;

void printUsage(const char* prog)
{
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --model <nn|knn>    Classifier type (default: nn)\n"
              << "  --train <count>     Training samples (default: 2000)\n"
              << "  --test <count>      Test samples (default: 500)\n"
              << "  --epochs <count>    Training epochs for NN (default: 10)\n"
              << "  --k <count>         K neighbors for KNN (default: 3)\n"
              << "  --help              Show this help\n";
}

int main(int argc, char* argv[])
{
    std::string model_type = "nn";
    int train_count = 2000;
    int test_count = 500;
    int epochs = 10;
    int k = 3;

    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc)
            model_type = argv[++i];
        else if (std::strcmp(argv[i], "--train") == 0 && i + 1 < argc)
            train_count = std::stoi(argv[++i]);
        else if (std::strcmp(argv[i], "--test") == 0 && i + 1 < argc)
            test_count = std::stoi(argv[++i]);
        else if (std::strcmp(argv[i], "--epochs") == 0 && i + 1 < argc)
            epochs = std::stoi(argv[++i]);
        else if (std::strcmp(argv[i], "--k") == 0 && i + 1 < argc)
            k = std::stoi(argv[++i]);
        else if (std::strcmp(argv[i], "--help") == 0)
        {
            printUsage(argv[0]);
            return 0;
        }
    }

    Dataset train_data("./data");
    Dataset test_data("./data");

    if (!train_data.loadTraining() || !test_data.loadTest())
    {
        std::cerr << "Failed to load dataset" << std::endl;
        return 1;
    }

    auto train_images = train_data.getImages(0, train_count);
    auto train_labels = train_data.getLabels(0, train_count);
    auto test_images = test_data.getImages(0, test_count);
    auto test_labels = test_data.getLabels(0, test_count);

    std::unique_ptr<Classifier> classifier;
    std::string model_path = "./models/";

    if (model_type == "knn")
    {
        classifier = std::make_unique<KNN>(k);
        model_path += "knn.model";
    }
    else
    {
        classifier = std::make_unique<NeuralNet>(epochs, 32);
        model_path += "neural_net.model";
    }

    std::cout << "\n[" << classifier->name() << "] Training with "
              << train_images.size() << " samples...\n" << std::endl;

    std::ifstream model_file(model_path);
    if (model_file.good() && model_type == "nn")
    {
        model_file.close();
        classifier->load(model_path);
        std::cout << "Loaded model: " << model_path << std::endl;
    }
    else
    {
        classifier->train(train_images, train_labels);
        if (model_type == "nn")
        {
            classifier->save(model_path);
            std::cout << "Saved model: " << model_path << std::endl;
        }
    }

    double accuracy = classifier->evaluate(test_images, test_labels);
    std::cout << "\nAccuracy: " << accuracy * 100 << "%" << std::endl;

    std::cout << "\nSample Predictions:\n";
    std::cout << std::string(40, '-') << std::endl;

    for (int i = 0; i < 5; ++i)
    {
        int predicted = classifier->predict(test_images[i]);
        bool correct = predicted == test_labels[i];
        std::cout << "Image " << i
                  << " | True: " << test_labels[i]
                  << " | Predicted: " << predicted
                  << (correct ? " [OK]" : " [X]") << std::endl;
        Dataset::printImage(test_images[i]);
        std::cout << std::string(40, '-') << std::endl;
    }

    return 0;
}
