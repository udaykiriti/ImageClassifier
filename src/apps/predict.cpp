#include "dataset.hpp"
#include "neural_net.hpp"
#include <iostream>
#include <fstream>
#include <cstring>

using namespace mnist;

void printUsage(const char* prog)
{
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --image <path>      Image text file (default: ./data/image.txt)\n"
              << "  --model <path>      Model file (default: ./models/neural_net.model)\n"
              << "  --label <digit>     True label for comparison (optional)\n"
              << "  --show              Show ASCII visualization\n"
              << "  --help              Show this help\n";
}

Image loadImageFromFile(const std::string& path)
{
    std::ifstream file(path);
    if (!file.is_open())
    {
        std::cerr << "Cannot open: " << path << std::endl;
        return {};
    }

    Image image;
    double value;
    while (file >> value)
    {
        image.push_back(value / 255.0);
    }

    if (image.size() != IMAGE_PIXELS)
    {
        std::cerr << "Expected " << IMAGE_PIXELS << " pixels, got "
                  << image.size() << std::endl;
        return {};
    }

    return image;
}

int main(int argc, char* argv[])
{
    std::string image_path = "./data/image.txt";
    std::string model_path = "./models/neural_net.model";
    int true_label = -1;
    bool show_image = false;

    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--image") == 0 && i + 1 < argc)
            image_path = argv[++i];
        else if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc)
            model_path = argv[++i];
        else if (std::strcmp(argv[i], "--label") == 0 && i + 1 < argc)
            true_label = std::stoi(argv[++i]);
        else if (std::strcmp(argv[i], "--show") == 0)
            show_image = true;
        else if (std::strcmp(argv[i], "--help") == 0)
        {
            printUsage(argv[0]);
            return 0;
        }
    }

    Image image = loadImageFromFile(image_path);
    if (image.empty())
        return 1;

    NeuralNet model;
    model.load(model_path);

    int prediction = model.predict(image);

    if (show_image)
    {
        std::cout << "\nASCII Image:\n";
        Dataset::printImage(image);
    }

    std::cout << "\nPredicted digit: " << prediction << std::endl;

    if (true_label >= 0)
    {
        std::cout << "True label: " << true_label << std::endl;
        std::cout << (prediction == true_label ? "CORRECT" : "INCORRECT") << std::endl;
    }

    return 0;
}
