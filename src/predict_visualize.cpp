#include "neural_network.hpp"
#include <fstream>
#include <iostream>
#include <vector>

void printAsciiImage(const std::vector<double>& image)
{
    for (int r = 0; r < 28; ++r)
    {
        for (int c = 0; c < 28; ++c)
        {
            double val = image[r * 28 + c];
            if (val > 0.75)
                std::cout << "@";
            else if (val > 0.5)
                std::cout << "#";
            else if (val > 0.25)
                std::cout << "*";
            else
                std::cout << ".";
        }
        std::cout << "\n";
    }
}

int main(int argc, char* argv[])
{
    const std::string image_path = "./data/image.txt";
    const std::string model_path = "./models/neural_network.model";

    int true_label = -1;
    if (argc > 1)
    {
        true_label = std::stoi(argv[1]);
    }

    std::ifstream file(image_path);
    if (!file.is_open())
    {
        std::cerr << "Error: Cannot open " << image_path << std::endl;
        return 1;
    }

    std::vector<double> image;
    double value;
    while (file >> value)
    {
        image.push_back(value / 255.0);
    }
    file.close();

    if (image.size() != 784)
    {
        std::cerr << "Error: Expected 784 pixels, got " << image.size() << std::endl;
        return 1;
    }

    NeuralNetwork model;
    model.load(model_path);

    int prediction = model.predict(image);

    std::cout << "\nASCII Image:\n";
    printAsciiImage(image);
    std::cout << "\nPredicted digit: " << prediction << std::endl;

    if (true_label >= 0)
    {
        std::cout << "True label: " << true_label << std::endl;
        std::cout << (prediction == true_label ? "CORRECT" : "INCORRECT") << std::endl;
    }

    return 0;
}
