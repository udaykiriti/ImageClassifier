#include "neural_network.hpp"
#include <fstream>
#include <iostream>
#include <vector>

int main()
{
    const std::string image_path = "./data/image.txt";
    const std::string model_path = "./models/neural_network.model";

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
    std::cout << "Predicted digit: " << prediction << std::endl;

    return 0;
}
