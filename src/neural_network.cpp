#include "neural_network.hpp"
#include <iostream>
#include <algorithm>

NeuralNetwork::NeuralNetwork()
{
    buildNetwork();
}

void NeuralNetwork::buildNetwork()
{
    using namespace tiny_dnn;

    network_ << fully_connected_layer(784, 128)
             << relu_layer()
             << fully_connected_layer(128, 10)
             << softmax_layer();
}

void NeuralNetwork::train(const std::vector<std::vector<double>>& images,
                          const std::vector<int>& labels,
                          int epochs,
                          int batch_size)
{
    using namespace tiny_dnn;

    adagrad optimizer;
    std::vector<vec_t> input_images;
    std::vector<label_t> input_labels;

    input_images.reserve(images.size());
    input_labels.reserve(labels.size());

    for (size_t i = 0; i < images.size(); ++i)
    {
        input_images.emplace_back(images[i].begin(), images[i].end());
        input_labels.push_back(static_cast<label_t>(labels[i]));
    }

    network_.train<mse>(optimizer, input_images, input_labels, batch_size, epochs);
    std::cout << "Training completed." << std::endl;
}

int NeuralNetwork::predict(const std::vector<double>& image)
{
    tiny_dnn::vec_t input(image.begin(), image.end());
    auto result = network_.predict(input);
    return static_cast<int>(std::distance(result.begin(),
                            std::max_element(result.begin(), result.end())));
}

double NeuralNetwork::evaluate(const std::vector<std::vector<double>>& images,
                               const std::vector<int>& labels)
{
    int correct = 0;
    for (size_t i = 0; i < images.size(); ++i)
    {
        if (predict(images[i]) == labels[i])
            ++correct;
    }
    return static_cast<double>(correct) / images.size();
}

void NeuralNetwork::save(const std::string& path)
{
    network_.save(path);
}

void NeuralNetwork::load(const std::string& path)
{
    network_.load(path);
}
