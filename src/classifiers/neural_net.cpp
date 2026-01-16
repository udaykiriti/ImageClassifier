#include "neural_net.hpp"
#include <iostream>
#include <algorithm>

namespace mnist {

NeuralNet::NeuralNet(int epochs, int batch_size)
    : epochs_(epochs), batch_size_(batch_size)
{
    buildNetwork();
}

void NeuralNet::buildNetwork()
{
    using namespace tiny_dnn;
    network_ << fully_connected_layer(IMAGE_PIXELS, 128)
             << relu_layer()
             << fully_connected_layer(128, NUM_CLASSES)
             << softmax_layer();
}

void NeuralNet::train(const ImageSet& images, const Labels& labels)
{
    using namespace tiny_dnn;

    std::vector<vec_t> input;
    std::vector<label_t> output;
    input.reserve(images.size());
    output.reserve(labels.size());

    for (size_t i = 0; i < images.size(); ++i)
    {
        input.emplace_back(images[i].begin(), images[i].end());
        output.push_back(static_cast<label_t>(labels[i]));
    }

    adagrad optimizer;
    network_.train<mse>(optimizer, input, output, batch_size_, epochs_);
    std::cout << "Training completed (" << epochs_ << " epochs)" << std::endl;
}

int NeuralNet::predict(const Image& image)
{
    tiny_dnn::vec_t input(image.begin(), image.end());
    auto result = network_.predict(input);
    return static_cast<int>(std::distance(result.begin(),
                            std::max_element(result.begin(), result.end())));
}

void NeuralNet::save(const std::string& path)
{
    network_.save(path);
}

void NeuralNet::load(const std::string& path)
{
    network_.load(path);
}

}
