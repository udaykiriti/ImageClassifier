#ifndef NEURAL_NET_HPP
#define NEURAL_NET_HPP

#include "classifier.hpp"
#include <tiny_dnn/tiny_dnn.h>

namespace mnist {

class NeuralNet : public Classifier
{
private:
    tiny_dnn::network<tiny_dnn::sequential> network_;
    int epochs_;
    int batch_size_;

public:
    NeuralNet(int epochs = 10, int batch_size = 32);

    void train(const ImageSet& images, const Labels& labels) override;
    int predict(const Image& image) override;
    void save(const std::string& path) override;
    void load(const std::string& path) override;
    std::string name() const override { return "NeuralNet"; }

private:
    void buildNetwork();
};

}

#endif
