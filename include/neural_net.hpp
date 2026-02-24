#ifndef NEURAL_NET_HPP
#define NEURAL_NET_HPP

#include "classifier.hpp"

#include <vector>

namespace mnist {

class NeuralNet : public Classifier
{
private:
    int epochs_;
    int batch_size_;
    int hidden_size_;
    double learning_rate_;
    double l2_lambda_;
    int random_seed_;

    std::vector<double> w1_;
    std::vector<double> b1_;
    std::vector<double> w2_;
    std::vector<double> b2_;

public:
    NeuralNet(int epochs = 10,
              int batch_size = 32,
              int hidden_size = 128,
              double learning_rate = 0.01,
              double l2_lambda = 1e-4,
              int random_seed = 42);

    void train(const ImageSet& images, const Labels& labels) override;
    int predict(const Image& image) override;
    std::vector<double> predictProba(const Image& image) const;
    void save(const std::string& path) override;
    void load(const std::string& path) override;
    std::string name() const override { return "NeuralNet"; }

private:
    void buildNetwork();
    std::vector<double> forward(const Image& image, std::vector<double>& hidden_pre, std::vector<double>& hidden_act) const;
};

}

#endif
