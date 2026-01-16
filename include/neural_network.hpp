#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <vector>
#include <string>
#include <tiny_dnn/tiny_dnn.h>

class NeuralNetwork
{
private:
    tiny_dnn::network<tiny_dnn::sequential> network_;

    void buildNetwork();

public:
    NeuralNetwork();

    void train(const std::vector<std::vector<double>>& images,
               const std::vector<int>& labels,
               int epochs = 10,
               int batch_size = 32);

    int predict(const std::vector<double>& image);

    double evaluate(const std::vector<std::vector<double>>& images,
                    const std::vector<int>& labels);

    void save(const std::string& path);
    void load(const std::string& path);
};

#endif
