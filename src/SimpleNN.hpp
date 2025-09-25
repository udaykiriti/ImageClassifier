#ifndef SIMPLENN_HPP
#define SIMPLENN_HPP

#include "dataset.hpp"
#include <vector>
#include <string>
#include <tiny_dnn/tiny_dnn.h>

class SimpleNN
{
private:
  tiny_dnn::network<tiny_dnn::sequential> nn;

public:
  SimpleNN();
  void buildNetwork(); // define layers
  void train(const std::vector<std::vector<double>> &images,
             const std::vector<int> &labels,
             int epochs = 5, int batch_size = 32);
  int predict(const std::vector<double> &image);
  double score(const std::vector<std::vector<double>> &images,
               const std::vector<int> &labels);
  void saveModel(const std::string &path);
  void loadModel(const std::string &path);
};

#endif
