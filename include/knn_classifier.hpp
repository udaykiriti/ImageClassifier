#ifndef KNN_CLASSIFIER_HPP
#define KNN_CLASSIFIER_HPP

#include <vector>

class KNNClassifier
{
private:
  int k;
  std::vector<std::vector<double>> X; // training images
  std::vector<int> y;                 // training labels

  double distance(const std::vector<double> &a, const std::vector<double> &b);

public:
  KNNClassifier(int neighbors = 3) : k(neighbors) {}

  void fit(const std::vector<std::vector<double>> &X_train,
           const std::vector<int> &y_train);

  int predict(const std::vector<double> &x);

  double score(const std::vector<std::vector<double>> &X_test,
               const std::vector<int> &y_test);
};

#endif
