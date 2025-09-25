#ifndef KNN_CLASSIFIER_HPP
#define KNN_CLASSIFIER_HPP

#include <vector>

class KNNClassifier
{
private:
  int k;
  std::vector<std::vector<double>> train_images;
  std::vector<int> train_labels;

  double euclidean_distance(const std::vector<double> &a, const std::vector<double> &b) const;

public:
  KNNClassifier(int neighbors = 3);

  void fit(const std::vector<std::vector<double>> &images, const std::vector<int> &labels);
  int predict(const std::vector<double> &image) const;
  double score(const std::vector<std::vector<double>> &test_images, const std::vector<int> &test_labels) const;
};

#endif
