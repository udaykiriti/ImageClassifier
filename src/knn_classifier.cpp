#include "knn_classifier.hpp"
#include <cmath>
#include <algorithm>
#include <limits>
#include <omp.h> // OpenMP

double KNNClassifier::distance(const std::vector<double> &a,
                               const std::vector<double> &b)
{
  double sum = 0.0;
  for (size_t i = 0; i < a.size(); i++)
  {
    double diff = a[i] - b[i];
    sum += diff * diff;
  }
  return std::sqrt(sum);
}

void KNNClassifier::fit(const std::vector<std::vector<double>> &X_train,
                        const std::vector<int> &y_train)
{
  X = X_train;
  y = y_train;
}

int KNNClassifier::predict(const std::vector<double> &x)
{
  std::vector<std::pair<double, int>> dist_label(X.size());

#pragma omp parallel for
  for (size_t i = 0; i < X.size(); i++)
  {
    dist_label[i] = {distance(X[i], x), y[i]};
  }

  std::nth_element(dist_label.begin(), dist_label.begin() + k, dist_label.end());
  dist_label.resize(k);

  std::vector<int> counts(10, 0); // MNIST digits
  for (auto &p : dist_label)
    counts[p.second]++;

  return std::distance(counts.begin(), std::max_element(counts.begin(), counts.end()));
}

double KNNClassifier::score(const std::vector<std::vector<double>> &X_test,
                            const std::vector<int> &y_test)
{
  int correct = 0;

#pragma omp parallel for reduction(+ : correct)
  for (size_t i = 0; i < X_test.size(); i++)
  {
    if (predict(X_test[i]) == y_test[i])
      correct++;
  }

  return static_cast<double>(correct) / X_test.size();
}
