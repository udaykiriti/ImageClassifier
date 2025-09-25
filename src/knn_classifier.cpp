#include "knn_classifier.hpp"
#include <cmath>
#include <algorithm>
#include <map>

KNNClassifier::KNNClassifier(int neighbors) : k(neighbors) {}

void KNNClassifier::fit(const std::vector<std::vector<double>> &images, const std::vector<int> &labels)
{
  train_images = images;
  train_labels = labels;
}

double KNNClassifier::euclidean_distance(const std::vector<double> &a, const std::vector<double> &b) const
{
  double sum = 0.0;
  for (size_t i = 0; i < a.size(); i++)
    sum += (a[i] - b[i]) * (a[i] - b[i]);
  return std::sqrt(sum);
}

int KNNClassifier::predict(const std::vector<double> &image) const
{
  std::vector<std::pair<double, int>> distances;

  for (size_t i = 0; i < train_images.size(); i++)
  {
    double dist = euclidean_distance(image, train_images[i]);
    distances.push_back({dist, train_labels[i]});
  }

  std::sort(distances.begin(), distances.end());

  std::map<int, int> votes;
  for (int i = 0; i < k; i++)
    votes[distances[i].second]++;

  int max_vote = 0, prediction = -1;
  for (auto &v : votes)
  {
    if (v.second > max_vote)
    {
      max_vote = v.second;
      prediction = v.first;
    }
  }
  return prediction;
}

double KNNClassifier::score(const std::vector<std::vector<double>> &test_images, const std::vector<int> &test_labels) const
{
  int correct = 0;
  for (size_t i = 0; i < test_images.size(); i++)
    if (predict(test_images[i]) == test_labels[i])
      correct++;
  return double(correct) / test_images.size();
}
