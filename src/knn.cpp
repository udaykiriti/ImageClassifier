#include "knn.hpp"
#include <cmath>
#include <algorithm>
#include <omp.h>

double KNN::euclideanDistance(const std::vector<double>& a,
                              const std::vector<double>& b) const
{
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

void KNN::fit(const std::vector<std::vector<double>>& images,
              const std::vector<int>& labels)
{
    train_images_ = images;
    train_labels_ = labels;
}

int KNN::predict(const std::vector<double>& image)
{
    std::vector<std::pair<double, int>> distances(train_images_.size());

    #pragma omp parallel for
    for (size_t i = 0; i < train_images_.size(); ++i)
    {
        distances[i] = {euclideanDistance(train_images_[i], image), train_labels_[i]};
    }

    std::nth_element(distances.begin(), distances.begin() + k_, distances.end());
    distances.resize(k_);

    std::vector<int> votes(10, 0);
    for (const auto& d : distances)
    {
        ++votes[d.second];
    }

    return static_cast<int>(std::distance(votes.begin(),
                            std::max_element(votes.begin(), votes.end())));
}

double KNN::evaluate(const std::vector<std::vector<double>>& images,
                     const std::vector<int>& labels)
{
    int correct = 0;

    #pragma omp parallel for reduction(+:correct)
    for (size_t i = 0; i < images.size(); ++i)
    {
        if (predict(images[i]) == labels[i])
            ++correct;
    }

    return static_cast<double>(correct) / images.size();
}
