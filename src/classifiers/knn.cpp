#include "knn.hpp"
#include <cmath>
#include <algorithm>
#include <omp.h>

namespace mnist {

double KNN::distance(const Image& a, const Image& b) const
{
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

void KNN::train(const ImageSet& images, const Labels& labels)
{
    train_images_ = images;
    train_labels_ = labels;
}

int KNN::predict(const Image& image)
{
    std::vector<std::pair<double, int>> distances(train_images_.size());

    #pragma omp parallel for
    for (size_t i = 0; i < train_images_.size(); ++i)
    {
        distances[i] = {distance(train_images_[i], image), train_labels_[i]};
    }

    std::nth_element(distances.begin(), distances.begin() + k_, distances.end());
    distances.resize(k_);

    std::vector<int> votes(NUM_CLASSES, 0);
    for (const auto& d : distances)
    {
        ++votes[d.second];
    }

    return static_cast<int>(std::distance(votes.begin(),
                            std::max_element(votes.begin(), votes.end())));
}

}
