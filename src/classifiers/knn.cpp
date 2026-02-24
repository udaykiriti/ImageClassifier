#include "knn.hpp"
#include <array>
#include <algorithm>
#include <cstddef>
#include <limits>
#include <omp.h>

namespace mnist {

double KNN::distanceSquared(const Image& a, const Image& b) const
{
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

void KNN::train(const ImageSet& images, const Labels& labels)
{
    const size_t count = std::min(images.size(), labels.size());
    train_images_.assign(images.begin(), images.begin() + count);
    train_labels_.assign(labels.begin(), labels.begin() + count);
}

int KNN::predict(const Image& image)
{
    if (train_images_.empty() || image.size() != IMAGE_PIXELS)
    {
        return -1;
    }

    const int k_effective = std::clamp(k_, 1, static_cast<int>(train_images_.size()));
    std::vector<std::pair<double, int>> distances(train_images_.size());
    const std::ptrdiff_t n = static_cast<std::ptrdiff_t>(train_images_.size());

    #pragma omp parallel for
    for (std::ptrdiff_t i = 0; i < n; ++i)
    {
        distances[i] = {distanceSquared(train_images_[i], image), train_labels_[i]};
    }

    if (k_effective < static_cast<int>(distances.size()))
    {
        std::nth_element(distances.begin(), distances.begin() + k_effective, distances.end());
    }
    distances.resize(static_cast<size_t>(k_effective));

    std::array<double, NUM_CLASSES> votes{};
    for (const auto& d : distances)
    {
        const double weight = weighted_votes_
            ? 1.0 / (d.first + std::numeric_limits<double>::epsilon())
            : 1.0;
        votes[d.second] += weight;
    }

    return static_cast<int>(std::distance(votes.begin(),
                            std::max_element(votes.begin(), votes.end())));
}

}
