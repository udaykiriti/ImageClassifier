#include "classifier.hpp"
#include <algorithm>

namespace mnist {

double Classifier::evaluate(const ImageSet& images, const Labels& labels)
{
    const size_t count = std::min(images.size(), labels.size());
    if (count == 0)
        return 0.0;

    int correct = 0;
    for (size_t i = 0; i < count; ++i)
    {
        if (predict(images[i]) == labels[i])
            ++correct;
    }
    return static_cast<double>(correct) / static_cast<double>(count);
}

}
