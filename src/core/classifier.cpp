#include "classifier.hpp"

namespace mnist {

double Classifier::evaluate(const ImageSet& images, const Labels& labels)
{
    int correct = 0;
    for (size_t i = 0; i < images.size(); ++i)
    {
        if (predict(images[i]) == labels[i])
            ++correct;
    }
    return static_cast<double>(correct) / images.size();
}

}
