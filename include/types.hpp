#ifndef TYPES_HPP
#define TYPES_HPP

#include <vector>

namespace mnist {

using Image = std::vector<double>;
using ImageSet = std::vector<Image>;
using Labels = std::vector<int>;

constexpr int IMAGE_SIZE = 28;
constexpr int IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE;
constexpr int NUM_CLASSES = 10;

}

#endif
