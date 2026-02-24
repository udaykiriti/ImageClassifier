#ifndef TYPES_HPP
#define TYPES_HPP

#include <array>
#include <string_view>
#include <vector>

namespace mnist {

using Image = std::vector<double>;
using ImageSet = std::vector<Image>;
using Labels = std::vector<int>;

constexpr int IMAGE_SIZE = 28;
constexpr int IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE;
constexpr int NUM_CLASSES = 10;

constexpr std::array<std::string_view, NUM_CLASSES> FASHION_CLASS_NAMES = {
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
};

inline std::string_view className(int label)
{
    if (label < 0 || label >= NUM_CLASSES)
        return "Unknown";
    return FASHION_CLASS_NAMES[static_cast<size_t>(label)];
}

}

#endif
