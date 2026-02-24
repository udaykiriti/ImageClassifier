#include "dataset.hpp"

#include <algorithm>
#include <fstream>
#include <limits>
#include <vector>

namespace mnist {
namespace {

constexpr uint32_t kImageMagic = 2051;
constexpr uint32_t kLabelMagic = 2049;

bool isExpectedDimensions(int rows, int cols) {
    return rows == IMAGE_SIZE && cols == IMAGE_SIZE;
}

bool canRepresentImageBuffer(uint32_t count, size_t image_pixels) {
    if (image_pixels == 0) {
        return false;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    return static_cast<size_t>(count) <= (max_size / image_pixels);
}

}  // namespace

bool Dataset::readBigEndian(std::ifstream& file, uint32_t& value) {
    unsigned char bytes[4];
    if (!file.read(reinterpret_cast<char*>(bytes), sizeof(bytes))) {
        return false;
    }

    value = (static_cast<uint32_t>(bytes[0]) << 24) |
            (static_cast<uint32_t>(bytes[1]) << 16) |
            (static_cast<uint32_t>(bytes[2]) << 8) |
            static_cast<uint32_t>(bytes[3]);
    return true;
}

Dataset::Dataset(const std::string& path) : path_(path) {}

bool Dataset::load(const std::string& images_file, const std::string& labels_file) {
    const std::string img_path = path_ + "/" + images_file;
    std::ifstream img_file(img_path, std::ios::binary);
    if (!img_file) {
        std::cerr << "Cannot open: " << img_path << std::endl;
        return false;
    }

    uint32_t magic = 0;
    uint32_t count = 0;
    uint32_t rows = 0;
    uint32_t cols = 0;

    if (!readBigEndian(img_file, magic) || magic != kImageMagic) {
        std::cerr << "Invalid image file header: " << img_path << std::endl;
        return false;
    }
    if (!readBigEndian(img_file, count) ||
        !readBigEndian(img_file, rows) ||
        !readBigEndian(img_file, cols)) {
        std::cerr << "Failed to read image metadata: " << img_path << std::endl;
        return false;
    }

    rows_ = static_cast<int>(rows);
    cols_ = static_cast<int>(cols);
    if (!isExpectedDimensions(rows_, cols_)) {
        std::cerr << "Unsupported image dimensions: " << rows_ << "x" << cols_
                  << " (expected " << IMAGE_SIZE << "x" << IMAGE_SIZE << ")" << std::endl;
        return false;
    }

    const size_t image_pixels = static_cast<size_t>(rows_) * static_cast<size_t>(cols_);
    if (!canRepresentImageBuffer(count, image_pixels)) {
        std::cerr << "Image buffer size overflow for " << img_path << std::endl;
        return false;
    }
    const size_t image_bytes = static_cast<size_t>(count) * image_pixels;

    std::vector<unsigned char> image_buffer(image_bytes);
    if (!img_file.read(reinterpret_cast<char*>(image_buffer.data()), static_cast<std::streamsize>(image_buffer.size()))) {
        std::cerr << "Failed to read image data from: " << img_path << std::endl;
        return false;
    }

    const std::string lbl_path = path_ + "/" + labels_file;
    std::ifstream lbl_file(lbl_path, std::ios::binary);
    if (!lbl_file) {
        std::cerr << "Cannot open: " << lbl_path << std::endl;
        return false;
    }

    uint32_t lbl_magic = 0;
    uint32_t lbl_count = 0;
    if (!readBigEndian(lbl_file, lbl_magic) || lbl_magic != kLabelMagic) {
        std::cerr << "Invalid label file header: " << lbl_path << std::endl;
        return false;
    }
    if (!readBigEndian(lbl_file, lbl_count)) {
        std::cerr << "Failed to read label metadata: " << lbl_path << std::endl;
        return false;
    }

    std::vector<unsigned char> label_buffer(lbl_count);
    if (!lbl_file.read(reinterpret_cast<char*>(label_buffer.data()), static_cast<std::streamsize>(label_buffer.size()))) {
        std::cerr << "Failed to read label data from: " << lbl_path << std::endl;
        return false;
    }

    if (count != lbl_count) {
        std::cerr << "Image/label count mismatch: images=" << count
                  << ", labels=" << lbl_count << std::endl;
    }

    const size_t usable_count = std::min(static_cast<size_t>(count), static_cast<size_t>(lbl_count));

    constexpr double inv255 = 1.0 / 255.0;
    images_.assign(usable_count, Image(image_pixels));
    for (size_t i = 0; i < usable_count; ++i) {
        const size_t offset = i * image_pixels;
        for (size_t j = 0; j < image_pixels; ++j) {
            images_[i][j] = static_cast<double>(image_buffer[offset + j]) * inv255;
        }
    }

    labels_.resize(usable_count);
    for (size_t i = 0; i < usable_count; ++i) {
        const int label = static_cast<int>(label_buffer[i]);
        if (label < 0 || label >= NUM_CLASSES) {
            std::cerr << "Invalid label value " << label << " at index " << i << std::endl;
            return false;
        }
        labels_[i] = label;
    }

    std::cout << "Loaded " << usable_count << " images (" << rows_ << "x" << cols_ << ")" << std::endl;
    return true;
}

bool Dataset::loadTraining() {
    return load("train-images-idx3-ubyte", "train-labels-idx1-ubyte");
}

bool Dataset::loadTest() {
    return load("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");
}

int Dataset::label(size_t index) const {
    return index < labels_.size() ? labels_[index] : -1;
}

void Dataset::printImage(size_t index, std::ostream& out) const {
    if (index < images_.size()) {
        printImage(images_[index], out);
    }
}

void Dataset::printImage(const Image& image, std::ostream& out) {
    if (image.size() < IMAGE_PIXELS) {
        out << "[invalid image: expected at least " << IMAGE_PIXELS
            << " values, got " << image.size() << "]\n";
        return;
    }

    for (int r = 0; r < IMAGE_SIZE; ++r) {
        for (int c = 0; c < IMAGE_SIZE; ++c) {
            const double val = image[r * IMAGE_SIZE + c];
            if (val > 0.75) {
                out << "@";
            } else if (val > 0.5) {
                out << "#";
            } else if (val > 0.25) {
                out << "*";
            } else {
                out << ".";
            }
        }
        out << "\n";
    }
}

ImageSet Dataset::getImages(size_t start, size_t count) const {
    if (start >= images_.size()) {
        return {};
    }

    const size_t end = std::min(images_.size(), start + std::min(count, images_.size() - start));
    return ImageSet(images_.begin() + start, images_.begin() + end);
}

Labels Dataset::getLabels(size_t start, size_t count) const {
    if (start >= labels_.size()) {
        return {};
    }

    const size_t end = std::min(labels_.size(), start + std::min(count, labels_.size() - start));
    return Labels(labels_.begin() + start, labels_.begin() + end);
}

}  // namespace mnist
