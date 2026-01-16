#include "dataset.hpp"
#include <fstream>
#include <algorithm>

namespace mnist {

uint32_t Dataset::readBigEndian(std::ifstream& file)
{
    unsigned char bytes[4];
    file.read(reinterpret_cast<char*>(bytes), 4);
    return (static_cast<uint32_t>(bytes[0]) << 24) |
           (static_cast<uint32_t>(bytes[1]) << 16) |
           (static_cast<uint32_t>(bytes[2]) << 8) |
           static_cast<uint32_t>(bytes[3]);
}

Dataset::Dataset(const std::string& path) : path_(path) {}

bool Dataset::load(const std::string& images_file, const std::string& labels_file)
{
    std::string img_path = path_ + "/" + images_file;
    std::ifstream img_file(img_path, std::ios::binary);
    if (!img_file)
    {
        std::cerr << "Cannot open: " << img_path << std::endl;
        return false;
    }

    uint32_t magic = readBigEndian(img_file);
    if (magic != 2051)
    {
        std::cerr << "Invalid image file" << std::endl;
        return false;
    }

    uint32_t count = readBigEndian(img_file);
    rows_ = static_cast<int>(readBigEndian(img_file));
    cols_ = static_cast<int>(readBigEndian(img_file));

    images_.resize(count, Image(rows_ * cols_));
    for (uint32_t i = 0; i < count; ++i)
    {
        for (int j = 0; j < rows_ * cols_; ++j)
        {
            unsigned char pixel;
            img_file.read(reinterpret_cast<char*>(&pixel), 1);
            images_[i][j] = pixel / 255.0;
        }
    }

    std::string lbl_path = path_ + "/" + labels_file;
    std::ifstream lbl_file(lbl_path, std::ios::binary);
    if (!lbl_file)
    {
        std::cerr << "Cannot open: " << lbl_path << std::endl;
        return false;
    }

    magic = readBigEndian(lbl_file);
    if (magic != 2049)
    {
        std::cerr << "Invalid label file" << std::endl;
        return false;
    }

    uint32_t lbl_count = readBigEndian(lbl_file);
    labels_.resize(lbl_count);
    for (uint32_t i = 0; i < lbl_count; ++i)
    {
        unsigned char label;
        lbl_file.read(reinterpret_cast<char*>(&label), 1);
        labels_[i] = static_cast<int>(label);
    }

    std::cout << "Loaded " << count << " images (" << rows_ << "x" << cols_ << ")" << std::endl;
    return true;
}

bool Dataset::loadTraining()
{
    return load("train-images-idx3-ubyte", "train-labels-idx1-ubyte");
}

bool Dataset::loadTest()
{
    return load("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");
}

int Dataset::label(size_t index) const
{
    return index < labels_.size() ? labels_[index] : -1;
}

void Dataset::printImage(size_t index, std::ostream& out) const
{
    if (index < images_.size())
        printImage(images_[index], out);
}

void Dataset::printImage(const Image& image, std::ostream& out)
{
    for (int r = 0; r < IMAGE_SIZE; ++r)
    {
        for (int c = 0; c < IMAGE_SIZE; ++c)
        {
            double val = image[r * IMAGE_SIZE + c];
            if (val > 0.75) out << "@";
            else if (val > 0.5) out << "#";
            else if (val > 0.25) out << "*";
            else out << ".";
        }
        out << "\n";
    }
}

ImageSet Dataset::getImages(size_t start, size_t count) const
{
    size_t end = std::min(start + count, images_.size());
    return ImageSet(images_.begin() + start, images_.begin() + end);
}

Labels Dataset::getLabels(size_t start, size_t count) const
{
    size_t end = std::min(start + count, labels_.size());
    return Labels(labels_.begin() + start, labels_.begin() + end);
}

}
