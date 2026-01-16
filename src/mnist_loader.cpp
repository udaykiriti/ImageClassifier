#include "mnist_loader.hpp"
#include <fstream>
#include <iostream>
#include <numeric>
#include <ctime>

namespace {
    uint32_t readBigEndian(std::ifstream& file)
    {
        unsigned char bytes[4];
        file.read(reinterpret_cast<char*>(bytes), 4);
        return (static_cast<uint32_t>(bytes[0]) << 24) |
               (static_cast<uint32_t>(bytes[1]) << 16) |
               (static_cast<uint32_t>(bytes[2]) << 8) |
               static_cast<uint32_t>(bytes[3]);
    }
}

MnistLoader::MnistLoader(const std::string& name, const std::string& path)
    : name_(name), path_(path), rows_(0), cols_(0)
{
    required_files_ = {
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte"
    };
    std::cout << "MNIST loader initialized: " << name_ << std::endl;
}

bool MnistLoader::checkFiles() const
{
    bool all_exist = true;
    for (const auto& file : required_files_)
    {
        std::string full_path = path_ + "/" + file;
        std::ifstream f(full_path);
        if (!f)
        {
            std::cerr << "Missing: " << full_path << std::endl;
            all_exist = false;
        }
    }
    return all_exist;
}

bool MnistLoader::loadImages(const std::string& filename)
{
    std::string filepath = path_ + "/" + filename;
    std::ifstream file(filepath, std::ios::binary);
    if (!file)
    {
        std::cerr << "Cannot open: " << filepath << std::endl;
        return false;
    }

    uint32_t magic = readBigEndian(file);
    if (magic != 2051)
    {
        std::cerr << "Invalid MNIST image file (magic=" << magic << ")" << std::endl;
        return false;
    }

    uint32_t count = readBigEndian(file);
    rows_ = static_cast<int>(readBigEndian(file));
    cols_ = static_cast<int>(readBigEndian(file));

    images_.resize(count, std::vector<double>(rows_ * cols_));

    for (uint32_t i = 0; i < count; ++i)
    {
        for (int j = 0; j < rows_ * cols_; ++j)
        {
            unsigned char pixel;
            file.read(reinterpret_cast<char*>(&pixel), 1);
            images_[i][j] = pixel / 255.0;
        }
    }

    std::cout << "Loaded " << count << " images (" << rows_ << "x" << cols_ << ")" << std::endl;
    return true;
}

bool MnistLoader::loadLabels(const std::string& filename)
{
    std::string filepath = path_ + "/" + filename;
    std::ifstream file(filepath, std::ios::binary);
    if (!file)
    {
        std::cerr << "Cannot open: " << filepath << std::endl;
        return false;
    }

    uint32_t magic = readBigEndian(file);
    if (magic != 2049)
    {
        std::cerr << "Invalid MNIST label file (magic=" << magic << ")" << std::endl;
        return false;
    }

    uint32_t count = readBigEndian(file);
    labels_.resize(count);

    for (uint32_t i = 0; i < count; ++i)
    {
        unsigned char label;
        file.read(reinterpret_cast<char*>(&label), 1);
        labels_[i] = static_cast<int>(label);
    }

    std::cout << "Loaded " << count << " labels" << std::endl;
    return true;
}

int MnistLoader::getLabel(int index) const
{
    if (index < 0 || index >= static_cast<int>(labels_.size()))
        return -1;
    return labels_[index];
}

void MnistLoader::printInfo() const
{
    std::cout << "Dataset: " << name_ << std::endl;
    std::cout << "Path: " << path_ << std::endl;
    std::cout << "Images: " << images_.size() << std::endl;
}

void MnistLoader::printImage(int index) const
{
    if (index < 0 || index >= static_cast<int>(images_.size()))
        return;

    for (int r = 0; r < rows_; ++r)
    {
        for (int c = 0; c < cols_; ++c)
        {
            double val = images_[index][r * cols_ + c];
            if (val > 0.75)
                std::cout << "@";
            else if (val > 0.5)
                std::cout << "#";
            else if (val > 0.25)
                std::cout << "*";
            else
                std::cout << ".";
        }
        std::cout << "\n";
    }
}

void MnistLoader::showSamples(int count) const
{
    if (images_.empty()) return;

    int limit = std::min(count, static_cast<int>(images_.size()));
    for (int i = 0; i < limit; ++i)
    {
        std::cout << "Image " << i << " - Label: " << getLabel(i) << "\n";
        printImage(i);
        std::cout << "--------------------\n";
    }
}

void MnistLoader::showRandomSamples(int count) const
{
    if (images_.empty()) return;

    count = std::min(count, static_cast<int>(images_.size()));
    std::vector<int> indices(images_.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::mt19937 rng(static_cast<unsigned int>(std::time(nullptr)));
    std::shuffle(indices.begin(), indices.end(), rng);

    for (int i = 0; i < count; ++i)
    {
        int idx = indices[i];
        std::cout << "Image " << idx << " - Label: " << getLabel(idx) << "\n";
        printImage(idx);
        std::cout << "--------------------\n";
    }
}
