#ifndef DATASET_HPP
#define DATASET_HPP

#include "types.hpp"
#include <string>
#include <iostream>
#include <fstream>
#include <cstdint>

namespace mnist {

class Dataset
{
private:
    std::string path_;
    ImageSet images_;
    Labels labels_;
    int rows_ = 0;
    int cols_ = 0;

    static uint32_t readBigEndian(std::ifstream& file);

public:
    explicit Dataset(const std::string& path = "./data");

    bool load(const std::string& images_file, const std::string& labels_file);
    bool loadTraining();
    bool loadTest();

    void printImage(size_t index, std::ostream& out = std::cout) const;
    static void printImage(const Image& image, std::ostream& out = std::cout);

    size_t size() const { return images_.size(); }
    int rows() const { return rows_; }
    int cols() const { return cols_; }

    const ImageSet& images() const { return images_; }
    const Labels& labels() const { return labels_; }
    int label(size_t index) const;

    ImageSet getImages(size_t start, size_t count) const;
    Labels getLabels(size_t start, size_t count) const;
};

}

#endif
