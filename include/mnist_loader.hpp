#ifndef MNIST_LOADER_HPP
#define MNIST_LOADER_HPP

#include <string>
#include <vector>
#include <cstdint>
#include <random>
#include <algorithm>

class MnistLoader
{
private:
    std::string name_;
    std::string path_;
    std::vector<std::string> required_files_;
    std::vector<std::vector<double>> images_;
    std::vector<int> labels_;
    int rows_;
    int cols_;

public:
    MnistLoader(const std::string& name, const std::string& path);

    bool loadImages(const std::string& filename);
    bool loadLabels(const std::string& filename);
    bool checkFiles() const;

    void printInfo() const;
    void printImage(int index) const;
    void showSamples(int count = 5) const;
    void showRandomSamples(int count = 5) const;

    int getLabel(int index) const;
    int numImages() const { return static_cast<int>(images_.size()); }
    int rows() const { return rows_; }
    int cols() const { return cols_; }

    const std::vector<std::vector<double>>& images() const { return images_; }
    const std::vector<int>& labels() const { return labels_; }
};

#endif
