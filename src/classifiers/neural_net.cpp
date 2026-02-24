#include "neural_net.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace mnist {
namespace {

constexpr const char* kModelHeader = "ICNN_V2";
constexpr double kEpsilon = 1e-12;

double dotProduct(const double* a, const double* b, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

void writeVector(std::ofstream& out, const std::vector<double>& vec) {
    const uint64_t n = static_cast<uint64_t>(vec.size());
    out.write(reinterpret_cast<const char*>(&n), sizeof(n));
    out.write(reinterpret_cast<const char*>(vec.data()), static_cast<std::streamsize>(n * sizeof(double)));
}

bool readVector(std::ifstream& in, std::vector<double>& vec) {
    uint64_t n = 0;
    if (!in.read(reinterpret_cast<char*>(&n), sizeof(n))) {
        return false;
    }

    vec.resize(static_cast<size_t>(n));
    return static_cast<bool>(in.read(reinterpret_cast<char*>(vec.data()), static_cast<std::streamsize>(n * sizeof(double))));
}

}  // namespace

NeuralNet::NeuralNet(int epochs,
                     int batch_size,
                     int hidden_size,
                     double learning_rate,
                     double l2_lambda,
                     int random_seed)
    : epochs_(epochs),
      batch_size_(batch_size),
      hidden_size_(hidden_size),
      learning_rate_(learning_rate),
      l2_lambda_(l2_lambda),
      random_seed_(random_seed) {
    if (epochs_ <= 0) {
        throw std::invalid_argument("epochs must be > 0");
    }
    if (batch_size_ <= 0) {
        throw std::invalid_argument("batch_size must be > 0");
    }
    if (hidden_size_ <= 0) {
        throw std::invalid_argument("hidden_size must be > 0");
    }
    if (!(learning_rate_ > 0.0)) {
        throw std::invalid_argument("learning_rate must be > 0");
    }
    if (l2_lambda_ < 0.0) {
        throw std::invalid_argument("l2_lambda must be >= 0");
    }

    buildNetwork();
}

void NeuralNet::buildNetwork() {
    if (!w1_.empty()) {
        return;
    }

    std::mt19937 rng(static_cast<uint32_t>(random_seed_));
    const double scale1 = std::sqrt(2.0 / static_cast<double>(IMAGE_PIXELS));
    const double scale2 = std::sqrt(2.0 / static_cast<double>(hidden_size_));
    std::normal_distribution<double> dist1(0.0, scale1);
    std::normal_distribution<double> dist2(0.0, scale2);

    w1_.resize(static_cast<size_t>(hidden_size_) * IMAGE_PIXELS);
    b1_.assign(static_cast<size_t>(hidden_size_), 0.0);
    w2_.resize(static_cast<size_t>(NUM_CLASSES) * static_cast<size_t>(hidden_size_));
    b2_.assign(NUM_CLASSES, 0.0);

    for (double& w : w1_) {
        w = dist1(rng);
    }
    for (double& w : w2_) {
        w = dist2(rng);
    }
}

std::vector<double> NeuralNet::forward(const Image& image,
                                       std::vector<double>& hidden_pre,
                                       std::vector<double>& hidden_act) const {
    if (image.size() != IMAGE_PIXELS) {
        throw std::runtime_error("NeuralNet expects 784 input values");
    }

    hidden_pre.assign(static_cast<size_t>(hidden_size_), 0.0);
    hidden_act.assign(static_cast<size_t>(hidden_size_), 0.0);

    for (int h = 0; h < hidden_size_; ++h) {
        const double* w_row = &w1_[static_cast<size_t>(h) * IMAGE_PIXELS];
        const double z = dotProduct(w_row, image.data(), IMAGE_PIXELS) + b1_[static_cast<size_t>(h)];
        hidden_pre[static_cast<size_t>(h)] = z;
        hidden_act[static_cast<size_t>(h)] = std::max(0.0, z);
    }

    std::vector<double> probs(NUM_CLASSES, 0.0);
    for (int o = 0; o < NUM_CLASSES; ++o) {
        const double* w_row = &w2_[static_cast<size_t>(o) * static_cast<size_t>(hidden_size_)];
        probs[static_cast<size_t>(o)] = dotProduct(w_row, hidden_act.data(), static_cast<size_t>(hidden_size_)) + b2_[static_cast<size_t>(o)];
    }

    const double max_logit = *std::max_element(probs.begin(), probs.end());
    double sum_exp = 0.0;
    for (double& v : probs) {
        v = std::exp(v - max_logit);
        sum_exp += v;
    }

    if (sum_exp <= 0.0) {
        throw std::runtime_error("Numerical instability in softmax");
    }

    for (double& v : probs) {
        v /= sum_exp;
    }

    return probs;
}

void NeuralNet::train(const ImageSet& images, const Labels& labels) {
    const size_t count = std::min(images.size(), labels.size());
    if (count == 0) {
        throw std::runtime_error("No training data provided");
    }

    buildNetwork();

    std::vector<size_t> indices(count);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(static_cast<uint32_t>(random_seed_ + 1));

    std::vector<double> gw1(w1_.size(), 0.0);
    std::vector<double> gb1(b1_.size(), 0.0);
    std::vector<double> gw2(w2_.size(), 0.0);
    std::vector<double> gb2(b2_.size(), 0.0);
    std::vector<double> hidden_pre;
    std::vector<double> hidden_act;
    std::vector<double> dz2(NUM_CLASSES, 0.0);
    std::vector<double> dhidden(static_cast<size_t>(hidden_size_), 0.0);

    for (int epoch = 0; epoch < epochs_; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), rng);

        double epoch_loss = 0.0;
        int epoch_correct = 0;

        for (size_t batch_start = 0; batch_start < count; batch_start += static_cast<size_t>(batch_size_)) {
            const size_t batch_end = std::min(count, batch_start + static_cast<size_t>(batch_size_));
            const size_t batch_n = batch_end - batch_start;

            std::fill(gw1.begin(), gw1.end(), 0.0);
            std::fill(gb1.begin(), gb1.end(), 0.0);
            std::fill(gw2.begin(), gw2.end(), 0.0);
            std::fill(gb2.begin(), gb2.end(), 0.0);

            for (size_t bi = batch_start; bi < batch_end; ++bi) {
                const size_t idx = indices[bi];
                const Image& x = images[idx];
                const int y = labels[idx];
                if (y < 0 || y >= NUM_CLASSES) {
                    throw std::runtime_error("Invalid class label in training data");
                }

                const std::vector<double> probs = forward(x, hidden_pre, hidden_act);
                const int predicted = static_cast<int>(std::distance(probs.begin(), std::max_element(probs.begin(), probs.end())));
                if (predicted == y) {
                    ++epoch_correct;
                }
                epoch_loss += -std::log(std::max(probs[static_cast<size_t>(y)], kEpsilon));

                dz2 = probs;
                dz2[static_cast<size_t>(y)] -= 1.0;

                for (int o = 0; o < NUM_CLASSES; ++o) {
                    gb2[static_cast<size_t>(o)] += dz2[static_cast<size_t>(o)];
                    for (int h = 0; h < hidden_size_; ++h) {
                        gw2[static_cast<size_t>(o) * static_cast<size_t>(hidden_size_) + static_cast<size_t>(h)] +=
                            dz2[static_cast<size_t>(o)] * hidden_act[static_cast<size_t>(h)];
                    }
                }

                std::fill(dhidden.begin(), dhidden.end(), 0.0);
                for (int h = 0; h < hidden_size_; ++h) {
                    double grad = 0.0;
                    for (int o = 0; o < NUM_CLASSES; ++o) {
                        grad += w2_[static_cast<size_t>(o) * static_cast<size_t>(hidden_size_) + static_cast<size_t>(h)] * dz2[static_cast<size_t>(o)];
                    }
                    if (hidden_pre[static_cast<size_t>(h)] <= 0.0) {
                        grad = 0.0;
                    }
                    dhidden[static_cast<size_t>(h)] = grad;
                }

                for (int h = 0; h < hidden_size_; ++h) {
                    gb1[static_cast<size_t>(h)] += dhidden[static_cast<size_t>(h)];
                    for (size_t i = 0; i < IMAGE_PIXELS; ++i) {
                        gw1[static_cast<size_t>(h) * IMAGE_PIXELS + i] += dhidden[static_cast<size_t>(h)] * x[i];
                    }
                }
            }

            const double lr = (learning_rate_ / (1.0 + 0.05 * static_cast<double>(epoch))) / static_cast<double>(batch_n);
            for (size_t i = 0; i < w1_.size(); ++i) {
                const double l2_grad = l2_lambda_ * w1_[i];
                w1_[i] -= lr * (gw1[i] + l2_grad);
            }
            for (size_t i = 0; i < b1_.size(); ++i) {
                b1_[i] -= lr * gb1[i];
            }
            for (size_t i = 0; i < w2_.size(); ++i) {
                const double l2_grad = l2_lambda_ * w2_[i];
                w2_[i] -= lr * (gw2[i] + l2_grad);
            }
            for (size_t i = 0; i < b2_.size(); ++i) {
                b2_[i] -= lr * gb2[i];
            }
        }

        const double avg_loss = epoch_loss / static_cast<double>(count);
        const double train_acc = static_cast<double>(epoch_correct) * 100.0 / static_cast<double>(count);
        std::cout << "Epoch " << (epoch + 1) << "/" << epochs_
                  << " | loss=" << avg_loss
                  << " | train_acc=" << train_acc << "%" << std::endl;
    }

    std::cout << "Training completed (" << epochs_ << " epochs)" << std::endl;
}

int NeuralNet::predict(const Image& image) {
    std::vector<double> hidden_pre;
    std::vector<double> hidden_act;
    const std::vector<double> probs = forward(image, hidden_pre, hidden_act);

    return static_cast<int>(std::distance(probs.begin(), std::max_element(probs.begin(), probs.end())));
}

std::vector<double> NeuralNet::predictProba(const Image& image) const {
    std::vector<double> hidden_pre;
    std::vector<double> hidden_act;
    return forward(image, hidden_pre, hidden_act);
}

void NeuralNet::save(const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open model file for write: " + path);
    }

    out.write(kModelHeader, static_cast<std::streamsize>(std::strlen(kModelHeader)));

    const int32_t input_size = IMAGE_PIXELS;
    const int32_t hidden = hidden_size_;
    const int32_t output = NUM_CLASSES;
    const int32_t seed = random_seed_;
    const double lr = learning_rate_;
    const double l2 = l2_lambda_;

    out.write(reinterpret_cast<const char*>(&input_size), sizeof(input_size));
    out.write(reinterpret_cast<const char*>(&hidden), sizeof(hidden));
    out.write(reinterpret_cast<const char*>(&output), sizeof(output));
    out.write(reinterpret_cast<const char*>(&seed), sizeof(seed));
    out.write(reinterpret_cast<const char*>(&lr), sizeof(lr));
    out.write(reinterpret_cast<const char*>(&l2), sizeof(l2));

    writeVector(out, w1_);
    writeVector(out, b1_);
    writeVector(out, w2_);
    writeVector(out, b2_);

    if (!out) {
        throw std::runtime_error("failed while writing model: " + path);
    }
}

void NeuralNet::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open:" + path);
    }

    char header[8] = {};
    if (!in.read(header, static_cast<std::streamsize>(std::strlen(kModelHeader)))) {
        throw std::runtime_error("invalid model file header");
    }
    if (std::string(header, std::strlen(kModelHeader)) != kModelHeader) {
        throw std::runtime_error("unsupported model format");
    }

    int32_t input_size = 0;
    int32_t hidden = 0;
    int32_t output = 0;
    int32_t seed = 0;
    double lr = 0.0;
    double l2 = 0.0;

    if (!in.read(reinterpret_cast<char*>(&input_size), sizeof(input_size)) ||
        !in.read(reinterpret_cast<char*>(&hidden), sizeof(hidden)) ||
        !in.read(reinterpret_cast<char*>(&output), sizeof(output)) ||
        !in.read(reinterpret_cast<char*>(&seed), sizeof(seed)) ||
        !in.read(reinterpret_cast<char*>(&lr), sizeof(lr)) ||
        !in.read(reinterpret_cast<char*>(&l2), sizeof(l2))) {
        throw std::runtime_error("corrupt model metadata");
    }

    if (input_size != IMAGE_PIXELS || output != NUM_CLASSES || hidden <= 0) {
        throw std::runtime_error("model architecture mismatch");
    }
    if (!(lr > 0.0) || l2 < 0.0) {
        throw std::runtime_error("invalid optimization metadata in model");
    }

    hidden_size_ = hidden;
    random_seed_ = seed;
    learning_rate_ = lr;
    l2_lambda_ = l2;

    if (!readVector(in, w1_) || !readVector(in, b1_) || !readVector(in, w2_) || !readVector(in, b2_)) {
        throw std::runtime_error("corrupt model weights");
    }

    const size_t expected_w1 = static_cast<size_t>(hidden_size_) * IMAGE_PIXELS;
    const size_t expected_b1 = static_cast<size_t>(hidden_size_);
    const size_t expected_w2 = static_cast<size_t>(NUM_CLASSES) * static_cast<size_t>(hidden_size_);
    const size_t expected_b2 = NUM_CLASSES;

    if (w1_.size() != expected_w1 || b1_.size() != expected_b1 || w2_.size() != expected_w2 || b2_.size() != expected_b2) {
        throw std::runtime_error("model weight sizes do not match metadata");
    }
}

}  // namespace mnist
