#include "dataset.hpp"
#include "neural_net.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

namespace mnist {
namespace {

struct PredictConfig {
    std::string image_path = "./data/image.txt";
    std::string model_path = "./models/neural_net.model";
    int true_label = -1;
    bool show_image = false;
    int top_k = 0;
};

enum class ParseResult {
    kOk,
    kHelp,
    kError,
};

void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --image <path>      Input image path (.txt or image file, default: ./data/image.txt)\n"
              << "  --model <path>      Model file (default: ./models/neural_net.model)\n"
              << "  --label <class_id>  True Fashion-MNIST class id (0-9, optional)\n"
              << "  --topk <count>      Show top-k class probabilities (default: 0)\n"
              << "  --show              Show ASCII visualization\n"
              << "  --help              Show this help\n";
}

std::string shellQuote(const std::string& value) {
    std::string escaped = "'";
    for (char ch : value) {
        if (ch == '\'') {
            escaped += "'\\''";
        } else {
            escaped += ch;
        }
    }
    escaped += "'";
    return escaped;
}

bool parseInt(const char* value, const char* flag_name, int& out) {
    try {
        out = std::stoi(value);
        return true;
    } catch (const std::exception&) {
        std::cerr << "Invalid value for " << flag_name << ": " << value << std::endl;
        return false;
    }
}

ParseResult parseArgs(int argc, char* argv[], PredictConfig& config) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0) {
            printUsage(argv[0]);
            return ParseResult::kHelp;
        }

        if (std::strcmp(argv[i], "--image") == 0 && i + 1 < argc) {
            config.image_path = argv[++i];
            continue;
        }
        if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            config.model_path = argv[++i];
            continue;
        }
        if (std::strcmp(argv[i], "--label") == 0 && i + 1 < argc) {
            if (!parseInt(argv[++i], "--label", config.true_label)) {
                return ParseResult::kError;
            }
            continue;
        }
        if (std::strcmp(argv[i], "--topk") == 0 && i + 1 < argc) {
            if (!parseInt(argv[++i], "--topk", config.top_k)) {
                return ParseResult::kError;
            }
            continue;
        }
        if (std::strcmp(argv[i], "--show") == 0) {
            config.show_image = true;
            continue;
        }

        std::cerr << "Unknown or incomplete argument: " << argv[i] << std::endl;
        printUsage(argv[0]);
        return ParseResult::kError;
    }

    if (config.true_label < -1 || config.true_label >= NUM_CLASSES) {
        std::cerr << "--label must be between 0 and " << (NUM_CLASSES - 1)
                  << " (or omitted)" << std::endl;
        return ParseResult::kError;
    }
    if (config.top_k < 0 || config.top_k > NUM_CLASSES) {
        std::cerr << "--topk must be between 0 and " << NUM_CLASSES << std::endl;
        return ParseResult::kError;
    }

    return ParseResult::kOk;
}

bool normalizePixel(double raw_value, double& normalized) {
    if (!std::isfinite(raw_value)) {
        return false;
    }

    // Accept either [0, 255] integer-like values or [0, 1] normalized values.
    if (raw_value >= 0.0 && raw_value <= 1.0) {
        normalized = raw_value;
        return true;
    }
    if (raw_value >= 0.0 && raw_value <= 255.0) {
        normalized = raw_value / 255.0;
        return true;
    }

    return false;
}

Image loadImageFromTextFile(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Cannot open: " << path << std::endl;
        return {};
    }

    Image image;
    image.reserve(IMAGE_PIXELS);

    double value = 0.0;
    while (file >> value) {
        double normalized = 0.0;
        if (!normalizePixel(value, normalized)) {
            std::cerr << "Invalid pixel value " << value
                      << " in " << path << " (expected 0..255 or 0..1)" << std::endl;
            return {};
        }
        image.push_back(normalized);
    }

    if (image.size() != IMAGE_PIXELS) {
        std::cerr << "Expected " << IMAGE_PIXELS << " pixels, got "
                  << image.size() << std::endl;
        return {};
    }

    return image;
}

Image loadImageFromAnyFile(const std::string& path) {
    namespace fs = std::filesystem;

    const fs::path input_path(path);
    if (!fs::exists(input_path)) {
        std::cerr << "Input file does not exist: " << path << std::endl;
        return {};
    }

    std::string ext = input_path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (ext == ".txt") {
        return loadImageFromTextFile(path);
    }

    const fs::path tmp_path = fs::temp_directory_path() / "imageclassifier_predict_input.txt";
    const std::string command = "python3 scripts/preprocess.py " +
        shellQuote(input_path.string()) + " " + shellQuote(tmp_path.string());

    const int rc = std::system(command.c_str());
    if (rc != 0 || !fs::exists(tmp_path)) {
        std::cerr << "Failed to preprocess image via scripts/preprocess.py (exit code " << rc << ")" << std::endl;
        std::cerr << "Hint: install preprocessing deps with: pip install pillow numpy" << std::endl;
        return {};
    }

    Image image = loadImageFromTextFile(tmp_path.string());
    std::error_code ec;
    fs::remove(tmp_path, ec);
    return image;
}

}  // namespace
}  // namespace mnist

int main(int argc, char* argv[]) {
    mnist::PredictConfig config;
    const mnist::ParseResult parse_result = mnist::parseArgs(argc, argv, config);
    if (parse_result == mnist::ParseResult::kHelp) {
        return 0;
    }
    if (parse_result == mnist::ParseResult::kError) {
        return 1;
    }

    const mnist::Image image = mnist::loadImageFromAnyFile(config.image_path);
    if (image.empty()) {
        return 1;
    }

    mnist::NeuralNet model;
    try {
        model.load(config.model_path);
    } catch (const std::exception& ex) {
        std::cerr << "Failed to load model from " << config.model_path
                  << ": " << ex.what() << std::endl;
        return 1;
    }

    int prediction = -1;
    std::vector<double> probs;
    try {
        prediction = model.predict(image);
        if (config.top_k > 0) {
            probs = model.predictProba(image);
        }
    } catch (const std::exception& ex) {
        std::cerr << "Prediction failed: " << ex.what() << std::endl;
        return 1;
    }

    if (config.show_image) {
        std::cout << "\nASCII Image:\n";
        mnist::Dataset::printImage(image);
    }

    std::cout << "\nPredicted class: " << prediction
              << " (" << mnist::className(prediction) << ")" << std::endl;

    if (!probs.empty()) {
        std::vector<int> classes(mnist::NUM_CLASSES);
        for (int i = 0; i < mnist::NUM_CLASSES; ++i) {
            classes[static_cast<size_t>(i)] = i;
        }
        std::partial_sort(classes.begin(),
                          classes.begin() + config.top_k,
                          classes.end(),
                          [&probs](int a, int b) { return probs[static_cast<size_t>(a)] > probs[static_cast<size_t>(b)]; });
        std::cout << "Top-" << config.top_k << " probabilities:\n";
        for (int rank = 0; rank < config.top_k; ++rank) {
            const int cls = classes[static_cast<size_t>(rank)];
            std::cout << "  " << cls << " (" << mnist::className(cls) << "): "
                      << std::fixed << std::setprecision(2) << (probs[static_cast<size_t>(cls)] * 100.0) << "%\n";
        }
    }

    if (config.true_label >= 0) {
        std::cout << "True class: " << config.true_label
                  << " (" << mnist::className(config.true_label) << ")" << std::endl;
        std::cout << (prediction == config.true_label ? "CORRECT" : "INCORRECT") << std::endl;
    }

    return 0;
}
