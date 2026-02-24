#include "dataset.hpp"
#include "knn.hpp"
#include "neural_net.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

namespace mnist {
namespace {

struct TrainConfig {
    std::string model_type = "nn";
    int train_count = 2000;
    int test_count = 500;
    int epochs = 10;
    int batch_size = 32;
    int hidden_size = 128;
    int seed = 42;
    int k = 3;
    double learning_rate = 0.01;
    double l2_lambda = 1e-4;
    bool weighted_knn = true;
};

enum class ParseResult {
    kOk,
    kHelp,
    kError,
};

void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --model <nn|knn>    Classifier type (default: nn)\n"
              << "  --train <count>     Training samples (default: 2000)\n"
              << "  --test <count>      Test samples (default: 500)\n"
              << "  --epochs <count>    Training epochs for NN (default: 10)\n"
              << "  --batch <count>     Batch size for NN (default: 32)\n"
              << "  --hidden <count>    Hidden layer size for NN (default: 128)\n"
              << "  --lr <value>        Learning rate for NN (default: 0.01)\n"
              << "  --l2 <value>        L2 regularization for NN (default: 1e-4)\n"
              << "  --seed <value>      Random seed for NN (default: 42)\n"
              << "  --k <count>         K neighbors for KNN (default: 3)\n"
              << "  --uniform-knn       Use uniform voting instead of distance weighting\n"
              << "  --help              Show this help\n";
}

bool parsePositiveInt(const char* value, const char* flag_name, int& out) {
    try {
        const int parsed = std::stoi(value);
        if (parsed <= 0) {
            std::cerr << flag_name << " must be > 0" << std::endl;
            return false;
        }
        out = parsed;
        return true;
    } catch (const std::exception&) {
        std::cerr << "Invalid value for " << flag_name << ": " << value << std::endl;
        return false;
    }
}

bool parseDouble(const char* value, const char* flag_name, double& out) {
    try {
        out = std::stod(value);
        return true;
    } catch (const std::exception&) {
        std::cerr << "Invalid value for " << flag_name << ": " << value << std::endl;
        return false;
    }
}

ParseResult parseArgs(int argc, char* argv[], TrainConfig& config) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0) {
            printUsage(argv[0]);
            return ParseResult::kHelp;
        }

        if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            config.model_type = argv[++i];
            continue;
        }
        if (std::strcmp(argv[i], "--train") == 0 && i + 1 < argc) {
            if (!parsePositiveInt(argv[++i], "--train", config.train_count)) {
                return ParseResult::kError;
            }
            continue;
        }
        if (std::strcmp(argv[i], "--test") == 0 && i + 1 < argc) {
            if (!parsePositiveInt(argv[++i], "--test", config.test_count)) {
                return ParseResult::kError;
            }
            continue;
        }
        if (std::strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
            if (!parsePositiveInt(argv[++i], "--epochs", config.epochs)) {
                return ParseResult::kError;
            }
            continue;
        }
        if (std::strcmp(argv[i], "--batch") == 0 && i + 1 < argc) {
            if (!parsePositiveInt(argv[++i], "--batch", config.batch_size)) {
                return ParseResult::kError;
            }
            continue;
        }
        if (std::strcmp(argv[i], "--hidden") == 0 && i + 1 < argc) {
            if (!parsePositiveInt(argv[++i], "--hidden", config.hidden_size)) {
                return ParseResult::kError;
            }
            continue;
        }
        if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            if (!parsePositiveInt(argv[++i], "--seed", config.seed)) {
                return ParseResult::kError;
            }
            continue;
        }
        if (std::strcmp(argv[i], "--lr") == 0 && i + 1 < argc) {
            if (!parseDouble(argv[++i], "--lr", config.learning_rate)) {
                return ParseResult::kError;
            }
            continue;
        }
        if (std::strcmp(argv[i], "--l2") == 0 && i + 1 < argc) {
            if (!parseDouble(argv[++i], "--l2", config.l2_lambda)) {
                return ParseResult::kError;
            }
            continue;
        }
        if (std::strcmp(argv[i], "--k") == 0 && i + 1 < argc) {
            if (!parsePositiveInt(argv[++i], "--k", config.k)) {
                return ParseResult::kError;
            }
            continue;
        }
        if (std::strcmp(argv[i], "--uniform-knn") == 0) {
            config.weighted_knn = false;
            continue;
        }

        std::cerr << "Unknown or incomplete argument: " << argv[i] << std::endl;
        printUsage(argv[0]);
        return ParseResult::kError;
    }

    if (config.model_type != "nn" && config.model_type != "knn") {
        std::cerr << "Invalid model type: " << config.model_type << " (expected nn or knn)" << std::endl;
        return ParseResult::kError;
    }
    if (!(config.learning_rate > 0.0)) {
        std::cerr << "--lr must be > 0" << std::endl;
        return ParseResult::kError;
    }
    if (config.l2_lambda < 0.0) {
        std::cerr << "--l2 must be >= 0" << std::endl;
        return ParseResult::kError;
    }

    return ParseResult::kOk;
}

std::unique_ptr<Classifier> createClassifier(const TrainConfig& config, std::string& model_path) {
    model_path = "./models/";

    if (config.model_type == "knn") {
        model_path += "knn.model";
        return std::make_unique<KNN>(config.k, config.weighted_knn);
    }

    model_path += "neural_net.model";
    return std::make_unique<NeuralNet>(config.epochs,
                                       config.batch_size,
                                       config.hidden_size,
                                       config.learning_rate,
                                       config.l2_lambda,
                                       config.seed);
}

void printSamplePredictions(Classifier& classifier, const ImageSet& test_images, const Labels& test_labels) {
    std::cout << "\nSample Predictions:\n";
    std::cout << std::string(40, '-') << std::endl;

    const size_t sample_count = std::min<size_t>(5, test_images.size());
    for (size_t i = 0; i < sample_count; ++i) {
        const int predicted = classifier.predict(test_images[i]);
        const bool correct = predicted == test_labels[i];

        std::cout << "Image " << i
                  << " | True: " << test_labels[i] << " (" << className(test_labels[i]) << ")"
                  << " | Predicted: " << predicted << " (" << className(predicted) << ")"
                  << (correct ? " [OK]" : " [X]") << std::endl;

        Dataset::printImage(test_images[i]);
        std::cout << std::string(40, '-') << std::endl;
    }
}

void printConfusionSummary(Classifier& classifier, const ImageSet& test_images, const Labels& test_labels) {
    std::array<std::array<int, NUM_CLASSES>, NUM_CLASSES> confusion{};
    const size_t count = std::min(test_images.size(), test_labels.size());
    for (size_t i = 0; i < count; ++i) {
        const int truth = test_labels[i];
        const int pred = classifier.predict(test_images[i]);
        if (truth >= 0 && truth < NUM_CLASSES && pred >= 0 && pred < NUM_CLASSES) {
            ++confusion[static_cast<size_t>(truth)][static_cast<size_t>(pred)];
        }
    }

    std::cout << "\nPer-class recall:\n";
    for (int c = 0; c < NUM_CLASSES; ++c) {
        int row_total = 0;
        for (int p = 0; p < NUM_CLASSES; ++p) {
            row_total += confusion[static_cast<size_t>(c)][static_cast<size_t>(p)];
        }
        const double recall = row_total > 0
            ? static_cast<double>(confusion[static_cast<size_t>(c)][static_cast<size_t>(c)]) / static_cast<double>(row_total)
            : 0.0;
        std::cout << "  " << c << " (" << className(c) << "): "
                  << std::fixed << std::setprecision(2) << (recall * 100.0) << "%\n";
    }
}

}  // namespace
}  // namespace mnist

int main(int argc, char* argv[]) {
    mnist::TrainConfig config;
    const mnist::ParseResult parse_result = mnist::parseArgs(argc, argv, config);
    if (parse_result == mnist::ParseResult::kHelp) {
        return 0;
    }
    if (parse_result == mnist::ParseResult::kError) {
        return 1;
    }

    mnist::Dataset train_data("./data");
    mnist::Dataset test_data("./data");

    if (!train_data.loadTraining() || !test_data.loadTest()) {
        std::cerr << "Failed to load dataset" << std::endl;
        return 1;
    }

    const mnist::ImageSet train_images = train_data.getImages(0, static_cast<size_t>(config.train_count));
    const mnist::Labels train_labels = train_data.getLabels(0, static_cast<size_t>(config.train_count));
    const mnist::ImageSet test_images = test_data.getImages(0, static_cast<size_t>(config.test_count));
    const mnist::Labels test_labels = test_data.getLabels(0, static_cast<size_t>(config.test_count));

    if (train_images.empty() || train_labels.empty() || test_images.empty() || test_labels.empty()) {
        std::cerr << "Requested train/test slice is empty" << std::endl;
        return 1;
    }

    std::string model_path;
    std::unique_ptr<mnist::Classifier> classifier = mnist::createClassifier(config, model_path);

    std::cout << "\n[" << classifier->name() << "] Training with "
              << train_images.size() << " samples...\n" << std::endl;

    std::ifstream model_file(model_path);
    if (model_file.good() && config.model_type == "nn") {
        try {
            classifier->load(model_path);
            std::cout << "Loaded model: " << model_path << std::endl;
        } catch (const std::exception& ex) {
            std::cout << "Existing model could not be loaded (" << ex.what() << "). Retraining..." << std::endl;
            classifier->train(train_images, train_labels);
            classifier->save(model_path);
            std::cout << "Saved model: " << model_path << std::endl;
        }
    } else {
        classifier->train(train_images, train_labels);
        if (config.model_type == "nn") {
            classifier->save(model_path);
            std::cout << "Saved model: " << model_path << std::endl;
        }
    }

    const double accuracy = classifier->evaluate(test_images, test_labels);
    std::cout << "\nAccuracy: " << accuracy * 100.0 << "%" << std::endl;
    mnist::printConfusionSummary(*classifier, test_images, test_labels);

    mnist::printSamplePredictions(*classifier, test_images, test_labels);
    return 0;
}
