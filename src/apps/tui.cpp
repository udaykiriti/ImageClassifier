#include "dataset.hpp"
#include "knn.hpp"
#include "neural_net.hpp"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
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
    int sample_predictions = 3;
};

struct PredictConfig {
    std::string image_path = "./data/image.txt";
    std::string model_path = "./models/neural_net.model";
    std::optional<int> true_label;
    bool show_image = true;
};

struct DatasetCache {
    Dataset train{"./data"};
    Dataset test{"./data"};
    bool loaded = false;
};

std::string trim(std::string s) {
    auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    return s;
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

void printDivider() {
    std::cout << "\n" << std::string(64, '=') << "\n";
}

void printHeader() {
    printDivider();
    std::cout << "Fashion-MNIST Image Classifier TUI\n";
    printDivider();
}

std::string askLine(const std::string& prompt, const std::string& default_value = "") {
    std::cout << prompt;
    if (!default_value.empty()) {
        std::cout << " [" << default_value << "]";
    }
    std::cout << ": ";

    std::string line;
    std::getline(std::cin, line);
    if (!std::cin) {
        return default_value;
    }

    line = trim(line);
    if (line.empty()) {
        return default_value;
    }
    return line;
}

int askInt(const std::string& prompt, int default_value, int min_value, int max_value) {
    while (true) {
        const std::string raw = askLine(prompt, std::to_string(default_value));
        try {
            const int value = std::stoi(raw);
            if (value < min_value || value > max_value) {
                std::cout << "Please enter a value between " << min_value << " and " << max_value << ".\n";
                continue;
            }
            return value;
        } catch (const std::exception&) {
            std::cout << "Invalid number. Try again.\n";
        }
    }
}

bool askYesNo(const std::string& prompt, bool default_value) {
    while (true) {
        const std::string default_text = default_value ? "Y/n" : "y/N";
        std::string value = askLine(prompt + " (" + default_text + ")", "");
        if (value.empty()) {
            return default_value;
        }

        for (char& ch : value) {
            ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        }

        if (value == "y" || value == "yes") {
            return true;
        }
        if (value == "n" || value == "no") {
            return false;
        }

        std::cout << "Please answer yes or no.\n";
    }
}

void waitForEnter() {
    std::cout << "\nPress Enter to continue...";
    std::string line;
    std::getline(std::cin, line);
}

bool ensureDatasetLoaded(DatasetCache& cache) {
    if (cache.loaded) {
        return true;
    }

    std::cout << "Loading Fashion-MNIST data from ./data ...\n";
    if (!cache.train.loadTraining() || !cache.test.loadTest()) {
        std::cout << "Failed to load dataset. Run ./scripts/download_fashion_mnist.sh and try again.\n";
        return false;
    }

    cache.loaded = true;
    return true;
}

bool normalizePixel(double raw_value, double& normalized) {
    if (!std::isfinite(raw_value)) {
        return false;
    }
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
        std::cout << "Cannot open image file: " << path << "\n";
        return {};
    }

    Image image;
    image.reserve(IMAGE_PIXELS);

    double value = 0.0;
    while (file >> value) {
        double normalized = 0.0;
        if (!normalizePixel(value, normalized)) {
            std::cout << "Invalid pixel value " << value << " in " << path
                      << " (expected 0..255 or 0..1).\n";
            return {};
        }
        image.push_back(normalized);
    }

    if (image.size() != IMAGE_PIXELS) {
        std::cout << "Expected " << IMAGE_PIXELS << " pixels, got " << image.size() << ".\n";
        return {};
    }

    return image;
}

Image loadImageFromAnyFile(const std::string& path) {
    namespace fs = std::filesystem;

    const fs::path input_path(path);
    if (!fs::exists(input_path)) {
        std::cout << "Input file does not exist: " << path << "\n";
        return {};
    }

    std::string ext = input_path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (ext == ".txt") {
        return loadImageFromTextFile(path);
    }

    const fs::path tmp_path = fs::temp_directory_path() / "imageclassifier_tui_input.txt";
    const std::string command = "python3 scripts/preprocess.py " +
        shellQuote(input_path.string()) + " " + shellQuote(tmp_path.string());
    const int rc = std::system(command.c_str());
    if (rc != 0 || !fs::exists(tmp_path)) {
        std::cout << "Failed to preprocess image via scripts/preprocess.py (exit code " << rc << ")\n";
        std::cout << "Hint: install preprocessing deps with: pip install pillow numpy\n";
        return {};
    }

    Image image = loadImageFromTextFile(tmp_path.string());
    std::error_code ec;
    fs::remove(tmp_path, ec);
    return image;
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

void printClassLegend() {
    printDivider();
    std::cout << "Fashion-MNIST Classes\n";
    for (int i = 0; i < NUM_CLASSES; ++i) {
        std::cout << "  " << i << " -> " << className(i) << "\n";
    }
}

TrainConfig promptTrainConfig() {
    TrainConfig config;
    printDivider();
    std::cout << "Train / Evaluate Model\n";

    while (true) {
        const std::string model = askLine("Model type (nn/knn)", config.model_type);
        if (model == "nn" || model == "knn") {
            config.model_type = model;
            break;
        }
        std::cout << "Please enter nn or knn.\n";
    }

    config.train_count = askInt("Training samples", config.train_count, 1, 60000);
    config.test_count = askInt("Test samples", config.test_count, 1, 10000);
    if (config.model_type == "nn") {
        config.epochs = askInt("Epochs", config.epochs, 1, 1000);
        config.batch_size = askInt("Batch size", config.batch_size, 1, 4096);
        config.hidden_size = askInt("Hidden size", config.hidden_size, 1, 2048);
        while (true) {
            const std::string raw = askLine("Learning rate", std::to_string(config.learning_rate));
            try {
                config.learning_rate = std::stod(raw);
                if (config.learning_rate <= 0.0) {
                    std::cout << "Learning rate must be > 0.\n";
                    continue;
                }
                break;
            } catch (const std::exception&) {
                std::cout << "Invalid learning rate.\n";
            }
        }
        while (true) {
            const std::string raw = askLine("L2 regularization", std::to_string(config.l2_lambda));
            try {
                config.l2_lambda = std::stod(raw);
                if (config.l2_lambda < 0.0) {
                    std::cout << "L2 regularization must be >= 0.\n";
                    continue;
                }
                break;
            } catch (const std::exception&) {
                std::cout << "Invalid L2 value.\n";
            }
        }
        config.seed = askInt("Random seed", config.seed, 1, 1000000000);
    } else {
        config.k = askInt("K neighbors", config.k, 1, 10000);
        config.weighted_knn = askYesNo("Use distance-weighted voting", config.weighted_knn);
    }
    config.sample_predictions = askInt("Show sample predictions", config.sample_predictions, 0, 10);

    return config;
}

void runTrainingWorkflow(DatasetCache& cache) {
    if (!ensureDatasetLoaded(cache)) {
        return;
    }

    const TrainConfig config = promptTrainConfig();

    const ImageSet train_images = cache.train.getImages(0, static_cast<size_t>(config.train_count));
    const Labels train_labels = cache.train.getLabels(0, static_cast<size_t>(config.train_count));
    const ImageSet test_images = cache.test.getImages(0, static_cast<size_t>(config.test_count));
    const Labels test_labels = cache.test.getLabels(0, static_cast<size_t>(config.test_count));

    if (train_images.empty() || train_labels.empty() || test_images.empty() || test_labels.empty()) {
        std::cout << "Requested train/test slice is empty.\n";
        return;
    }

    std::string model_path;
    std::unique_ptr<Classifier> classifier = createClassifier(config, model_path);

    printDivider();
    std::cout << "[" << classifier->name() << "] Training with " << train_images.size() << " samples...\n";

    try {
        std::ifstream model_file(model_path);
        if (config.model_type == "nn" && model_file.good() && askYesNo("Load existing NN model instead of retraining", true)) {
            try {
                classifier->load(model_path);
                std::cout << "Loaded model: " << model_path << "\n";
            } catch (const std::exception& ex) {
                std::cout << "Existing model could not be loaded (" << ex.what() << "). Retraining...\n";
                classifier->train(train_images, train_labels);
                classifier->save(model_path);
                std::cout << "Saved model: " << model_path << "\n";
            }
        } else {
            classifier->train(train_images, train_labels);
            if (config.model_type == "nn") {
                classifier->save(model_path);
                std::cout << "Saved model: " << model_path << "\n";
            }
        }

        const double accuracy = classifier->evaluate(test_images, test_labels);
        std::cout << "Accuracy: " << (accuracy * 100.0) << "%\n";

        const size_t sample_count = std::min(static_cast<size_t>(config.sample_predictions), test_images.size());
        if (sample_count > 0) {
            printDivider();
            std::cout << "Sample Predictions\n";
            for (size_t i = 0; i < sample_count; ++i) {
                const int predicted = classifier->predict(test_images[i]);
                const bool correct = predicted == test_labels[i];
                std::cout << "Image " << i
                          << " | True: " << test_labels[i] << " (" << className(test_labels[i]) << ")"
                          << " | Predicted: " << predicted << " (" << className(predicted) << ")"
                          << (correct ? " [OK]" : " [X]") << "\n";
            }
        }
    } catch (const std::exception& ex) {
        std::cout << "Operation failed: " << ex.what() << "\n";
    }
}

PredictConfig promptPredictConfig() {
    PredictConfig config;

    printDivider();
    std::cout << "Predict from Image\n";
    config.image_path = askLine("Image text path", config.image_path);
    config.model_path = askLine("NN model path", config.model_path);
    config.show_image = askYesNo("Show ASCII image", config.show_image);

    const std::string label_raw = askLine("True class label (0-9, optional)", "");
    if (!label_raw.empty()) {
        try {
            const int label = std::stoi(label_raw);
            if (label < 0 || label >= NUM_CLASSES) {
                std::cout << "Ignoring invalid label (must be 0-9).\n";
            } else {
                config.true_label = label;
            }
        } catch (const std::exception&) {
            std::cout << "Ignoring invalid label input.\n";
        }
    }

    return config;
}

void runPredictionWorkflow() {
    const PredictConfig config = promptPredictConfig();

    const Image image = loadImageFromAnyFile(config.image_path);
    if (image.empty()) {
        return;
    }

    try {
        NeuralNet model;
        model.load(config.model_path);

        const int prediction = model.predict(image);

        if (config.show_image) {
            printDivider();
            std::cout << "ASCII Image\n";
            Dataset::printImage(image);
        }

        printDivider();
        std::cout << "Predicted class: " << prediction << " (" << className(prediction) << ")\n";
        if (config.true_label.has_value()) {
            const int true_label = *config.true_label;
            std::cout << "True class: " << true_label << " (" << className(true_label) << ")\n";
            std::cout << (true_label == prediction ? "CORRECT" : "INCORRECT") << "\n";
        }
    } catch (const std::exception& ex) {
        std::cout << "Prediction failed: " << ex.what() << "\n";
    }
}

int askMenuChoice() {
    printHeader();
    std::cout << "1) Train / Evaluate model\n";
    std::cout << "2) Predict from image\n";
    std::cout << "3) Show class labels\n";
    std::cout << "4) Exit\n\n";
    return askInt("Select an option", 1, 1, 4);
}

}  // namespace
}  // namespace mnist

int main() {
    mnist::DatasetCache cache;

    while (true) {
        const int choice = mnist::askMenuChoice();

        if (choice == 1) {
            mnist::runTrainingWorkflow(cache);
            mnist::waitForEnter();
            continue;
        }
        if (choice == 2) {
            mnist::runPredictionWorkflow();
            mnist::waitForEnter();
            continue;
        }
        if (choice == 3) {
            mnist::printClassLegend();
            mnist::waitForEnter();
            continue;
        }

        std::cout << "Bye.\n";
        return 0;
    }
}
