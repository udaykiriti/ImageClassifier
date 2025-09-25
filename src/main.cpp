#include "dataset.hpp"
#include "knn_classifier.hpp"
#include <iostream>
#include <algorithm> // for std::min
using namespace std;

int main()
{
    // Initialize datasets
    Dataset TrainData("MNIST Training Data", "./data");
    Dataset TestData("MNIST Test Data", "./data");

    // Check required files
    if (!TrainData.CheckFiles() || !TestData.CheckFiles())
        return 1;

    // Load images and labels
    if (!TrainData.LoadImages("train-images-idx3-ubyte") ||
        !TrainData.LoadLabels("train-labels-idx1-ubyte") ||
        !TestData.LoadImages("t10k-images-idx3-ubyte") ||
        !TestData.LoadLabels("t10k-labels-idx1-ubyte"))
        return 1;

    // Limit number of training and test samples for faster k-NN
    int train_limit = min(1000, TrainData.NumImages());
    int test_limit  = min(200, TestData.NumImages());

    // Prepare subsets
    vector<vector<double>> train_images_subset(
        TrainData.GetImages().begin(), TrainData.GetImages().begin() + train_limit);
    vector<int> train_labels_subset(
        TrainData.GetLabels().begin(), TrainData.GetLabels().begin() + train_limit);

    vector<vector<double>> test_images_subset(
        TestData.GetImages().begin(), TestData.GetImages().begin() + test_limit);
    vector<int> test_labels_subset(
        TestData.GetLabels().begin(), TestData.GetLabels().begin() + test_limit);

    // Train k-NN classifier
    KNNClassifier knn(3);
    knn.fit(train_images_subset, train_labels_subset);

    // Evaluate on test subset
    double acc = knn.score(test_images_subset, test_labels_subset);
    cout << "Accuracy on subset: " << acc * 100 << "%" << endl;

    // Display sample predictions
    cout << "\nSample Predictions:\n";
    for (int i = 0; i < 5; i++)
    {
        int pred = knn.predict(test_images_subset[i]);
        cout << "Test Image " << i << " - True Label: " << test_labels_subset[i]
             << ", Predicted: " << pred << endl;
        TestData.PrintImage(i);
        cout << "--------------------\n";
    }

    return 0;
}
