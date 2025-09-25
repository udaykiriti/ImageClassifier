#include "dataset.hpp"
#include "knn_classifier.hpp"
#include <iostream>
#include <algorithm>
using namespace std;

int main()
{
  // Initialize datasets
  Dataset TrainData("MNIST Training Data", "./data");
  Dataset TestData("MNIST Test Data", "./data");

  // Check required files
  if (!TrainData.CheckFiles() || !TestData.CheckFiles())
    return 1;

  // Load all images and labels
  if (!TrainData.LoadImages("train-images-idx3-ubyte") ||
      !TrainData.LoadLabels("train-labels-idx1-ubyte") ||
      !TestData.LoadImages("t10k-images-idx3-ubyte") ||
      !TestData.LoadLabels("t10k-labels-idx1-ubyte"))
    return 1;

  cout << "Training on " << TrainData.NumImages() << " images, testing on "
       << TestData.NumImages() << " images.\n";

  // Train k-NN classifier with k=3
  KNNClassifier knn(3);

  // Fit on full training set
  knn.fit(TrainData.GetImages(), TrainData.GetLabels());

  // Evaluate on full test set
  double acc = knn.score(TestData.GetImages(), TestData.GetLabels());
  cout << "Accuracy on full test set: " << acc * 100 << "%" << endl;

  // Show first 10 misclassified images (if any)
  cout << "\nSample Misclassified Images:\n";
  int mis_count = 0;
  for (size_t i = 0; i < TestData.NumImages(); i++)
  {
    int pred = knn.predict(TestData.GetImages()[i]);
    int true_label = TestData.GetLabels()[i];
    if (pred != true_label && mis_count < 10)
    {
      cout << "Test Image " << i << " - True Label: " << true_label
           << ", Predicted: " << pred << endl;
      TestData.PrintImage(i);
      cout << "--------------------\n";
      mis_count++;
    }
  }

  if (mis_count == 0)
    cout << "No misclassified images in first 10 samples.\n";

  return 0;
}
