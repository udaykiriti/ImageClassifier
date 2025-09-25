#include "dataset.hpp"
#include "SimpleNN.hpp"
#include <iostream>
#include <algorithm>
using namespace std;

int main()
{
  Dataset TrainData("MNIST Training Data", "./data");
  Dataset TestData("MNIST Test Data", "./data");

  TrainData.LoadImages("train-images-idx3-ubyte");
  TrainData.LoadLabels("train-labels-idx1-ubyte");
  TestData.LoadImages("t10k-images-idx3-ubyte");
  TestData.LoadLabels("t10k-labels-idx1-ubyte");

  // Take smaller subset for faster training
  int train_limit = min(2000, TrainData.NumImages());
  int test_limit = min(500, TestData.NumImages());

  vector<vector<double>> train_images(
      TrainData.GetImages().begin(), TrainData.GetImages().begin() + train_limit);
  vector<int> train_labels(
      TrainData.GetLabels().begin(), TrainData.GetLabels().begin() + train_limit);

  vector<vector<double>> test_images(
      TestData.GetImages().begin(), TestData.GetImages().begin() + test_limit);
  vector<int> test_labels(
      TestData.GetLabels().begin(), TestData.GetLabels().begin() + test_limit);

  SimpleNN nn;

  // Check if model exists and load it, otherwise train and save
  const string model_path = "./build/simple_nn_model";
  ifstream f(model_path);
  if (f.good())
  {
    f.close();
    nn.loadModel(model_path);
    cout << "Loaded pre-trained model from " << model_path << endl;
  }
  else
  {
    nn.train(train_images, train_labels, 10, 32);
    nn.saveModel(model_path);
    cout << "Training completed and model saved to " << model_path << endl;
  }

  double acc = nn.score(test_images, test_labels);
  cout << "Accuracy: " << acc * 100 << "%" << endl;

  // Display true and predicted labels for first 10 test images
  cout << "\nSample Predictions:\n";
  for (int i = 0; i < 10; i++)
  {
    int pred = nn.predict(test_images[i]);
    cout << "Test Image " << i
         << " - True Label: " << test_labels[i]
         << ", Predicted: " << pred << endl;

    // Optional: display ASCII image
    TestData.PrintImage(i);
    cout << "--------------------\n";
  }

  return 0;
}
