#include "dataset.hpp"
#include "SimpleNN.hpp"
#include <iostream>
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

  // Train NN
  SimpleNN nn;
  nn.train(train_images, train_labels, 10, 32);

  double acc = nn.score(test_images, test_labels);
  cout << "Accuracy: " << acc * 100 << "%" << endl;

  return 0;
}
