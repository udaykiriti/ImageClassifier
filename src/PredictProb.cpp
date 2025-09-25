#include "SimpleNN.hpp"
#include "dataset.hpp"
#include <fstream>
#include <iostream>
#include <vector>
using namespace std;

/*
 g++ src/PredictProb.cpp src/SimpleNN.cpp src/dataset.cpp -ID:\ImageClassifier\tiny-dnn -Iinclude -O2 -std=c++17 -fopenmp -o build/ImagePredict.exe
 build\ImagePredict.exe
*/

int main()
{
  // Load image from text file
  string image_path = "./data/image.txt"; // change path if needed
  ifstream fin(image_path);
  if (!fin.is_open())
  {
    cerr << "Error: cannot open " << image_path << endl;
    return 1;
  }

  vector<double> image;
  double val;
  while (fin >> val)
  {
    image.push_back(val / 255.0); // normalize
  }
  fin.close();

  if (image.size() != 28 * 28)
  {
    cerr << "Error: image size must be 28x28 = 784 values, got " << image.size() << endl;
    return 1;
  }

  // Load trained neural network
  SimpleNN nn;
  nn.loadModel("./build/simple_nn_model");

  // Predict
  int pred = nn.predict(image);

  // Set the true label for this image
  int true_label = 5; // <-- replace with actual true label

  cout << "Predicted label: " << pred << endl;
  cout << "True label: " << true_label << endl;
  cout << (pred == true_label ? "Prediction is correct!" : "Prediction is incorrect!") << endl;

  return 0;
}
