#include "SimpleNN.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <sstream>
using namespace std;

/*
 g++ src/Predict.cpp src/SimpleNN.cpp src/dataset.cpp -ID:\ImageClassifier\tiny-dnn -Iinclude -O2 -std=c++17 -fopenmp -o build/ImagePredict.exe
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
    // Normalize to [0,1]
    image.push_back(val / 255.0);
  }

  if (image.size() != 28 * 28)
  {
    cerr << "Error: image size must be 28x28 = 784 values, got " << image.size() << endl;
    return 1;
  }

  fin.close();

  // Load your trained neural network
  SimpleNN nn;
  nn.loadModel("./build/simple_nn_model"); // path to saved model

  // Predict
  int pred = nn.predict(image);
  cout << "Predicted label: " << pred << endl;

  return 0;
}
