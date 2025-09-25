#include "SimpleNN.hpp"
#include <iostream>
#include <algorithm> // for std::max_element
using namespace tiny_dnn;
using namespace std;

SimpleNN::SimpleNN()
{
  buildNetwork();
}

void SimpleNN::buildNetwork()
{
  nn << fully_connected_layer(28 * 28, 128)
     << relu_layer()
     << fully_connected_layer(128, 10)
     << softmax_layer();
}

void SimpleNN::train(const vector<vector<double>> &images,
                     const vector<int> &labels,
                     int epochs, int batch_size)
{
  adagrad optimizer;

  vector<vec_t> input_images;
  vector<label_t> input_labels;

  // Convert dataset to tiny-dnn format
  for (size_t i = 0; i < images.size(); i++)
  {
    // Convert double -> float for tiny-dnn
    vec_t img(images[i].begin(), images[i].end());
    input_images.push_back(img);

    // Convert int -> label_t
    input_labels.push_back(static_cast<label_t>(labels[i]));
  }

  nn.train<mse>(optimizer, input_images, input_labels, batch_size, epochs);

  cout << "Training completed." << endl;
}

int SimpleNN::predict(const vector<double> &image)
{
  // Convert double -> float
  vec_t img(image.begin(), image.end());
  auto result = nn.predict(img);

  return distance(result.begin(), max_element(result.begin(), result.end()));
}

double SimpleNN::score(const vector<vector<double>> &images,
                       const vector<int> &labels)
{
  int correct = 0;
  for (size_t i = 0; i < images.size(); i++)
  {
    if (predict(images[i]) == labels[i])
      correct++;
  }
  return double(correct) / images.size();
}

void SimpleNN::saveModel(const string &path)
{
  nn.save(path);
}

void SimpleNN::loadModel(const string &path)
{
  nn.load(path);
}
