#ifndef DATASET_HPP
#define DATASET_HPP

#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <random>
#include <algorithm>

class Dataset
{
private:
  std::string name;
  std::string path;
  std::vector<std::string> ReqFiles;

  std::vector<std::vector<double>> images;
  std::vector<int> labels;
  int rows, cols;

public:
  Dataset(const std::string &DatasetName, const std::string &DatasetPath);

  void PrintInfo() const;
  void PrintPath() const;
  bool CheckFiles() const;

  bool LoadImages(const std::string &FileName);
  bool LoadLabels(const std::string &FileName);
  void PrintImage(int idx) const;

  int GetLabel(int idx) const;

  int NumImages() const { return images.size(); }
  int Rows() const { return rows; }
  int Cols() const { return cols; }

  void head(int n = 5) const;
  void tail(int n = 5) const;
  void sample(int idx) const;
  void RandomSample(int n = 5) const;

  const std::vector<std::vector<double>> &GetImages() const { return images; }
  const std::vector<int> &GetLabels() const { return labels; }
};

#endif