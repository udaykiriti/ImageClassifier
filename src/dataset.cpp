#include "dataset.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <ctime>

Dataset::Dataset(const std::string &DatasetName, const std::string &DatasetPath)
    : name(DatasetName), path(DatasetPath)
{
  std::cout << "Dataset module initialized: " << name << std::endl;

  ReqFiles = {
      "train-images-idx3-ubyte",
      "train-labels-idx1-ubyte",
      "t10k-images-idx3-ubyte",
      "t10k-labels-idx1-ubyte"};
}

void Dataset::PrintInfo() const
{
  std::cout << "Dataset: " << name << std::endl;
}

void Dataset::PrintPath() const
{
  std::cout << "Dataset path: " << path << std::endl;
}

bool Dataset::CheckFiles() const
{
  bool all_exist = true;
  for (const auto &file : ReqFiles)
  {
    std::string full_path = path + "/" + file;
    std::ifstream f(full_path);
    if (!f)
    {
      std::cout << "Missing file: " << full_path << std::endl;
      all_exist = false;
    }
  }
  if (all_exist)
    std::cout << "All required files exist in " << path << std::endl;
  return all_exist;
}

uint32_t read_uint32(std::ifstream &f)
{
  unsigned char b[4];
  f.read((char *)b, 4);
  return (uint32_t(b[0]) << 24) | (uint32_t(b[1]) << 16) | (uint32_t(b[2]) << 8) | uint32_t(b[3]);
}

bool Dataset::LoadImages(const std::string &file_name)
{
  std::string full_path = path + "/" + file_name;
  std::ifstream f(full_path, std::ios::binary);
  if (!f)
  {
    std::cerr << "Cannot open image file: " << full_path << std::endl;
    return false;
  }

  uint32_t magic = read_uint32(f);
  if (magic != 2051)
  {
    std::cerr << "Invalid MNIST image file (magic = " << magic << ")\n";
    return false;
  }

  uint32_t n_images = read_uint32(f);
  rows = read_uint32(f);
  cols = read_uint32(f);

  images.resize(n_images, std::vector<double>(rows * cols));

  for (uint32_t i = 0; i < n_images; i++)
  {
    for (uint32_t j = 0; j < rows * cols; j++)
    {
      unsigned char pixel;
      f.read((char *)&pixel, 1);
      images[i][j] = pixel / 255.0;
    }
  }

  std::cout << "Loaded " << n_images << " images of size " << rows << "x" << cols << std::endl;
  return true;
}

bool Dataset::LoadLabels(const std::string &file_name)
{
  std::string full_path = path + "/" + file_name;
  std::ifstream f(full_path, std::ios::binary);
  if (!f)
  {
    std::cerr << "Cannot open label file: " << full_path << std::endl;
    return false;
  }

  uint32_t magic = read_uint32(f);
  if (magic != 2049)
  {
    std::cerr << "Invalid MNIST label file (magic = " << magic << ")\n";
    return false;
  }

  uint32_t n_labels = read_uint32(f);
  labels.resize(n_labels);

  for (uint32_t i = 0; i < n_labels; i++)
  {
    unsigned char lab;
    f.read((char *)&lab, 1);
    labels[i] = int(lab);
  }

  std::cout << "Loaded " << n_labels << " labels" << std::endl;
  return true;
}

int Dataset::GetLabel(int idx) const
{
  if (idx < 0 || idx >= labels.size())
    return -1;
  return labels[idx];
}

void Dataset::PrintImage(int idx) const
{
  if (idx < 0 || idx >= images.size())
    return;

  for (int r = 0; r < rows; r++)
  {
    for (int c = 0; c < cols; c++)
    {
      double val = images[idx][r * cols + c];
      if (val > 0.75)
        std::cout << "@";
      else if (val > 0.5)
        std::cout << "#";
      else if (val > 0.25)
        std::cout << "*";
      else
        std::cout << ".";
    }
    std::cout << "\n";
  }
}

void Dataset::head(int n) const
{
  if (images.empty())
    return;
  int limit = std::min(n, (int)images.size());
  for (int i = 0; i < limit; i++)
  {
    std::cout << "Image " << i << " Label: " << GetLabel(i) << "\n";
    PrintImage(i);
    std::cout << "---------------------\n";
  }
}

void Dataset::tail(int n) const
{
  if (images.empty())
    return;
  int start = std::max(0, (int)images.size() - n);
  for (int i = start; i < images.size(); i++)
  {
    std::cout << "Image " << i << " Label: " << GetLabel(i) << "\n";
    PrintImage(i);
    std::cout << "---------------------\n";
  }
}

void Dataset::sample(int idx) const
{
  if (images.empty())
    return;
  if (idx < 0 || idx >= images.size())
  {
    std::cout << "Invalid index " << idx << std::endl;
    return;
  }
  std::cout << "Image " << idx << " Label: " << GetLabel(idx) << "\n";
  PrintImage(idx);
  std::cout << "---------------------\n";
}

void Dataset::RandomSample(int n) const
{
  if (images.empty())
    return;
  n = std::min(n, (int)images.size());

  std::vector<int> indices(images.size());
  std::iota(indices.begin(), indices.end(), 0);

  std::mt19937 rng(static_cast<unsigned int>(std::time(nullptr)));
  std::shuffle(indices.begin(), indices.end(), rng);

  for (int i = 0; i < n; i++)
  {
    sample(indices[i]);
  }
}
