#ifndef KNN_HPP
#define KNN_HPP

#include <vector>

class KNN
{
private:
    int k_;
    std::vector<std::vector<double>> train_images_;
    std::vector<int> train_labels_;

    double euclideanDistance(const std::vector<double>& a,
                             const std::vector<double>& b) const;

public:
    explicit KNN(int neighbors = 3) : k_(neighbors) {}

    void fit(const std::vector<std::vector<double>>& images,
             const std::vector<int>& labels);

    int predict(const std::vector<double>& image);

    double evaluate(const std::vector<std::vector<double>>& images,
                    const std::vector<int>& labels);
};

#endif
