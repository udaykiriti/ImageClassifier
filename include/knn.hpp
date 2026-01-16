#ifndef KNN_HPP
#define KNN_HPP

#include "classifier.hpp"

namespace mnist {

class KNN : public Classifier
{
private:
    int k_;
    ImageSet train_images_;
    Labels train_labels_;

    double distance(const Image& a, const Image& b) const;

public:
    explicit KNN(int k = 3) : k_(k) {}

    void train(const ImageSet& images, const Labels& labels) override;
    int predict(const Image& image) override;
    std::string name() const override { return "KNN"; }
};

}

#endif
