#ifndef KNN_HPP
#define KNN_HPP

#include "classifier.hpp"

namespace mnist {

class KNN : public Classifier
{
private:
    int k_;
    bool weighted_votes_;
    ImageSet train_images_;
    Labels train_labels_;

    double distanceSquared(const Image& a, const Image& b) const;

public:
    explicit KNN(int k = 3, bool weighted_votes = true)
        : k_(k), weighted_votes_(weighted_votes) {}

    void train(const ImageSet& images, const Labels& labels) override;
    int predict(const Image& image) override;
    std::string name() const override { return "KNN"; }
};

}

#endif
