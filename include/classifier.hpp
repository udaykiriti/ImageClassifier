#ifndef CLASSIFIER_HPP
#define CLASSIFIER_HPP

#include "types.hpp"
#include <string>

namespace mnist {

class Classifier
{
public:
    virtual ~Classifier() = default;

    virtual void train(const ImageSet& images, const Labels& labels) = 0;
    virtual int predict(const Image& image) = 0;
    virtual double evaluate(const ImageSet& images, const Labels& labels);
    virtual void save(const std::string& path) { (void)path; }
    virtual void load(const std::string& path) { (void)path; }
    virtual std::string name() const = 0;
};

}

#endif
