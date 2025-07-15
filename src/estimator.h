#ifndef ESTIMATOR_H_
#define ESTIMATOR_H_

class Estimator {
   public:
    virtual ~Estimator() = default;
    virtual auto Estimate() -> double = 0;
};

#endif