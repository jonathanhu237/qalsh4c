#ifndef ESTIMATOR_H_
#define ESTIMATOR_H_

class IEstimator {
   public:
    virtual ~IEstimator() = default;
    virtual auto Estimate() -> double = 0;
};

#endif