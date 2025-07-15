#ifndef ESTIMATOR_H_
#define ESTIMATOR_H_

class Estimator {
   public:
    virtual ~Estimator() = default;
    virtual double Estimate() = 0;
};

#endif