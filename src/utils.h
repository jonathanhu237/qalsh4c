#ifndef UTILS_H_
#define UTILS_H_

#include <spdlog/spdlog.h>

#include <Eigen/Eigen>

#include "point_set.h"

class Utils {
   public:
    static double L1Distance(const Point &pt1, const Point &pt2);
};

#endif