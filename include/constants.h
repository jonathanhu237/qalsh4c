#ifndef CONST_H_
#define CONST_H_

#include <cmath>
#include <numbers>

namespace qalsh_chamfer {

const double kEpsilon = 1e-6;
const unsigned int kDefaultPageSize = 4096;

const int kDefaultLeftBoundary = -128;  // For dataset generation
const int kDefaultRightBoundary = 128;  // For dataset generation
const double kDefaultErrorProbability = 1.0 / std::numbers::e_v<double>;

}  // namespace qalsh_chamfer

#endif