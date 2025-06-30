#ifndef CONST_H_
#define CONST_H_

#include <cmath>
#include <string>

namespace qalsh_chamfer {

const float kEpsilon = 1e-6;

const std::string kDefaultParentDirectory = "data";
const int kDefaultLeftBoundary = -128;  // For dataset generation
const int kDefaultRightBoundary = 128;  // For dataset generation
const float kDefaultErrorProbability = 1.0 / std::numbers::e_v<float>;

}  // namespace qalsh_chamfer

#endif