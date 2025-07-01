#ifndef UTILS_H_
#define UTILS_H_

#include <Eigen/Dense>

namespace qalsh_chamfer {

class Utils {
   public:
    auto static CalculateChamfer(const std::vector<std::vector<double>>& from_set,
                                 const std::vector<std::vector<double>>& to_set, const std::string& from_set_name = "",
                                 const std::string& to_set_name = "", bool verbose = false) -> double;

   private:
    auto static ToEigenMatrix(const std::vector<std::vector<double>>& set)
        -> Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
};

}  // namespace qalsh_chamfer

#endif