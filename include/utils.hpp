#ifndef UTILS_H_
#define UTILS_H_

#include <Eigen/Dense>
#include <filesystem>
#include <vector>

namespace qalsh_chamfer {

namespace fs = std::filesystem;

class Utils {
   public:
    auto static WriteSetToFile(const fs::path& file_path, const std::vector<std::vector<double>>& set,
                               const std::string& set_name = "", bool verbose = false) -> void;

    auto static CalculateChamfer(const std::vector<std::vector<double>>& from_set,
                                 const std::vector<std::vector<double>>& to_set, const std::string& from_set_name = "",
                                 const std::string& to_set_name = "", bool verbose = false) -> double;

    auto static DotProduct(const std::vector<double>& vec1, const std::vector<double>& vec2) -> double;

   private:
    auto static ToEigenMatrix(const std::vector<std::vector<double>>& set)
        -> Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
};

}  // namespace qalsh_chamfer

#endif