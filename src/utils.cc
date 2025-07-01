#include "utils.hpp"

#include <Eigen/Dense>
#include <cstddef>
#include <format>
#include <iostream>

namespace qalsh_chamfer {

auto Utils::CalculateChamfer(const std::vector<std::vector<double>>& from_set,
                             const std::vector<std::vector<double>>& to_set, const std::string& from_set_name,
                             const std::string& to_set_name, bool verbose) -> double {
    size_t num_points = from_set.size();
    size_t num_dimensions = from_set.empty() ? 0 : from_set[0].size();

    Eigen::MatrixXd from_matrix = ToEigenMatrix(from_set);
    Eigen::MatrixXd to_matrix = ToEigenMatrix(to_set);

    double chamfer_distance = 0.0;
    for (unsigned int i = 0; i < num_points; ++i) {
        if (verbose) {
            std::cout << std::format("Calculating Chamfer distance ({} to {})... ({}/{})\r", from_set_name, to_set_name,
                                     i + 1, num_points)
                      << std::flush;
        }
        chamfer_distance += (to_matrix.rowwise() - from_matrix.row(i)).rowwise().lpNorm<1>().minCoeff();
    }
    if (verbose) {
        std::cout << "\n";
    }
    return chamfer_distance;
}

auto Utils::ToEigenMatrix(const std::vector<std::vector<double>>& set)
    -> Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> {
    size_t num_points = set.size();
    size_t num_dimensions = set.empty() ? 0 : set[0].size();

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrix(num_points, num_dimensions);
    for (unsigned int i = 0; i < num_points; ++i) {
        matrix.row(i) = Eigen::Map<const Eigen::VectorXd>(set[i].data(), static_cast<Eigen::Index>(num_dimensions));
    }
    return matrix;
}

}  // namespace qalsh_chamfer