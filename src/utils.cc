#include "utils.h"

#include <Eigen/Dense>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <ios>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace qalsh_chamfer {

auto Utils::WriteSetToFile(const fs::path& file_path, const std::vector<std::vector<double>>& set,
                           const std::string& set_name, bool verbose) -> void {
    std::ofstream ofs(file_path, std::ios::binary | std::ios::trunc);

    if (!ofs.is_open()) {
        throw std::runtime_error(std::format("Could not open file for writing: {}", file_path.string()));
    }

    size_t num_dimensions = set.empty() ? 0 : set[0].size();

    for (const auto& point : set) {
        if (verbose) {
            std::cout << std::format("Writing set {} to file ... ({}/{})\r", set_name, &point - set.data() + 1,
                                     set.size())
                      << std::flush;
        }

        ofs.write(reinterpret_cast<const char*>(point.data()),
                  static_cast<std::streamsize>(sizeof(double) * num_dimensions));
    }
}

auto Utils::ReadSetFromFile(const fs::path& file_path, unsigned int num_points, unsigned int num_dimensions,
                            const std::string& set_name, bool verbose) -> std::vector<std::vector<double>> {
    std::ifstream ifs(file_path, std::ios::binary);

    if (!ifs.is_open()) {
        throw std::runtime_error(std::format("Could not open file for reading: {}", file_path.string()));
    }

    std::uintmax_t file_size = fs::file_size(file_path);
    if (file_size != static_cast<std::uintmax_t>(sizeof(double) * num_points * num_dimensions)) {
        throw std::runtime_error(std::format("File size does not match expected size for {} points and {} dimensions.",
                                             num_points, num_dimensions));
    }

    std::vector<std::vector<double>> set(num_points, std::vector<double>(num_dimensions));

    for (unsigned int i = 0; i < num_points; i++) {
        if (verbose) {
            std::cout << std::format("Reading set {} from file ... ({}/{})\r", set_name, i + 1, num_points)
                      << std::flush;
        }

        ifs.read(reinterpret_cast<char*>(set[i].data()), static_cast<std::streamsize>(sizeof(double) * num_dimensions));
    }

    if (verbose) {
        std::cout << "\n";
    }

    return set;
}

auto Utils::WriteArrayToFile(const fs::path& file_path, const std::vector<double>& array) -> void {
    std::ofstream ofs(file_path, std::ios::binary | std::ios::trunc);
    if (!ofs.is_open()) {
        throw std::runtime_error(std::format("Could not open file for writing: {}", file_path.string()));
    }

    ofs.write(reinterpret_cast<const char*>(array.data()), static_cast<std::streamsize>(sizeof(double) * array.size()));
}

auto Utils::ReadArrayFromFile(const fs::path& file_path, unsigned int num_entries) -> std::vector<double> {
    std::vector<double> array(num_entries);
    std::ifstream ifs(file_path, std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error(std::format("Could not open file for reading: {}", file_path.string()));
    }

    ifs.read(reinterpret_cast<char*>(array.data()), static_cast<std::streamsize>(sizeof(double) * num_entries));
    return array;
}

auto Utils::CalculateChamfer(const std::vector<std::vector<double>>& from_set,
                             const std::vector<std::vector<double>>& to_set, const std::string& from_set_name,
                             const std::string& to_set_name, bool verbose) -> double {
    size_t num_points = from_set.size();

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

auto Utils::DotProduct(const std::vector<double>& vec1, const std::vector<double>& vec2) -> double {
    if (vec1.size() != vec2.size()) {
        throw std::invalid_argument("Vectors must be of the same size for dot product.");
    }

    Eigen::Map<const Eigen::VectorXd> eigen_vec1(vec1.data(), static_cast<Eigen::Index>(vec1.size()));
    Eigen::Map<const Eigen::VectorXd> eigen_vec2(vec2.data(), static_cast<Eigen::Index>(vec2.size()));

    return eigen_vec1.dot(eigen_vec2);
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