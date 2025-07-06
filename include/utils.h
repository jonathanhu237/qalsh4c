#ifndef UTILS_H_
#define UTILS_H_

#include <Eigen/Dense>
#include <filesystem>
#include <span>
#include <vector>

namespace qalsh_chamfer {

namespace fs = std::filesystem;

class Utils {
   public:
    auto static WriteSetToFile(const fs::path& file_path, const std::vector<std::vector<double>>& set,
                               const std::string& set_name = "", bool verbose = false) -> void;

    auto static ReadSetFromFile(const fs::path& file_path, unsigned int num_points, unsigned int num_dimensions,
                                const std::string& set_name = "", bool verbose = false)
        -> std::vector<std::vector<double>>;

    auto static WriteArrayToFile(const fs::path& file_path, const std::vector<double>& array) -> void;

    auto static ReadArrayFromFile(const fs::path& file_path, unsigned int num_entries) -> std::vector<double>;

    auto static CalculateChamfer(const std::vector<std::vector<double>>& from_set,
                                 const std::vector<std::vector<double>>& to_set, const std::string& from_set_name = "",
                                 const std::string& to_set_name = "", bool verbose = false) -> double;
    auto static CalculateManhattan(const std::vector<double>& vec1, const std::vector<double>& vec2) -> double;

    auto static DotProduct(const std::vector<double>& vec1, const std::vector<double>& vec2) -> double;

    template <typename T>
    auto static AppendToBuffer(std::vector<char>& buffer, const T& data) -> void {
        const auto* data_ptr = reinterpret_cast<const char*>(&data);
        std::span<const char> data_span(data_ptr, sizeof(T));

        buffer.insert(buffer.end(), data_span.begin(), data_span.end());
    }

   private:
    auto static ToEigenMatrix(const std::vector<std::vector<double>>& set)
        -> Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
};

}  // namespace qalsh_chamfer

#endif