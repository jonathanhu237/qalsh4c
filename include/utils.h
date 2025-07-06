#ifndef UTILS_H_
#define UTILS_H_

#include <Eigen/Dense>
#include <filesystem>
#include <stdexcept>
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
    auto static WriteToBuffer(std::vector<char>& buffer, size_t& offset, const T& data) -> void {
        if (offset + sizeof(T) > buffer.size()) {
            throw std::out_of_range("Not enough space in buffer to write.");
        }
        std::memcpy(&buffer[offset], &data, sizeof(T));
        offset += sizeof(T);
    }

    template <typename T>
    auto static ReadFromBuffer(const std::vector<char>& buffer, size_t& offset) -> T {
        if (offset + sizeof(T) > buffer.size()) {
            throw std::out_of_range("Not enough data in buffer to read.");
        }

        T data;
        std::memcpy(&data, &buffer[offset], sizeof(T));
        offset += sizeof(T);

        return data;
    }

   private:
    auto static ToEigenMatrix(const std::vector<std::vector<double>>& set)
        -> Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
};

}  // namespace qalsh_chamfer

#endif