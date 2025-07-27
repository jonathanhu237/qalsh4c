#ifndef UTILS_H_
#define UTILS_H_

#include <spdlog/spdlog.h>
#include <toml++/toml.h>

#include <Eigen/Eigen>
#include <format>
#include <fstream>
#include <string_view>
#include <vector>

class Utils {
   public:
    template <typename T1, typename T2>
    static double CalculateL1Distance(const std::vector<T1> &vec1, const std::vector<T2> &vec2);

    template <typename T>
    static T GetValueFromTomlTable(const toml::table &tbl, std::string_view key);

    template <typename T1, typename T2>
    static double DotProduct(const std::vector<T1> &vec1, const std::vector<T2> &vec2);

    template <typename T>
    void static WriteToBuffer(std::vector<char> &buffer, size_t &offset, const T &data);

    template <typename T>
    T static ReadFromBuffer(const std::vector<char> &buffer, size_t &offset);

    auto static SampleFromWeights(const std::vector<double> &weights) -> unsigned int;

    template <typename T>
    void static WritePoints(const std::vector<std::vector<T>> &points, const std::string &path);

    std::vector<std::vector<uint8_t>> static ReadBmpGrayscale(const std::string &path);
};

template <typename T1, typename T2>
double Utils::CalculateL1Distance(const std::vector<T1> &vec1, const std::vector<T2> &vec2) {
    if (vec1.size() != vec2.size()) {
        spdlog::error("Vectors must be of the same size");
    }

    Eigen::Map<const Eigen::Matrix<T1, Eigen::Dynamic, 1>> eigen_vec1(vec1.data(), vec1.size());
    Eigen::Map<const Eigen::Matrix<T2, Eigen::Dynamic, 1>> eigen_vec2(vec2.data(), vec2.size());

    return (eigen_vec1.template cast<double>() - eigen_vec2.template cast<double>()).template lpNorm<1>();
}

template <typename T>
T Utils::GetValueFromTomlTable(const toml::table &tbl, std::string_view key) {
    auto value = tbl.get(key)->value<T>();
    if (!value) {
        spdlog::error(std::format("Key '{}' not found in TOML table", key));
    }
    return *value;
}

template <typename T1, typename T2>
double Utils::DotProduct(const std::vector<T1> &vec1, const std::vector<T2> &vec2) {
    if (vec1.size() != vec2.size()) {
        spdlog::error("Vectors must be of the same size for dot product. vec1.size(): {}, vec2.size(): {}", vec1.size(),
                      vec2.size());
    }

    Eigen::Map<const Eigen::Matrix<T1, Eigen::Dynamic, 1>> eigen_vec1(vec1.data(), vec1.size());
    Eigen::Map<const Eigen::Matrix<T2, Eigen::Dynamic, 1>> eigen_vec2(vec2.data(), vec2.size());

    return eigen_vec1.template cast<double>().dot(eigen_vec2.template cast<double>());
}

template <typename T>
void Utils::WriteToBuffer(std::vector<char> &buffer, size_t &offset, const T &data) {
    if (offset + sizeof(T) > buffer.size()) {
        spdlog::error("Not enough space in buffer to write.");
    }
    std::memcpy(&buffer[offset], &data, sizeof(T));
    offset += sizeof(T);
}

template <typename T>
T Utils::ReadFromBuffer(const std::vector<char> &buffer, size_t &offset) {
    if (offset + sizeof(T) > buffer.size()) {
        spdlog::error("Not enough data in buffer to read.");
    }

    T data;
    std::memcpy(&data, &buffer[offset], sizeof(T));
    offset += sizeof(T);

    return data;
}

template <typename TData>
void Utils::WritePoints(const std::vector<std::vector<TData>> &points, const std::string &path) {
    std::ofstream ofs(path, std::ios::binary | std::ios::out);

    for (size_t i = 0; i < points.size(); i++) {
        ofs.write(reinterpret_cast<const char *>(points[i].data()), sizeof(TData) * points[i].size());
    }
    ofs.close();
}

#endif