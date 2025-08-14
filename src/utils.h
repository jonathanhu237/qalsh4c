#ifndef UTILS_H_
#define UTILS_H_

#include <spdlog/spdlog.h>

#include <Eigen/Eigen>
#include <random>

#include "types.h"

class Utils {
   public:
    static double L1Distance(const Point &pt1, const Point &pt2);
    static double L2Distance(const Point &pt1, const Point &pt2);
    static double DotProduct(const Point &pt1, const Point &pt2);
    static DatasetMetadata LoadDatasetMetadata(const std::filesystem::path &file_path);
    static std::vector<Point> LoadPointsFromFile(const std::filesystem::path &file_path, unsigned int num_points,
                                                 unsigned int num_dimensions);
    static Point ReadPoint(std::ifstream &ifs, unsigned int num_dimensions, unsigned int point_id);
    static void RegularizeQalshConfig(QalshConfig &config, unsigned int num_points, double norm_order);
    static void SaveQalshConfig(QalshConfig &config, const std::filesystem::path &file_path);
    static QalshConfig LoadQalshConfig(const std::filesystem::path &file_path);
    static unsigned int SampleFromWeights(const std::vector<double> &weights);
    static double GetMemoryUsage();
    static std::mt19937 CreateSeededGenerator();
    static double CalculateL1Probability(double x);
    static double CalculateL2Probability(double x);

    template <typename T>
    static T ReadFromBuffer(const std::vector<char> &buffer, size_t &offset);

    template <typename T>
    static std::vector<T> ReadVectorFromBuffer(const std::vector<char> &buffer, size_t &offset, size_t count);

    template <typename T>
    static void WriteToBuffer(std::vector<char> &buffer, size_t &offset, const T &data);
};

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

template <typename T>
std::vector<T> Utils::ReadVectorFromBuffer(const std::vector<char> &buffer, size_t &offset, size_t count) {
    const size_t num_bytes_to_read = count * sizeof(T);
    std::vector<T> result(count);
    std::memcpy(result.data(), &buffer[offset], num_bytes_to_read);

    offset += num_bytes_to_read;
    return result;
}

template <typename T>
void Utils::WriteToBuffer(std::vector<char> &buffer, size_t &offset, const T &data) {
    if (offset + sizeof(T) > buffer.size()) {
        spdlog::error("Not enough space in buffer to write.");
    }
    std::memcpy(&buffer[offset], &data, sizeof(T));
    offset += sizeof(T);
}

#endif