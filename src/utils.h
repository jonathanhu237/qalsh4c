#ifndef UTILS_H_
#define UTILS_H_

#include <spdlog/spdlog.h>

#include <Eigen/Eigen>

#include "point_set.h"

class Utils {
   public:
    static double L1Distance(const Point &pt1, const Point &pt2);
    static double DotProduct(const Point &pt1, const Point &pt2);
    static unsigned int SampleFromWeights(const std::vector<double> &weights);
    static double GetMemoryUsage();

    template <typename T>
    static T ReadFromBuffer(const std::vector<char> &buffer, size_t &offset);

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
void Utils::WriteToBuffer(std::vector<char> &buffer, size_t &offset, const T &data) {
    if (offset + sizeof(T) > buffer.size()) {
        spdlog::error("Not enough space in buffer to write.");
    }
    std::memcpy(&buffer[offset], &data, sizeof(T));
    offset += sizeof(T);
}

#endif