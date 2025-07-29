#ifndef GLOBAL_H_
#define GLOBAL_H_

#include <cmath>
#include <numbers>
#include <string_view>

class Global {
   public:
    // Global constants
    static constexpr double kEpsilon = 1e-6;
    static constexpr unsigned int kDefaultPageSize = 4096;

    static constexpr double kDefaultLeftBoundary = -128.0;                               // For dataset generation
    static constexpr double kDefaultRightBoundary = 128.0;                               // For dataset generation
    static constexpr double kDefaultErrorProbability = 1.0 / std::numbers::e_v<double>;  // For QALSH

    // Dataset attributes
    static constexpr unsigned int kSiftNumPoints = 1000000;
    static constexpr unsigned int kSiftNumDimensions = 128;
    static constexpr std::string_view kSiftDataType = "uint8";

    static constexpr unsigned int kGistNumPoints = 1000000;
    static constexpr unsigned int kGistNumDimensions = 960;
    static constexpr std::string_view kGistDataType = "float";

    static constexpr unsigned int kTreviNumPoints = 101120;
    static constexpr unsigned int kTreviNumDimensions = 4096;
    static constexpr std::string_view kTreviDataType = "uint8";

    static constexpr unsigned int kP53NumPoints = 31159;
    static constexpr unsigned int kP53NumDimensions = 5408;
    static constexpr std::string_view kP53DataType = "float";

    // Global variables
    static bool high_memory_mode;
    static bool measure_time;

    // Variables for time measurement.
    static double linear_scan_search_time_ms;
};

#endif