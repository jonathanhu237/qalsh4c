#ifndef GLOBAL_H_
#define GLOBAL_H_

#include <cmath>
#include <numbers>

class Global {
   public:
    // Global constants
    static constexpr double kEpsilon = 1e-6;
    static constexpr unsigned int kDefaultPageSize = 4096;

    static constexpr double kDefaultLeftBoundary = -128.0;                               // For dataset generation
    static constexpr double kDefaultRightBoundary = 128.0;                               // For dataset generation
    static constexpr double kDefaultErrorProbability = 1.0 / std::numbers::e_v<double>;  // For QALSH

    // Global variables
    static bool high_memory_mode;
    static bool measure_time;

    // Variables for time measurement.
    static double linear_scan_search_time_ms;
};

#endif