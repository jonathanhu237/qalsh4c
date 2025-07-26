#include "timer.h"

#include <chrono>

#include "global.h"

Timer::Timer(double* time_ms_ref) {
    if (Global::measure_time) {
        time_ms_ref_ = time_ms_ref;
        start_time_ = std::chrono::high_resolution_clock::now();
    }
}

Timer::~Timer() {
    if (Global::measure_time) {
        auto end_time = std::chrono::high_resolution_clock::now();
        *time_ms_ref_ += std::chrono::duration<double, std::milli>(end_time - start_time_).count();
    }
}