#ifndef TIMER_H_
#define TIMER_H_

#include <chrono>
class Timer {
   public:
    Timer(double *time_ms_ref);
    ~Timer();

   private:
    double *time_ms_ref_;
    std::chrono::high_resolution_clock::time_point start_time_;
};

#endif