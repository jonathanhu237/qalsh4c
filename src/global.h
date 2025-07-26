#ifndef GLOBAL_H_
#define GLOBAL_H_

class Global {
   public:
    static bool high_memory_mode;
    static bool measure_time;

    // Variables for time measurement.
    static double linear_scan_search_time_ms;
};

#endif