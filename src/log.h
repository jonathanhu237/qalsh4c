#pragma once

#include <cstdlib>
#include <print>
#include <utility>

class Log {
   public:
    template <typename... Args>
    static void info(std::format_string<Args...> fmt, Args&&... args) {
        std::println(stderr, fmt, std::forward<Args>(args)...);
    }

    template <typename... Args>
    static void warn(std::format_string<Args...> fmt, Args&&... args) {
        std::println(stderr, fmt, std::forward<Args>(args)...);
    }

    template <typename... Args>
    [[noreturn]] static void error(std::format_string<Args...> fmt, Args&&... args) {
        std::println(stderr, fmt, std::forward<Args>(args)...);
        std::exit(EXIT_FAILURE);
    }
};
