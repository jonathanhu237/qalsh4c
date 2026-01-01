#pragma once

#include <cstdio>
#include <cstdlib>
#include <format>
#include <string>
#include <utility>

class Log {
   public:
    template <typename... Args>
    static void info(std::format_string<Args...> fmt, Args&&... args) {
        PrintLine(fmt, std::forward<Args>(args)...);
    }

    template <typename... Args>
    static void warn(std::format_string<Args...> fmt, Args&&... args) {
        PrintLine(fmt, std::forward<Args>(args)...);
    }

    template <typename... Args>
    [[noreturn]] static void error(std::format_string<Args...> fmt, Args&&... args) {
        PrintLine(fmt, std::forward<Args>(args)...);
        std::exit(EXIT_FAILURE);
    }

   private:
    template <typename... Args>
    static void PrintLine(std::format_string<Args...> fmt, Args&&... args) {
        const std::string message = std::format(fmt, std::forward<Args>(args)...);
        std::fprintf(stderr, "%s\n", message.c_str());
    }
};
