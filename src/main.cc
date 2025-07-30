#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>

#include "sink.h"

int main(int argc, char** argv) {
    CLI::App app{"Fast Chamfer Distance Approximation via Query-Aware Locality-Sensitive Hashing (QALSH)."};

    std::string log_level;
    app.add_option("-l,--log_level", log_level, "Set the logging level (default: warn)")
        ->default_val("warn")
        ->check(CLI::IsMember({"debug", "info", "warn", "error"}))
        ->each([&](const std::string& level) {
            auto console_sink = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
            auto terminating_sink = std::make_shared<TerminatingSink<std::mutex>>(console_sink);
            auto logger = std::make_shared<spdlog::logger>("qalsh_chamfer", terminating_sink);
            logger->set_level(spdlog::level::from_str(level));
            spdlog::set_default_logger(logger);
        });

    CLI11_PARSE(app, argc, argv);

    return 0;
}