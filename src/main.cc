#include <spdlog/common.h>
#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>
#include <exception>
#include <format>

auto main(int argc, char** argv) -> int {
    CLI::App app{"Fast Chamfer Distance Approximation via Query-Aware Locality-Sensitive Hashing (QALSH)."};

    app.add_flag_callback(
        "-v,--verbose", []() { spdlog::set_level(spdlog::level::debug); }, "Enable verbose (debug) logging");

    try {
        CLI11_PARSE(app, argc, argv);
    } catch (std::exception& e) {
        std::cerr << std::format("Error: {}\n", e.what());
        return 1;
    }

    return 0;
}