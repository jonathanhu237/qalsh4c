#include "config.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace {

std::string Quote(const std::filesystem::path& path) {
    std::string value = path.string();
    std::string quoted = "'";
    for (char ch : value) {
        if (ch == '\'') {
            quoted += "'\\''";
        } else {
            quoted += ch;
        }
    }
    quoted += "'";
    return quoted;
}

std::string ReadFile(const std::filesystem::path& path) {
    std::ifstream ifs(path);
    std::ostringstream oss;
    oss << ifs.rdbuf();
    return oss.str();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: dataset_cuda_cli_test <qalsh4c-bin> <temp-dir>\n";
        return 1;
    }

    const std::filesystem::path qalsh4c_bin = argv[1];
    const std::filesystem::path temp_dir = argv[2];
    const std::filesystem::path dataset_dir = temp_dir / "dataset";
    const std::filesystem::path stdout_path = temp_dir / "stdout.txt";
    const std::filesystem::path stderr_path = temp_dir / "stderr.txt";
    std::filesystem::remove_all(temp_dir);
    std::filesystem::create_directories(temp_dir);

    std::ostringstream command;
    command << Quote(qalsh4c_bin) << " dataset generate gmm"
            << " --num-points-a 2"
            << " --num-points-b 2"
            << " --num-dimensions 2"
            << " --seed 31"
            << " --ground-truth l2:ba"
            << " --ground-truth-device cuda"
            << " --cuda-batch-size 1"
            << " --no-progress"
            << " -o " << Quote(dataset_dir)
            << " > " << Quote(stdout_path)
            << " 2> " << Quote(stderr_path);

    const int result = std::system(command.str().c_str());
    const std::string stderr_output = ReadFile(stderr_path);
    if (result != 0) {
        if (stderr_output.find("CUDA ground-truth computation required but unavailable") != std::string::npos) {
            std::cerr << stderr_output;
            return 77;
        }
        std::cerr << "CUDA CLI command failed\n" << stderr_output;
        return 2;
    }

    DatasetMetadata metadata;
    metadata.Load(dataset_dir / "metadata.json");
    if (!metadata.chamfer_distance_l2_ba.has_value()) {
        std::cerr << "Expected CUDA CLI command to store l2:ba ground truth\n";
        return 3;
    }
    if (metadata.chamfer_distance_l2_ab.has_value()) {
        std::cerr << "Expected CUDA CLI command to leave unrequested l2:ab ground truth empty\n";
        return 4;
    }

    return 0;
}
