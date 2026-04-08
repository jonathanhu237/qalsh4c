#include "b_plus_tree.h"

#include <filesystem>
#include <fstream>
#include <vector>

namespace {
std::vector<char> ReadFile(const std::filesystem::path& path) {
    std::ifstream ifs(path, std::ios::binary);
    return std::vector<char>(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
}
}  // namespace

int main() {
    const std::filesystem::path temp_dir = std::filesystem::current_path() / "b_plus_tree_stream_test_tmp";
    std::filesystem::remove_all(temp_dir);
    std::filesystem::create_directories(temp_dir);

    const std::vector<DotProductPointIdPair> data = {
        {.dot_product = -2.0, .point_id = 5},
        {.dot_product = -0.5, .point_id = 7},
        {.dot_product = 0.0, .point_id = 2},
        {.dot_product = 1.2, .point_id = 8},
        {.dot_product = 3.4, .point_id = 1},
        {.dot_product = 5.6, .point_id = 9},
    };

    const std::filesystem::path regular_file = temp_dir / "regular.bin";
    const std::filesystem::path streamed_file = temp_dir / "streamed.bin";

    {
        BPlusTreeBulkLoader loader(regular_file, 128);
        loader.Build(data);
    }
    {
        BPlusTreeBulkLoader loader(streamed_file, 128);
        size_t index = 0;
        loader.BuildFromStream(data.size(), [&]() { return data.at(index++); });
    }

    if (ReadFile(regular_file) != ReadFile(streamed_file)) {
        return 1;
    }

    std::filesystem::remove_all(temp_dir);
    return 0;
}
