#include "external_sorter.h"

#include <filesystem>
#include <vector>

int main() {
    const std::filesystem::path temp_dir = std::filesystem::current_path() / "external_sorter_test_tmp";
    std::filesystem::remove_all(temp_dir);
    std::filesystem::create_directories(temp_dir);

    std::vector<DotProductPointIdPair> merged;
    {
        ExternalSorter sorter(temp_dir, 2);
        sorter.Add({.dot_product = 3.5, .point_id = 0});
        sorter.Add({.dot_product = 1.0, .point_id = 1});
        sorter.Add({.dot_product = 2.0, .point_id = 2});
        sorter.Add({.dot_product = -1.0, .point_id = 3});
        sorter.Add({.dot_product = 4.2, .point_id = 4});
        sorter.Finish();
        sorter.MergeTo([&](const DotProductPointIdPair& pair) { merged.push_back(pair); });
    }

    if (merged.size() != 5) {
        return 1;
    }
    for (size_t i = 1; i < merged.size(); ++i) {
        if (merged[i - 1].dot_product > merged[i].dot_product) {
            return 1;
        }
    }
    for (const auto& entry : std::filesystem::directory_iterator(temp_dir)) {
        if (entry.is_regular_file()) {
            return 1;
        }
    }

    std::filesystem::remove_all(temp_dir);
    return 0;
}
