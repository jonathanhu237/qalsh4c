#include "external_sorter.h"

#include <filesystem>
#include <vector>

int main() {
    const std::filesystem::path temp_dir = std::filesystem::current_path() / "external_sorter_prefix_test_tmp";
    std::filesystem::remove_all(temp_dir);
    std::filesystem::create_directories(temp_dir);

    {
        ExternalSorter sorter_a(temp_dir, "a", 2);
        ExternalSorter sorter_b(temp_dir, "b", 2);

        sorter_a.Add({.dot_product = 3.0, .point_id = 30});
        sorter_a.Add({.dot_product = 1.0, .point_id = 10});
        sorter_a.Add({.dot_product = 2.0, .point_id = 20});
        sorter_a.Finish();

        sorter_b.Add({.dot_product = 6.0, .point_id = 60});
        sorter_b.Add({.dot_product = 4.0, .point_id = 40});
        sorter_b.Add({.dot_product = 5.0, .point_id = 50});
        sorter_b.Finish();

        const auto reader_a = sorter_a.MakeMergeReader();
        const auto reader_b = sorter_b.MakeMergeReader();

        const std::vector<DotProductPointIdPair> expected_a = {
            {.dot_product = 1.0, .point_id = 10},
            {.dot_product = 2.0, .point_id = 20},
            {.dot_product = 3.0, .point_id = 30},
        };
        const std::vector<DotProductPointIdPair> expected_b = {
            {.dot_product = 4.0, .point_id = 40},
            {.dot_product = 5.0, .point_id = 50},
            {.dot_product = 6.0, .point_id = 60},
        };

        for (const auto& expected : expected_a) {
            if (const auto actual = reader_a();
                actual.dot_product != expected.dot_product || actual.point_id != expected.point_id) {
                std::filesystem::remove_all(temp_dir);
                return 1;
            }
        }
        for (const auto& expected : expected_b) {
            if (const auto actual = reader_b();
                actual.dot_product != expected.dot_product || actual.point_id != expected.point_id) {
                std::filesystem::remove_all(temp_dir);
                return 1;
            }
        }
    }

    std::filesystem::remove_all(temp_dir);
    return 0;
}
