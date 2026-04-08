#include "b_plus_tree.h"

#include <filesystem>
#include <fstream>
#include <vector>

namespace {
unsigned int ReadRootPageNum(const std::filesystem::path& path) {
    std::ifstream ifs(path, std::ios::binary);
    unsigned int root_page_num = 0;
    ifs.read(reinterpret_cast<char*>(&root_page_num), static_cast<std::streamsize>(sizeof(root_page_num)));
    return root_page_num;
}

void OverwritePage(const std::filesystem::path& path, unsigned int page_num, unsigned int page_size) {
    std::fstream fs(path, std::ios::binary | std::ios::in | std::ios::out);
    std::vector<char> zero_page(page_size, 0);
    fs.seekp(static_cast<std::streamoff>(page_num) * page_size, std::ios::beg);
    fs.write(zero_page.data(), static_cast<std::streamsize>(zero_page.size()));
}

bool FindExactMatch(BPlusTreeSearcher& searcher, double query_key, unsigned int expected_point_id) {
    std::vector<unsigned int> visited;
    searcher.Init(query_key);
    searcher.IncrementalSearch(0.0, [&](unsigned int point_id, [[maybe_unused]] double dot_product) {
        visited.push_back(point_id);
    });
    return visited.size() == 1 && visited.front() == expected_point_id;
}
}  // namespace

int main() {
    const std::filesystem::path temp_dir = std::filesystem::current_path() / "b_plus_tree_searcher_cache_test_tmp";
    std::filesystem::remove_all(temp_dir);
    std::filesystem::create_directories(temp_dir);

    const unsigned int page_size = 48;
    const std::filesystem::path tree_file = temp_dir / "tree.bin";
    const std::vector<DotProductPointIdPair> data = {
        {.dot_product = 0.0, .point_id = 10},
        {.dot_product = 1.0, .point_id = 11},
        {.dot_product = 2.0, .point_id = 12},
        {.dot_product = 3.0, .point_id = 13},
        {.dot_product = 4.0, .point_id = 14},
        {.dot_product = 5.0, .point_id = 15},
    };

    {
        BPlusTreeBulkLoader loader(tree_file, page_size);
        loader.Build(data);
    }

    const unsigned int root_page_num = ReadRootPageNum(tree_file);

    BPlusTreeSearcher searcher(tree_file, page_size);
    if (!FindExactMatch(searcher, 1.0, 11)) {
        std::filesystem::remove_all(temp_dir);
        return 1;
    }

    OverwritePage(tree_file, 0, page_size);
    OverwritePage(tree_file, root_page_num, page_size);

    try {
        if (!FindExactMatch(searcher, 5.0, 15)) {
            std::filesystem::remove_all(temp_dir);
            return 1;
        }
    } catch (...) {
        std::filesystem::remove_all(temp_dir);
        return 1;
    }

    std::filesystem::remove_all(temp_dir);
    return 0;
}
