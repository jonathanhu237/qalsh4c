#ifndef B_PLUS_TREE_H_
#define B_PLUS_TREE_H_

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <optional>
#include <utility>
#include <vector>

namespace qalsh_chamfer {

namespace fs = std::filesystem;

class InternalNode {
   public:
    friend class BPlusTree;
    friend class BPlusTreeBulkLoader;
    friend class BPlusTreeSearcher;

   private:
    InternalNode();
    InternalNode(unsigned int order);
    InternalNode(const std::vector<char>& buffer);

    auto static GetHeaderSize() -> size_t;
    auto Serialize(std::vector<char>& buffer) const -> void;

    // Header
    unsigned int num_children_;

    // Data
    std::vector<double> keys_;
    std::vector<unsigned int> pointers_;
};

class LeafNode {
   public:
    friend class BPlusTree;
    friend class BPlusTreeBulkLoader;
    friend class BPlusTreeSearcher;

   private:
    LeafNode();
    LeafNode(unsigned int order);
    LeafNode(const std::vector<char>& buffer);

    auto static GetHeaderSize() -> size_t;
    auto Serialize(std::vector<char>& buffer) const -> void;

    // Header
    unsigned int num_entries_;
    unsigned int prev_leaf_page_num_;
    unsigned int next_leaf_page_num_;

    // Data
    std::vector<double> keys_;
    std::vector<unsigned int> values_;
};

// class BPlusTree {
//    public:
//     using KeyValuePair = std::pair<double, unsigned int>;
//     using SearchLocation = std::pair<LeafNode, size_t>;

//     BPlusTree(Pager&& pager);

//     auto BulkLoad(std::vector<KeyValuePair>& data) -> void;

//     auto InitIncrementalSearch(double key) -> void;
//     auto IncrementalSearch(double key, double bound) -> std::vector<unsigned int>;

//    private:
//     auto LocateLeafMayContainKey(double key) -> LeafNode;
//     auto LocateLeafByPageNum(unsigned int page_num) -> LeafNode;

//     Pager pager_;

//     // Header
//     unsigned int root_page_num_;
//     unsigned int level_;
//     unsigned int internal_node_order_;
//     unsigned int leaf_node_order_;

//     // Variables for incremental search
//     std::optional<SearchLocation> left_search_location_;
//     std::optional<SearchLocation> right_search_location_;
// };

class BPlusTreeBulkLoader {
   public:
    using KeyValuePair = std::pair<double, unsigned int>;

    BPlusTreeBulkLoader(const fs::path& file_path, unsigned int page_size);
    // TODO: should I implement a destructor to close the file?

    auto Build(const std::vector<KeyValuePair>& data) -> void;

   private:
    auto AllocatePage() -> unsigned int;
    auto WritePage(unsigned int page_num, const std::vector<char>& buffer) -> void;

    std::ofstream ofs_;
    unsigned int page_size_;
    unsigned int num_page_;
    unsigned int next_page_num_;

    // Header
    unsigned int root_page_num_;
    unsigned int level_;
    unsigned int internal_node_order_;
    unsigned int leaf_node_order_;
};

class BPlusTreeSearcher {
   public:
    using SearchLocation = std::pair<LeafNode, size_t>;

    BPlusTreeSearcher(const fs::path& file_path, unsigned int page_size, double key);

    auto IncrementalSearch(double bound) -> std::vector<unsigned int>;

   private:
    auto LocateLeafMayContainKey() -> LeafNode;
    auto LocateLeafByPageNum(unsigned int page_num) -> LeafNode;

    auto ReadPage(unsigned int page_num) -> std::vector<char>;

    std::ifstream ifs_;
    unsigned int page_size_;
    double key_;
    std::optional<SearchLocation> left_search_location_;
    std::optional<SearchLocation> right_search_location_;

    // Header
    unsigned int root_page_num_;
    unsigned int level_;
    unsigned int internal_node_order_;
    unsigned int leaf_node_order_;
};

}  // namespace qalsh_chamfer

#endif