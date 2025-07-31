#ifndef QALSH_HASH_TABLE_H
#define QALSH_HASH_TABLE_H

#include <cstddef>
#include <optional>
#include <unordered_map>
#include <vector>

#include "b_plus_tree.h"
#include "types.h"

// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions)
class QalshHashTable {
   public:
    virtual ~QalshHashTable() = default;

    virtual void Init(double query_dot_product) = 0;
    virtual std::optional<unsigned int> LeftFindNext(double bound) = 0;
    virtual std::optional<unsigned int> RightFindNext(double bound) = 0;
};

class InMemoryQalshHashTable : public QalshHashTable {
   public:
    InMemoryQalshHashTable(const std::vector<DotProductPointIdPair>& data);

    void Init(double key) override;
    std::optional<unsigned int> LeftFindNext(double bound) override;
    std::optional<unsigned int> RightFindNext(double bound) override;

   private:
    double key_{0.0};
    std::vector<DotProductPointIdPair> data_;

    std::optional<unsigned int> left_;
    std::optional<unsigned int> right_;
};

class DiskQalshHashTable : public QalshHashTable {
   public:
    struct SearchLocation {
        LeafNode leaf_node;
        size_t index;
    };

    DiskQalshHashTable(const std::filesystem::path& file_path, unsigned int page_size);
    void Init(double key) override;
    std::optional<unsigned int> LeftFindNext(double bound) override;
    std::optional<unsigned int> RightFindNext(double bound) override;

   private:
    LeafNode LocateLeafMayContainKey();
    LeafNode LocateLeafByPageNum(unsigned int page_num);

    std::vector<char> ReadPage(unsigned int page_num);

    std::ifstream ifs_;
    unsigned int page_size_{0};
    double key_{0.0};
    std::optional<SearchLocation> left_;
    std::optional<SearchLocation> right_;

    // Header
    unsigned int root_page_num_{0};
    unsigned int level_{0};
    unsigned int internal_node_order_{0};
    unsigned int leaf_node_order_{0};

    // Performance optimizations
    std::unordered_map<unsigned int, std::vector<char>> page_cache_;
};

#endif