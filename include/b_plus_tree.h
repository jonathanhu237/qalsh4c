#ifndef B_PLUS_TREE_H_
#define B_PLUS_TREE_H_

#include <filesystem>
#include <vector>

#include "constants.h"

namespace qalsh_chamfer {

namespace fs = std::filesystem;

class InternalNode {
   public:
    friend class BPlusTreeBulkLoader;

   private:
    InternalNode();
    auto Serialize(std::vector<char> &buffer) -> void;

    NodeType node_type_;
};

class LeafNode {
   public:
    friend class BPlusTreeBulkLoader;

   private:
    LeafNode();
    auto Serialize(std::vector<char> &buffer) -> void;

    NodeType node_type_;
};

class BPlusTreeBulkLoader {
   public:
    BPlusTreeBulkLoader(fs::path file_path, unsigned int page_size, std::vector<double> &dot_vector);
    auto BulkLoad() -> void;

   private:
    fs::path file_path_;
    unsigned int page_size_;
    std::vector<double> dot_vector_;
};

}  // namespace qalsh_chamfer

#endif