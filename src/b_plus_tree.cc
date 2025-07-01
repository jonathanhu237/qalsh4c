#include "b_plus_tree.h"

#include "constants.h"

namespace qalsh_chamfer {

// ----- InternalNode Implementation -----
InternalNode::InternalNode() : node_type_(NodeType::kInternalNode) {};

auto InternalNode::Serialize(std::vector<char> &buffer) -> void {
    // TODO: Implement this function.
}

// ----- LeafNode Implementation -----
LeafNode::LeafNode() : node_type_(NodeType::kLeafNode) {};

auto LeafNode::Serialize(std::vector<char> &buffer) -> void {
    // TODO: Implement this function.
}

// ----- BPlusTreeBulkLoader Implementation -----
BPlusTreeBulkLoader::BPlusTreeBulkLoader(fs::path file_path, unsigned int page_size, std::vector<double> &dot_vector)
    : file_path_(std::move(file_path)), page_size_(page_size), dot_vector_(dot_vector) {}

auto BPlusTreeBulkLoader::BulkLoad() -> void {
    // TODO: Implement this function.
}

}  // namespace qalsh_chamfer