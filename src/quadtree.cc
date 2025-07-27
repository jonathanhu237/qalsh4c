#include "quadtree.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <ranges>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "point_set.h"
#include "types.h"

void QuadtreeNode::SetCluster(const std::vector<unsigned int>& new_cluster) { cluster = new_cluster; }

void QuadtreeNode::AddChild(std::unique_ptr<QuadtreeNode> node) { children.emplace_back(std::move(node)); }

Quadtree::Quadtree(const std::filesystem::path& dataset_directory, unsigned int max_level, bool need_random_shift)
    : max_level_(max_level), need_random_shift_(need_random_shift) {
    // Load dataset metadata.
    DatasetMetadata dataset_metadata;
    dataset_metadata.Load(dataset_directory / "metadata.toml");

    // Initialize members.
    base_reader_ = PointSetReaderFactory::Create(dataset_directory / "base.bin", dataset_metadata.data_type,
                                                 dataset_metadata.base_num_points, dataset_metadata.num_dimensions);
    num_points_ = dataset_metadata.base_num_points;
    num_dimensions_ = dataset_metadata.num_dimensions;

    // Initialize shift.
    shift_.resize(num_dimensions_, 0);
}

void Quadtree::Build() {
    // Get the number of dimensions.
    unsigned int num_dimensions = base_reader_->get_num_dimensions();

    // Suppose the `dataset` is[[2, 8], [6, 2], [7, 4], [3, 6], [9, 7]],
    // then `lower` will be [2, 2] and `upper` will be [9, 8], that is,
    // the `lower` is the coordinates of the southwest corner of the dataset,
    // and the `upper` is the northeast.
    std::vector<double> lower(num_dimensions_, std::numeric_limits<double>::max());
    std::vector<double> upper(num_dimensions_, std::numeric_limits<double>::lowest());

    for (unsigned int i = 0; i < base_reader_->get_num_points(); i++) {
        PointVariant point = base_reader_->GetPoint(i);

        std::visit(
            [&](const auto& concrete_point) {
                for (unsigned int dim = 0; dim < num_dimensions; dim++) {
                    lower[dim] = std::min(lower[dim], static_cast<double>(concrete_point[dim]));
                    upper[dim] = std::max(upper[dim], static_cast<double>(concrete_point[dim]));
                }
            },
            point);
    }

    // Initialize the shift if set need_random_shift_ as true.
    if (need_random_shift_) {
        ComputeShift(lower, upper);
    }

    // Build the Quadtree
    std::vector<unsigned int> cluster(num_points_);
    std::iota(cluster.begin(), cluster.end(), 0);
    root_ = BuildTreeAux(cluster, lower, upper);
}

void Quadtree::ComputeShift(const std::vector<double>& lower, std::vector<double>& upper) {
    std::mt19937 gen(std::random_device{}());

    for (unsigned int dim = 0; dim < num_dimensions_; dim++) {
        double spread = upper[dim] - lower[dim];
        std::uniform_real_distribution<double> dist(0, spread);
        shift_[dim] = dist(gen);
        upper[dim] += shift_[dim];
    }
}

std::unique_ptr<QuadtreeNode> Quadtree::BuildTreeAux(const std::vector<unsigned int>& cluster,
                                                     const std::vector<double>& lower,
                                                     const std::vector<double>& upper) {
    auto node = std::make_unique<QuadtreeNode>();

    if (max_level_ == 1 || cluster.size() <= 1) {
        node->SetCluster(cluster);
        return node;
    }

    // Get the midpoint of cell.
    std::vector<double> mid_point(num_dimensions_, 0);
    for (unsigned int dim = 0; dim < num_dimensions_; dim++) {
        mid_point[dim] = 0.5 * (lower[dim] + upper[dim]);
    }

    // Generate the sub clusters
    std::unordered_map<std::vector<bool>, std::vector<unsigned int>> sub_clusters_map;
    for (auto point_id : cluster) {
        PointVariant point = base_reader_->GetPoint(point_id);
        std::vector<bool> edge(num_dimensions_);

        std::visit(
            [&](const auto& concrete_point) {
                for (unsigned int dim = 0; dim < num_dimensions_; dim++) {
                    edge[dim] = (concrete_point[dim] <= mid_point[dim]);
                }
            },
            point);

        sub_clusters_map[edge].emplace_back(point_id);
    }

    // Loop over each non-empty sub-cells.
    for (const auto& [edge, sub_cluster] : sub_clusters_map) {
        auto [sub_lower, sub_upper] = FindBounds(edge, lower, mid_point, upper);
        node->AddChild(BuildTreeAux(sub_cluster, sub_lower, sub_upper));
    }

    return node;
}

std::pair<std::vector<double>, std::vector<double>> Quadtree::FindBounds(const std::vector<bool>& edge,
                                                                         const std::vector<double>& lower,
                                                                         const std::vector<double>& mid_point,
                                                                         const std::vector<double>& upper) const {
    std::vector<double> sub_lower(num_dimensions_);
    std::vector<double> sub_upper(num_dimensions_);

    for (unsigned int dim = 0; dim < num_dimensions_; dim++) {
        if (edge[dim]) {
            sub_lower[dim] = lower[dim];
            sub_upper[dim] = mid_point[dim];
        } else {
            sub_lower[dim] = mid_point[dim];
            sub_upper[dim] = upper[dim];
        }
    }

    return {sub_lower, sub_upper};
}

std::vector<unsigned int> Quadtree::DepthFirstSearch() const {
    std::vector<unsigned int> dfs_order(num_points_);
    std::stack<const QuadtreeNode*> stack;
    std::unordered_set<const QuadtreeNode*> visited;

    stack.emplace(root_);
    while (!stack.empty()) {
        const QuadtreeNode* vertex = stack.top();
        stack.pop();

        if (visited.contains(vertex)) {
            continue;
        }
        visited.emplace(vertex);

        if (vertex->children.empty()) {
            if (vertex->cluster.has_value()) {
                dfs_order.insert(dfs_order.end(), vertex->cluster->begin(), vertex->cluster->end());
            }
        }

        for (const auto& it : std::ranges::reverse_view(vertex->children)) {
            stack.emplace(it.get());
        }
    }

    return dfs_order;
}