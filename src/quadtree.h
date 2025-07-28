#ifndef QUADTREE_H_
#define QUADTREE_H_

#include <filesystem>
#include <memory>
#include <utility>
#include <vector>

#include "point_set.h"

struct QuadtreeNode {
    std::vector<std::unique_ptr<QuadtreeNode>> children;
    std::optional<std::vector<unsigned int>> cluster;

    void SetCluster(const std::vector<unsigned int>& new_cluster);
    void AddChild(std::unique_ptr<QuadtreeNode>);
};

class Quadtree {
   public:
    Quadtree(const std::filesystem::path& dataset_directory, unsigned int max_level, bool need_random_shift);
    void Build();
    [[nodiscard]] std::vector<unsigned int> DepthFirstSearch() const;

   private:
    void ComputeShift(const std::vector<double>& lower, std::vector<double>& upper);
    [[nodiscard]] std::unique_ptr<QuadtreeNode> BuildTreeAux(const std::vector<unsigned int>& cluster,
                                                             const std::vector<double>& lower,
                                                             const std::vector<double>& upper,
                                                             unsigned int cur_max_level);
    [[nodiscard]] std::pair<std::vector<double>, std::vector<double>> FindBounds(
        const std::vector<bool>& edge, const std::vector<double>& lower, const std::vector<double>& mid_point,
        const std::vector<double>& upper) const;

    std::unique_ptr<CombinePointSetReader> combine_reader_;

    unsigned int num_points_;
    unsigned int num_dimensions_;
    unsigned int max_level_;

    bool need_random_shift_;
    std::vector<double> shift_;

    std::unique_ptr<QuadtreeNode> root_;
};

#endif