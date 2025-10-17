#pragma once

#include <filesystem>
#include <fstream>
#include <vector>

using Coordinate = double;
using Point = std::vector<Coordinate>;

struct PointSetMetadata {
    std::filesystem::path file_path;
    unsigned int num_points{0};
    unsigned int num_dimensions{0};
};

class PointSet {
   public:
    PointSet() = default;
    [[nodiscard]] unsigned int get_num_points() const { return num_points_; };
    [[nodiscard]] unsigned int get_num_dimensions() const { return num_dimensions_; };

    void Init(const PointSetMetadata& metadata);
    Point ReadPoint(unsigned int point_id);
    static std::vector<Point> LoadPointsFromFile(const std::filesystem::path& file_path, unsigned int num_points,
                                                 unsigned int num_dimensions);

   private:
    unsigned int num_points_{0};
    unsigned int num_dimensions_{0};

    std::ifstream ifs_;
};
