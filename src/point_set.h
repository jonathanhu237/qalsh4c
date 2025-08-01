#ifndef POINT_SET_H_
#define POINT_SET_H_

#include <spdlog/spdlog.h>

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

// ---------------------------------------------
// PointSet Definition
// ---------------------------------------------
class PointSet {
   public:
    virtual ~PointSet() = default;

    [[nodiscard]] virtual unsigned int get_num_points() const = 0;
    [[nodiscard]] virtual unsigned int get_num_dimensions() const = 0;

    virtual Point GetPoint(unsigned int index) = 0;
};

// ---------------------------------------------
// InMemoryPointSet Definition
// ---------------------------------------------
class InMemoryPointSet : public PointSet {
   public:
    InMemoryPointSet(const PointSetMetadata& point_set_metadata);

    [[nodiscard]] unsigned int get_num_points() const override;
    [[nodiscard]] unsigned int get_num_dimensions() const override;

    Point GetPoint(unsigned int index) override;

   private:
    unsigned int num_points_;
    unsigned int num_dimensions_;

    std::vector<Point> points_;
};

// ---------------------------------------------
// DiskPointSet Definition
// ---------------------------------------------
class DiskPointSet : public PointSet {
   public:
    DiskPointSet(const PointSetMetadata& point_set_metadata);

    [[nodiscard]] unsigned int get_num_points() const override;
    [[nodiscard]] unsigned int get_num_dimensions() const override;

    Point GetPoint(unsigned int index) override;

   private:
    unsigned int num_points_;
    unsigned int num_dimensions_;

    std::ifstream ifs_;
};

#endif