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
// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions)
class PointSetReader {
   public:
    virtual ~PointSetReader() = default;

    [[nodiscard]] virtual unsigned int get_num_points() const = 0;
    [[nodiscard]] virtual unsigned int get_num_dimensions() const = 0;

    virtual Point GetPoint(unsigned int index) = 0;
};

// ---------------------------------------------
// InMemoryPointSet Definition
// ---------------------------------------------
class InMemoryPointSetReader : public PointSetReader {
   public:
    InMemoryPointSetReader(const PointSetMetadata& point_set_metadata);

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
class DiskPointSetReader : public PointSetReader {
   public:
    DiskPointSetReader(const PointSetMetadata& point_set_metadata);

    [[nodiscard]] unsigned int get_num_points() const override;
    [[nodiscard]] unsigned int get_num_dimensions() const override;

    Point GetPoint(unsigned int index) override;

   private:
    unsigned int num_points_;
    unsigned int num_dimensions_;

    std::ifstream ifs_;
};

#endif