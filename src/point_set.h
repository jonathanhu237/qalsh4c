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

// ---------------------------------------------
// PointSetWriter Definition
// ---------------------------------------------
class PointSetWriter {
   public:
    virtual ~PointSetWriter() = default;
    virtual void AddPoint(const Point& point) = 0;
    virtual void Flush() = 0;
};

// ---------------------------------------------
// InMemoryPointSetWriter Definition
// ---------------------------------------------
class InMemoryPointSetWriter : public PointSetWriter {
   public:
    InMemoryPointSetWriter(std::filesystem::path file_path);

    void AddPoint(const Point& point) override;
    void Flush() override;

   private:
    std::filesystem::path file_path_;
    std::vector<Point> points_;
};

// ---------------------------------------------
// DiskPointSetWriter Definition
// ---------------------------------------------
class DiskPointSetWriter : public PointSetWriter {
   public:
    DiskPointSetWriter(const std::filesystem::path& file_path);

    void AddPoint(const Point& point) override;
    void Flush() override;

   private:
    std::ofstream ofs_;
};

#endif