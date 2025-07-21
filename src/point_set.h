#ifndef POINT_SET_H_
#define POINT_SET_H_

#include <spdlog/spdlog.h>

#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <utility>
#include <vector>

template <typename T>
using Point = std::vector<T>;

using PointVariant = std::variant<Point<uint8_t>, Point<int>, Point<double>>;

// ---------------------------------------------
// PointSetWriter Definition
// ---------------------------------------------

class PointSetWriter {
   public:
    virtual ~PointSetWriter() = default;
    virtual void AddPoint(const PointVariant& point) = 0;
    virtual void Flush() = 0;
};

// ---------------------------------------------
// InMemoryPointSetWriter Definition
// ---------------------------------------------

template <typename T>
class InMemoryPointSetWriter : public PointSetWriter {
   public:
    InMemoryPointSetWriter(std::filesystem::path file_path, unsigned int num_dimensions);

    void AddPoint(const PointVariant& point) override;
    void Flush() override;

   private:
    std::filesystem::path file_path_;
    unsigned int num_dimensions_;
    std::vector<Point<T>> points_;
};

// ---------------------------------------------
// DiskPointSetWriter Definition
// ---------------------------------------------

template <typename T>
class DiskPointSetWriter : public PointSetWriter {
   public:
    DiskPointSetWriter(const std::filesystem::path& file_path, unsigned int num_dimensions);
    ~DiskPointSetWriter();

    DiskPointSetWriter(const DiskPointSetWriter&) = delete;
    DiskPointSetWriter& operator=(const DiskPointSetWriter&) = delete;
    DiskPointSetWriter(DiskPointSetWriter&&) noexcept = default;
    DiskPointSetWriter& operator=(DiskPointSetWriter&&) noexcept = default;

    void AddPoint(const PointVariant& point) override;
    void Flush() override;

   private:
    std::ofstream ofs_;
    unsigned int num_dimensions_;
};

// ---------------------------------------------
// PointSetWriterFactory Definition
// ---------------------------------------------
class PointSetWriterFactory {
   public:
    static std::unique_ptr<PointSetWriter> Create(const std::filesystem::path& file_path, const std::string& data_type,
                                                  unsigned int num_dimensions);
};

// ---------------------------------------------
// PointSetReader Definition
// ---------------------------------------------

class PointSetReader {
   public:
    virtual ~PointSetReader() = default;

    [[nodiscard]] virtual unsigned int get_num_points() const = 0;
    [[nodiscard]] virtual unsigned int get_num_dimensions() const = 0;

    virtual PointVariant GetPoint(unsigned int index) = 0;
};

// ---------------------------------------------
// InMemoryPointSetReader Definition
// ---------------------------------------------

template <typename T>
class InMemoryPointSetReader : public PointSetReader {
   public:
    InMemoryPointSetReader(const std::filesystem::path& file_path, unsigned int num_points,
                           unsigned int num_dimensions);

    [[nodiscard]] unsigned int get_num_points() const override;
    [[nodiscard]] unsigned int get_num_dimensions() const override;

    PointVariant GetPoint(unsigned int index) override;

   private:
    unsigned int num_points_;
    unsigned int num_dimensions_;
    std::vector<Point<T>> points_;
};

// ---------------------------------------------
// DiskPointSetReader Definition
// ---------------------------------------------

template <typename T>
class DiskPointSetReader : public PointSetReader {
   public:
    DiskPointSetReader(const std::filesystem::path& file_path, unsigned int num_points, unsigned int num_dimensions);
    ~DiskPointSetReader();

    DiskPointSetReader(const DiskPointSetReader&) = delete;
    DiskPointSetReader& operator=(const DiskPointSetReader&) = delete;
    DiskPointSetReader(DiskPointSetReader&&) noexcept = default;
    DiskPointSetReader& operator=(DiskPointSetReader&&) noexcept = default;

    [[nodiscard]] unsigned int get_num_points() const override;
    [[nodiscard]] unsigned int get_num_dimensions() const override;

    PointVariant GetPoint(unsigned int index) override;

   private:
    std::ifstream ifs_;
    unsigned int num_points_;
    unsigned int num_dimensions_;
};

// ---------------------------------------------
// PointSetReaderFactory Definition
// ---------------------------------------------
class PointSetReaderFactory {
   public:
    static std::unique_ptr<PointSetReader> Create(const std::filesystem::path& file_path, const std::string& data_type,
                                                  unsigned int num_points, unsigned int num_dimensions);
};

// ---------------------------------------------
// InMemoryPointSetWriter Implementation
// ---------------------------------------------

template <typename T>
InMemoryPointSetWriter<T>::InMemoryPointSetWriter(std::filesystem::path file_path, unsigned int num_dimensions)
    : file_path_(std::move(file_path)), num_dimensions_(num_dimensions) {}

template <typename T>
void InMemoryPointSetWriter<T>::AddPoint(const PointVariant& point) {
    const auto& concrete_point = std::get<Point<T>>(point);
    points_.emplace_back(concrete_point);
}

template <typename T>
void InMemoryPointSetWriter<T>::Flush() {
    std::ofstream ofs(file_path_, std::ios::binary | std::ios::trunc);
    if (!ofs.is_open()) {
        spdlog::error(std::format("Failed to open file: {}", file_path_.string()));
    }
    for (const auto& point : points_) {
        ofs.write(reinterpret_cast<const char*>(point.data()), static_cast<std::streamsize>(sizeof(T) * point.size()));
    }
}

// ---------------------------------------------
// DiskPointSetWriter Implementation
// ---------------------------------------------

template <typename T>
DiskPointSetWriter<T>::DiskPointSetWriter(const std::filesystem::path& file_path, unsigned int num_dimensions)
    : num_dimensions_(num_dimensions) {
    ofs_.open(file_path, std::ios::binary | std::ios::trunc);
    if (!ofs_.is_open()) {
        spdlog::error(std::format("Failed to open file: {}", file_path.string()));
    }
}

template <typename T>
DiskPointSetWriter<T>::~DiskPointSetWriter() {
    if (ofs_.is_open()) {
        ofs_.close();
    }
}

template <typename T>
void DiskPointSetWriter<T>::AddPoint(const PointVariant& point) {
    const auto& concrete_point = std::get<Point<T>>(point);

    if (concrete_point.size() != num_dimensions_) {
        spdlog::error(std::format("Point dimensions do not match the set dimensions: expected {}, got {}",
                                  num_dimensions_, concrete_point.size()));
    }

    ofs_.seekp(0, std::ios::end);
    ofs_.write(reinterpret_cast<const char*>(concrete_point.data()),
               static_cast<std::streamsize>(sizeof(T) * concrete_point.size()));
}

template <typename T>
void DiskPointSetWriter<T>::Flush() {
    // Do nothing here, as we write directly to the file on each AddPoint call.
}

// ---------------------------------------------
// InMemoryPointSetReader Implementation
// ---------------------------------------------

template <typename T>
InMemoryPointSetReader<T>::InMemoryPointSetReader(const std::filesystem::path& file_path, unsigned int num_points,
                                                  unsigned int num_dimensions)
    : num_points_(num_points), num_dimensions_(num_dimensions) {
    std::ifstream ifs(file_path, std::ios::binary);
    if (!ifs.is_open()) {
        spdlog::error(std::format("Failed to open file: {}", file_path.string()));
    }

    points_.resize(num_points_);
    for (unsigned int i = 0; i < num_points_; ++i) {
        points_[i].resize(num_dimensions_);
        ifs.read(reinterpret_cast<char*>(points_[i].data()), sizeof(T) * num_dimensions_);
    }
}

template <typename T>
unsigned int InMemoryPointSetReader<T>::get_num_points() const {
    return num_points_;
}

template <typename T>
unsigned int InMemoryPointSetReader<T>::get_num_dimensions() const {
    return num_dimensions_;
}

template <typename T>
PointVariant InMemoryPointSetReader<T>::GetPoint(unsigned int index) {
    return points_.at(index);
}

// ---------------------------------------------
// DiskPointSetReader Implementation
// ---------------------------------------------

template <typename T>
DiskPointSetReader<T>::DiskPointSetReader(const std::filesystem::path& file_path, unsigned int num_points,
                                          unsigned int num_dimensions)
    : num_points_(num_points), num_dimensions_(num_dimensions) {
    ifs_.open(file_path, std::ios::binary);
    if (!ifs_.is_open()) {
        spdlog::error(std::format("Failed to open file: {}", file_path.string()));
    }
}

template <typename T>
DiskPointSetReader<T>::~DiskPointSetReader() {
    if (ifs_.is_open()) {
        ifs_.close();
    }
}

template <typename T>
unsigned int DiskPointSetReader<T>::get_num_points() const {
    return num_points_;
}

template <typename T>
unsigned int DiskPointSetReader<T>::get_num_dimensions() const {
    return num_dimensions_;
}

template <typename T>
PointVariant DiskPointSetReader<T>::GetPoint(unsigned int index) {
    ifs_.seekg(index * sizeof(T) * num_dimensions_, std::ios::beg);
    Point<T> point(num_dimensions_);
    ifs_.read(reinterpret_cast<char*>(point.data()), sizeof(T) * num_dimensions_);
    return point;
}

#endif