#ifndef POINT_SET_H_
#define POINT_SET_H_

#include <filesystem>
#include <format>
#include <fstream>
#include <vector>

template <typename T>
class PointSetWriter {
   public:
    explicit PointSetWriter(const std::filesystem::path& file_path);
    ~PointSetWriter();

    PointSetWriter(const PointSetWriter&) = delete;
    auto operator=(const PointSetWriter&) -> PointSetWriter& = delete;
    PointSetWriter(PointSetWriter&&) noexcept = default;
    auto operator=(PointSetWriter&&) noexcept -> PointSetWriter& = default;

    auto AddPoint(const std::vector<T>& point) -> void;

   private:
    std::ofstream ofs_;
};

template <typename T>
PointSetWriter<T>::PointSetWriter(const std::filesystem::path& file_path) {
    ofs_.open(file_path, std::ios::binary | std::ios::trunc);
    if (!ofs_.is_open()) {
        throw std::runtime_error(std::format("Failed to open file: {}", file_path.string()));
    }
}

template <typename T>
PointSetWriter<T>::~PointSetWriter() {
    if (ofs_.is_open()) {
        ofs_.close();
    }
}

template <typename T>
auto PointSetWriter<T>::AddPoint(const std::vector<T>& point) -> void {
    ofs_.seekp(0, std::ios::end);
    ofs_.write(reinterpret_cast<const char*>(point.data()), sizeof(T) * point.size());
}

#endif