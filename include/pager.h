#ifndef PAGER_H_
#define PAGER_H_

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>

namespace qalsh_chamfer {

namespace fs = std::filesystem;

class Pager {
   public:
    enum class PagerMode : uint8_t { kRead, kWrite };

    Pager(const fs::path& file_path, unsigned int page_size, PagerMode mode);
    ~Pager();

    Pager(const Pager&) = delete;
    auto operator=(const Pager&) -> Pager& = delete;
    Pager(Pager&&) noexcept = default;
    auto operator=(Pager&&) noexcept -> Pager& = default;

    auto get_mode() const -> PagerMode;
    auto get_page_size() const -> unsigned int;
    auto get_num_page() const -> unsigned int;
    auto get_next_page_num() const -> unsigned int;

    // Only used in write mode
    auto Allocate() -> unsigned int;
    auto WritePage(unsigned int page_num, const std::vector<char>& buffer) -> void;

    // Only used in read mode
    auto ReadPage(unsigned int page_num) -> std::vector<char>;

   private:
    std::ofstream ofs_;
    std::ifstream ifs_;

    PagerMode mode_;
    unsigned int page_size_;

    // Only used in write mode
    unsigned int num_page_;
    unsigned int next_page_num_;
};

}  // namespace qalsh_chamfer

#endif