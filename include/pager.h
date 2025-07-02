#ifndef PAGER_H_
#define PAGER_H_

#include <filesystem>
#include <fstream>
#include <vector>

namespace qalsh_chamfer {

namespace fs = std::filesystem;

class Pager {
   public:
    Pager(const fs::path& file_path, unsigned int page_size);
    ~Pager();

    Pager(const Pager&) = delete;
    auto operator=(const Pager&) -> Pager& = delete;
    Pager(Pager&&) noexcept = default;
    auto operator=(Pager&&) noexcept -> Pager& = default;

    auto get_page_size() const -> unsigned int;
    auto get_num_page() const -> unsigned int;
    auto get_next_page_num() const -> unsigned int;

    auto Allocate() -> unsigned int;
    auto WritePage(unsigned int page_num, const std::vector<char>& buffer) -> void;

   private:
    std::ofstream ofs_;
    unsigned int page_size_;
    unsigned int num_page_;
    unsigned int next_page_num_;
};

}  // namespace qalsh_chamfer

#endif