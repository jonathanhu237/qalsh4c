#include "pager.h"

#include <ios>
#include <vector>

namespace qalsh_chamfer {

Pager::Pager(const fs::path& file_path, unsigned int page_size)
    : page_size_(page_size), num_page_(0), next_page_num_(0) {
    ofs_.open(file_path, std::ios::binary);
}

Pager::~Pager() {
    if (ofs_.is_open()) {
        ofs_.close();
    }
}

auto Pager::get_page_size() const -> unsigned int { return page_size_; }

auto Pager::get_num_page() const -> unsigned int { return num_page_; }

auto Pager::get_next_page_num() const -> unsigned int { return next_page_num_; }

auto Pager::Allocate() -> unsigned int {
    unsigned int new_page_num = next_page_num_++;

    ofs_.seekp(static_cast<std::streamoff>(new_page_num * page_size_));
    std::vector<char> empty_page(page_size_, 0);
    ofs_.write(empty_page.data(), static_cast<std::streamsize>(empty_page.size()));
    num_page_++;

    return new_page_num;
}

auto Pager::WritePage(unsigned int page_num, const std::vector<char>& buffer) -> void {
    ofs_.seekp(static_cast<std::streamoff>(page_num * page_size_));
    ofs_.write(buffer.data(), static_cast<std::streamsize>(buffer.size()));
}

}  // namespace qalsh_chamfer