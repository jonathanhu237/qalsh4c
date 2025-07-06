#include "pager.h"

#include <ios>
#include <iostream>
#include <vector>

namespace qalsh_chamfer {

Pager::Pager(const fs::path& file_path, unsigned int page_size, PagerMode mode)
    : mode_(mode), page_size_(page_size), num_page_(0), next_page_num_(0) {
    if (mode_ == PagerMode::kWrite) {
        ofs_.open(file_path, std::ios::binary);
        if (!ofs_.is_open()) {
            throw std::ios_base::failure("Failed to open file for writing: " + file_path.string());
        }
    } else if (mode_ == PagerMode::kRead) {
        ifs_.open(file_path, std::ios::binary);
        if (!ifs_.is_open()) {
            throw std::ios_base::failure("Failed to open file for reading: " + file_path.string());
        }
    }
}

Pager::~Pager() {
    if (ofs_.is_open()) {
        ofs_.close();
    }
    if (ifs_.is_open()) {
        ifs_.close();
    }
}

auto Pager::get_mode() const -> PagerMode { return mode_; }

auto Pager::get_page_size() const -> unsigned int { return page_size_; }

auto Pager::get_num_page() const -> unsigned int { return num_page_; }

auto Pager::get_next_page_num() const -> unsigned int { return next_page_num_; }

auto Pager::Allocate() -> unsigned int {
    if (mode_ != PagerMode::kWrite) {
        throw std::runtime_error("Allocate can only be called in write mode.");
    }

    unsigned int new_page_num = next_page_num_++;

    ofs_.seekp(static_cast<std::streamoff>(new_page_num * page_size_));
    std::vector<char> empty_page(page_size_, 0);
    ofs_.write(empty_page.data(), static_cast<std::streamsize>(empty_page.size()));
    num_page_++;

    return new_page_num;
}

auto Pager::WritePage(unsigned int page_num, const std::vector<char>& buffer) -> void {
    if (mode_ != PagerMode::kWrite) {
        throw std::runtime_error("WritePage can only be called in write mode.");
    }

    ofs_.seekp(static_cast<std::streamoff>(page_num * page_size_));
    ofs_.write(buffer.data(), static_cast<std::streamsize>(buffer.size()));
}

auto Pager::ReadPage(unsigned int page_num) -> std::vector<char> {
    if (mode_ != PagerMode::kRead) {
        throw std::runtime_error("ReadPage can only be called in read mode.");
    }

    std::vector<char> buffer(page_size_, 0);
    ifs_.seekg(static_cast<std::streamoff>(page_num * page_size_));
    ifs_.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    return buffer;
}

}  // namespace qalsh_chamfer