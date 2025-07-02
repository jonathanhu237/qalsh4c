#ifndef PAGER_H_
#define PAGER_H_

#include <filesystem>
#include <fstream>

namespace qalsh_chamfer {

namespace fs = std::filesystem;

class Pager {
   public:
    Pager(fs::path file_path, unsigned int page_size);
    ~Pager();

    Pager(const Pager&) = delete;
    auto operator=(const Pager&) -> Pager& = delete;
    Pager(Pager&&) noexcept = default;
    auto operator=(Pager&&) noexcept -> Pager& = default;

   private:
    std::ifstream ifs_;
    unsigned int page_size_;
};

}  // namespace qalsh_chamfer

#endif