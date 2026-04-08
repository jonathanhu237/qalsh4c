#pragma once

#include <filesystem>
#include <functional>
#include <vector>

#include "b_plus_tree.h"

class ExternalSorter {
   public:
    ExternalSorter(const std::filesystem::path& temp_dir, size_t chunk_size = 1'000'000);

    void Add(const DotProductPointIdPair& pair);
    void Finish();
    [[nodiscard]] std::function<DotProductPointIdPair()> MakeMergeReader() const;

    template <typename Consumer>
    void MergeTo(Consumer&& consumer) {
        auto next = MakeMergeReader();
        for (size_t i = 0; i < total_pairs_; ++i) {
            std::forward<Consumer>(consumer)(next());
        }
    }

    ~ExternalSorter();

   private:
    void FlushRun();

    std::filesystem::path temp_dir_;
    size_t chunk_size_;
    std::vector<DotProductPointIdPair> buffer_;
    std::vector<std::filesystem::path> run_files_;
    size_t total_pairs_{0};
};
