#include "external_sorter.h"

#include <algorithm>
#include <format>
#include <fstream>
#include <queue>
#include <utility>

#include "log.h"

ExternalSorter::ExternalSorter(const std::filesystem::path& temp_dir, size_t chunk_size)
    : temp_dir_(temp_dir), chunk_size_(chunk_size) {
    buffer_.reserve(chunk_size_);
}

void ExternalSorter::Add(const DotProductPointIdPair& pair) {
    buffer_.emplace_back(pair);
    total_pairs_++;

    if (buffer_.size() >= chunk_size_) {
        FlushRun();
    }
}

void ExternalSorter::Finish() {
    if (!buffer_.empty()) {
        FlushRun();
    }
}

std::function<DotProductPointIdPair()> ExternalSorter::MakeMergeReader() const {
    struct HeapEntry {
        DotProductPointIdPair pair;
        size_t run_index{0};
    };

    struct HeapEntryGreater {
        bool operator()(const HeapEntry& lhs, const HeapEntry& rhs) const {
            return DotProductPointIdPairLess{}(rhs.pair, lhs.pair);
        }
    };

    auto streams = std::make_shared<std::vector<std::ifstream>>();
    streams->reserve(run_files_.size());

    auto heap = std::make_shared<std::priority_queue<HeapEntry, std::vector<HeapEntry>, HeapEntryGreater>>();

    for (size_t run_index = 0; run_index < run_files_.size(); ++run_index) {
        streams->emplace_back(run_files_[run_index], std::ios::binary);
        if (!streams->back().is_open()) {
            Log::error("Failed to open run file: {}", run_files_[run_index].string());
        }

        DotProductPointIdPair pair;
        streams->back().read(reinterpret_cast<char*>(&pair), static_cast<std::streamsize>(sizeof(pair)));
        if (streams->back().gcount() == static_cast<std::streamsize>(sizeof(pair))) {
            heap->emplace(HeapEntry{.pair = pair, .run_index = run_index});
        }
    }

    return [streams = std::move(streams), heap = std::move(heap)]() mutable {
        if (heap->empty()) {
            Log::error("Merge reader exhausted unexpectedly.");
        }

        HeapEntry entry = heap->top();
        heap->pop();

        std::ifstream& stream = streams->at(entry.run_index);
        DotProductPointIdPair next_pair;
        stream.read(reinterpret_cast<char*>(&next_pair), static_cast<std::streamsize>(sizeof(next_pair)));
        if (stream.gcount() == static_cast<std::streamsize>(sizeof(next_pair))) {
            heap->emplace(HeapEntry{.pair = next_pair, .run_index = entry.run_index});
        }

        return entry.pair;
    };
}

ExternalSorter::~ExternalSorter() {
    for (const auto& run_file : run_files_) {
        std::error_code error;
        std::filesystem::remove(run_file, error);
    }
}

void ExternalSorter::FlushRun() {
    std::ranges::sort(buffer_, DotProductPointIdPairLess{});

    std::filesystem::path run_file = temp_dir_ / std::format("_run_{}.tmp", run_files_.size());
    std::ofstream ofs(run_file, std::ios::binary | std::ios::trunc);
    if (!ofs.is_open()) {
        Log::error("Failed to open run file for writing: {}", run_file.string());
    }

    ofs.write(reinterpret_cast<const char*>(buffer_.data()),
              static_cast<std::streamsize>(buffer_.size() * sizeof(DotProductPointIdPair)));
    run_files_.emplace_back(run_file);
    buffer_.clear();
}
