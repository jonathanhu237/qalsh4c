#pragma once

#include <cstddef>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <cstdint>
#include <concepts>
#include <unordered_map>
#include <utility>
#include <vector>

#include "config.h"
#include "disk_qalsh_index.h"
#include "global.h"
#include "log.h"
#include "point_set.h"
#include "qalsh_index.h"

struct AnnResult {
    Distance distance{0.0};
    unsigned int point_id{0};
};

struct CompareAnnResult {
    bool operator()(const AnnResult& a, const AnnResult& b) const { return a.distance > b.distance; }
};

// ---------------------------------------------
// AnnSearcher Definition
// ---------------------------------------------
class AnnSearcher {
   public:
    virtual ~AnnSearcher() = default;
    [[nodiscard]] virtual bool RequiresIndex() const = 0;
    [[nodiscard]] virtual std::unique_ptr<AnnSearcher> Clone() const = 0;
    virtual void Init(const PointSetMetadata& metadata, float norm_order) = 0;
    virtual AnnResult Search(const Point& query_point) = 0;
};

// ---------------------------------------------
// LinearScanAnnSearcher Definition
// ---------------------------------------------
class LinearScanAnnSearcher : public AnnSearcher {
   public:
    LinearScanAnnSearcher() = default;
    [[nodiscard]] bool RequiresIndex() const override;
    [[nodiscard]] std::unique_ptr<AnnSearcher> Clone() const override;
    void Init(const PointSetMetadata& metadata, float norm_order) override;
    void InitLoaded(std::shared_ptr<const std::vector<Point>> base_points, unsigned int num_dimensions,
                    float norm_order);
    AnnResult Search(const Point& query_point) override;
    AnnResult SearchWithUpperBound(const Point& query_point, Distance initial_upper_bound);
    std::vector<AnnResult> SearchBatchWithUpperBounds(const std::vector<Point>& query_points,
                                                      const std::vector<Distance>& initial_upper_bounds);

   private:
    float norm_order_{0.0};
    unsigned int num_points_{0};
    unsigned int num_dimensions_{0};
    std::shared_ptr<const std::vector<Point>> base_points_;
};

// ---------------------------------------------
// DiskBasedLinearScanAnnSearcher Definition
// ---------------------------------------------
class DiskBasedLinearScanAnnSearcher : public AnnSearcher {
   public:
    explicit DiskBasedLinearScanAnnSearcher(unsigned int block_size = Global::kDefaultDiskBlockSize);
    [[nodiscard]] bool RequiresIndex() const override;
    [[nodiscard]] std::unique_ptr<AnnSearcher> Clone() const override;
    void Init(const PointSetMetadata& metadata, float norm_order) override;
    AnnResult Search(const Point& query_point) override;
    AnnResult SearchWithUpperBound(const Point& query_point, Distance initial_upper_bound);

   private:
    float norm_order_{0.0};
    PointSetMetadata metadata_;
    unsigned int block_size_{Global::kDefaultDiskBlockSize};
    bool initialized_{false};
    std::vector<Coordinate> block_buffer_;
};

struct QalshAnnIndexData {
    float norm_order{0.0};
    unsigned int num_points{0};
    unsigned int num_dimensions{0};
    QalshConfig qalsh_config;
    std::vector<Point> base_points;
    std::vector<Point> dot_vectors;
    std::vector<InMemoryQalshTable> searchers;
};

struct QalshSearchState {
    unsigned int collision_count{0};
    unsigned int collision_stamp{0};
    unsigned int visited_stamp{0};
};

class DiskQalshTableMapping {
   public:
    DiskQalshTableMapping(const DiskQalsh::DiskQalshEntry* entries, size_t num_entries, size_t byte_size);
    ~DiskQalshTableMapping();
    DiskQalshTableMapping(const DiskQalshTableMapping&) = delete;
    DiskQalshTableMapping& operator=(const DiskQalshTableMapping&) = delete;
    DiskQalshTableMapping(DiskQalshTableMapping&&) = delete;
    DiskQalshTableMapping& operator=(DiskQalshTableMapping&&) = delete;

    [[nodiscard]] const DiskQalsh::DiskQalshEntry* entries() const;
    [[nodiscard]] size_t num_entries() const;

   private:
    const DiskQalsh::DiskQalshEntry* entries_{nullptr};
    size_t num_entries_{0};
    size_t byte_size_{0};
};

// ---------------------------------------------
// QalshAnnSearcher Definition
// ---------------------------------------------
class QalshAnnSearcher : public AnnSearcher {
   public:
    explicit QalshAnnSearcher(float approximation_ratio = Global::kDefaultApproximationRatio);
    QalshAnnSearcher(float approximation_ratio, std::mt19937::result_type random_seed);
    [[nodiscard]] bool RequiresIndex() const override;
    [[nodiscard]] std::unique_ptr<AnnSearcher> Clone() const override;
    [[nodiscard]] std::shared_ptr<const std::vector<Point>> TargetPoints() const;
    void Init(const PointSetMetadata& metadata, float norm_order) override;
    AnnResult Search(const Point& query_point) override;

   private:
    float approximation_ratio_{Global::kDefaultApproximationRatio};
    std::mt19937::result_type random_seed_{Global::kDefaultSeed};
    std::shared_ptr<const QalshAnnIndexData> index_data_;
    std::vector<InMemoryQalshTableCursor> search_cursors_;
    std::vector<QalshSearchState> search_states_;
    unsigned int search_epoch_{0};

    void ResetSearchState();
    unsigned int NextSearchEpoch();
};

struct DiskBasedQalshIndexData {
    float norm_order{0.0F};
    PointSetMetadata target_metadata;
    std::filesystem::path index_directory;
    DiskQalsh::IndexMetadata metadata;
    QalshConfig qalsh_config;
    std::vector<Point> dot_vectors;
    std::shared_ptr<const DiskQalshTableMapping> table_mapping;
};

struct DiskQalshSearchOptions {
    unsigned int table_block_size{Global::kDefaultDiskQalshTableBlockSize};
    unsigned int table_cache_blocks{Global::kDefaultDiskQalshTableCacheBlocks};
    bool profile{false};
};

struct DiskQalshProfileSnapshot {
    std::uint64_t queries{0};
    std::uint64_t table_entry_reads{0};
    std::uint64_t table_block_reads{0};
    std::uint64_t table_cache_hits{0};
    std::uint64_t table_cache_misses{0};
    std::uint64_t candidate_point_reads{0};
    std::uint64_t candidates_evaluated{0};
    double total_search_time_ms{0.0};

    void Merge(const DiskQalshProfileSnapshot& other);
};

// ---------------------------------------------
// DiskBasedQalshAnnSearcher Definition
// ---------------------------------------------
class DiskBasedQalshAnnSearcher : public AnnSearcher {
   public:
    explicit DiskBasedQalshAnnSearcher(std::optional<float> expected_approximation_ratio = std::nullopt,
                                       DiskQalshSearchOptions options = {});
    ~DiskBasedQalshAnnSearcher() override;
    DiskBasedQalshAnnSearcher(const DiskBasedQalshAnnSearcher&) = delete;
    DiskBasedQalshAnnSearcher& operator=(const DiskBasedQalshAnnSearcher&) = delete;
    DiskBasedQalshAnnSearcher(DiskBasedQalshAnnSearcher&&) = delete;
    DiskBasedQalshAnnSearcher& operator=(DiskBasedQalshAnnSearcher&&) = delete;

    [[nodiscard]] bool RequiresIndex() const override;
    [[nodiscard]] std::unique_ptr<AnnSearcher> Clone() const override;
    void Init(const PointSetMetadata& metadata, float norm_order) override;
    AnnResult Search(const Point& query_point) override;
    void SearchBatchContiguous(const Coordinate* query_data, unsigned int query_count, unsigned int num_dimensions,
                               std::vector<AnnResult>& results);
    void PreflightIndex(const PointSetMetadata& metadata, float norm_order) const;
    [[nodiscard]] bool ProfilingEnabled() const;
    void ResetProfile();
    void MergeProfile(const DiskQalshProfileSnapshot& snapshot);
    [[nodiscard]] DiskQalshProfileSnapshot ProfileSnapshot() const;
    void PrintProfile() const;

   private:
    struct TableBlockCacheEntry {
        std::uint64_t block_id{0};
        std::uint64_t last_used{0};
    };
    struct TableBlockView {
        const DiskQalsh::DiskQalshEntry* entries{nullptr};
        size_t size{0};
    };

    std::optional<float> expected_approximation_ratio_;
    DiskQalshSearchOptions options_;
    std::shared_ptr<const DiskBasedQalshIndexData> index_data_;
    int dot_vectors_fd_{-1};
    int target_fd_{-1};
    std::vector<InMemoryQalshTableCursor> search_cursors_;
    std::vector<QalshSearchState> search_states_;
    std::vector<Coordinate> candidate_buffer_;
    unsigned int search_epoch_{0};
    size_t table_entries_per_block_{0};
    mutable std::vector<TableBlockCacheEntry> table_cache_;
    mutable std::unordered_map<std::uint64_t, size_t> table_cache_positions_;
    mutable std::uint64_t table_cache_clock_{0};
    mutable std::optional<size_t> last_table_cache_slot_;
    mutable DiskQalshProfileSnapshot profile_;

    void ValidateOptions() const;
    void ResetTableCache();
    void OpenFiles();
    void CloseFiles();
    void ResetSearchState();
    unsigned int NextSearchEpoch();
    [[nodiscard]] std::uint64_t TableLinearIndex(unsigned int table_id, size_t entry_index) const;
    [[nodiscard]] TableBlockView ReadTableBlock(std::uint64_t block_id) const;
    void CountTableEntryRead() const;
    void ValidateTableEntry(const DiskQalsh::DiskQalshEntry& entry) const;
    [[nodiscard]] DiskQalsh::DiskQalshEntry ReadTableEntry(unsigned int table_id, size_t entry_index) const;
    [[nodiscard]] InMemoryQalshTableCursor InitCursor(unsigned int table_id, Projection query_key) const;
    template <std::invocable<unsigned int, Projection> Handler>
    [[nodiscard]] bool IncrementalSearch(unsigned int table_id, InMemoryQalshTableCursor& cursor, Projection bound,
                                         Handler&& handler) const;
    void ReadCandidatePoint(unsigned int point_id);
};

template <std::invocable<unsigned int, Projection> Handler>
bool DiskBasedQalshAnnSearcher::IncrementalSearch(unsigned int table_id, InMemoryQalshTableCursor& cursor,
                                                  Projection bound, Handler&& handler) const {
    bool left_finished = true;
    if (cursor.left.has_value()) {
        left_finished = false;
        size_t entry_index = *cursor.left;
        unsigned int scan_count = 0;
        bool stop_left = false;
        while (scan_count < Global::kScanSize && !stop_left) {
            const std::uint64_t linear_index = TableLinearIndex(table_id, entry_index);
            const std::uint64_t block_id = linear_index / static_cast<std::uint64_t>(table_entries_per_block_);
            size_t block_entry_index =
                static_cast<size_t>(linear_index % static_cast<std::uint64_t>(table_entries_per_block_));
            const TableBlockView block = ReadTableBlock(block_id);

            while (scan_count < Global::kScanSize) {
                if (options_.profile) {
                    ++profile_.table_entry_reads;
                }
                if (block_entry_index >= block.size) {
                    Log::error("Disk QALSH table mmap entry is out of range: block {}, entry {}.", block_id,
                               block_entry_index);
                }
                const DiskQalsh::DiskQalshEntry entry = block.entries[block_entry_index];
                if (entry.point_id >= index_data_->metadata.num_points) {
                    Log::error("Disk QALSH table entry has invalid point id {}.", entry.point_id);
                }
                if (cursor.query_key - entry.projection > bound) {
                    cursor.left = entry_index;
                    left_finished = true;
                    stop_left = true;
                    break;
                }

                std::forward<Handler>(handler)(entry.point_id, entry.projection);
                ++scan_count;

                if (entry_index == 0) {
                    cursor.left = std::nullopt;
                    left_finished = true;
                    stop_left = true;
                    break;
                }

                --entry_index;
                cursor.left = entry_index;
                if (block_entry_index == 0) {
                    break;
                }
                --block_entry_index;
            }
        }
    }

    bool right_finished = true;
    if (cursor.right.has_value()) {
        right_finished = false;
        size_t entry_index = *cursor.right;
        unsigned int scan_count = 0;
        bool stop_right = false;
        while (scan_count < Global::kScanSize && !stop_right) {
            const std::uint64_t linear_index = TableLinearIndex(table_id, entry_index);
            const std::uint64_t block_id = linear_index / static_cast<std::uint64_t>(table_entries_per_block_);
            size_t block_entry_index =
                static_cast<size_t>(linear_index % static_cast<std::uint64_t>(table_entries_per_block_));
            const TableBlockView block = ReadTableBlock(block_id);

            while (scan_count < Global::kScanSize) {
                if (options_.profile) {
                    ++profile_.table_entry_reads;
                }
                if (block_entry_index >= block.size) {
                    Log::error("Disk QALSH table mmap entry is out of range: block {}, entry {}.", block_id,
                               block_entry_index);
                }
                const DiskQalsh::DiskQalshEntry entry = block.entries[block_entry_index];
                if (entry.point_id >= index_data_->metadata.num_points) {
                    Log::error("Disk QALSH table entry has invalid point id {}.", entry.point_id);
                }
                if (entry.projection - cursor.query_key > bound) {
                    cursor.right = entry_index;
                    right_finished = true;
                    stop_right = true;
                    break;
                }

                std::forward<Handler>(handler)(entry.point_id, entry.projection);
                ++scan_count;

                if (entry_index + 1 >= index_data_->metadata.num_points) {
                    cursor.right = std::nullopt;
                    right_finished = true;
                    stop_right = true;
                    break;
                }

                ++entry_index;
                cursor.right = entry_index;
                ++block_entry_index;
                if (block_entry_index >= block.size) {
                    break;
                }
            }
        }
    }

    return left_finished && right_finished;
}
