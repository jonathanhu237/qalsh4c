#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <vector>

#include <qalsh/qalsh.h>

#include "config.h"
#include "disk_qalsh_index.h"
#include "global.h"
#include "point_set.h"

struct AnnResult {
    Distance distance{0.0F};
    unsigned int point_id{0};
};

struct CompareAnnResult {
    bool operator()(const AnnResult& a, const AnnResult& b) const { return a.distance > b.distance; }
};

class AnnSearcher {
   public:
    virtual ~AnnSearcher() = default;
    [[nodiscard]] virtual bool RequiresIndex() const = 0;
    [[nodiscard]] virtual std::unique_ptr<AnnSearcher> Clone() const = 0;
    virtual void Init(const PointSetMetadata& metadata, float norm_order) = 0;
    virtual AnnResult Search(const Point& query_point) = 0;
};

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
    // True when exact batch searches go through ExactScan: L1 and L2, unless --scalar-exact-scan is set.
    [[nodiscard]] bool UsesBatchedScan() const;
    // Same results as SearchBatchWithUpperBounds for all queries at once, through ExactScan. It runs in parallel over
    // target blocks, so call it outside parallel regions.
    [[nodiscard]] std::vector<AnnResult> SearchBatched(const std::vector<Point>& query_points,
                                                       const std::vector<Distance>& initial_upper_bounds) const;

   private:
    float norm_order_{0.0F};
    unsigned int num_points_{0};
    unsigned int num_dimensions_{0};
    std::shared_ptr<const std::vector<Point>> base_points_;
};

class DiskBasedLinearScanAnnSearcher : public AnnSearcher {
   public:
    explicit DiskBasedLinearScanAnnSearcher(unsigned int block_size = Global::kDefaultDiskBlockSize);
    [[nodiscard]] bool RequiresIndex() const override;
    [[nodiscard]] std::unique_ptr<AnnSearcher> Clone() const override;
    void Init(const PointSetMetadata& metadata, float norm_order) override;
    AnnResult Search(const Point& query_point) override;
    AnnResult SearchWithUpperBound(const Point& query_point, Distance initial_upper_bound);

   private:
    float norm_order_{0.0F};
    PointSetMetadata metadata_;
    unsigned int block_size_{Global::kDefaultDiskBlockSize};
    bool initialized_{false};
    std::vector<Coordinate> block_buffer_;
};

struct QalshAnnIndexData {
    float norm_order{0.0F};
    unsigned int num_points{0};
    unsigned int num_dimensions{0};
    QalshConfig qalsh_config;
    std::shared_ptr<const std::vector<Point>> base_points;
    std::shared_ptr<const qalsh::ProjectionIndex> index;
};

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
    std::unique_ptr<qalsh::SearchEngine> engine_;
};

struct DiskBasedQalshIndexData {
    float norm_order{0.0F};
    PointSetMetadata target_metadata;
    std::filesystem::path index_path;
    DiskQalsh::IndexMetadata metadata;
    QalshConfig qalsh_config;
    std::shared_ptr<const qalsh::ProjectionIndex> index;
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

class DiskBasedQalshAnnSearcher : public AnnSearcher {
   public:
    explicit DiskBasedQalshAnnSearcher(std::optional<float> expected_approximation_ratio = std::nullopt,
                                       DiskQalshSearchOptions options = {});
    ~DiskBasedQalshAnnSearcher() override;
    [[nodiscard]] bool RequiresIndex() const override;
    [[nodiscard]] std::unique_ptr<AnnSearcher> Clone() const override;
    void Init(const PointSetMetadata& metadata, float norm_order) override;
    AnnResult Search(const Point& query_point) override;
    void PreflightIndex(const PointSetMetadata& metadata, float norm_order) const;
    [[nodiscard]] bool ProfilingEnabled() const;
    void ResetProfile();
    void MergeProfile(const DiskQalshProfileSnapshot& snapshot);
    [[nodiscard]] DiskQalshProfileSnapshot ProfileSnapshot() const;
    void PrintProfile() const;

   private:
    void InitializeFromIndexData(std::shared_ptr<const DiskBasedQalshIndexData> index_data);

    std::optional<float> expected_approximation_ratio_;
    DiskQalshSearchOptions options_;
    std::shared_ptr<const DiskBasedQalshIndexData> index_data_;
    mutable PointSet target_points_;
    int target_fd_{-1};
    std::unique_ptr<qalsh::SearchEngine> engine_;
    mutable DiskQalshProfileSnapshot profile_;
    mutable std::mutex profile_mutex_;
};
