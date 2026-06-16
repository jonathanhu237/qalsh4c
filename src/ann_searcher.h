#pragma once

#include <memory>
#include <random>
#include <vector>

#include "config.h"
#include "global.h"
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
