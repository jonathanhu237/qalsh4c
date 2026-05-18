#pragma once

#include <memory>
#include <optional>
#include <vector>

#include "config.h"
#include "global.h"
#include "point_set.h"

struct ProjectionPointIdPair {
    double dot_product{0.0};
    unsigned int point_id{0};
};

struct ProjectionPointIdPairLess {
    bool operator()(const ProjectionPointIdPair& lhs, const ProjectionPointIdPair& rhs) const {
        if (lhs.dot_product != rhs.dot_product) {
            return lhs.dot_product < rhs.dot_product;
        }
        return lhs.point_id < rhs.point_id;
    }
};

struct InMemoryQalshIndex {
    QalshConfig config;
    PointSetMetadata metadata;
    std::vector<Point> points;
    std::vector<Point> dot_vectors;
    std::vector<std::vector<ProjectionPointIdPair>> tables;
};

struct AnnResult {
    double distance{0.0};
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
    virtual void Init(const PointSetMetadata& metadata, double norm_order) = 0;
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
    void Init(const PointSetMetadata& metadata, double norm_order) override;
    AnnResult Search(const Point& query_point) override;

   private:
    PointSet base_;
    double norm_order_{0.0};
};

// ---------------------------------------------
// QalshAnnSearcher Definition
// ---------------------------------------------
class QalshAnnSearcher : public AnnSearcher {
   public:
    explicit QalshAnnSearcher(bool candidate_limit = false,
                              double approximation_ratio = Global::kDefaultApproximationRatio);
    QalshAnnSearcher(std::shared_ptr<const InMemoryQalshIndex> index, bool candidate_limit = false,
                     double approximation_ratio = Global::kDefaultApproximationRatio);
    [[nodiscard]] bool RequiresIndex() const override;
    [[nodiscard]] std::unique_ptr<AnnSearcher> Clone() const override;
    void Init(const PointSetMetadata& metadata, double norm_order) override;
    AnnResult Search(const Point& query_point) override;
    [[nodiscard]] std::optional<QalshConfig> GetConfig() const;

   private:
    [[nodiscard]] bool HasReusableIndex(const PointSetMetadata& metadata, double norm_order) const;
    std::shared_ptr<const InMemoryQalshIndex> BuildIndex(const PointSetMetadata& metadata, double norm_order) const;

    bool candidate_limit_{false};
    double approximation_ratio_{Global::kDefaultApproximationRatio};
    double norm_order_{0.0};
    std::shared_ptr<const InMemoryQalshIndex> index_;
};
