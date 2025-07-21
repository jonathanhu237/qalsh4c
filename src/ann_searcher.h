#ifndef ANN_SEARCHER_H_
#define ANN_SEARCHER_H_

#include "point_set.h"
#include "types.h"

class AnnSearcher {
   public:
    virtual ~AnnSearcher() = default;
    virtual AnnResult Search(const PointVariant& query_point) = 0;
};

class LinearScanAnnSearcher : public AnnSearcher {
   public:
    LinearScanAnnSearcher(PointSetReader* base_reader);

    AnnResult Search(const PointVariant& query_point) override;

   private:
    PointSetReader* base_reader_;
};

class AnnSearcherFactory {
   public:
    static std::unique_ptr<AnnSearcher> Create(PointSetReader* base_reader, const std::string& searcher_type) {
        if (searcher_type == "linear_scan") {
            return std::make_unique<LinearScanAnnSearcher>(base_reader);
        }
        spdlog::error("Unknown ANN searcher type: {}", searcher_type);
        return nullptr;
    }
};

#endif