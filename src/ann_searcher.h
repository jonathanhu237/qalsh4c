#ifndef ANN_SEARCHER_H_
#define ANN_SEARCHER_H_

#include <fstream>
#include <memory>
#include <vector>

#include "b_plus_tree.h"
#include "types.h"

// ---------------------------------------------
// AnnSearcher Definition
// ---------------------------------------------
class AnnSearcher {
   public:
    virtual ~AnnSearcher() = default;
    virtual void Init(const PointSetMetadata& base_metadata, double norm_order) = 0;
    virtual AnnResult Search(const Point& query_point) = 0;
};

// ---------------------------------------------
// LinearScanAnnSearcher Definition
// ---------------------------------------------
class LinearScanAnnSearcher : public AnnSearcher {
   public:
    LinearScanAnnSearcher() = default;
    void Init(const PointSetMetadata& base_metadata, double norm_order) override;
    AnnResult Search(const Point& query_point) override;

   private:
    std::ifstream base_file_;
    unsigned int num_points_{0};
    unsigned int num_dimensions_{0};
    double norm_order_{0.0};
};

// ---------------------------------------------
// QalshAnnSearcher Definition
// ---------------------------------------------
class QalshAnnSearcher : public AnnSearcher {
   public:
    struct SearchRecord {
        std::shared_ptr<LeafNode> leaf_node;
        unsigned int index{0};
    };

    QalshAnnSearcher() = default;
    void Init(const PointSetMetadata& base_metadata, double norm_order) override;
    AnnResult Search(const Point& query_point) override;

   private:
    std::shared_ptr<LeafNode> LocateLeafMayContainKey(std::ifstream& ifs, double key);
    std::shared_ptr<LeafNode> LocateLeafByPageNum(std::ifstream& ifs, unsigned int page_num);
    void ReadPage(std::ifstream& ifs, unsigned int page_num);

    std::ifstream base_file_;
    unsigned int num_points_{0};
    unsigned int num_dimensions_{0};
    double norm_order_{0.0};
    QalshConfig qalsh_config_;
    std::vector<Point> dot_vectors_;
    std::vector<std::ifstream> hash_tables_;
    std::vector<char> buffer_;
};

#endif