#ifndef ANN_SEARCHER_H_
#define ANN_SEARCHER_H_

#include <cstdint>
#include <fstream>
#include <memory>
#include <unordered_map>
#include <vector>

#include "b_plus_tree.h"
#include "types.h"

// ---------------------------------------------
// AnnSearcher Definition
// ---------------------------------------------
class AnnSearcher {
   public:
    virtual ~AnnSearcher() = default;
    virtual void Init(const PointSetMetadata& base_metadata) = 0;
    virtual AnnResult Search(const Point& query_point) = 0;
};

// ---------------------------------------------
// InMemoryLinearScanAnnSearcher Definition
// ---------------------------------------------
class InMemoryLinearScanAnnSearcher : public AnnSearcher {
   public:
    InMemoryLinearScanAnnSearcher() = default;
    void Init(const PointSetMetadata& base_metadata) override;
    AnnResult Search(const Point& query_point) override;

   private:
    std::vector<Point> base_points_;
};

// ---------------------------------------------
// DiskLinearScanAnnSearcher Definition
// ---------------------------------------------
class DiskLinearScanAnnSearcher : public AnnSearcher {
   public:
    DiskLinearScanAnnSearcher() = default;
    void Init(const PointSetMetadata& base_metadata) override;
    AnnResult Search(const Point& query_point) override;

   private:
    std::ifstream base_file_;
    unsigned int num_points_{0};
    unsigned int num_dimensions_{0};
};

// ---------------------------------------------
// InMemoryQalshAnnSearcher Definition
// ---------------------------------------------
class InMemoryQalshAnnSearcher : public AnnSearcher {
   public:
    InMemoryQalshAnnSearcher(double approximation_ratio);
    void Init(const PointSetMetadata& base_metadata) override;
    AnnResult Search(const Point& query_point) override;

   private:
    std::vector<Point> base_points_;
    QalshConfig qalsh_config_;
    std::vector<Point> dot_vectors_;
    std::vector<std::vector<DotProductPointIdPair>> hash_tables_;
};

// ---------------------------------------------
// DiskQalshAnnSearcher Definition
// ---------------------------------------------
class DiskQalshAnnSearcher : public AnnSearcher {
   public:
    struct SearchRecord {
        std::shared_ptr<LeafNode> leaf_node;
        unsigned int index{0};
    };

    DiskQalshAnnSearcher() = default;
    void Init(const PointSetMetadata& base_metadata) override;
    AnnResult Search(const Point& query_point) override;

   private:
    std::shared_ptr<LeafNode> LocateLeafMayContainKey(std::ifstream& ifs, unsigned int table_idx, double key);
    std::shared_ptr<LeafNode> LocateLeafByPageNum(std::ifstream& ifs, unsigned int table_idx, unsigned int page_num);
    void ReadPage(std::ifstream& ifs, unsigned int page_num);

    std::ifstream base_file_;
    unsigned int num_points_{0};
    unsigned int num_dimensions_{0};
    QalshConfig qalsh_config_;
    std::vector<Point> dot_vectors_;
    std::vector<std::ifstream> hash_tables_;
    std::vector<char> buffer_;
    std::unordered_map<uint64_t, std::shared_ptr<LeafNode>> leaf_nodes_cache_;
};

#endif