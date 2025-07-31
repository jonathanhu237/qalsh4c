#ifndef QALSH_HASH_TABLE_H
#define QALSH_HASH_TABLE_H

#include <optional>
#include <vector>

struct DotProductPointIdPair {
    double dot_product{0.0};
    unsigned int point_id{0};
};

// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions)
class QalshHashTable {
   public:
    virtual ~QalshHashTable() = default;

    virtual void Init(double query_dot_product) = 0;
    virtual std::optional<unsigned int> LeftFindNext(double bound) = 0;
    virtual std::optional<unsigned int> RightFindNext(double bound) = 0;
};

class InMemoryQalshHashTable : public QalshHashTable {
   public:
    InMemoryQalshHashTable(const std::vector<DotProductPointIdPair>& data);

    void Init(double key) override;
    std::optional<unsigned int> LeftFindNext(double bound) override;
    std::optional<unsigned int> RightFindNext(double bound) override;

   private:
    double key_{0.0};
    std::vector<DotProductPointIdPair> data_;

    std::optional<unsigned int> left_;
    std::optional<unsigned int> right_;
};

#endif