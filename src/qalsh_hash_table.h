#ifndef QALSH_HASH_TABLE_H
#define QALSH_HASH_TABLE_H

#include <cstddef>
#include <optional>
#include <vector>

struct KeyValuePair {
    double dot_product{0.0};
    unsigned int point_id{0};
};

// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions)
class QalshHashTable {
   public:
    virtual ~QalshHashTable() = default;

    virtual void Init(double key) = 0;
    virtual std::optional<unsigned> FindNext(double bound) = 0;
};

class InmemoryQalshHashTable : public QalshHashTable {
   public:
    struct SearchRecord {
        double distance;
        size_t index;
    };

    InmemoryQalshHashTable(const std::vector<KeyValuePair>& data);

    void Init(double key) override;
    std::optional<unsigned int> FindNext(double bound) override;

   private:
    std::optional<unsigned int> FindNextLeft(double bound);
    std::optional<unsigned int> FindNextRight(double bound);

    double key_{0.0};
    std::vector<KeyValuePair> data_;

    std::optional<SearchRecord> left_;
    std::optional<SearchRecord> right_;
};

#endif