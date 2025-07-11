#ifndef INDEXER_H_
#define INDEXER_H_

#include <filesystem>
class Indexer {
   public:
    virtual ~Indexer() = default;
    virtual auto BuildIndex(const std::filesystem::path& dataset_directory) -> void = 0;
};

#endif