#ifndef INDEXER_H_
#define INDEXER_H_

class Indexer {
   public:
    virtual ~Indexer() = default;
    virtual auto BuildIndex() -> void = 0;
};

#endif