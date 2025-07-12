#include "index_command.h"

#include <stdexcept>

#include "indexer.h"

IndexCommand::IndexCommand(Indexer* indexer) : indexer_(indexer) {}

auto IndexCommand::Execute() -> void {
    if (indexer_ == nullptr) {
        throw std::runtime_error("Indexer is not set.");
    }

    indexer_->BuildIndex();
};