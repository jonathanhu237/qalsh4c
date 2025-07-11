#include "index_command.h"

#include "indexer.h"

IndexCommand::IndexCommand(Indexer* indexer, std::filesystem::path dataset_directory)
    : indexer_(indexer), dataset_directory_(std::move(dataset_directory)) {}

auto IndexCommand::Execute() -> void {
    if (indexer_ == nullptr) {
        throw std::runtime_error("Indexer is not set.");
    }

    indexer_->BuildIndex(dataset_directory_);
};