#ifndef INDEX_COMMAND_H_
#define INDEX_COMMAND_H_

#include "command.h"
#include "indexer.h"

class IndexCommand : public Command {
   public:
    explicit IndexCommand(Indexer* indexer);
    auto Execute() -> void override;

   private:
    Indexer* indexer_;
};

#endif