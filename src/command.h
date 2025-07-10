#ifndef COMMAND_H_
#define COMMAND_H_

namespace qalsh_chamfer {

class Command {
   public:
    virtual ~Command() = default;
    virtual auto Execute() -> void = 0;
};

};  // namespace qalsh_chamfer

#endif