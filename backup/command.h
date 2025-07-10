#ifndef COMMAND_H_
#define COMMAND_H_

namespace qalsh_chamfer {

// Command Interface
class Command {
   public:
    virtual ~Command() = default;
    Command(const Command&) = delete;
    auto operator=(const Command&) -> Command& = delete;
    Command(Command&&) = delete;
    auto operator=(Command&&) -> Command& = delete;

    virtual auto Execute() -> void = 0;

   protected:
    Command() = default;
};

}  // namespace qalsh_chamfer

#endif