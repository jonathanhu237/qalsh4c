#ifndef COMMAND_H_
#define COMMAND_H_

class Command {
   public:
    virtual ~Command() = default;
    virtual auto Execute() -> void = 0;
};

#endif