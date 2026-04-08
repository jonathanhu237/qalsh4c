#include "log.h"

int main() {
    Log::info("smoke {}", 1);
    Log::warn("warn {}", 2);
    return 0;
}
