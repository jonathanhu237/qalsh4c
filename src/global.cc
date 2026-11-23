#include "global.h"

bool Global::kUseFixedSeed = false;
bool Global::kUseScalarExactScan = false;
float Global::kInitialRadius = 1.0F;
bool Global::kQalshStopAtRoundBoundary = true;
float Global::kQalshMidRoundRatio = 1.0F;
float Global::kQalshBeta = 1.0F;
unsigned int Global::kQalshTables = 0;
unsigned int Global::kQalshThreshold = 0;
float Global::kQalshBucketWidth = 0.0F;
unsigned int Global::kQalshScanQuantum = 2 * Global::kScanSize;
