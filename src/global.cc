#include "global.h"

bool Global::kUseFixedSeed = false;
long long Global::kSeed = -1;
bool Global::kUseScalarExactScan = false;
float Global::kInitialRadius = 0x1p-149F;
bool Global::kQalshStopAtRoundBoundary = true;
float Global::kQalshMidRoundRatio = 1.0F;
unsigned int Global::kQalshSlabsPerRound = 2;
std::string Global::kQalshScanOrder = "distance";
bool Global::kQalshCheckAfterRadiusAdvance = true;
std::string Global::kQalshStopRule = "continuous";
float Global::kQalshContinuousRatio = 0.0F;
