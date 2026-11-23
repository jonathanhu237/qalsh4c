#!/usr/bin/env bash
# Dataset screening (2026-11-21): for each candidate instance and norm, exact NN distances (exact_nn), one timed
# exact linear scan (batched kernel) and one cold QALSH4C run with beta = 1 (throwaway prof build).
set -u
R=/home/jonathanhu237/qalsh4c-results/dasfaa-20261120
S=$R/screen
BIN_PROF=~/code/qalsh4c-prof/build/release/qalsh4c
BIN_TMP=~/code/qalsh4c-tmp/build/release/qalsh4c
EXACT=$R/bin/exact_nn
R0=1.401298464324817e-45
DS_ROOT=~/datasets/qalsh4c
mkdir -p $S/exact $S/logs
OUT=$S/screen.txt
declare -A T=( [1]=1137 [2]=1121 )
for ds in "$@"; do
  dir=$DS_ROOT/$ds
  for p in 1 2; do
    ex=$S/exact/${ds}_l$p.bin
    [ -s $ex ] || $EXACT $dir $p $ex > $S/logs/exact-$ds-$p.log 2>&1
    lin=$($BIN_TMP estimate -p $p -d $dir --direction ab ann linear_scan 2>&1 | grep "Time Consumed" | sed "s/.*Time Consumed: //; s/ ms//")
    W=$S/w-$ds-$p; rm -rf $W; mkdir -p $W; for f in A.bin B.bin metadata.json; do ln -s $dir/$f $W/$f; done
    log=$S/logs/beta1-$ds-$p.log
    QALSH4C_BETA=1 $BIN_PROF --initial-radius $R0 estimate -p $p -d $W --direction ab sampling -n ${T[$p]} --use-cache qalsh -c 2 > $log 2>&1
    ms=$(grep "Time Consumed" $log | sed "s/.*Time Consumed: //; s/ ms//")
    stats=$(grep "QALSH stats" $log | sed "s/.*QALSH stats: //")
    ratio=$(python3 -c "import numpy as np; w=np.fromfile('$W/l${p}_qalsh_weights_A.bin',dtype=np.float32).astype(np.float64); e=np.fromfile('$ex',dtype=np.float64); print('%.4f bad=%d' % (w.sum()/e.sum(), int((w < e*(1-1e-5)-1e-6).sum())))")
    echo "$ds l$p exact_ms=$lin beta1_ms=$ms speedup=$(python3 -c "print('%.2f' % ($lin/$ms))") D/CH=$ratio stats=[$stats]" >> $OUT
    rm -rf $W
  done
done
echo DONE >> $OUT
