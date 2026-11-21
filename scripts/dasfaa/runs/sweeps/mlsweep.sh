#!/usr/bin/env bash
# (m, l) sweep (2026-11-21): QALSH4C with explicit table count m and threshold l, T from kappa(m, l) at eps = delta = 0.1.
# One cold run per cell. Waits for the dataset screening to finish so the two do not share the CPU.
set -u
R=/home/jonathanhu237/qalsh4c-results/dasfaa-20261120
S=$R/mlsweep
BIN=~/code/qalsh4c-tmp/build/release-opt/qalsh4c
R0=1.401298464324817e-45
mkdir -p $S/logs
while ! grep -q DONE $R/screen/screen.txt 2>/dev/null; do sleep 30; done
OUT=$S/mlsweep.txt
: > $OUT
declare -A DIR=( [mnist-a20k]=~/datasets/qalsh4c/mnist-a20k [trevi-a20k]=~/datasets/qalsh4c/trevi-a20k
                 [enron-a20k]=~/datasets/qalsh4c/enron-a20k [gist]=$R/datasets/gist )
declare -A EX=( [mnist-a20k]=$R/screen/exact/mnist-a20k [trevi-a20k]=$R/screen/exact/trevi-a20k
                [enron-a20k]=$R/screen/exact/enron-a20k [gist]=$R/exact/gist )
L1="36:18:1137 28:14:1176 24:12:1200 20:10:1229 16:8:1266 12:6:1315 10:5:1347 8:4:1388"
L2="17:12:1121 14:9:1047 12:8:1086 10:6:1038 8:5:1071 6:4:1140 5:3:1080 12:6:1004 6:3:1021 4:2:1037"
for ds in mnist-a20k enron-a20k trevi-a20k gist; do
  dir=${DIR[$ds]}
  for p in 1 2; do
    if [ $p = 1 ]; then list=$L1; else list=$L2; fi
    ex=${EX[$ds]}_l$p.bin
    for cfg in $list; do
      m=${cfg%%:*}; rest=${cfg#*:}; l=${rest%%:*}; T=${rest#*:}
      W=$S/w; rm -rf $W; mkdir -p $W; for f in A.bin B.bin metadata.json; do ln -s $dir/$f $W/$f; done
      log=$S/logs/$ds-$p-$m-$l.log
      $BIN --initial-radius $R0 --qalsh-tables $m --qalsh-threshold $l estimate -p $p -d $W --direction ab sampling -n $T --use-cache qalsh -c 2 > $log 2>&1
      ms=$(grep "Time Consumed" $log | sed "s/.*Time Consumed: //; s/ ms//")
      st=$(grep "QALSH stage times" $log | sed "s/.*QALSH stage times: //")
      wt=$(grep "Stage time: weights_ms" $log | sed "s/.*weights_ms=//")
      stats=$(grep "QALSH stats" $log | sed "s/.*QALSH stats: //")
      ratio=$(python3 -c "import numpy as np; w=np.fromfile('$W/l${p}_qalsh_weights_A.bin',dtype=np.float32).astype(np.float64); e=np.fromfile('$ex',dtype=np.float64); print('%.4f bad=%d' % (w.sum()/e.sum(), int((w < e*(1-1e-5)-1e-6).sum())))")
      echo "$ds l$p m=$m l=$l T=$T total_ms=$ms weights_ms=$wt $st D/CH=$ratio stats=[$stats]" >> $OUT
    done
  done
done
rm -rf $S/w
echo DONE >> $OUT
