#!/usr/bin/env bash
# Add-on to mlsweep.sh: GPT's direct (w, m, l) configurations and the beta = 1 defaults on the id-array build.
set -u
R=/home/jonathanhu237/qalsh4c-results/dasfaa-20261120
S=$R/mlsweep
BIN=~/code/qalsh4c-tmp/build/release-soa/qalsh4c
R0=1.401298464324817e-45
while ! grep -q DONE $S/mlsweep.txt 2>/dev/null; do sleep 20; done
OUT=$S/mlsweep2.txt
: > $OUT
declare -A DIR=( [mnist-a20k]=~/datasets/qalsh4c/mnist-a20k [trevi-a20k]=~/datasets/qalsh4c/trevi-a20k
                 [enron-a20k]=~/datasets/qalsh4c/enron-a20k [gist]=$R/datasets/gist )
declare -A EX=( [mnist-a20k]=$R/screen/exact/mnist-a20k [trevi-a20k]=$R/screen/exact/trevi-a20k
                [enron-a20k]=$R/screen/exact/enron-a20k [gist]=$R/exact/gist )
# label:w:m:l:T (w = 0 keeps QALSH's width)
L1="b1soa:0:36:18:1137 gpt:2.25:36:16:1200"
L2="b1soa:0:17:12:1121 gpt:2.85:12:9:1200"
for ds in mnist-a20k enron-a20k trevi-a20k gist; do
  dir=${DIR[$ds]}
  for p in 1 2; do
    if [ $p = 1 ]; then list=$L1; else list=$L2; fi
    ex=${EX[$ds]}_l$p.bin
    for cfg in $list; do
      IFS=: read -r label w m l T <<< "$cfg"
      W=$S/w2; rm -rf $W; mkdir -p $W; for f in A.bin B.bin metadata.json; do ln -s $dir/$f $W/$f; done
      log=$S/logs/$ds-$p-$label.log
      $BIN --initial-radius $R0 --qalsh-bucket-width $w --qalsh-tables $m --qalsh-threshold $l estimate -p $p -d $W --direction ab sampling -n $T --use-cache qalsh -c 2 > $log 2>&1
      ms=$(grep "Time Consumed" $log | sed "s/.*Time Consumed: //; s/ ms//")
      st=$(grep "QALSH stage times" $log | sed "s/.*QALSH stage times: //")
      wt=$(grep "Stage time: weights_ms" $log | sed "s/.*weights_ms=//")
      stats=$(grep "QALSH stats" $log | sed "s/.*QALSH stats: //")
      ratio=$(python3 -c "import numpy as np; w=np.fromfile('$W/l${p}_qalsh_weights_A.bin',dtype=np.float32).astype(np.float64); e=np.fromfile('$ex',dtype=np.float64); print('%.4f bad=%d' % (w.sum()/e.sum(), int((w < e*(1-1e-5)-1e-6).sum())))")
      echo "$ds l$p $label w=$w m=$m l=$l T=$T total_ms=$ms weights_ms=$wt $st D/CH=$ratio stats=[$stats]" >> $OUT
    done
  done
done
rm -rf $S/w2
echo DONE >> $OUT
