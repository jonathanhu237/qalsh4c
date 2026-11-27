#!/usr/bin/env bash
# VLDB v2: every timed and randomized experiment again, with the binary of 2026-11-25 (the sampled exact distances use
# the same kernel as the exact baseline, D = 0 returns 0, quadtree and precomputed crude distances prune the exact
# stage, and the bfloat16 margin is 2^-7; the crude search itself is unchanged). The phases run one
# after another, in the order that keeps the paper complete at every point: the main cost table, the 50-run accuracy
# test, the ANN comparison, equal time, the stopping-rule ablation, scaling, threads, the eps sweep and memory.
# Usage: nohup bash v2.sh > /dev/null 2>&1 &   (VLDB_ROUNDEND_EXTRA="--qalsh-mid-round-ratio 0" for the pure round-end
# rule in the ablation)
set -u
export VLDB_V=/home/jonathanhu237/qalsh4c-results/vldb-v2-20261125
export VLDB_BIN=$VLDB_V/bin/qalsh4c
# The nested subsets of the scaling phases are data, not results: share those of the first run.
mkdir -p $VLDB_V
[ -e $VLDB_V/data ] || ln -s /home/jonathanhu237/qalsh4c-results/vldb-20261124/data $VLDB_V/data
D=$(dirname $(readlink -f $0))
source $D/common.sh
ALL="p53 trevi notredame enron flickr2coco coco2flickr dbpedia1m"
echo "v2 start $(date +%H:%M) binary md5 $(md5sum < $BIN | cut -c1-12)" >> $V/progress.log
# The certified kappa and T of this binary, to compare with kappa.txt (the certification code did not change).
for p in 1 2; do $BIN --initial-radius $R0 kappa -p $p -c 2 -e 0.3 0.1 0.01 >> $V/logs/kappa.log 2>&1; done
bash $D/realcost.sh $ALL
bash $D/randomness.sh $ALL
bash $D/ann.sh notredame flickr2coco coco2flickr dbpedia1m
bash $D/equaltime.sh $ALL
bash $D/ablation.sh $ALL
bash $D/dscaling.sh
bash $D/bscaling.sh
bash $D/threads.sh trevi flickr2coco
mkdir -p $V/epsweep
for ds in $ALL; do
  for p in 1 2; do
    grep -q "\"dataset\": \"$ds\", \"norm\": $p," $V/epsweep/epsweep.jsonl 2>/dev/null && continue
    cat ${DIR[$ds]}/A.bin ${DIR[$ds]}/B.bin > /dev/null
    python3 $D/epsweep.py --name $ds --dataset ${DIR[$ds]} --exact $(exact_file $ds $p) --norm $p --binary $BIN \
      --weights-dir $V/randomness/weights/${ds}_l$p --out $V/epsweep/epsweep.jsonl --upper-bounds \
      >> $V/logs/epsweep.log 2>&1 || echo "epsweep failed $ds $p" >> $V/progress.log
  done
done
echo "epsweep done $(date +%H:%M)" >> $V/progress.log
bash $D/memory.sh $ALL
echo "v2 done $(date +%H:%M)" >> $V/progress.log
