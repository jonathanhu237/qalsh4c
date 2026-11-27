#!/usr/bin/env bash
# VLDB v3: every timed and randomized experiment with the default stopping ratio gamma = c^1.5 of the binary of
# 2026-11-27 (crude search otherwise unchanged from v2). The exact scan and FAISS are timed again in the cost phase on
# a quiet machine; the scaling, thread and memory phases reuse the single exact runs of v2 (same exact-scan code) and
# rerun only QALSH4C. The phases run one after another, in the order that keeps the paper complete at every point.
# Usage: nohup bash v3.sh > /dev/null 2>&1 &
set -u
export VLDB_V=/home/jonathanhu237/qalsh4c-results/vldb-v3-20261127
export VLDB_BIN=$VLDB_V/bin/qalsh4c
export VLDB_KAPPA=$VLDB_V/kappa.txt
export VLDB_ROUNDEND_EXTRA="--qalsh-mid-round-ratio 0"
V2=/home/jonathanhu237/qalsh4c-results/vldb-v2-20261125
mkdir -p $VLDB_V
[ -e $VLDB_V/data ] || ln -s /home/jonathanhu237/qalsh4c-results/vldb-20261124/data $VLDB_V/data
D=$(dirname $(readlink -f $0))
source $D/common.sh
ALL="p53 trevi notredame enron flickr2coco coco2flickr dbpedia1m"
{
  echo "v3 start $(date +%H:%M) binary md5 $(md5sum < $BIN | cut -c1-12)"
  echo "load at start: $(uptime | sed 's/.*load average/load average/')"
  ps -eo pcpu,user,comm --sort=-pcpu | head -6 | sed 's/^/  /'
} >> $V/progress.log

# Certified kappa and T of every rule from this binary, in the format of the DASFAA kappa.txt.
record() {  # label c eps-list [options of the rule]
  local label=$1 c=$2 eps=$3 p out k ml; shift 3
  for p in 1 2; do
    out=$($BIN "$@" kappa -p $p -c $c -e $eps) || { echo "kappa failed $label $p $c" >> $V/progress.log; continue; }
    echo "$out" >> $V/logs/kappa.log
    k=$(echo "$out" | grep -o "kappa_up=[0-9.]*" | cut -d= -f2)
    ml=$(echo "$out" | grep -o " m=[0-9]* l=[0-9]*")
    echo "$out" | grep "^T " | while read -r _ e _ t; do echo "$label norm=$p c=$c $e $t kappa_up=$k$ml"; done >> $KAPPA
  done
}
if [ ! -s $KAPPA ]; then
  EPS="0.3 0.2 0.1 0.05 0.02 0.01"
  record main 2 "$EPS"
  record cont2 2 0.1 --qalsh-continuous-ratio 2
  record cont4 2 0.1 --qalsh-continuous-ratio 4
  record within 2 0.1 $WITHIN
  record roundend 2 0.1 $ROUNDEND
  record main 1.5 0.1
  record main 3 0.1
fi
echo "kappa: $(grep 'epsilon=0.1 ' $KAPPA | awk '{print $1, $2, $3, $5}' | tr '\n' ';')" >> $V/progress.log

# The exact rows of the scaling, thread and memory phases come from v2.
for f in dscaling.txt bscaling.txt threads.txt; do
  [ -s $V/$f ] || { head -1 $V2/$f; grep " exact " $V2/$f; } > $V/$f
done
[ -s $V/memory.txt ] || { head -1 $V2/memory.txt; grep " exact " $V2/memory.txt; } > $V/memory.txt

bash $D/realcost.sh $ALL
bash $D/randomness.sh $ALL
bash $D/ann.sh $ALL
bash $D/equaltime.sh $ALL
bash $D/ablation.sh $ALL
bash $D/dscaling.sh
bash $D/bscaling.sh
bash $D/threads.sh trevi flickr2coco
bash $D/csweep.sh $ALL
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
echo "v3 done $(date +%H:%M)" >> $V/progress.log
