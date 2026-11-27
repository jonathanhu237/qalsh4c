#!/usr/bin/env bash
# VLDB phase: ablation of the stopping rule (Sect. 4.2) on every dataset and norm, all without criterion T2 and with
# the same m, l and w. The continuous front with gamma = c (cont2), the default gamma = c^1.5 (main) and gamma = c^2
# (cont4), QALSH's own test on every new candidate (within, gamma = c^2) and the round rule with end-of-round tests
# (roundend, gamma = c). Each rule uses the T it certifies at eps = delta = 0.1. Run rep of every rule uses the seed
# 1000 + rep, so the rules are compared on the same projections. Cold runs at 24 threads: ABL_RUNS (20) per cell, and
# 3 for the round-end rule, whose runs are capped at 600 s.
set -u
source $(dirname $(readlink -f $0))/common.sh
OUT=$V/ablation.txt
[ -s $OUT ] || echo "# dataset norm rule rep T total_ms index_ms queries_ms weights_ms D/C stats" > $OUT
W=$V/work-ablation
CAP=600
RUNS=${ABL_RUNS:-20}
for ds in ${@:-$LINEUP}; do
  cat ${DIR[$ds]}/A.bin ${DIR[$ds]}/B.bin > /dev/null
  for p in 1 2; do
    ex=$(exact_file $ds $p)
    for rule in cont2 main cont4 within roundend; do
      case $rule in
        cont2) opt="--qalsh-continuous-ratio 2" ;;
        main) opt="" ;;
        cont4) opt="--qalsh-continuous-ratio 4" ;;
        within) opt="$WITHIN" ;;
        roundend) opt="$ROUNDEND" ;;
      esac
      T=$(tval $p $rule)
      reps=$RUNS; [ $rule = roundend ] && reps=3
      for rep in $(seq 1 $reps); do
        grep -q "^$ds $p $rule $rep " $OUT && continue
        fresh $ds $W
        log=$V/logs/ablation-$ds-$p-$rule-$rep.log
        # A run of the round rule can take many minutes on DBpedia; it is stopped at $CAP s and reported as such.
        if timeout $CAP $BIN -T 24 --seed $((1000 + rep)) $opt estimate -p $p -d $W --direction ab sampling -n $T \
             --use-cache qalsh -c 2 > $log 2>&1; then
          st=$(grep "QALSH stats" $log | sed "s/.*QALSH stats: //")
          echo "$ds $p $rule $rep $T $(tc $log) $(field index_ms $log) $(field queries_ms $log)" \
               "$(field weights_ms $log) $(dch $W/l${p}_qalsh_weights_A.bin $ex) [$st]" >> $OUT
        else
          echo "$ds $p $rule $rep $T >${CAP}000 - - - - -" >> $OUT
        fi
      done
    done
  done
done
rm -rf $W
echo "ablation done $(date +%H:%M)" >> $V/progress.log
