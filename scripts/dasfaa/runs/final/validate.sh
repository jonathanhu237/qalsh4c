#!/usr/bin/env bash
# Follow-up checks after the GPT r3 review (2026-11-21):
# (a) time uniform and quadtree sampling directly at the modeled equal-time counts (and with T = 1 for the fixed cost),
# (b) end-to-end totals of the two stopping rules with their own certified T (same projections),
# (c) end-to-end totals for c = 1.5 and 3 with their own T, three runs each with new projections.
set -u
source $(dirname $0)/common.sh
OUT=$F/validate.txt
: > $OUT
# dataset norm budget_s T_uniform T_quadtree (modeled at QALSH4C's median time)
BUDGETS="dbpedia:1:7108:4197 dbpedia:2:16904:10706 enron:1:5992:3792 enron:2:5780:2183 gist:1:6547:4679 gist:2:7486:3776
notredame:1:4099:2478 notredame:2:4383:1013 p53:1:2654:186 p53:2:4408:0 trevi:1:3370:1834 trevi:2:4240:993"
for spec in $BUDGETS; do
  IFS=: read -r ds p tu tq <<< "$spec"
  cat ${DIR[$ds]}/A.bin ${DIR[$ds]}/B.bin > /dev/null
  for threads in 12 24; do
    for n in 1 $tu; do
      log=$F/logs/val-$ds-$p-uniform-$threads-$n.log
      $BIN -T $threads estimate -p $p -d ${DIR[$ds]} --direction ab sampling -n $n uniform > $log 2>&1
      echo "$ds $p uniform $threads $n $(tc $log)" >> $OUT
    done
  done
  if [ "$tq" != 0 ]; then
    for n in 1 $tq; do
      W=$F/work-val; fresh $ds $W
      log=$F/logs/val-$ds-$p-quadtree-$n.log
      $BIN -T 24 estimate -p $p -d $W --direction ab sampling -n $n --use-cache quadtree --max-levels 3 > $log 2>&1
      echo "$ds $p quadtree 24 $n $(tc $log)" >> $OUT
    done
  fi
done
echo "equal-time timing done $(date +%H:%M)" >> $F/progress.log
for ds in trevi gist enron; do for p in 1 2; do
  for rule in hybrid within; do
    if [ $rule = hybrid ]; then ro=""; t=$(tval $p); else ro="--qalsh-stop-within-round"; if [ $p = 1 ]; then t=3273; else t=3242; fi; fi
    W=$F/work-val; fresh $ds $W
    log=$F/logs/val-stop-$ds-$p-$rule.log
    $BIN --use-fixed-seed --initial-radius $R0 $ro estimate -p $p -d $W --direction ab sampling -n $t --use-cache qalsh -c 2 > $log 2>&1
    echo "$ds $p stop-$rule T=$t $(tc $log)" >> $OUT
  done
done; done
echo "stop totals done $(date +%H:%M)" >> $F/progress.log
for ds in trevi enron; do for p in 1 2; do for c in 1.5 3; do
  if [ $c = 1.5 ]; then if [ $p = 1 ]; then t=550; else t=551; fi; else if [ $p = 1 ]; then t=2301; else t=2052; fi; fi
  for rep in 1 2 3; do
    W=$F/work-val; fresh $ds $W
    log=$F/logs/val-c-$ds-$p-$c-$rep.log
    $BIN --initial-radius $R0 estimate -p $p -d $W --direction ab sampling -n $t --use-cache qalsh -c $c > $log 2>&1
    echo "$ds $p c=$c T=$t rep=$rep $(tc $log)" >> $OUT
  done
done; done; done
rm -rf $F/work-val
echo "validate done $(date +%H:%M)" >> $F/progress.log
