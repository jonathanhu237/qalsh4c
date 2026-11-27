#!/usr/bin/env bash
# VLDB phase 2: scaling in |B| with A fixed, on nested prefixes of B. The exact scan once at 12 threads (faster than 24
# for the exact scan) and QALSH4C three times at 24 threads (cold), both norms. The question is whether the speedup
# grows with |B|, that is, whether a crude query visits a shrinking share of the m|B| projection entries.
set -u
source $(dirname $(readlink -f $0))/common.sh
OUT=$V/bscaling.txt
[ -s $OUT ] || echo "# dataset norm na nb method rep total_ms index_ms queries_ms weights_ms stats" > $OUT
for spec in "dbpedia 20000 12500 25000 50000 100000 200000" "trevi 20000 5000 10000 20000 40000 80801"; do
  set -- $spec; ds=$1; na=$2; shift 2
  for nb in "$@"; do
    d=$(subset_dir $ds $na $nb)
    cat $d/A.bin $d/B.bin > /dev/null
    for p in 1 2; do
      # A run directory may hold the exact rows of an earlier run (the exact scan does not depend on QALSH4C).
      grep -q "^$ds $p $na $nb exact " $OUT || {
        log=$V/logs/bscaling-$ds-$nb-$p-exact.log
        $BIN -T 12 estimate -p $p -d $d --direction ab ann linear_scan > $log 2>&1
        echo "$ds $p $na $nb exact 1 $(tc $log)" >> $OUT; }
      for rep in 1 2 3; do
        grep -q "^$ds $p $na $nb qalsh4c $rep " $OUT && continue
        W=$V/work-bscaling; rm -rf $W; mkdir -p $W; for f in A.bin B.bin metadata.json; do ln -s $d/$f $W/$f; done
        log=$V/logs/bscaling-$ds-$nb-$p-$rep-qalsh4c.log
        $BIN -T 24 --initial-radius $R0 estimate -p $p -d $W --direction ab sampling -n $(tval $p) --use-cache \
          qalsh -c 2 > $log 2>&1
        st=$(grep "QALSH stats" $log | sed "s/.*QALSH stats: //")
        echo "$ds $p $na $nb qalsh4c $rep $(tc $log) $(field index_ms $log) $(field queries_ms $log)" \
             "$(field weights_ms $log) [$st]" >> $OUT
      done
    done
  done
done
rm -rf $V/work-bscaling
echo "bscaling done $(date +%H:%M)" >> $V/progress.log
