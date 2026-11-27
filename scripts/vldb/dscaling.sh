#!/usr/bin/env bash
# VLDB phase: two-sided scaling on the full DBpedia set. A = 50,000 with B growing to 500,000 (dbpedia1m-scale
# prefixes) and then to about 950,000 (dbpedia1m); B = 500,000 with A growing to 500,000. The exact scan once at 12
# threads and QALSH4C three times (cold) at 24 threads, both norms.
set -u
source $(dirname $(readlink -f $0))/common.sh
DIR[dbpedia1m-scale]=~/datasets/qalsh4c/dbpedia1m-scale
OUT=$V/dscaling.txt
[ -s $OUT ] || echo "# dataset norm na nb method rep total_ms index_ms queries_ms weights_ms stats" > $OUT
point() {  # label dir na nb
  local label=$1 d=$2 na=$3 nb=$4
  cat $d/A.bin $d/B.bin > /dev/null
  for p in 1 2; do
    grep -q "^$label $p $na $nb exact " $OUT || {
      log=$V/logs/dscaling-$na-$nb-$p-exact.log
      $BIN -T 12 estimate -p $p -d $d --direction ab ann linear_scan > $log 2>&1
      echo "$label $p $na $nb exact 1 $(tc $log)" >> $OUT; }
    for rep in 1 2 3; do
      grep -q "^$label $p $na $nb qalsh4c $rep " $OUT && continue
      W=$V/work-dscaling; rm -rf $W; mkdir -p $W; for f in A.bin B.bin metadata.json; do ln -s $d/$f $W/$f; done
      log=$V/logs/dscaling-$na-$nb-$p-$rep-qalsh4c.log
      $BIN -T 24 --initial-radius $R0 estimate -p $p -d $W --direction ab sampling -n $(tval $p) --use-cache \
        qalsh -c 2 > $log 2>&1
      st=$(grep "QALSH stats" $log | sed "s/.*QALSH stats: //")
      echo "$label $p $na $nb qalsh4c $rep $(tc $log) $(field index_ms $log) $(field queries_ms $log)" \
           "$(field weights_ms $log) [$st]" >> $OUT
    done
  done
}
for nb in 62500 125000 250000 500000; do point dbpedia1m-scale $(subset_dir dbpedia1m-scale 50000 $nb) 50000 $nb; done
point dbpedia1m ${DIR[dbpedia1m]} 50000 $(python3 -c "import json;print(json.load(open('${DIR[dbpedia1m]}/metadata.json'))['num_points']['b'])")
for na in 12500 125000 250000 500000; do point dbpedia1m-scale $(subset_dir dbpedia1m-scale $na 500000) $na 500000; done
rm -rf $V/work-dscaling
echo "dscaling done $(date +%H:%M)" >> $V/progress.log
