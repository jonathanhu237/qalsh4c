#!/usr/bin/env bash
# Final chain: certified kappa and T -> cost (lineup and boundary datasets) -> HNSW summary -> full randomness ->
# summaries -> equal-time timing, stopping rules and c end to end -> extras (thread check, |A| scaling, c sweep).
set -u
D=$(dirname $(readlink -f $0))
source $D/common.sh
echo "chain start $(date +%H:%M)" >> $F/progress.log
# Inputs that do not depend on QALSH4C, from the runs of 2026-11-20/21: FAISS times, relative contrast, HNSW crude
# distances and graph timings.
[ -s $F/faiss.jsonl ] || cp $OLD/final/faiss.jsonl $F/faiss.jsonl
cp -n $OLD/final/contrast.json $OLD/final/contrast.txt $F/
mkdir -p $F/hnsw && cp -rn $OLD/final/hnsw/. $F/hnsw/
# Certified kappa and T of every rule and c used below, from the binary.
: > $F/kappa.txt
record_kappa main 2
record_kappa main 1.5
record_kappa main 3
record_kappa roundend 2 $ROUNDEND
record_kappa within 2 $WITHIN
echo "kappa done $(date +%H:%M)" >> $F/progress.log
bash $D/cost.sh $LINEUP mnist federalist
python3 $SCRIPTS/summarize_cost.py $F --json $F/cost_summary.json > $F/cost_summary.txt 2>&1
python3 $SCRIPTS/hnsw_summary.py $F $BIN gist=${DIR[gist]} notredame=${DIR[notredame]} > $F/hnsw_summary.txt 2>&1
echo "cost summary done $(date +%H:%M)" >> $F/progress.log
bash $D/randomness.sh
python3 $SCRIPTS/summarize_failure.py $F/randomness/*.jsonl > $F/randomness_summary.txt 2>&1
echo "randomness summary done $(date +%H:%M)" >> $F/progress.log
bash $D/validate.sh
python3 $D/recalib.py $F $BIN $(for ds in $LINEUP; do echo -n "$ds=${DIR[$ds]} "; done) > $F/recalib.txt 2>&1
echo "recalib done $(date +%H:%M)" >> $F/progress.log
bash $D/extras.sh
echo "ALL DONE $(date +%H:%M)" >> $F/progress.log
