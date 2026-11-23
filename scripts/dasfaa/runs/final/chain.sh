#!/usr/bin/env bash
# Final chain: cost (lineup and boundary datasets) -> full randomness -> summaries -> extras.
set -u
D=$(dirname $(readlink -f $0))
source $D/common.sh
echo "chain start $(date +%H:%M)" >> $F/progress.log
bash $D/cost.sh $LINEUP mnist federalist
python3 $SCRIPTS/summarize_cost.py $F --json $F/cost_summary.json > $F/cost_summary.txt 2>&1
mkdir -p $F/ds; for ds in $LINEUP mnist federalist; do ln -sfn ${DIR[$ds]} $F/ds/$ds; done
python3 $D/contrast.py $F/ds $F/exact $F/contrast.json > $F/contrast.txt 2>&1
echo "cost and contrast done $(date +%H:%M)" >> $F/progress.log
bash $D/randomness.sh
python3 $SCRIPTS/summarize_failure.py $F/randomness/*.jsonl > $F/randomness_summary.txt 2>&1
python3 $SCRIPTS/time_accuracy.py $F $F/cost_summary.json $F/time_accuracy.json $T_1 $T_2 > $F/time_accuracy.txt 2>&1
echo "summaries done $(date +%H:%M)" >> $F/progress.log
bash $D/extras.sh
echo "ALL DONE $(date +%H:%M)" >> $F/progress.log
