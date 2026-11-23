#!/usr/bin/env bash
# Continues a chain that was stopped after the cost phase: tops the cold QALSH4C runs up to $QALSH_RUNS per dataset and
# norm, summarizes the cost phase by means, and runs the remaining phases of chain.sh (randomness resumes from the runs
# already written).
set -u
D=$(dirname $(readlink -f $0))
source $D/common.sh
echo "resume start $(date +%H:%M)" >> $F/progress.log
bash $D/topup.sh
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
