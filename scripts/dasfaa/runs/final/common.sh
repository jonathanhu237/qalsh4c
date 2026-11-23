# Shared settings of the final DASFAA experiments (sourced by the phase scripts).
# QALSH-CrudeNN with the continuous front (the binary's default rule). Every estimate, the certified kappa and T come
# from the C++ binary; the scripts only run it and summarize. 2026-11-24.
R=/home/jonathanhu237/qalsh4c-results/dasfaa-20261124-cpp
F=$R/final
# The runs of 2026-11-20/21: their FAISS times, relative contrast and HNSW crude distances do not depend on QALSH4C
# and are reused.
OLD=/home/jonathanhu237/qalsh4c-results/dasfaa-20261120
BIN=$R/bin/qalsh4c
# The drivers of this run (a snapshot under $R/code, so later edits do not change a running chain).
SCRIPTS=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
R0=1.401298464324817e-45
# Extra binary options per norm (none: c = 2, beta = 1 and the continuous front are the defaults).
QOPT_1=""; QOPT_2=""
# The stopping rules compared in Sect. 4.2: the round rule with end-of-round tests (gamma = c) and QALSH's test on every
# candidate (gamma = c^2).
ROUNDEND="--qalsh-scan-order quanta --qalsh-stop-rule round --qalsh-advance-check false"
WITHIN="$ROUNDEND --qalsh-stop-within-round"
LINEUP="dbpedia notredame p53 trevi gist enron"
# Cold QALSH4C runs per dataset and norm in the cost phase; the summaries report their mean (and fastest and slowest).
QALSH_RUNS=20
declare -A DIR=( [dbpedia]=~/datasets/qalsh4c/dbpedia3072-b200k [notredame]=~/datasets/qalsh4c/notredame-a20k
                 [p53]=~/datasets/qalsh4c/p53-half [trevi]=~/datasets/qalsh4c/trevi-a20k
                 [gist]=/data/jonathanhu237/attic-20260915/datasets/qalsh4c/gist [enron]=~/datasets/qalsh4c/enron-a20k
                 [mnist]=~/datasets/qalsh4c/mnist-a20k [federalist]=$OLD/datasets/federalist )
mkdir -p $F/exact $F/logs
# Fresh working directory with symlinks, so no cached weights are reused.
fresh() { rm -rf "$2"; mkdir -p "$2"; for f in A.bin B.bin metadata.json; do ln -s "${DIR[$1]}/$f" "$2/$f"; done; }
tc() { grep "Time Consumed" "$1" | sed "s/.*Time Consumed: //; s/ ms//"; }
qopt() { if [ "$1" = 1 ]; then echo "$QOPT_1"; else echo "$QOPT_2"; fi; }
# Certified kappa and T from the binary, one line per rule label, norm, c and epsilon in $F/kappa.txt, for example
# "main norm=1 c=2 epsilon=0.1 T=417 kappa_up=1.416132 m=36 l=18".
record_kappa() {  # label c [global options of the rule]
  local label=$1 c=$2 p out k; shift 2
  for p in 1 2; do
    out=$($BIN --initial-radius $R0 "$@" kappa -p $p -c $c -e 0.3 0.1 0.01) || { echo "kappa failed $label $p $c" >> $F/progress.log; continue; }
    echo "$out" >> $F/logs/kappa.log
    k=$(echo "$out" | grep -o "kappa_up=[0-9.]*" | cut -d= -f2)
    ml=$(echo "$out" | grep -o " m=[0-9]* l=[0-9]*")
    echo "$out" | grep "^T " | while read -r _ e _ t; do echo "$label norm=$p c=$c $e $t kappa_up=$k$ml"; done >> $F/kappa.txt
  done
}
# T(eps) of a rule from kappa.txt: tval norm [label] [c] [eps].
tval() { grep "^${2:-main} norm=$1 c=${3:-2} epsilon=${4:-0.1} " $F/kappa.txt | grep -o " T=[0-9]*" | cut -d= -f2; }
kval() { grep "^${2:-main} norm=$1 c=${3:-2} epsilon=0.1 " $F/kappa.txt | grep -o "kappa_up=[0-9.]*" | cut -d= -f2; }
# D / CH of a weights file against the exact per-point distances.
dch() { python3 -c "import numpy as np; w=np.fromfile('$1',dtype=np.float32).astype(np.float64); e=np.fromfile('$2',dtype=np.float64); print('%.4f' % (w.sum()/e.sum()))"; }
# d(a, B) of every source point, written once by the binary's exact scan (--save-distances), for D / CH and for the
# experiments that repeat the estimator.
exact_file() {
  local out=$F/exact/${1}_l$2.bin
  [ -s $out ] || $BIN -T 24 estimate -p $2 -d ${DIR[$1]} --direction ab ann --save-distances $out linear_scan > $F/logs/exact-$1-$2.log 2>&1
  echo $out
}
