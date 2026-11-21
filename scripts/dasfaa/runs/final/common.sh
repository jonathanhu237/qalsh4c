# Shared settings of the final DASFAA experiments (sourced by the phase scripts).
R=/home/jonathanhu237/qalsh4c-results/dasfaa-20261120
F=$R/final
BIN=~/code/qalsh4c-tmp/build/release-soa/qalsh4c
PY=$R/venv-hnsw/bin/python
SCRIPTS=~/code/qalsh4c-tmp/scripts/dasfaa
R0=1.401298464324817e-45
# Final QALSH4C parameters per norm: extra binary options, certified kappa and T(eps = 0.1).
QOPT_1=""; QOPT_2=""
KAPPA_1=2.136466; KAPPA_2=2.120719
T_1=1137; T_2=1121
LINEUP="dbpedia notredame p53 trevi gist enron"
declare -A DIR=( [dbpedia]=~/datasets/qalsh4c/dbpedia3072-b200k [notredame]=~/datasets/qalsh4c/notredame-a20k
                 [p53]=~/datasets/qalsh4c/p53-half [trevi]=~/datasets/qalsh4c/trevi-a20k
                 [gist]=/data/jonathanhu237/attic-20260915/datasets/qalsh4c/gist [enron]=~/datasets/qalsh4c/enron-a20k
                 [mnist]=~/datasets/qalsh4c/mnist-a20k [federalist]=$R/datasets/federalist )
mkdir -p $F/exact $F/logs
# Fresh working directory with symlinks, so no cached weights are reused.
fresh() { rm -rf "$2"; mkdir -p "$2"; for f in A.bin B.bin metadata.json; do ln -s "${DIR[$1]}/$f" "$2/$f"; done; }
tc() { grep "Time Consumed" "$1" | sed "s/.*Time Consumed: //; s/ ms//"; }
qopt() { if [ "$1" = 1 ]; then echo "$QOPT_1"; else echo "$QOPT_2"; fi; }
tval() { if [ "$1" = 1 ]; then echo "$T_1"; else echo "$T_2"; fi; }
kval() { if [ "$1" = 1 ]; then echo "$KAPPA_1"; else echo "$KAPPA_2"; fi; }
# Exact per-point distances: reuse earlier exact_nn outputs when the instance is the same.
exact_file() {
  local ds=$1 p=$2 out=$F/exact/${1}_l$2.bin
  if [ ! -s $out ]; then
    case $ds in
      dbpedia) src=$R/screen/exact/dbpedia3072-b200k_l$p.bin ;;
      notredame) src=$R/exact/notredame_l$p.bin ;;
      p53) src=$R/screen/exact/p53-half_l$p.bin ;;
      trevi) src=$R/screen/exact/trevi-a20k_l$p.bin ;;
      gist) src=$R/exact/gist_l$p.bin ;;
      enron) src=$R/screen/exact/enron-a20k_l$p.bin ;;
      mnist) src=$R/screen/exact/mnist-a20k_l$p.bin ;;
      federalist) src=$R/exact/federalist_l$p.bin ;;
    esac
    if [ -s "$src" ]; then cp "$src" $out; else $R/bin/exact_nn ${DIR[$ds]} $p $out > $F/logs/exact-$ds-$p.log 2>&1; fi
  fi
  echo $out
}
