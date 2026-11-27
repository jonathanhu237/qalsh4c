# Shared settings of the VLDB experiments. They reuse the binary, the datasets, the certified T and the exact
# distances of the DASFAA final runs (scripts/dasfaa/runs/final/common.sh) and write to their own directory.
source $(dirname $(readlink -f ${BASH_SOURCE[0]}))/../dasfaa/runs/final/common.sh
# VLDB_V and VLDB_BIN select another run directory and binary (v2.sh: the rerun with the binary of 2026-11-25).
V=${VLDB_V:-/home/jonathanhu237/qalsh4c-results/vldb-20261124}
BIN=${VLDB_BIN:-$BIN}
# VLDB_KAPPA selects another table of certified T (v3.sh: the binary's own table for the default gamma = c^1.5).
KAPPA=${VLDB_KAPPA:-$F/kappa.txt}
tval() { grep "^${2:-main} norm=$1 c=${3:-2} epsilon=${4:-0.1} " $KAPPA | grep -o " T=[0-9]*" | cut -d= -f2; }
kval() { grep "^${2:-main} norm=$1 c=${3:-2} epsilon=0.1 " $KAPPA | grep -o "kappa_up=[0-9.]*" | cut -d= -f2; }
# Extra options of the round-end rule in the ablation (v2.sh may add --qalsh-mid-round-ratio 0).
ROUNDEND="$ROUNDEND ${VLDB_ROUNDEND_EXTRA:-}"
# A file roundend_extra in the run directory adds options too, so the rule can be chosen while a chain runs.
[ -s $V/roundend_extra ] && ROUNDEND="$ROUNDEND $(cat $V/roundend_extra)"
# Real collection pairs (Flickr30K and COCO, VGG-16 fc7, 4096-d) and the full DBpedia set.
DIR[flickr2coco]=~/datasets/qalsh4c/flickr2coco
DIR[coco2flickr]=~/datasets/qalsh4c/coco2flickr
DIR[dbpedia1m]=~/datasets/qalsh4c/dbpedia1m
VENV=/home/jonathanhu237/qalsh4c-results/dasfaa-20261120/venv-hnsw/bin/python
mkdir -p $V/logs $V/data
# A number such as index_ms=12.3 from a binary log.
field() { grep -o "$1=[0-9.]*" $2 | head -1 | cut -d= -f2; }
# A dataset directory with the first na source points and the first nb target points of a dataset (nested prefixes;
# each collection was shuffled once when it was built, so prefixes are random subsets).
subset_dir() {  # dataset na nb -> directory
  local base=${DIR[$1]} na=$2 nb=$3 d=$V/data/$1-A$2-B$3 dim
  if [ ! -s $d/metadata.json ]; then
    mkdir -p $d
    dim=$(python3 -c "import json;print(json.load(open('$base/metadata.json'))['num_dimensions'])")
    head -c $((na * dim * 4)) $base/A.bin > $d/A.bin
    head -c $((nb * dim * 4)) $base/B.bin > $d/B.bin
    python3 -c "
import json; m=json.load(open('$base/metadata.json')); m['num_points']={'a': $na, 'b': $nb}
m['chamfer_distances']={k: {'ab': None, 'ba': None} for k in m['chamfer_distances']}
json.dump(m, open('$d/metadata.json', 'w'), indent=4)"
  fi
  echo $d
}
