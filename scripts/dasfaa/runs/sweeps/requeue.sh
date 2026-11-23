#!/usr/bin/env bash
# After DBpedia l2 finishes, stop the screening (the remaining instances are not needed) and screen p53-half only.
R=/home/jonathanhu237/qalsh4c-results/dasfaa-20261120
until grep -q "dbpedia3072-b200k l2" $R/screen/screen.txt; do sleep 10; done
pkill -f "screen/screen.sh|screen\.sh enron" ; pkill -f "^bash ./screen.sh"; sleep 2
pkill -f "datasets/qalsh4c/gist-b200k"; pkill -f "w-gist-b200k"; sleep 1
cd $R/screen && ./screen.sh p53-half > screen-p53.nohup 2>&1
