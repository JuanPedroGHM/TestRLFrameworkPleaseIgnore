#!/bin/sh
REPORT=$1

for nRefs in 1 2 3
do
    for discount in 0.3 0.7 
    do
        python -m trlfpi.experiments.MBACD $REPORT --episodes 500 --nRefs $nRefs --discount $discount
    done
done
