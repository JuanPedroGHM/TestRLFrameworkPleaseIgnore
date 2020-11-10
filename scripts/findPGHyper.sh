#!/bin/sh
REPORT=$1

for nRefs in 1 2 3
do
    for aCost in 0 0.001 0.1 1
    do
        for WD in 0 0.001 0.1 1
        do
            python -m trlfpi.experiments.reinforce $REPORT --episodes 500 --a_lr '1e-5' --nRefs $nRefs --aCost $aCost --weightDecay $WD
        done
    done
done
