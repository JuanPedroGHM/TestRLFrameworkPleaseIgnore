#!/bin/sh
REPORT=$1

for nRefs in 1 2
do
    for tau in 0.001 0.0005
    do
        for aCost in 0.001 0.0001
        do
            python -m trlfpi.experiments.DPG $REPORT --episodes 100 --batch_size 1024 --c_lr 0.001 --a_lr 0.00005 --nRefs $nRefs --discount 0.7 --tau $tau --weightDecay 0.001 --aCost $aCost
        done
    done
done
