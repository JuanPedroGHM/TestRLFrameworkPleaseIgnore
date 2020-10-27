#!/bin/sh

for nRefs in 1 2 3
do
    for aCost in 0 0.001 0.1 1
    do
        for WD in 0 0.001 0.1 1
        do
            python -m trlfpi.experiments.refTrackingPG --episodes 1000 --a_lr '1e-5' --plots --plot_freq 25 --nRefs $nRefs --aCost $aCost --weightDecay $WD
        done
    done
done
