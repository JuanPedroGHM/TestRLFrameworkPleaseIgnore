#!/bin/sh

for nRefs in 1 2 3
do
    for discount in 0 0.3 0.7 0.9
    do
        for tau in 0.001 0.005 0.01 0.05 
        do
            python -m trlfpi.experiments.refTrackingAC --episodes 500 --nRefs $nRefs --discount $discount --tau $tau
        done
    done
done
