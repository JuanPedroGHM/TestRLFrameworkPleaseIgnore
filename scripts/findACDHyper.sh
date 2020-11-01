#!/bin/sh

for nRefs in 1 2 3
do
    for discount in 0.0 0.3 0.7 0.9
    do
        python -m trlfpi.experiments.refTrackingACD --episodes 500 --nRefs $nRefs --discount $discount
    done
done
