#!/bin/sh

for nRefs in 1
do
    for discount in 0.3 0.5 0.7 
    do
        for tau in 0.005 0.001
        do
            python -m trlfpi.experiments.AC --episodes 200 --nRefs $nRefs --discount $discount --tau $tau --c_lr 0.001 --a_lr 0.00005
        done
    done
done
