#!/bin/bash
REPORT=$1

for x in {0..10}
do
    python -m trlfpi.experiments.reinforce $REPORT --episodes 200 --buffer_size 5000 --batch_size 512 --discount 0.7 --a_lr '1e-5' --nRefs 1 --aCost '1e-3' --weightDecay '1e-3' 
done
