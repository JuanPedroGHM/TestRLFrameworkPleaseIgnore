#!/bin/bash
EXPERIMENT=$1
REPORT=$2
ENV=$3

for nRefs {1..5}
do
    for x in {0..5}
    do
        python -m trlfpi.experiments.$EXPERIMENT $REPORT --env $ENV --episodes 500
    done
done

