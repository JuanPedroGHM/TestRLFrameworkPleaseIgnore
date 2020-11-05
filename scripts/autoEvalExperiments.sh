TYPE=$1
EXPERIMENT=$2

FILES=results/$EXPERIMENT/*
for i in $FILES
do
    f=$(basename $i)
    echo "Evaluating $EXPERIMENT $f"
    python -m trlfpi.eval.$TYPE $EXPERIMENT $f --plots
done
