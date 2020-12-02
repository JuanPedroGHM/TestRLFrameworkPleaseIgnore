TYPE=$1
EXPERIMENT=$2
ENV=$3

FILES=results/$EXPERIMENT/*
for i in $FILES
do
    f=$(basename $i)
    echo "Evaluating $EXPERIMENT $f"
    python -m trlfpi.eval.$TYPE $EXPERIMENT $f $ENV --plots
done
