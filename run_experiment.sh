EXPERIMENT=$2
ARGS=$3
GPU=$1
VOLUME=/serverhome/ghm/code/TestRLFrameworkPleaseIgnore

docker run --rm --gpus $GPU -v $VOLUME:/code/ trlfpi python $EXPERIMENT $ARGS
