EXPERIMENT=$2
ARGS=$3
GPU=$1
VOLUME=/serverhome/ghm/code/TestRLFrameworkPleaseIgnore

docker run --rm -it \
            --gpus $GPU \
            -v $VOLUME:/code/ \
            trlfpi \
            python -m trlfpi.$EXPERIMENT $ARGS
