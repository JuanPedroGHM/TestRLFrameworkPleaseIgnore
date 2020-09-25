EXPERIMENT=$1
ARGS=$2
GPU=$3
VOLUME=/serverhome/ghm/code/TestRLFrameworkPleaseIgnore

docker run --rm -it \
            --gpus $GPU \
            -v $VOLUME:/code/ \
            trlfpi \
            python -m trlfpi.$EXPERIMENT $ARGS
