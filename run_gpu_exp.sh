GPU=$1
EXPERIMENT=$2
ARGS=$3
VOLUME=$(pwd)

docker run --rm -it \
            --gpus $GPU \
            -v $VOLUME:/code/ \
            trlfpi \
            python -m trlfpi.$EXPERIMENT $ARGS
