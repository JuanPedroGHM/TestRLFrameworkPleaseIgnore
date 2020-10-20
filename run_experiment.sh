EXPERIMENT=$1
ARGS=$2
VOLUME=$(pwd)

docker run --rm -it \
            -v $VOLUME:/code/ \
            trlfpi \
            python -m trlfpi.$EXPERIMENT $ARGS
