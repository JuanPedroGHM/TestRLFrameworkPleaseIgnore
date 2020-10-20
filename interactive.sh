GPU=$1
VOLUME=$(pwd)

docker run --rm -it \
            --gpus $GPU \
            -v $VOLUME:/code/ \
            trlfpi 
