GPU=$1
VOLUME=$(pwd)

echo "GPU: ${GPU}"
echo "ARGS: ${@:2}"
docker run --rm -it \
            --gpus $GPU \
            -v $VOLUME:/code/ \
            trlfpi ${@:2}
            
