EXPERIMENT=$1
ARGS=$2
VOLUME=/serverhome/ghm/code/TestRLFrameworkPleaseIgnore

docker run --rm -it \
            -v $VOLUME:/code/ \
            trlfpi \
            python -m trlfpi.$EXPERIMENT $ARGS
