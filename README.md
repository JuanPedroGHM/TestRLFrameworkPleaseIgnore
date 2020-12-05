# TestRLFrameworkPleaseIgnore

## Setup

1) Install docker
2) Run 

```
docker build --no-cache -t trlfpi .
```

## Conda env

To work locally without docker, the following cmd can be used to create a conda env with all dependencies

```
conda env create -f environment.yml
```

## TestRLFrameworkPleaseIgnore cli

1) Create experiment file (see experiments/test.yml)

2) To train model (5 times of 1 by default)

    ```trlfpi -c experiments/test.yml train -n 5```

3) To test the model

    ```trlfpi -c experiments/test.yml test```

## Usage with docker

Different files are provided depending on the task.

1) Using docker compose (gpus not supported)

    ``` docker-compose run --rm trlfpi -c experiments/test.yml train -n 2  ```

1) Using run_trlfpi_gpu.sh to run on the gpu (if available):

    ``` ./run_trflfpi_gpu.sh 0 -c experiments/test.yml train -n 2 ```

## Experiments
