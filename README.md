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

## Usage

Different files are provided depending on the task.

1) interactive.sh : If you want to open a shell on the container

2) run_experiment.sh : Runs a python module on the cpu, can be any file inside trlfpi dir

    ``` ./run_experiment.sh experiments.refTrackingPG '--nRefs 1 --plots' ```

3) run_gpu_exp.sh : Same as run_experiment.sh, but a gpu can be specified

    ``` ./run_gpu_exp.sh 0 experiments.refTrackingPG '--nRefs 1 --plots' ```

4) To run an experiment outside the docker container, use the following cmd format

    ``` python -m trlfpi.experiments.refTrackingPG --nRefs 1 --plots ```

## Experiments

#### refTrackingPG

#### refTrackingAC

#### refTrackingACD

#### refTrackingDGPQ
