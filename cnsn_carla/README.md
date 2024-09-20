# Normalization Enhances Generalization in Visual Reinforcement Learning

## Requirements
the simplest way to install all required dependencies is to create an anaconda environment by running:
```
conda env create -f conda_env.yml
```
or create the environment and pip install requirements.txt, and then install suitable versions of `torch` and `torchvision` based on your CUDA version

```
conda create  --n cnsn_carla python=3.6
pip install -r requirements.txt
### install torch and torchvision
```

After the installation ends you can activate your environment with:

```
conda activate cnsn_carla
```

## Instructions
Download CARLA from https://github.com/carla-simulator/carla/releases, e.g.:
1. https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.6.tar.gz

and merge the directories to CARLA_0.9.6 folder in our code.

Add to your python path like(change /path/to/location/):

```
export PYTHONPATH=$PYTHONPATH:/path/to/location/CARLA_0.9.6/PythonAPI
export PYTHONPATH=$PYTHONPATH:/path/to/location/CARLA_0.9.6/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:/path/to/location/CARLA_0.9.6/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg
```
Terminal 1:
```
cd CARLA_0.9.6
bash CarlaUE4.sh -fps 20 -carla-rpc-port=2000
```

Terminal 2:

```
cd CARLA_0.9.6
bash CarlaUE4.sh -fps 20 -carla-rpc-port=2002
```

Terminal 3:

```
bash ./carla_train_drqv2cnsn.sh
```

After completing the training phase, you are able to evaluate the model's performance across a total of 7 weather conditions. This includes the weather condition used during training as well as 6 additional, unseen weather conditions.

```
bash ./eval_weather_drqv2cnsn.sh
```

