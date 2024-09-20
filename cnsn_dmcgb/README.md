# Normalization Enhances Generalization in Visual Reinforcement Learning

Our code for the DMControl Generalization Benchmark (DMC-GB) is built upon the codebase of DMC-GB. For more details of DMC-GB, please refer to the original DMC-GB repository: [[DMC-GB](https://github.com/nicklashansen/dmcontrol-generalization-benchmark/)]

**Setup**

All dependencies can be installed with the following commands:

```
conda env create -f setup/conda.yml

conda activate dmcgb

sh setup/install_envs.sh
```

**Training & Evaluation**

you can run the following command to train the agent with DrQ+CNSN on Walker Walk environment and evaluate on three test environments: color hard, video easy, video hard

```
bash ./train_drqcnsn.sh
```

You can easily modify the shell script according to tasks and settings. 