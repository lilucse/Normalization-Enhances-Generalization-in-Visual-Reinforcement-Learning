#!/bin/bash

DOMAIN=carla096
TASK=highway
SEED=5
TRAIN_WEATHER=WetCloudySunset
SAVEDIR=./save_${TRAIN_WEATHER}_drqv2_cn4sn_style/drqv2_cn4sn_seed${SEED}
mkdir -p ${SAVEDIR}

python train.py \
    --domain_name ${DOMAIN} \
    --task_name ${TASK} \
    --agent drqv2_cnsn \
    --init_steps 100 \
    --num_train_steps 200000 \
    --action_repeat 4 \
    --replay_buffer_capacity 100000 \
    --total_frames 10000 \
    --batch_size 128 \
    --work_dir ${SAVEDIR} \
    --seed ${SEED} $@ \
    --frame_stack 3 \
    --image_size 84 \
    --eval_freq 2000 \
    --port 2000 \
    --act_cn 4 \
    --weather ${TRAIN_WEATHER} \
    --save_model >> ${SAVEDIR}/output.txt 
