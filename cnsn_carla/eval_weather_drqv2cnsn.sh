#!/bin/bash

DOMAIN=carla096
TASK=highway
SEED=5
DECODER_TYPE=identity
TRANSITION_MODEL=deterministic
WEATHER=WetCloudySunset
SAVEDIR=./save_${WEATHER}_drqv2_cn4sn_style/drqv2_cn4sn_seed${SEED}
mkdir -p ${SAVEDIR}
eval_weather_list=('WetCloudySunset' 'WetNoon' 'MidRainSunset' 'SoftRainNoon' 'HardRainSunset' 'MidRainyNoon' 'HardRainNoon')
for weather in ${eval_weather_list[@]}; do
    python eval.py \
        --domain_name ${DOMAIN} \
        --task_name ${TASK} \
        --agent drqv2_cnsn \
        --init_steps 100 \
        --num_eval_episodes 50 \
        --num_train_steps 200000 \
        --action_repeat 4 \
        --replay_buffer_capacity 100000 \
        --total_frames 10000 \
        --batch_size 128 \
        --work_dir ${SAVEDIR} \
        --seed ${SEED} $@ \
        --frame_stack 3 \
        --image_size 84 \
        --eval_freq 100 \
        --port 2000 \
        --weather ${weather} \
        --save_model >> ${SAVEDIR}/output_eval.txt 
done