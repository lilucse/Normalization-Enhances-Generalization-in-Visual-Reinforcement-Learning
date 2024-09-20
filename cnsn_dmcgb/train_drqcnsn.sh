python src/train.py \
            --algorithm drq_cnsn \
            --seed 0 \
            --domain_name walker \
            --task_name walk \
            --train_steps 500k \
            --active_cn 5 \
            --eval_mode video_hard \
            --log_dir ./drq_cn5sn/ 