<!--Batch size=424 (as given in paper)
microbatch=106-->
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=12233 \
    --use_env run_train.py \
    --diff_steps 2000 \
    --lr 0.0001 \
    --learning_steps 50000 \
    --save_interval 10000 \
    --seed 102 \
    --noise_schedule sqrt \
    --hidden_dim 128 \
    --bsz 424 \
    --microbatch 106 \
    --dataset qqp \
    --data_dir "datasets/QQP" \
    --vocab bert \
    --config_name bert-base-uncased \
    --seq_len 128 \
    --schedule_sampler lossaware \
    --use_fp16 \
    --notes qqp_timing_probe
