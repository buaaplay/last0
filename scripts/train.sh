cd /media/liuzhuoyang/clash-for-linux
bash start.sh
source /etc/profile.d/clash.sh
proxy_on
cd /media/chenhao/double_rl/scripts
source /media/miniconda3/bin/activate /media/miniconda3/envs/double_rl
export PATH=/media/miniconda3/envs/double_rl/bin:$PATH
export HF_HOME=/media/huggingface
export PYTHONPATH=/media/chenhao/double_rl:$PYTHONPATH
# export CUDA_VISIBLE_DEVICES=4,5,6,7

accelerate launch --config_file ../config/sft.yaml \
    --num_processes 8  \
    --num_machines 1 \
    --machine_rank 0 \
    --deepspeed_multinode_launcher standard train_janus_no_siglip.py \
    --model_path FreedomIntelligence/Janus-4o-7B \
    --data_path ../training_data/json/4tasks_train.json \
    --n_epochs 30 \
    --action_dim 7 \
    --train_bsz_per_gpu 1 \
    --learning_rate 1e-4 \
    --weight_decay 1e-4 \
    --gradient_accumulation_steps 8 \
    --output_dir ../exp \
    --log_dir ../exp \
    --experiment_name action_image \
    --run_name "janus_4o_lr_1e-4_weightdecay_1e-4" \


# FreedomIntelligence/Janus-4o-7B   deepseek-ai/Janus-Pro-7B