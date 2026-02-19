
export http_proxy=http://192.168.32.28:18000 && export https_proxy=http://192.168.32.28:18000 # for baidu

cd /media/liuzhuoyang/LCoT_VLA_MOT/scripts
source /media/miniconda3/bin/activate /media/miniconda3/envs/janus_cot
export PATH=/media/miniconda3/envs/janus_cot/bin:$PATH
export HF_HOME=/media/huggingFace
export PYTHONPATH=/media/liuzhuoyang/LCoT_VLA_MOT:$PYTHONPATH
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
# export NCCL_SOCKET_IFNAME=eth0

run_name="janus_pro_siglip_1B_1e-4_mot_s2_2view_latent4_chunk4_state1_1211"

accelerate launch --config_file ../config/sft.yaml \
    --num_processes 8  \
    --num_machines 1 \
    --machine_rank 0 \
    --deepspeed_multinode_launcher standard train_janus_siglip_ar_state.py \
    --model_path /media/liuzhuoyang/LCoT_VLA_MOT/exp/latent_cot_mot_2view/janus_pro_siglip_1B_1e-4_mot_s0_2view_latent4_chunk4_state1_1211/checkpoint-59-3420/tfmr \
    --data_path /media/liuzhuoyang/LCoT_VLA_MOT/training_data/json/4tasks_2view_chunk4_img+pc+state_train.json \
    --data_root "" \
    --n_epochs 100 \
    --action_dim 7 \
    --train_bsz_per_gpu 4 \
    --learning_rate 1e-4 \
    --min_lr_ratio 0 \
    --weight_decay 0 \
    --gradient_accumulation_steps 1 \
    --output_dir ../exp \
    --log_dir ../exp \
    --experiment_name latent_cot_mot_2view \
    --load_action_from_latent 1 \
    --freeze_latent 1 \
    --use_latent 1 \
    --latent_size 20 \
    --compress_strategy average \
    --run_name ${run_name} \

# FreedomIntelligence/Janus-4o-7B   deepseek-ai/Janus-Pro-7B

