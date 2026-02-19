cd /media/liuzhuoyang/LCoT_VLA_MOT/scripts
source /media/miniconda3/bin/activate /media/miniconda3/envs/janus_cot
export COPPELIASIM_ROOT=/media/Programs/CoppeliaSim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

export PATH=/media/miniconda3/envs/janus_cot/bin:$PATH
export HF_HOME=/media/huggingFace
export PYTHONPATH=/media/liuzhuoyang/LCoT_VLA_MOT:$PYTHONPATH

models=("/media/liuzhuoyang/LCoT_VLA_MOT/exp/latent_cot_mot_flow/janus_pro_siglip_uni3d_1B_1e-4_mot_pretrain1220e1_franka_single_put_egg_view1+2_sparse_slow1_fast4_pc_state_12_1224/stage2/checkpoint-399-40000/tfmr")

for model in "${models[@]}"; do
  python /media/liuzhuoyang/LCoT_VLA_MOT/scripts/test_franka_siglip_flow_fastslow.py \
    --model-path ${model} \
    --data-path /media/liuzhuoyang/data/franka/npy/put_egg_on_bread_0829_keyframe/0829_133246.npy \
    --cuda 0 \
    --dataset-name 'rlbench' \
    --use_robot_state 0 \
    --use_latent 1 \
    --latent_size 12 \
    --compress_strategy average \
    --fs_ratio 4 \
    --action-chunk 1
done

