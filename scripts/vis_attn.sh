cd /media/liuzhuoyang/LCoT_VLA_MOT/scripts
source /media/miniconda3/bin/activate /media/miniconda3/envs/janus_cot

export PATH=/media/miniconda3/envs/janus_cot/bin:$PATH
export HF_HOME=/media/huggingFace
export PYTHONPATH=/media/liuzhuoyang/LCoT_VLA_MOT:$PYTHONPATH

python vis_attn.py \
  --model-path /media/liuzhuoyang/LCoT_VLA_MOT/exp/latent_cot_mot_flow/janus_pro_siglip_uni3d_1B_1e-4_mot_pretrain1220e5_12tasks_view1_sparse_slow1_fast4_pc_state_12_0102/stage2/checkpoint-399-60000/tfmr \
  --data-path /media/liuzhuoyang/data/rlbench/data_rlbench_npy/keyframe_delta_position_abs_euler_1024_4view_lcot_chunk4_img+pc+state_1209/for_rlds/place_wine_at_rack_location/episode7.npy \
  --output-dir /media/liuzhuoyang/LCoT_VLA_MOT/vis_results \
  --cuda 0