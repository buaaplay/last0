source /media/miniconda3/bin/activate /media/miniconda3/envs/double_rl
export COPPELIASIM_ROOT=/media/Programs/CoppeliaSim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7  ## for our machine
export PATH=/media/miniconda3/envs/double_rl/bin:$PATH
export HF_HOME=/media/huggingface
export PYTHONPATH=/media/chenhao/double_rl/LIFT3D:/media/chenhao/double_rl:$PYTHONPATH

# export CUDA_VISIBLE_DEVICES=4,5,6,7

N=0
Xvfb :$N -screen 0 1024x768x24 &
export DISPLAY=:$N

models=("/media/chenhao/double_rl/exp/action_image/janus_pro_1e-4/checkpoint-29-6720/tfmr")
# tasks=("close_box" "close_laptop_lid")
# tasks=("toilet_seat_down" "sweep_to_dustpan")
# tasks=("close_fridge" "place_wine_at_rack_location")
# tasks=("water_plants" "phone_on_base")
# tasks=("take_umbrella_out_of_umbrella_stand" "take_frame_off_hanger")

tasks=("close_box" "close_laptop_lid" "sweep_to_dustpan" "phone_on_base")

for model in "${models[@]}"; do
  exp_name=$(echo "$model" | awk -F'/' '{print $(NF-3)"_"$(NF-2)"_"$(NF-1)}')
  for task in "${tasks[@]}"; do
    python /media/chenhao/double_rl/scripts/test_rlbench.py \
      --model-path ${model} \
      --task-name ${task} \
      --exp-name ${exp_name} \
      --replay-or-predict 'predict' \
      --result-dir ${model} \
      --cuda $N \
      --use_robot_state 1 \
      --max-steps 10 \
      --num-episodes 20 \
      --load-pointcloud 0 \
      --dataset-name 'rlbench' \
      --result-dir /media/chenhao/test_result/ch_test_0814 \
      --action-chunk 1
  done
done