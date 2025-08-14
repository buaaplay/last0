# cd /media/liuzhuoyang/clash-for-linux
# bash start.sh
# source /etc/profile.d/clash.sh
# proxy_on
export CUDA_VISIBLE_DEVICES=4,5,6,7
source /media/miniconda3/bin/activate /media/miniconda3/envs/double_rl
export PATH=/media/miniconda3/envs/double_rl/bin:$PATH
export HF_HOME=/media/huggingface
export PYTHONPATH=/media/chenhao/double_rl:$PYTHONPATH
cd /media/chenhao/double_rl/scripts
python test_toy.py