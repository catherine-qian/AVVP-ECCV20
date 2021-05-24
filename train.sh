#!/bin/bash
#SBATCH --job-name=avvp
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --gres=gpu:1

/home/qian/anaconda3/envs/avvp/bin/python main_avvp.py --mode train --audio_dir feats/vggish/ --video_dir feats/res152/ --st_dir feats/r2plus1d_18/ > log/logtrain-`date +%F-%T`.txt


