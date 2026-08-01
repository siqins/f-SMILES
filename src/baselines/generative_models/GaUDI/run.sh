#!/bin/bash

cd /home/ssq233/项目/PAS/diffusion
conda activate py311pt21

#run_file="train_edm.py"
run_file="cond_prediction/train_cond_predictor.py"

#nohup python -u $run_file > "train_edm.txt" 2>&1 &

nohup python -u $run_file > "train_cond_prediction.txt" 2>&1 &